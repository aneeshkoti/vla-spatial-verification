"""LIBERO-Spatial helpers: environments, episode set-up, the seven test conditions and analysis-only ground truth.

Simulator positions are used ONLY to build the perturbations and to grade results afterwards. The gate never sees them.
"""
import random

import numpy as np

from .config import IMG_H, IMG_W
from .geometry import CameraModel, depth_as_2d, mujoco_depth_to_meters
from .parsing import as_list, is_table_center, parse_goal, parse_instruction

MAX_STEPS = 220                             # SmolVLA episode length used in all experiments
NUM_TASKS = 10
KINDS = ["clean", "absent_reference", "absent_target", "relation_violation", "distractor", "position_change", "occlusion"]
VALID_KINDS = ["clean", "distractor", "position_change", "occlusion"]
INVALID_KINDS = ["absent_reference", "absent_target", "relation_violation"]       # the instruction no longer matches the scene
SCENE_PERTURBATIONS = ("relation_violation", "distractor", "position_change", "occlusion")
ABSENT_OBJECTS = ["red mug", "banana", "blue teapot"]
REL_PHRASE = {"on": "on the {}", "near": "next to the {}", "in": "in the top drawer of the {}", "at": "next to the {}",
              "between": "between the {} and the {}"}

_SUITE = None


def libero_suite():
    """the LIBERO-Spatial benchmark (10 tasks, 50 initial states each)"""
    global _SUITE
    if _SUITE is None:
        from libero.libero import benchmark
        _SUITE = benchmark.get_benchmark_dict()["libero_spatial"]()
    return _SUITE


def make_env(task_id):
    """-> (env, task, instruction); 256 x 256 camera images with depth"""
    import os
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = libero_suite().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True)
    return env, task, task.language


def load_init_states(task_id):
    import torch
    suite = libero_suite()
    try:
        return suite.get_task_init_states(task_id)
    except Exception:
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        try:
            return suite.get_task_init_states(task_id)
        finally:
            torch.load = _orig


def apply_obs_hook(env, obs):
    """the occlusion condition paints a grey patch into every camera image (set up by setup_episode)"""
    hook = getattr(env, "obs_hook", None)
    return hook(obs) if hook is not None else obs


# ---------- analysis-only ground truth ----------
def true_keys(env, obs):
    """target bowl + goal object keys, read from the task's goal in the BDDL file (fallback: bowl_1 / plate_1)."""
    tgt, goal = None, None
    try:
        for pred in env.env.parsed_problem["goal_state"]:
            if len(pred) >= 3:
                tgt, goal = pred[1] + "_pos", pred[2] + "_pos"
    except Exception:
        pass
    keys = [k for k in obs if k.endswith("_pos") and "_to_" not in k]
    if tgt not in obs:
        tgt = next((k for k in keys if "bowl_1" in k), None)
    if goal not in obs:
        goal = next((k for k in keys if "plate" in k), None)
    return tgt, goal


def object_keys(obs):
    return [k for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot0")
            and np.asarray(obs[k]).shape == (3,)]


def ref_key(phrase, obs):
    """simulator key for a reference phrase, e.g. 'cookie box' -> 'cookies_1_pos'."""
    if phrase is None or is_table_center(phrase):
        return None
    for w in [w for w in phrase.lower().split() if len(w) > 3][::-1]:
        for k in object_keys(obs):
            if w.rstrip("s") in k:
                return k
    return None


def contains(box, uv, pad=0.0):
    return uv is not None and box[0] - pad <= uv[0] <= box[2] + pad and box[1] - pad <= uv[1] <= box[3] + pad


# ---------- scene perturbations ----------
def _joint_of(env, name):
    try:
        return env.env.objects_dict[name].joints[0]
    except Exception:
        m = env.sim.model
        names = [m.joint_id2name(i) for i in range(m.njnt)]
        cand = [n for n in names if n and n.startswith(name)]
        return cand[0] if cand else None


def move_object(env, key, xyz):
    """teleport a free object (obs key like 'plate_1_pos') to world position xyz."""
    name = key[:-4]
    j = _joint_of(env, name)
    if j is None:
        raise RuntimeError(f"no joint found for {name}")
    q = np.array(env.sim.data.get_joint_qpos(j), dtype=float).copy()
    q[:3] = xyz
    env.sim.data.set_joint_qpos(j, q)
    try:
        env.sim.data.set_joint_qvel(j, np.zeros(6))
    except Exception:
        pass
    env.sim.forward()


def _clearance(key):
    return 0.20 if ("cabinet" in key or "stove" in key) else 0.11


def _free_spot_once(obs, exclude_key, far_from=None, far_min=0.20, env=None, z=None):
    """table position far from every other object (and >= far_min from far_from), visible to the camera."""
    far_list = [] if far_from is None else ([far_from] if np.asarray(far_from).ndim == 1 else list(far_from))
    vis = None
    if env is not None and z is not None:
        cam = CameraModel.from_mujoco(env)
        dF = depth_as_2d(mujoco_depth_to_meters(env, obs["agentview_depth"]))

        def vis(p):
            r = cam.world_to_f([p[0], p[1], z + 0.02])
            if r is None or not (4 <= r[0] < IMG_W - 4 and 4 <= r[1] < IMG_H - 4):
                return False
            return abs(dF[int(r[1]), int(r[0])] - r[2]) < 0.06        # nothing in front of that spot
    keys = [k for k in object_keys(obs) if k != exclude_key]
    pts = np.array([np.asarray(obs[k], float)[:2] for k in keys])
    allp = np.array([np.asarray(obs[k], float)[:2] for k in object_keys(obs)])
    lo, hi = allp.min(0) - 0.05, allp.max(0) + 0.05
    best, best_score = None, -1e9
    for x in np.arange(lo[0], hi[0] + 1e-6, 0.02):
        for y in np.arange(lo[1], hi[1] + 1e-6, 0.02):
            p = np.array([x, y])
            margins = [np.linalg.norm(p - q) - _clearance(k) for k, q in zip(keys, pts)]
            score = min(margins) if margins else 1.0
            if any(np.linalg.norm(p - np.asarray(f)[:2]) < far_min for f in far_list):
                continue
            if vis is not None and not vis(p):
                continue
            if score > best_score:
                best, best_score = p, score
    return best


def free_spot(obs, exclude_key, far_from=None, far_min=0.20, env=None, z=None):
    """free_spot with the fallbacks of fix v3.1 (relaxes the distance, then the visibility test, if no spot is found)"""
    for fm, use_vis in ((far_min, True), (0.75 * far_min, True), (0.75 * far_min, False), (0.10, False)):
        p = _free_spot_once(obs, exclude_key, far_from, fm, env if use_vis else None, z if use_vis else None)
        if p is not None:
            return p
    return _free_spot_once(obs, exclude_key, None, 0.0, None, None)


def table_height(obs):
    for k in object_keys(obs):
        if "plate" in k:
            return float(np.asarray(obs[k])[2])
    return float(min(np.asarray(obs[k])[2] for k in object_keys(obs)))


def make_occlusion_hook(box_F, color=(128, 128, 128)):
    x1, y1, x2, y2 = [int(round(v)) for v in box_F]

    def hook(obs):
        obs = dict(obs)
        img_F = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy()
        img_F[max(0, y1):min(IMG_H, y2), max(0, x1):min(IMG_W, x2)] = color
        obs["agentview_image"] = np.ascontiguousarray(img_F[::-1, ::-1])   # the flip is its own inverse
        return obs
    return hook


def apply_scene_perturbation(env, obs, kind, instruction):
    """Returns (obs, info). Deterministic, so the policy alone and the gated system see the same scene."""
    tkey, gkey = true_keys(env, obs)
    parsed = parse_instruction(instruction)
    rkeys = [k for k in (ref_key(rp, obs) for rp in as_list(parsed.get("reference"))) if k]
    info = {"kind": kind}
    tz = table_height(obs)
    if kind == "relation_violation":              # move the target away: the instruction's relation is now false
        p = free_spot(obs, tkey, far_from=[obs[tkey]] + [obs[k] for k in rkeys], far_min=0.22, env=env, z=tz)
        move_object(env, tkey, [p[0], p[1], tz + 0.03]); info["moved"] = tkey
    elif kind == "position_change":               # move the goal plate: relation still true, layout changed
        p = free_spot(obs, gkey, far_from=obs[gkey], far_min=0.15, env=env, z=tz)
        move_object(env, gkey, [p[0], p[1], tz + 0.01]); info["moved"] = gkey
    elif kind == "distractor":                    # put the other bowl right next to the reference -> ambiguity
        others = [k for k in object_keys(obs) if "bowl" in k and k != tkey]
        if others:
            anchor = np.asarray(obs[rkeys[0]] if rkeys else obs[tkey], float)
            d = anchor[:2] - np.asarray(obs[tkey], float)[:2]
            d = d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 1e-3 else np.array([0.0, 1.0])
            p = anchor[:2] + 0.13 * d
            move_object(env, others[0], [p[0], p[1], tz + 0.03]); info["moved"] = others[0]
    elif kind == "occlusion":                     # grey patch over the reference (or target) in the camera image
        cam = CameraModel.from_mujoco(env)
        k = rkeys[0] if rkeys else tkey
        c = np.asarray(obs[k], float) + np.array([0, 0, 0.03])
        r = cam.world_to_f(c)
        if r is not None:
            half = cam.K[0, 0] * 0.035 / r[2]
            env.obs_hook = make_occlusion_hook([r[0] - half, r[1] - half, r[0] + half, r[1] + half])
            info["occluded"] = k
    else:
        raise ValueError(kind)
    for _ in range(15):                           # let moved objects settle
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return apply_obs_hook(env, obs), info


def setup_episode(env, task_id, init_idx, seed=None, perturb=None, instruction=None, policy=None):
    """Identical start for every system: official LIBERO initial state + 10 settle steps (+ optional scene perturbation)
    + fixed random seeds (1000 * task + state). policy: anything with .reset() (reset before the episode)."""
    env.obs_hook = None
    seed = 1000 * task_id + init_idx if seed is None else seed
    env.seed(seed)
    env.reset()
    init_states = load_init_states(task_id)
    obs = env.set_init_state(init_states[init_idx % len(init_states)])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    env.perturb_info = {"kind": "clean"}
    if perturb in SCENE_PERTURBATIONS:
        obs, env.perturb_info = apply_scene_perturbation(env, obs, perturb, instruction or libero_suite().get_task(task_id).language)
    if policy is not None:
        policy.reset()
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    np.random.seed(seed); random.seed(seed)
    return obs


# ---------- instruction perturbations ----------
def perturb_instruction(instruction, kind, k=0):
    """'absent_target' / 'absent_reference': replace an object by one that is not on the table (red mug, banana, blue teapot)"""
    p, goal = parse_instruction(instruction), parse_goal(instruction)
    absent = ABSENT_OBJECTS[k % len(ABSENT_OBJECTS)]
    refs = as_list(p["reference"])
    goal_txt = goal["object"] or "plate"
    if kind == "absent_target":
        rel_txt = REL_PHRASE[p["relation"]].format(*refs) if p["relation"] != "at" else "from table center"
        return f"pick up the {absent} {rel_txt} and place it on the {goal_txt}"
    if kind == "absent_reference":
        if p["relation"] == "between":
            return f"pick up the {p['target']} " + REL_PHRASE["between"].format(absent, refs[1]) + f" and place it on the {goal_txt}"
        return f"pick up the {p['target']} " + REL_PHRASE.get(p["relation"], "next to the {}").format(absent) + f" and place it on the {goal_txt}"
    raise ValueError(kind)


def make_case(env, task_id, state, kind, policy=None):
    """one test case of the paper: the scene and the instruction for a task, an initial state and a condition.
    Returns (obs, instruction)."""
    clean_instr = libero_suite().get_task(task_id).language
    instr = perturb_instruction(clean_instr, kind, state) if kind in ("absent_reference", "absent_target") else clean_instr
    obs = setup_episode(env, task_id, state, perturb=kind if kind in SCENE_PERTURBATIONS else None, instruction=instr, policy=policy)
    return obs, instr
