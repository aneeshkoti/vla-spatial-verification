"""The evidence engine (parts 1-4: grounding, camera model, graded relations, scene verification -> G, S, D)
and the experiment helpers (episode set-up, the seven test conditions, analysis-only ground truth).
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# ============================================================================
# EVIDENCE ENGINE v3 — Part 1: config, helpers, robust grounding
# (final merged engine: replaces all earlier G/S/D/A code and all 'patch' cells)
# ============================================================================
import os, re, csv, json, time, math, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
warnings.filterwarnings("ignore", category=FutureWarning)   # silences the Grounding DINO 'labels' spam

IMG_H, IMG_W = 256, 256

CFG = {
    # ---- grounding ----
    "min_conf": [0.30, 0.25, 0.22],   # detection floor for re-ground attempt 0 / 1 / 2
    "text_threshold": 0.20,
    "conf_floor": 0.22,               # GDINO confidence -> calibrated [0,1]:  <=0.22 -> 0
    "conf_full": 0.40,                #                                         >=0.40 -> 1
    "max_area_frac": 0.25,            # boxes bigger than 25% of the image are not table-top objects
    "max_aspect": 3.0,                # very tall/thin boxes are usually the robot arm
    "nms_iou": 0.70,
    "same_box_iou": 0.85,             # a 'reference' box that IS the target box is rejected
    "max_candidates": 4,
    # ---- evidence score C = wG*G + wS*S + wD*D + wA*A ----
    "weights": (0.20, 0.30, 0.25, 0.25),
    "high": 0.75,
    "low": 0.50,
    # ---- action evidence ----
    "a_ema": 0.30,                    # smoothing of A across steps (one noisy step can't kill an episode)
    "a_small_motion": 0.10,           # |xyz action| below this = fine adjustment -> no new evidence (A kept)
    "a_above_radius": 0.07,           # m: gripper already above the object in xy -> any vertical motion OK
    "grasp_radius": 0.10,             # m
    "release_radius": 0.12,           # m
    "patience": 10,                   # mid-episode: consecutive low-C steps before ABSTAIN
    "resample_k": 3,                  # RE-GROUND mid-episode: extra SmolVLA samples, keep most consistent (0 = off)
    "max_re_ground": 2,               # RE-GROUND at verification: extra grounding attempts
    "strict_mid_band": False,         # True: unresolved RE-GROUND -> ABSTAIN; False -> EXECUTE best action
    "verbose": False,
}
CFG.update({
            "exclusive_iou": 0.6, "stack_iou": 0.85, "ambig_lo": 0.05, "ambig_hi": 0.30,
            "ground_min_conf": 0.22, "max_candidates": 6, "on_dz_max": (0.06, 0.08),     # 'on': bowl may be at most (half support height + 6 cm) above the measured support point
            "head_noun_scale": 0.9,          # 'bowl' detections count slightly less than 'black bowl'
            "contrast_iou": 0.70, "contrast_margin": 0.02, "max_ref_candidates": 4,
            # object vocabulary of the environment (used ONLY for the contrast check; edit for other scenes)
            "scene_vocab": ["black bowl", "plate", "ramekin", "cookie box", "wooden cabinet", "stove"]})
CFG.update({"mid_band_min_relation": 0.3,  # unresolved RE-GROUND at verification executes only if BOTH 2D and 3D support the relation
            "mid_band_min_unambiguity": 0.75,  # ... and the instruction is not ambiguous (two bowls fit equally well -> do not guess)
            "resample_below": 0.5,       # resample SmolVLA only if its action looks clearly wrong (A < 0.5) ...
            "resample_min_scene": 0.6})  # ... AND we trust the verified scene (normalised G,S,D >= 0.6)
STACK_RELATIONS = ("on", "in")


# ---------- soft (graded) comparison helpers: return values in [0,1] ----------
def ramp_down(x, lo, hi):
    """1 when x<=lo, 0 when x>=hi, linear in between."""
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return float((hi - x) / (hi - lo))

def ramp_up(x, lo, hi):
    """0 when x<=lo, 1 when x>=hi, linear in between."""
    return 1.0 - ramp_down(x, lo, hi)

def calibrate_conf(conf, cfg=None):
    cfg = cfg or CFG
    return ramp_up(float(conf), cfg["conf_floor"], cfg["conf_full"])

# ---------- box helpers ----------
def box_center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def box_wh(b):
    return (max(b[2] - b[0], 1e-3), max(b[3] - b[1], 1e-3))

def box_diag(b):
    w, h = box_wh(b)
    return math.hypot(w, h)

def frac_inside(inner, outer):
    """fraction of the INNER box's area that lies inside OUTER"""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    w, h = box_wh(inner)
    return inter / (w * h)

def box_iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = box_wh(a)[0] * box_wh(a)[1] + box_wh(b)[0] * box_wh(b)[1] - inter
    return inter / ua if ua > 0 else 0.0

# ---------- phrase handling ----------
SYNONYMS = {
    "black bowl":     ["black bowl", "dark bowl", "bowl"],
    "ramekin":        ["ramekin", "small white bowl", "white cup"],
    "plate":          ["plate", "white plate", "dish"],
    "cookie box":     ["cookie box", "box", "carton"],
    "wooden cabinet": ["wooden cabinet", "cabinet", "drawer"],
    "stove":          ["stove", "cooktop", "burner"],
}
TABLE_CENTER_NAMES = {"table center", "table centre", "center of the table", "centre of the table"}

def is_table_center(phrase):
    return phrase is not None and phrase.strip().lower() in TABLE_CENTER_NAMES

def phrase_variants(phrase, attempt):
    """attempt 0: the phrase itself; later attempts add known synonyms (re-grounding)."""
    base = phrase.strip().lower()
    variants = SYNONYMS.get(base, [base])   # unknown phrases are NOT loosened (avoids hallucinations)
    out = []
    for v in variants[: 1 + attempt]:
        if v not in out:
            out.append(v)
    return out

def clean_detections(dets, min_conf, allow_large=False, cfg=None):
    """Clip boxes, drop low-confidence / giant / arm-shaped boxes, NMS, sort by confidence."""
    cfg = cfg or CFG
    out = []
    for d in dets:
        x1, y1, x2, y2 = [float(v) for v in d["bbox"]]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(IMG_W), x2), min(float(IMG_H), y2)
        w, h = x2 - x1, y2 - y1
        if w < 2 or h < 2 or float(d["confidence"]) < min_conf:
            continue
        if not allow_large:
            if w * h > cfg["max_area_frac"] * IMG_W * IMG_H:
                continue
            if max(w / h, h / w) > cfg["max_aspect"]:
                continue
        out.append({**d, "bbox": [x1, y1, x2, y2], "confidence": float(d["confidence"])})
    out.sort(key=lambda d: d["confidence"], reverse=True)
    kept = []
    for d in out:
        if all(box_iou(d["bbox"], k["bbox"]) < cfg["nms_iou"] for k in kept):
            kept.append(d)
    return kept[: cfg["max_candidates"]]

def ground_phrase(image_pil, phrase, attempt=0, cfg=None, min_conf=None):
    """Grounding DINO over all phrase variants for this attempt -> cleaned candidate list."""
    cfg = cfg or CFG
    if min_conf is None:
        min_conf = cfg["min_conf"][min(attempt, len(cfg["min_conf"]) - 1)]
    dets = []
    for p in phrase_variants(phrase, attempt):
        for d in ground_object(image_pil, p, box_threshold=min_conf, text_threshold=cfg["text_threshold"]):
            dets.append({**d, "phrase_used": p})
    return clean_detections(dets, min_conf, allow_large=False, cfg=cfg)


# ---------- goal ("place it on the plate") parsing — the old parser ignored this half ----------
def parse_goal(instruction):
    text = instruction.lower().strip().rstrip(".")
    m = re.search(r"\b(?:place|put|set|move)\s+(?:it|them)\s+(.*)$", text)
    if not m:
        return {"object": None, "relation": None}
    rest = m.group(1)
    rel_phrase, rel = _find_relation(rest)
    if rel is None:
        return {"object": _clean(rest) or None, "relation": None}
    obj = rest[rest.find(rel_phrase) + len(rel_phrase):]
    obj = _clean(re.sub(r"^\s*(of|to)\s+", "", obj))
    return {"object": obj or None, "relation": rel}

print("Engine part 1 loaded (grounding model:", GDINO_MODEL_ID, ")")
for _i in ["pick up the black bowl on the ramekin and place it on the plate",
           "pick up the black bowl from table center and place it on the plate"]:
    print(" ", parse_goal(_i))


# ----------------------------------------------------------------------------------------------------------------------
# ============================================================================
# EVIDENCE ENGINE v3 — Part 2: camera model, image<->world conversion, self-calibration
# ----------------------------------------------------------------------------
# BUGS FIXED HERE:
#  * old pixel_to_3d returned CAMERA-frame coords (camera looks down at an angle,
#    so its 'y' mixes height and depth) -> 'above/below/left/right' were wrong,
#    and nothing could be compared with the robot's world-frame actions.
#  * old code looked up depth at ONE pixel (bbox center) -> often hit the table
#    or background (e.g. plate at 3.0 m). Now: median over the box's central patch.
#  * the image we give Grounding DINO is flipped [::-1, ::-1]; the flip convention
#    is now VERIFIED against the simulator (not assumed) by calibrate_camera().
# ============================================================================
CAMERA_FLIPS = None   # set once by calibrate_camera(); reused by every episode

class CameraModel:
    def __init__(self, env, camera_name="agentview", H=IMG_H, W=IMG_W):
        m, d = env.sim.model, env.sim.data
        cid = m.camera_name2id(camera_name)
        f = 0.5 * H / np.tan(np.deg2rad(float(m.cam_fovy[cid])) / 2.0)
        self.K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]])
        R_mj = np.array(d.cam_xmat[cid], dtype=float).reshape(3, 3)
        self.cam2world = np.eye(4)
        self.cam2world[:3, :3] = R_mj @ np.diag([1.0, -1.0, -1.0])   # MuJoCo cam (-z fwd, +y up) -> OpenCV (+z fwd, +y down)
        self.cam2world[:3, 3] = np.array(d.cam_xpos[cid], dtype=float)
        self.world2cam = np.linalg.inv(self.cam2world)
        self.H, self.W = H, W
        self.flip_u, self.flip_v = (CAMERA_FLIPS if CAMERA_FLIPS is not None else (True, False))

    # flipped-image coords (what GDINO sees) <-> true OpenCV pixel coords (an involution)
    def f_to_cv(self, u, v):
        return (self.W - u if self.flip_u else u, self.H - v if self.flip_v else v)

    def box_f_to_cv(self, b):
        x1, y1, x2, y2 = b
        if self.flip_u:
            x1, x2 = self.W - x2, self.W - x1
        if self.flip_v:
            y1, y2 = self.H - y2, self.H - y1
        return [x1, y1, x2, y2]

    def unproject(self, u_cv, v_cv, Z):
        X = (u_cv - self.K[0, 2]) * Z / self.K[0, 0]
        Y = (v_cv - self.K[1, 2]) * Z / self.K[1, 1]
        return (self.cam2world @ np.array([X, Y, Z, 1.0]))[:3]

    def project(self, p_world):
        pc = self.world2cam @ np.array([p_world[0], p_world[1], p_world[2], 1.0])
        if pc[2] <= 1e-6:
            return None
        return (self.K[0, 0] * pc[0] / pc[2] + self.K[0, 2], self.K[1, 1] * pc[1] / pc[2] + self.K[1, 2], pc[2])

    def world_to_f(self, p_world):
        pr = self.project(p_world)
        if pr is None:
            return None
        u, v = self.f_to_cv(pr[0], pr[1])
        return u, v, pr[2]

    def horizontal_axes(self):
        """camera 'right' and 'toward-scene' directions projected on the table plane (for left/right/front/behind)."""
        right = self.cam2world[:3, 0].copy(); right[2] = 0
        fwd = self.cam2world[:3, 2].copy(); fwd[2] = 0
        return right / (np.linalg.norm(right) + 1e-9), fwd / (np.linalg.norm(fwd) + 1e-9)


def depth_as_2d(depth_F):
    d = np.asarray(depth_F, dtype=float)
    return d[..., 0] if d.ndim == 3 else d

def box_depth(box_F, depth_F, shrink=0.4, max_valid=2.5):
    """Median depth of the central patch of a box (flipped-image coords)."""
    d2 = depth_as_2d(depth_F)
    H, W = d2.shape
    x1, y1, x2, y2 = box_F
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = max((x2 - x1) * shrink / 2.0, 1.0), max((y2 - y1) * shrink / 2.0, 1.0)
    c0, c1 = int(max(0, math.floor(cx - hw))), int(min(W, math.ceil(cx + hw)))
    r0, r1 = int(max(0, math.floor(cy - hh))), int(min(H, math.ceil(cy + hh)))
    patch = d2[r0:max(r1, r0 + 1), c0:max(c1, c0 + 1)]
    vals = patch[np.isfinite(patch) & (patch > 0) & (patch < max_valid)]
    if vals.size == 0:
        return float(d2[int(min(H - 1, max(0, cy))), int(min(W - 1, max(0, cx)))])
    return float(np.median(vals))

def measure_detection(det, depth_F, cam):
    """Detection (flipped-image box) -> record with calibrated confidence, true-view box, 3D world position, size."""
    box_F = [float(v) for v in det["bbox"]]
    box = cam.box_f_to_cv(box_F)
    uc, vc = box_center(box)
    Z = box_depth(box_F, depth_F)
    return {
        "phrase": det.get("phrase_used", det.get("label")),
        "box_F": box_F, "box": box, "center": (uc, vc),
        "conf": float(det["confidence"]), "g": calibrate_conf(det["confidence"]),
        "depth": Z, "pos": cam.unproject(uc, vc, Z),
        "size": float((box[2] - box[0]) * Z / cam.K[0, 0]),
        "height": float((box[3] - box[1]) * Z / cam.K[1, 1]),          # apparent height (m), used by 'on'
    }


def calibrate_camera(env, obs, inlier_tol=0.10, move_steps=12, verbose=True):
    """
    Finds how the flipped image relates to the real camera (4 flip hypotheses).
    Test 1 (decisive): move the gripper sideways (+y, then -y) and find WHERE in the depth image
            something new appeared; the right hypothesis projects the gripper's new position there.
            (A table scene looks almost the same mirrored, so a static check alone can be fooled.)
    Test 2 (sanity): simulator body positions projected into the depth map should match its depth.
    Uses ground truth ONLY for this one-time sensor calibration, never inside the evidence score.
    NOTE: this moves the robot -> call setup_episode() again afterwards.
    """
    global CAMERA_FLIPS
    cam = CameraModel(env)
    hyps = [(fu, fv) for fu in (False, True) for fv in (False, True)]

    # ---- test 1: motion ----
    moves = []
    prev = obs
    for direction, n in ((+1, move_steps), (-1, 2 * move_steps)):
        cur = prev
        for _ in range(n):
            cur, _, _, _ = env.step([0, 0.8 * direction, 0, 0, 0, 0, -1])
        d_prev = depth_as_2d(get_real_depth_map(env, prev["agentview_depth"]))
        d_cur = depth_as_2d(get_real_depth_map(env, cur["agentview_depth"]))
        appeared = (d_prev - d_cur) > 0.03                      # something is now closer to the camera
        if appeared.sum() >= 5:
            vv, uu = np.nonzero(appeared)
            moves.append((np.array([uu.mean() + 0.5, vv.mean() + 0.5]), np.asarray(cur["robot0_eef_pos"], float)))
        prev = cur
    motion_err = {}
    for h in hyps:
        cam.flip_u, cam.flip_v = h
        errs = []
        for centroid, eef in moves:
            r = cam.world_to_f(eef)
            if r is not None:
                errs.append(float(np.linalg.norm(np.array(r[:2]) - centroid)))
        motion_err[h] = float(np.mean(errs)) if errs else float("inf")

    # ---- test 2: static depth agreement ----
    depth_F = depth_as_2d(get_real_depth_map(env, obs["agentview_depth"]))
    pts = [np.array(p, dtype=float) for p in np.asarray(env.sim.data.body_xpos)[1:]]
    pts += [np.array(obs[k], dtype=float) for k in obs
            if k.endswith("_pos") and "_to_" not in k and np.asarray(obs[k]).shape == (3,)]
    static = {}
    for h in hyps:
        cam.flip_u, cam.flip_v = h
        errs = []
        for p in pts:
            r = cam.world_to_f(p)
            if r is None:
                continue
            iu, iv = int(math.floor(r[0])), int(math.floor(r[1]))
            if 2 <= iu < cam.W - 2 and 2 <= iv < cam.H - 2:
                errs.append(abs(depth_F[iv, iu] - r[2]))
        errs = np.array(errs)
        static[h] = (float(np.mean(errs < inlier_tol)) if errs.size else 0.0, float(np.median(errs)) if errs.size else float("inf"))

    best = min(hyps, key=lambda h: (motion_err[h], -static[h][0]))
    CAMERA_FLIPS = best
    if verbose:
        print("flip_u flip_v | motion_err(px)  static_inliers  static_median_err(m)")
        for h in hyps:
            print(f"{str(h[0]):6s} {str(h[1]):6s} | {motion_err[h]:14.1f}  {static[h][0]:14.2f}  {static[h][1]:18.3f}", "  <-- chosen" if h == best else "")
        if not moves:
            print("WARNING: the gripper motion was not visible in the depth image; calibration fell back to the static test.")
        elif motion_err[best] > 25:
            print("WARNING: best motion error > 25 px — calibration is uncertain; check the projection plot.")
    return best, {"motion_err": motion_err, "static": static}


def show_projection_check(env, obs, save_path=None):
    """Draws simulator object positions on the image with the calibrated convention: dots must sit on the objects."""
    cam = CameraModel(env)
    img = obs["agentview_image"][::-1, ::-1]
    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.imshow(img)
    for k in obs:
        if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot0") and np.asarray(obs[k]).shape == (3,):
            r = cam.world_to_f(obs[k])
            if r is not None:
                ax.plot(r[0], r[1], "r+", markersize=12, mew=2)
                ax.text(r[0] + 3, r[1] - 3, k.replace("_pos", ""), color="yellow", fontsize=7)
    ax.set_title(f"projection check (flip_u={cam.flip_u}, flip_v={cam.flip_v})")
    ax.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    plt.close(fig)

print("Engine part 2 loaded")


# ----------------------------------------------------------------------------------------------------------------------
# ============================================================================
# EVIDENCE ENGINE v3 — Part 3: GRADED spatial relations (2D image space and 3D world space)
# ----------------------------------------------------------------------------
# BUGS FIXED HERE:
#  * old S/D were 0/1 string matches of a single 'predicted relation' -> one pixel over a
#    threshold flipped evidence from 1 to 0. Now every relation returns a truth degree in [0,1].
#  * 'near' threshold was a fixed 60 px regardless of object size -> now scale-normalised
#    (2D: by box sizes, 3D: by the objects' physical sizes in metres).
#  * 'on' was treated as 'near'; now 'on' also requires horizontal support + height.
#  * 'between' had a separate hand-written 3D branch inside run_gated_step using camera x/y.
#  * 'at table center' could never work (Grounding DINO cannot box 'table center').
#  * 'on' = inside the support's footprint AND resting on its top (not 20 cm above it).
# ============================================================================
def _norm2(a):
    return float(math.hypot(a[0], a[1]))

def _between_score(p, a, b, perp_lo=0.25, perp_hi=0.60):
    """p between a and b (2D points/arrays): inside the segment AND close to the line."""
    p, a, b = np.asarray(p, float)[:2], np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return 0.0
    t = float(np.dot(p - a, ab) / (L * L))
    perp = float(abs(ab[0] * (p - a)[1] - ab[1] * (p - a)[0]) / L) / L
    inside = ramp_down(max(-t, t - 1.0), 0.0, 0.2)
    return inside * ramp_down(perp, perp_lo, perp_hi)

def relation_score_2d(rel, t, refs):
    """t, refs[i]: records with 'box' (true-view pixel box). Returns degree in [0,1]."""
    if t is None or not refs or any(r is None for r in refs):
        return 0.0
    r = refs[0]
    ct, cr = np.array(t["center"]), np.array(r["center"])
    scale = 0.5 * (box_diag(t["box"]) + box_diag(r["box"]))
    gap = float(np.linalg.norm(ct - cr)) / scale
    wt, ht = box_wh(t["box"])
    wr, hr = box_wh(r["box"])
    if rel == "between":
        return _between_score(ct, refs[0]["center"], refs[1]["center"]) if len(refs) >= 2 else 0.0
    if rel == "near":
        return ramp_down(gap, 1.0, 2.2)
    if rel == "on":                                            # FIX 1 (2D)
        horiz = ramp_down(abs(ct[0] - cr[0]) / (0.5 * wr), 0.7, 1.1)
        bottom = t["box"][3]                                   # bowl's bottom edge (image y grows down)
        out = max(r["box"][1] - bottom, bottom - r["box"][3], 0.0) / hr
        return horiz * ramp_down(out, 0.05, 0.35)
    if rel == "in":
        return max(ramp_up(frac_inside(t["box"], r["box"]), 0.3, 0.7), 0.8 * ramp_down(gap, 0.8, 2.0))
    if rel == "at":
        return ramp_down(float(np.linalg.norm(ct - cr)) / box_diag(t["box"]), 1.0, 3.0)
    if rel == "far":
        return ramp_up(gap, 2.5, 4.0)
    signed = {
        "left":   (cr[0] - ct[0]) / scale,
        "right":  (ct[0] - cr[0]) / scale,
        "above":  (cr[1] - ct[1]) / scale,
        "below":  (ct[1] - cr[1]) / scale,
        "front":  (ct[1] - cr[1]) / scale,
        "behind": (cr[1] - ct[1]) / scale,
    }.get(rel)
    return ramp_up(signed, 0.2, 0.8) if signed is not None else 0.0

def relation_score_3d(rel, t, refs, cam):
    """Same relations in the WORLD frame (metres, z up)."""
    if t is None or not refs or any(r is None for r in refs):
        return 0.0
    r = refs[0]
    pt, pr = np.asarray(t["pos"], float), np.asarray(r["pos"], float)
    h = _norm2(pt[:2] - pr[:2])
    dz = float(pt[2] - pr[2])
    st, sr = max(t["size"], 0.02), max(r["size"], 0.02)
    if rel == "between":
        return _between_score(pt, refs[0]["pos"], refs[1]["pos"]) if len(refs) >= 2 else 0.0
    if rel == "near":                                          # 'next to' = (almost) touching
        return ramp_down(h - 0.5 * (st + sr), 0.05, 0.14)
    if rel == "on":                                            # FIX 1 (3D): inside footprint AND raised
        inside = ramp_down(h, 0.5 * sr, 0.5 * sr + 0.04)
        margin, width = CFG["on_dz_max"]
        lo = 0.5 * max(r.get("height", 0.05), 0.02) + margin           # tall support (cabinet): its measured point is mid-front
        resting = ramp_up(dz, -0.02, 0.01) * ramp_down(dz, lo, lo + width)   # on its top, not far above it
        return inside * resting
    if rel == "in":
        return ramp_down(h, 0.5 * sr, 0.5 * sr + 0.10)
    if rel == "at":
        return ramp_down(h, 0.08, 0.25)
    if rel == "far":
        return ramp_up(h, 0.25, 0.40)
    scale = 0.5 * (st + sr) + 0.02
    right, fwd = cam.horizontal_axes()
    signed = {
        "left":   float(np.dot(pr - pt, right)) / scale,
        "right":  float(np.dot(pt - pr, right)) / scale,
        "front":  float(np.dot(pr - pt, fwd)) / scale,
        "behind": float(np.dot(pt - pr, fwd)) / scale,
        "above":  dz / scale,
        "below":  -dz / scale,
    }.get(rel)
    return ramp_up(signed, 0.2, 0.8) if signed is not None else 0.0

# quick self-test with made-up boxes (true-view pixel coords)
_t = {"box": [100, 100, 130, 125], "center": (115, 112.5)}
_r = {"box": [135, 100, 165, 125], "center": (150, 112.5)}
_far = {"box": [10, 200, 40, 225], "center": (25, 212.5)}
print("near (adjacent):", relation_score_2d("near", _t, [_r]), "| near (far away):", relation_score_2d("near", _far, [_r]))
print("left (adjacent):", relation_score_2d("left", _t, [_r]))


# ----------------------------------------------------------------------------------------------------------------------
# ============================================================================
# EVIDENCE ENGINE v3 — Part 4: scene verification -> G, S, D
# ----------------------------------------------------------------------------
# BUGS FIXED HERE:
#  * LIBERO-Spatial scenes contain TWO black bowls. The old code took the first/most
#    confident bowl and then checked the relation -> it often checked the WRONG bowl.
#    Now every bowl candidate is scored against the relation and the best one is chosen
#    (this is what the spatial relation in the instruction is for).
#  * the relation in "pick up the bowl ON THE RAMEKIN" describes where the bowl is
#    BEFORE it is picked. The old code re-checked it every step, so once the robot lifted
#    the bowl (i.e. did the right thing) S and D dropped to 0 and the episode ABSTAINED.
#    Now G/S/D are measured at the verification step (before the robot moves) and on RE-GROUND.
#  * relation evidence is weighted by how sure we are both objects were found
#    (a perfect relation between hallucinated boxes is not evidence).
#  * the goal object ('place it on the PLATE') is now grounded too; it feeds G and A.
#  * CONTRAST CHECK: a box only counts as 'stove' if Grounding DINO is not more confident it is
#    another scene object ('wooden cabinet') -> fixed cabinet->'stove', plate->'ramekin' confusions.
#  * the target is also searched by its head noun ('bowl'): the LIBERO bowls are silver-grey.
#  * (bowl, reference) pairs are chosen JOINTLY; ambiguity between two bowls lowers S and D.
# ============================================================================
import itertools

def _table_center_candidates(image_pil, attempt, cfg):
    min_conf = cfg["min_conf"][min(attempt, len(cfg["min_conf"]) - 1)]
    dets = [d for d in ground_object(image_pil, "table", box_threshold=min_conf, text_threshold=cfg["text_threshold"])]
    dets = clean_detections(dets, min_conf, allow_large=True, cfg=cfg)
    if not dets:   # fallback: image centre, low confidence
        return [{"bbox": [IMG_W / 2 - 4, IMG_H / 2 - 4, IMG_W / 2 + 4, IMG_H / 2 + 4], "confidence": cfg["conf_floor"] + 0.5 * (cfg["conf_full"] - cfg["conf_floor"]), "phrase_used": "image center"}]
    x1, y1, x2, y2 = dets[0]["bbox"]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [{"bbox": [cx - 4, cy - 4, cx + 4, cy + 4], "confidence": dets[0]["confidence"], "phrase_used": "table"}]

def _as_list(ref):
    if ref is None:
        return []
    return list(ref) if isinstance(ref, (list, tuple)) else [ref]

def _words(p):
    return set(p.lower().split())

def _pool(img, phrase, attempt, cfg, cache):
    """cleaned detections of one phrase (cached per image + attempt)."""
    key = (phrase, attempt)
    if key not in cache:
        cache[key] = ground_phrase(img, phrase, attempt, cfg, min_conf=cfg["ground_min_conf"])
    return cache[key]

def _contrast_filter(dets, phrase, competitors, img, cfg, cache):
    """FIX 5: drop a box if another scene object is detected on the SAME box more confidently."""
    keep = []
    for d in dets:
        rival = 0.0
        for q in competitors:
            for e in _pool(img, q, 0, cfg, cache):
                if box_iou(d["bbox"], e["bbox"]) > cfg["contrast_iou"]:
                    rival = max(rival, float(e["confidence"]))
        if rival <= float(d["confidence"]) + cfg["contrast_margin"]:
            keep.append(d)
    return keep

def _pair_ok(t, r, stacked, cfg):
    """may bowl t be paired with reference box r?"""
    iou = box_iou(r["box_F"], t["box_F"])
    if not stacked:
        return iou < cfg["exclusive_iou"]                            # next-to / between: different objects
    if iou < cfg["stack_iou"]:
        return True
    # (almost) the same box: only OK if the support is visible BELOW the bowl (hidden ramekin under a bowl)
    return (r["box"][3] - t["box"][3]) >= max(2.0, 0.05 * box_wh(t["box"])[1])

def perceive_scene(env, obs, parsed, goal, cam, attempt=0, cfg=None):
    cfg = cfg or CFG
    img = Image.fromarray(obs["agentview_image"][::-1, ::-1])
    depth_F = get_real_depth_map(env, obs["agentview_depth"])
    rel = parsed.get("relation")
    tphrase = parsed.get("target")
    ref_phrases = _as_list(parsed.get("reference"))
    gphrase = goal.get("object") if goal else None
    stacked = rel in STACK_RELATIONS
    cache = {}

    obj_phrases = [p for p in [tphrase] + ref_phrases + [gphrase] if p and not is_table_center(p)]
    vocab = list(dict.fromkeys(obj_phrases + list(cfg["scene_vocab"])))

    def competitors(p):
        out = []
        for q in vocab:
            if q == p or (_words(p) & _words(q)):
                continue                                             # same object type ('bowl' vs 'black bowl')
            if stacked and {p, q} <= set([tphrase] + ref_phrases):
                continue                                             # a stacked bowl+support box is legitimate
            out.append(q)
        return out

    def pool_for(p, with_head_noun=False):
        dets = list(_pool(img, p, attempt, cfg, cache))
        head = p.split()[-1]
        if with_head_noun and head != p:                             # FIX 4
            dets += [{**d, "confidence": d["confidence"] * cfg["head_noun_scale"], "phrase_used": head}
                     for d in _pool(img, head, attempt, cfg, cache)]
            dets = clean_detections(dets, 0.0, allow_large=False, cfg=cfg)
        dets = _contrast_filter(dets, p, competitors(p), img, cfg, cache)
        return [measure_detection(d, depth_F, cam) for d in dets]

    t_recs = pool_for(tphrase, with_head_noun=True) if tphrase else []
    ref_recs = {}
    for rp in ref_phrases:
        if is_table_center(rp):
            ref_recs[rp] = [measure_detection(d, depth_F, cam) for d in _table_center_candidates(img, attempt, cfg)]
        else:
            ref_recs[rp] = sorted(pool_for(rp), key=lambda r: r["conf"], reverse=True)[: cfg["max_ref_candidates"]]
    goal_recs = pool_for(gphrase) if gphrase else []

    # ---- FIX 6: choose (bowl, reference) pairs jointly by how well they satisfy the relation ----
    scored = []
    for t in t_recs:
        options = []
        for rp in ref_phrases:
            opts = [r for r in ref_recs[rp] if _pair_ok(t, r, stacked, cfg)]
            options.append(opts if opts else [None])
        best = None
        for combo in itertools.product(*options):
            combo = list(combo)
            if len(combo) == 2 and None not in combo and box_iou(combo[0]["box_F"], combo[1]["box_F"]) > cfg["exclusive_iou"]:
                continue                                             # 'between A and B': A and B must differ
            s_rel = relation_score_2d(rel, t, combo) if rel else 0.0
            d_rel = relation_score_3d(rel, t, combo, cam) if rel else 0.0
            w = float(np.mean([0.7 + 0.3 * r["g"] if r is not None else 0.0 for r in combo])) if combo else 0.0
            sc = 0.5 * (s_rel + d_rel) * w
            if best is None or sc > best[0]:
                best = (sc, combo, s_rel, d_rel)
        if best is None:
            best = (0.0, [None] * len(ref_phrases), 0.0, 0.0)
        scored.append((best[0], t, best[1], best[2], best[3]))
    scored.sort(key=lambda x: x[0] + 0.05 * x[1]["g"], reverse=True)

    ambig = 1.0
    if scored:
        best_rel, target, refs, S_rel, D_rel = scored[0]
        second = scored[1][0] if len(scored) > 1 else None
        if second is not None:                                        # FIX 3: ambiguity penalty
            ambig = 0.5 + 0.5 * ramp_up(best_rel - second, cfg["ambig_lo"], cfg["ambig_hi"])
        S_rel, D_rel = S_rel * ambig, D_rel * ambig
    else:
        target, S_rel, D_rel, second = None, 0.0, 0.0, None
        refs = [ref_recs[rp][0] if ref_recs[rp] else None for rp in ref_phrases]

    goal_rec = None
    gc = [g for g in goal_recs if target is None or box_iou(g["box_F"], target["box_F"]) < cfg["exclusive_iou"]]
    if gc:
        goal_rec = max(gc, key=lambda g: g["conf"])

    gs = [target["g"] if target else 0.0] + [r["g"] if r else 0.0 for r in refs]
    if gphrase:
        gs.append(goal_rec["g"] if goal_rec else 0.0)
    G = float(np.prod(gs) ** (1.0 / len(gs))) if gs else 0.0
    pair_g = min([target["g"] if target else 0.0] + [r["g"] if r else 0.0 for r in refs]) if refs else 0.0
    rel_trust = math.sqrt(max(pair_g, 0.0))
    S, D = S_rel * rel_trust, D_rel * rel_trust

    return {
        "G": round(G, 4), "S": round(S, 4), "D": round(D, 4),
        "S_rel": round(S_rel, 4), "D_rel": round(D_rel, 4), "pair_g": round(pair_g, 4),
        "target": target, "refs": refs, "goal": goal_rec, "relation": rel,
        "n_target_candidates": len(t_recs), "second_best": second, "ambiguity": round(ambig, 3),
        "candidates": [{"rel": round(x[0], 3), "s": round(x[3], 3), "d": round(x[4], 3), "conf": round(x[1]["conf"], 3),
                        "phrase": x[1]["phrase"], "box": [round(v) for v in x[1]["box_F"]],
                        "world": np.round(x[1]["pos"], 3).tolist()} for x in scored],
        "attempt": attempt,
    }


def scene_summary(scene):
    def fmt(r):
        if r is None:
            return "NOT FOUND"
        return f"conf={r['conf']:.2f} box={[round(v) for v in r['box_F']]} world={np.round(r['pos'], 3).tolist()}"
    lines = [f"G={scene['G']:.3f}  S={scene['S']:.3f} (rel {scene['S_rel']:.2f})  D={scene['D']:.3f} (rel {scene['D_rel']:.2f})  relation={scene['relation']}  attempt={scene['attempt']}",
             f"  target : {fmt(scene['target'])}   ({scene['n_target_candidates']} candidates)"]
    for i, r in enumerate(scene["refs"]):
        lines.append(f"  ref[{i}] : {fmt(r)}")
    lines.append(f"  goal   : {fmt(scene['goal'])}")
    return "\n".join(lines)

print("Engine part 4 loaded")


# ----------------------------------------------------------------------------------------------------------------------
# ============================================================================
# EXPERIMENT HELPERS — episode setup, scene/visual perturbations (Task 26), analysis-only checks
# Ground-truth simulator positions are used ONLY to (a) build perturbations and (b) grade/analyse
# results afterwards. They are NEVER used by the evidence engine or the gate.
# ============================================================================
V3_DIR = os.path.join(RESULTS_ROOT, "v3")          # all new results go here
BASELINE_CLEAN_CSV = os.path.join(RESULTS_ROOT, "v2", "baseline_clean.csv")   # reused (baseline never changes)
QUAL_DIR = os.path.join(V3_DIR, "qualitative")
for _d in (V3_DIR, QUAL_DIR):
    os.makedirs(_d, exist_ok=True)

OBS_HOOK = None          # optional function applied to every observation (used for the occlusion perturbation)

def apply_obs_hook(obs):
    return OBS_HOOK(obs) if OBS_HOOK is not None else obs

def load_init_states(task_id):
    try:
        return suite.get_task_init_states(task_id)
    except Exception:
        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        try:
            return suite.get_task_init_states(task_id)
        finally:
            torch.load = _orig

# ---------- analysis-only ground truth ----------
def _true_keys(env, obs):
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

def _object_keys(obs):
    return [k for k in obs if k.endswith("_pos") and "_to_" not in k and not k.startswith("robot0")
            and np.asarray(obs[k]).shape == (3,)]

def _ref_key(phrase, obs):
    """simulator key for a reference phrase, e.g. 'cookie box' -> 'cookies_1_pos'."""
    if phrase is None or is_table_center(phrase):
        return None
    for w in [w for w in phrase.lower().split() if len(w) > 3][::-1]:
        for k in _object_keys(obs):
            if w.rstrip("s") in k:
                return k
    return None

def _contains(box, uv, pad=0.0):
    return uv is not None and box[0] - pad <= uv[0] <= box[2] + pad and box[1] - pad <= uv[1] <= box[3] + pad

# ---------- scene perturbations (Task 26) ----------
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

def free_spot(obs, exclude_key, far_from=None, far_min=0.20, env=None, z=None):
    far_list = [] if far_from is None else ([far_from] if np.asarray(far_from).ndim == 1 else list(far_from))
    """table position far from every other object (and >= far_min from far_from), visible to the camera."""
    vis = None
    if env is not None and z is not None:
        cam = CameraModel(env)
        dF = depth_as_2d(get_real_depth_map(env, obs["agentview_depth"]))
        def vis(p):
            r = cam.world_to_f([p[0], p[1], z + 0.02])
            if r is None or not (4 <= r[0] < IMG_W - 4 and 4 <= r[1] < IMG_H - 4):
                return False
            return abs(dF[int(r[1]), int(r[0])] - r[2]) < 0.06        # nothing in front of that spot
    keys = [k for k in _object_keys(obs) if k != exclude_key]
    pts = np.array([np.asarray(obs[k], float)[:2] for k in keys])
    allp = np.array([np.asarray(obs[k], float)[:2] for k in _object_keys(obs)])
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

def table_height(obs):
    for k in _object_keys(obs):
        if "plate" in k:
            return float(np.asarray(obs[k])[2])
    return float(min(np.asarray(obs[k])[2] for k in _object_keys(obs)))

def make_occlusion_hook(box_F, color=(128, 128, 128)):
    x1, y1, x2, y2 = [int(round(v)) for v in box_F]
    def hook(obs):
        obs = dict(obs)
        img_F = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).copy()
        img_F[max(0, y1):min(IMG_H, y2), max(0, x1):min(IMG_W, x2)] = color
        obs["agentview_image"] = np.ascontiguousarray(img_F[::-1, ::-1])   # flip is its own inverse
        return obs
    return hook

SCENE_PERTURBATIONS = ("relation_violation", "distractor", "position_change", "occlusion")
INVALID_KINDS = ("relation_violation", "absent_reference", "absent_target")   # instruction no longer matches the scene

def apply_scene_perturbation(env, obs, kind, instruction):
    """Task 26 perturbations. Returns (obs, info). Deterministic, so baseline and gated see the same scene."""
    global OBS_HOOK
    tkey, gkey = _true_keys(env, obs)
    parsed = parse_instruction(instruction)
    rkeys = [k for k in (_ref_key(rp, obs) for rp in _as_list(parsed.get("reference"))) if k]
    info = {"kind": kind}
    tz = table_height(obs)
    if kind == "relation_violation":              # move the target away: the instruction's relation is now false
        p = free_spot(obs, tkey, far_from=[obs[tkey]] + [obs[k] for k in rkeys], far_min=0.22, env=env, z=tz)
        move_object(env, tkey, [p[0], p[1], tz + 0.03]); info["moved"] = tkey
    elif kind == "position_change":               # move the goal plate: relation still true, layout changed
        p = free_spot(obs, gkey, far_from=obs[gkey], far_min=0.15, env=env, z=tz)
        move_object(env, gkey, [p[0], p[1], tz + 0.01]); info["moved"] = gkey
    elif kind == "distractor":                    # put the other bowl right next to the reference -> ambiguity
        others = [k for k in _object_keys(obs) if "bowl" in k and k != tkey]
        if others:
            anchor = np.asarray(obs[rkeys[0]] if rkeys else obs[tkey], float)
            d = anchor[:2] - np.asarray(obs[tkey], float)[:2]
            d = d / (np.linalg.norm(d) + 1e-9) if np.linalg.norm(d) > 1e-3 else np.array([0.0, 1.0])
            p = anchor[:2] + 0.13 * d
            move_object(env, others[0], [p[0], p[1], tz + 0.03]); info["moved"] = others[0]
    elif kind == "occlusion":                     # grey patch over the reference (or target) in the camera image
        cam = CameraModel(env)
        k = rkeys[0] if rkeys else tkey
        c = np.asarray(obs[k], float) + np.array([0, 0, 0.03])
        r = cam.world_to_f(c)
        if r is not None:
            half = cam.K[0, 0] * 0.035 / r[2]
            OBS_HOOK = make_occlusion_hook([r[0] - half, r[1] - half, r[0] + half, r[1] + half])
            info["occluded"] = k
    else:
        raise ValueError(kind)
    for _ in range(15):                           # let moved objects settle
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return apply_obs_hook(obs), info

def setup_episode(env, task_id, init_idx, seed=None, perturb=None, instruction=None):
    """Identical start for baseline and gated runs: official LIBERO init state + 10 settle steps
    (+ optional scene perturbation) + fixed RNG seed."""
    global OBS_HOOK
    OBS_HOOK = None
    seed = 1000 * task_id + init_idx if seed is None else seed
    env.seed(seed)
    env.reset()
    init_states = load_init_states(task_id)
    obs = env.set_init_state(init_states[init_idx % len(init_states)])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    env.perturb_info = {"kind": "clean"}
    if perturb in SCENE_PERTURBATIONS:
        obs, env.perturb_info = apply_scene_perturbation(env, obs, perturb, instruction or suite.get_task(task_id).language)
    policy.reset()
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    return obs

# ---------- instruction perturbations (Task 26) ----------
ABSENT_OBJECTS = ["red mug", "banana", "blue teapot"]
REL_PHRASE = {"on": "on the {}", "near": "next to the {}", "in": "in the top drawer of the {}", "at": "next to the {}",
              "between": "between the {} and the {}"}

def perturb_instruction(instruction, kind, k=0):
    p, goal = parse_instruction(instruction), parse_goal(instruction)
    absent = ABSENT_OBJECTS[k % len(ABSENT_OBJECTS)]
    refs = _as_list(p["reference"])
    goal_txt = goal["object"] or "plate"
    if kind == "absent_target":
        rel_txt = REL_PHRASE[p["relation"]].format(*refs) if p["relation"] != "at" else "from table center"
        return f"pick up the {absent} {rel_txt} and place it on the {goal_txt}"
    if kind == "absent_reference":
        if p["relation"] == "between":
            return f"pick up the {p['target']} " + REL_PHRASE["between"].format(absent, refs[1]) + f" and place it on the {goal_txt}"
        return f"pick up the {p['target']} " + REL_PHRASE.get(p["relation"], "next to the {}").format(absent) + f" and place it on the {goal_txt}"
    raise ValueError(kind)

print("experiment helpers loaded; results ->", V3_DIR)
