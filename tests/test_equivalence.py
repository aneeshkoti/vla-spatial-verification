"""The clean package (vla_gate/) must give exactly the same answers as the experiment code that produced the paper's numbers
(experiments/setup/). Both are run on the same synthetic scenes with the same fake detectors, and every evidence value,
every chosen box and every decision is compared for equality.

Run:  python tests/test_equivalence.py      (or: pytest tests)
Needs only numpy, pandas, Pillow and matplotlib - no simulator, no models, no GPU.
"""
import ast
import contextlib
import copy
import io
import json
import os
import shutil
import sys
import tempfile
import types

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SETUP = os.path.join(ROOT, "experiments", "setup")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from synthetic import EXTRA_INSTRUCTIONS, LIBERO_INSTRUCTIONS, Scene  # noqa: E402

from vla_gate import config as pcfg  # noqa: E402
from vla_gate import decision as pdec  # noqa: E402
from vla_gate import sim as psim  # noqa: E402
from vla_gate import stats as pstats  # noqa: E402
from vla_gate.detectors import HybridDetector  # noqa: E402
from vla_gate.evidence import perceive_scene as p_perceive  # noqa: E402
from vla_gate.gate import SpatialGate  # noqa: E402
from vla_gate.geometry import CameraModel, mujoco_depth_to_meters  # noqa: E402
from vla_gate.parsing import parse_goal, parse_instruction  # noqa: E402
from vla_gate.policies import failure_reason  # noqa: E402

CURRENT = {"scene": None}


def fake_gdino(image_pil, phrase, box_threshold=0.35, text_threshold=0.25):
    return CURRENT["scene"].gdino(image_pil, phrase, box_threshold, text_threshold)


def fake_owl(image_pil, phrase, box_threshold=0.10, text_threshold=None):
    return CURRENT["scene"].owl(image_pil, phrase, box_threshold, text_threshold)


_NS = None


def original_code():
    """executes the experiment set-up scripts (the paper's code) with fake detectors; returns their namespace"""
    global _NS
    if _NS is not None:
        return _NS
    tmp = tempfile.mkdtemp(prefix="vla_gate_test_")
    os.makedirs(os.path.join(tmp, "v4"))
    shutil.copy(os.path.join(ROOT, "configs", "frozen_config.json"), os.path.join(tmp, "v4", "frozen_config.json"))
    ns = {"__name__": "experiment_code", "RESULTS_ROOT": tmp, "GDINO_MODEL_ID": "fake", "MAX_STEPS": 220, "device": "cpu",
          "ground_object": fake_gdino}
    # from s1 only the parser and the depth conversion (the rest of s1 loads SmolVLA, LIBERO and Grounding DINO)
    s1 = os.path.join(SETUP, "s1_models_and_simulator.py")
    tree = ast.parse(open(s1).read())
    wanted = {"RELATION_KEYWORDS", "SORTED_RELATIONS", "_find_relation", "_clean", "parse_instruction", "get_real_depth_map"}
    body = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in wanted)
            or (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in wanted for t in n.targets))
            or (isinstance(n, ast.Import) and n.names[0].name == "re")]
    assert len(body) == 7, [getattr(n, "name", None) for n in body]
    sys.modules.setdefault("transformers", types.SimpleNamespace(Owlv2Processor=None, Owlv2ForObjectDetection=None))
    ns["_owl_mdl"] = ns["_owl_proc"] = object()            # skip loading OWLv2 (replaced by the fake below)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(ast.Module(body, []), s1, "exec"), ns)
        for f in ("s2_engine.py", "s4_engine_fixes.py", "s5_policy_helpers.py", "s6_study1_setup.py", "s7_study2_setup.py",
                  "s8_two_policies_and_judge_setup.py"):
            p = os.path.join(SETUP, f)
            exec(compile(open(p).read(), p, "exec"), ns)
    ns["ground_object_owl"] = fake_owl
    assert ns["_GO_REAL"] is fake_gdino
    _NS = ns
    return ns


def original_gate(ns, scene, instr, mode):
    """exactly what gate_case7 + gate_decide do in the experiments (sweeps and the live attached run)"""
    CURRENT["scene"] = scene
    env, obs = scene.env, scene.obs
    parsed, goal = ns["parse_instruction"](instr), ns["parse_goal"](instr)
    cam = ns["CameraModel"](env)
    save = {k: copy.deepcopy(ns["CFG"][k]) for k in ns["_CFG_KEYS6"]}
    try:
        ns["ground_object"] = ns["DETECTORS7"][mode]["fn"]
        ns["_pool"] = ns["_POOL_REAL6"]
        if ns["DETECTORS7"][mode]["ramp"] == "owl":
            O = ns["OWL_RAMP"]
            ns["CFG"].update({"min_conf": [O[0]] * 3, "ground_min_conf": O[0], "conf_floor": O[0], "conf_full": O[1],
                              "scene_vocab": list(ns["HYB_SCENE_VOCAB"])})
        row = {"has_goal": int(bool(goal and goal.get("object")))}
        scenes = {}
        for sfx, att in (("0", 0), ("1", 1)):
            sc = ns["perceive_scene"](env, obs, parsed, goal, cam, att, ns["CFG"])
            row.update(ns["_scene_fields"](sc, 1.0, sfx))
            scenes[sfx] = sc
    finally:
        ns["ground_object"] = ns["_GO_REAL"]
        ns["_pool"] = ns["_POOL_REAL6"]
        ns["CFG"].update(save)
    cfg = ns["gate_cfg7"](ns["DETECTORS7"][mode]["ramp"], True)
    return row, scenes, ns["gate_decide"](row, cfg), cfg


def same_record(a, b):
    if a is None or b is None:
        return a is None and b is None
    for k in ("phrase", "box_F", "box", "center", "conf", "g", "depth", "size", "height"):
        if a[k] != b[k] and not (isinstance(a[k], (list, tuple)) and list(a[k]) == list(b[k])):
            return False
    return np.array_equal(a["pos"], b["pos"])


def same_scene(a, b):
    for k in ("G", "S", "D", "S_rel", "D_rel", "pair_g", "relation", "n_target_candidates", "second_best", "ambiguity",
              "candidates", "attempt"):
        if a[k] != b[k]:
            return False, k
    if not same_record(a["target"], b["target"]):
        return False, "target"
    if not same_record(a["goal"], b["goal"]):
        return False, "goal"
    if len(a["refs"]) != len(b["refs"]) or not all(same_record(x, y) for x, y in zip(a["refs"], b["refs"])):
        return False, "refs"
    return True, ""


# ----------------------------------------------------------------------------------------------------------- tests
def test_engine_settings_match():
    ns = original_code()
    for k, v in pcfg.ENGINE_CFG.items():
        assert ns["CFG"][k] == v, (k, ns["CFG"][k], v)
    assert tuple(ns["OWL_RAMP"]) == pcfg.OWL_RAMP and ns["GD_PHRASES"] == pcfg.GD_PHRASES and ns["GD_FALLBACK"] == pcfg.GD_FALLBACK
    for mode in ("gdino", "hybrid"):
        a, b = ns["gate_cfg7"]("owl" if mode == "hybrid" else "gdino", True), pcfg.gate_cfg(mode)
        for k in ("weights", "high", "low", "ramp", "reground", "mid_rule", "mid_min_rel", "mid_min_unamb"):
            assert tuple(a[k]) == tuple(b[k]) if isinstance(a[k], (list, tuple)) else a[k] == b[k], (mode, k)


def test_parser_and_instruction_changes():
    ns = original_code()
    for instr in LIBERO_INSTRUCTIONS + EXTRA_INSTRUCTIONS:
        assert ns["parse_instruction"](instr) == parse_instruction(instr), instr
        assert ns["parse_goal"](instr) == parse_goal(instr), instr
    for instr in LIBERO_INSTRUCTIONS:
        for kind in ("absent_target", "absent_reference"):
            for k in range(5):
                assert ns["perturb_instruction"](instr, kind, k) == psim.perturb_instruction(instr, kind, k)


def test_gate_decisions_match(n_scenes=40, verbose=False):
    ns = original_code()
    n, counts = 0, {"EXECUTE": 0, "ABSTAIN": 0, "regrounded": 0}
    instructions = LIBERO_INSTRUCTIONS + EXTRA_INSTRUCTIONS
    instructions += [psim.perturb_instruction(i, k, 1) for i in LIBERO_INSTRUCTIONS[:5] for k in ("absent_target", "absent_reference")]
    gates = {"gdino": SpatialGate(fake_gdino, "gdino"), "hybrid": SpatialGate(HybridDetector(fake_owl, fake_gdino), "hybrid")}
    for seed in range(n_scenes):
        scene = Scene(seed)
        for instr in instructions:
            for mode, gate in gates.items():
                row, scenes, dec, cfg = original_gate(ns, scene, instr, mode)
                CURRENT["scene"] = scene
                res = gate.check_libero(scene.env, scene.obs, instr)
                assert (res.decision, res.C, res.C_first, int(res.regrounded)) == dec[:4], (seed, instr, mode, res, dec)
                ok, k = same_scene(res.scene, scenes[dec[4]])
                assert ok, (seed, instr, mode, "scene used", k)
                v = ns["scene_vals"](row, dec[4], tuple(cfg["ramp"]))
                assert (res.G, res.S, res.D, res.S_rel, res.D_rel, res.ambiguity) == (v["G"], v["S"], v["D"], v["S_rel"], v["D_rel"], v["amb"])
                for sfx in res.scenes:                       # the package skips pass 1 only when the decision does not use it
                    ok, k = same_scene(res.scenes[sfx], scenes[sfx])
                    assert ok, (seed, instr, mode, sfx, k)
                if "1" not in res.scenes:
                    assert dec[3] == 0
                # pass 1 also on its own (the logs keep both passes)
                img_F = __import__("PIL.Image", fromlist=["Image"]).fromarray(scene.obs["agentview_image"][::-1, ::-1])
                sc1 = p_perceive(gate.detector, img_F, mujoco_depth_to_meters(scene.env, scene.obs["agentview_depth"]),
                                 CameraModel.from_mujoco(scene.env), parse_instruction(instr), parse_goal(instr), 1, gate.engine)
                ok, k = same_scene(sc1, scenes["1"])
                assert ok, (seed, instr, mode, "pass 1", k)
                n += 1
                counts[res.decision] += 1
                counts["regrounded"] += int(res.regrounded)
    if verbose:
        print(f"   {n} gate checks identical ({counts['EXECUTE']} EXECUTE, {counts['ABSTAIN']} STOP, {counts['regrounded']} re-grounded)")
    assert counts["EXECUTE"] > 0 and counts["ABSTAIN"] > 0 and counts["regrounded"] > 0


def random_rows(n, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        r = {"task_id": i % 10, "state": int(rng.integers(0, 5)), "kind": rng.choice(psim.KINDS), "has_goal": int(rng.random() < 0.9)}
        r["valid"] = int(r["kind"] in psim.VALID_KINDS)
        for sfx in ("0", "1"):
            nref = int(rng.integers(0, 3))
            r.update({f"G_{sfx}": 0.0, f"S_{sfx}": 0.0, f"D_{sfx}": 0.0,
                      f"S_rel_{sfx}": float(rng.choice([0.0, rng.random(), 1.0])), f"D_rel_{sfx}": float(rng.choice([0.0, rng.random(), 1.0])),
                      f"amb_{sfx}": float(rng.choice([1.0, 0.5 + 0.5 * rng.random()])), f"A_{sfx}": 1.0,
                      f"t_conf_{sfx}": None if rng.random() < 0.1 else round(float(rng.uniform(0.05, 0.6)), 4),
                      f"ref_confs_{sfx}": json.dumps([None if rng.random() < 0.1 else round(float(rng.uniform(0.05, 0.6)), 4) for _ in range(nref)]),
                      f"goal_conf_{sfx}": None if rng.random() < 0.1 else round(float(rng.uniform(0.05, 0.6)), 4)})
        rows.append(r)
    return pd.DataFrame(rows)


def test_offline_decisions_and_statistics_match():
    ns = original_code()
    df = random_rows(600)
    for mode in ("gdino", "hybrid"):
        for reground in (True, False):
            cfg = pcfg.gate_cfg(mode, reground)
            a, b = ns["decide_all"](df, cfg), pdec.decide_all(df, cfg)
            pd.testing.assert_frame_equal(a, b)
            assert ns["summarize"](a) == pdec.summarize(b)
            bsm = {(t, s): int((t + s) % 2) for t in range(10) for s in range(5)}
            assert ns["summarize"](a, bsm) == pdec.summarize(b, bsm)
    rng = np.random.default_rng(1)
    for _ in range(300):
        k, n = int(rng.integers(0, 50)), int(rng.integers(50, 120))
        assert ns["wilson"](k, n) == pstats.wilson(k, n)
        b_, c_ = int(rng.integers(0, 40)), int(rng.integers(0, 40))
        assert ns["mcnemar_exact"](b_, c_) == pstats.mcnemar_exact(b_, c_)
        a4 = [int(x) for x in rng.integers(0, 30, 4)]
        assert ns["fisher_exact7"](*a4) == pstats.fisher_exact(*a4)
        pos, neg = rng.random(int(rng.integers(1, 30))), rng.random(int(rng.integers(1, 30)))
        assert ns["auroc"](pos, neg) == pstats.auroc(pos, neg)
    assert ns["failure_reason"]({"success": 0, "other_lifted": "plate_1", "target_lifted": 0}) == failure_reason(
        {"success": 0, "other_lifted": "plate_1", "target_lifted": 0})


def test_simulator_helpers_match():
    ns = original_code()
    obs = {"agentview_image": np.random.default_rng(0).integers(0, 255, (256, 256, 3), dtype=np.uint8),
           "akita_black_bowl_1_pos": np.zeros(3), "akita_black_bowl_2_pos": np.ones(3), "plate_1_pos": np.ones(3),
           "cookies_1_pos": np.ones(3), "glazed_rim_porcelain_ramekin_1_pos": np.ones(3), "robot0_eef_pos": np.ones(3)}
    for ph in ("plate", "cookie box", "ramekin", "black bowl", "wooden cabinet", "table center", "red mug"):
        assert ns["_ref_key"](ph, obs) == psim.ref_key(ph, obs), ph
    a = ns["make_occlusion_hook"]([10.3, 20.7, 60.2, 80.9])(obs)["agentview_image"]
    b = psim.make_occlusion_hook([10.3, 20.7, 60.2, 80.9])(obs)["agentview_image"]
    assert np.array_equal(a, b)


def test_frozen_files():
    assert pcfg.frozen_hash(pcfg.FROZEN_STUDY1_JSON) == "a4a4d6846a"
    assert pcfg.frozen_hash(pcfg.FROZEN_HYBRID_JSON) == "a5f5591cec"


if __name__ == "__main__":
    for name, fn in [("frozen config files (hashes a4a4d6846a, a5f5591cec)", test_frozen_files),
                     ("perception settings and gate settings", test_engine_settings_match),
                     ("instruction parser and the impossible instructions", test_parser_and_instruction_changes),
                     ("offline decisions and statistics", test_offline_decisions_and_statistics_match),
                     ("simulator helpers", test_simulator_helpers_match),
                     ("live gate: evidence, boxes and decisions on synthetic scenes", lambda: test_gate_decisions_match(verbose=True))]:
        fn()
        print("OK  ", name)
    print("package and experiment code agree")
