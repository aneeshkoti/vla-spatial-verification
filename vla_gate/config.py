"""Settings of the verification layer.

All values are the ones used in the paper. ENGINE_CFG holds the perception settings (grounding, relations);
the gate settings (weights, thresholds) are read from the two frozen files in configs/, which were written
before the test scenes were touched.
"""
import copy
import json
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
FROZEN_STUDY1_JSON = os.path.join(CONFIG_DIR, "frozen_config.json")       # Study 1: weights, thresholds, ramp
FROZEN_HYBRID_JSON = os.path.join(CONFIG_DIR, "frozen_config_v6.json")    # Study 2: hybrid detector settings

IMG_H, IMG_W = 256, 256                     # LIBERO camera images used by the gate

# Perception settings. These are the final values of the notebook's CFG dictionary (engine part 1, fixes v3.1 / v3.2
# and the Study 1 setup). Keys that only mattered for the earlier step-by-step gating are left out.
ENGINE_CFG = {
    # grounding
    "min_conf": [0.30, 0.25, 0.22],         # detection floor for grounding attempt 0 / 1 / 2 (table-centre search)
    "ground_min_conf": 0.22,                # detection floor used for every named object
    "text_threshold": 0.20,
    "conf_floor": 0.22,                     # detector confidence -> calibrated g in [0, 1]: <= floor -> 0
    "conf_full": 0.40,                      #                                               >= full  -> 1
    "max_area_frac": 0.25,                  # boxes bigger than 25% of the image are not table-top objects
    "max_aspect": 3.0,                      # very tall / thin boxes are usually the robot arm
    "nms_iou": 0.70,
    "max_candidates": 6,
    "use_head_noun": False,                 # also search the target by its last word ("bowl"): off in the final method
    "head_noun_scale": 0.9,
    "scene_vocab": [],                      # extra object names for the contrast check (off: only the instruction's objects)
    "contrast_iou": 0.70,
    "contrast_margin": 0.02,
    "max_ref_candidates": 4,
    "ref_max_above_table": 0.30,            # a reference must be a table-top object (m above the table)
    # pairing and ambiguity
    "exclusive_iou": 0.6,
    "stack_iou": 0.85,
    "ambig_lo": 0.05,
    "ambig_hi": 0.30,
    # relations
    "on_dz_max": (0.06, 0.08),
}

# Hybrid detector (Study 2): OWLv2 for bowls / ramekin / plate, Grounding DINO for cabinet, stove and cookie box.
OWL_RAMP = (0.10, 0.25)                     # OWLv2 confidence scale: 0.10 = "seen", 0.25 = "sure"
GDINO_RAMP = (0.22, 0.40)
GD_PHRASES = {"wooden cabinet", "cabinet", "drawer", "stove", "cooktop", "burner", "cookie box", "box", "carton"}
GD_FALLBACK = {"plate", "white plate", "dish"}
HYBRID_OVERRIDES = {"min_conf": [OWL_RAMP[0]] * 3, "ground_min_conf": OWL_RAMP[0], "conf_floor": OWL_RAMP[0],
                    "conf_full": OWL_RAMP[1], "scene_vocab": []}

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
OWLV2_MODEL_ID = "google/owlv2-base-patch16-ensemble"
SMOLVLA_MODEL_ID = "HuggingFaceVLA/smolvla_libero"
XVLA_MODEL_ID = "lerobot/xvla-libero"


def engine_cfg(detector="hybrid"):
    """perception settings for a detector: 'gdino' (first gate) or 'hybrid' (OWLv2 + Grounding DINO)"""
    cfg = copy.deepcopy(ENGINE_CFG)
    if detector == "hybrid":
        cfg.update(copy.deepcopy(HYBRID_OVERRIDES))
    elif detector != "gdino":
        raise ValueError(f"unknown detector '{detector}' (use 'gdino' or 'hybrid')")
    return cfg


def gate_cfg(detector="hybrid", reground=True, path=FROZEN_STUDY1_JSON):
    """the frozen gate: C = 0.27 G + 0.40 S + 0.33 D, execute if C >= 0.70, stop if C < 0.50 after one re-grounding.
    Only the confidence ramp follows the detector (same as gate_cfg6 / gate_cfg7 in the experiment code)."""
    f = json.load(open(path))
    c = {**f, "weights": tuple(f["weights"]), "ramp": tuple(f["ramp"])}
    if detector == "hybrid":
        c["ramp"] = OWL_RAMP
    elif detector != "gdino":
        raise ValueError(f"unknown detector '{detector}' (use 'gdino' or 'hybrid')")
    c["reground"] = bool(reground)
    return c


def frozen_hash(path):
    """the 10-character fingerprint printed by the notebooks (a4a4d6846a for Study 1, a5f5591cec for the hybrid gate)"""
    import hashlib
    return hashlib.sha1(json.dumps(json.load(open(path)), sort_keys=True, default=float).encode()).hexdigest()[:10]
