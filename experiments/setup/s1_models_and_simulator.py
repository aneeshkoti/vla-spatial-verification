"""Environment variables, library versions, SmolVLA, the LIBERO helpers, the instruction parser,
Grounding DINO and a first test environment with depth.
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ----------------------------------------------------------------------------------------------------------------------
import torch, mujoco, importlib.metadata as md
print("torch:", torch.__version__, "| CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
print("mujoco:", mujoco.__version__)
print("lerobot:", md.version("lerobot"))
import libero.libero
print("LIBERO import OK")


# ----------------------------------------------------------------------------------------------------------------------
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

MODEL_ID = "HuggingFaceVLA/smolvla_libero"
device = "cuda"

policy = SmolVLAPolicy.from_pretrained(MODEL_ID).to(device).eval()
cfg = policy.config
preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=MODEL_ID)
print("Model + processors loaded OK")


# ----------------------------------------------------------------------------------------------------------------------
import numpy as np, json, time
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

MAX_STEPS = 220
suite = benchmark.get_benchmark_dict()["libero_spatial"]()

def get_task_env_and_instruction(task_id):

  task = suite.get_task(task_id)
  instruction = task.language
  bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
  env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, camera_depths=True)
  return env, task, instruction

def to_img(x):
    return torch.from_numpy(x.copy()).permute(2, 0, 1).float() / 255.0

def build_raw_batch(obs, instruction):
    img   = obs["agentview_image"][::-1, ::-1]
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)
    return {
        "observation.images.image":  to_img(img),
        "observation.images.image2": to_img(wrist),
        "observation.state": torch.from_numpy(state),
        "task": instruction,
    }


# ----------------------------------------------------------------------------------------------------------------------
import re

RELATION_KEYWORDS = {
    "in the top drawer of": "in", "in the bottom drawer of": "in",
    "between": "between", "next to": "near", "on top of": "above",
    "from table center": "at", "on": "on", "in": "in", "at": "at",
    "left of": "left", "to the left of": "left",
    "right of": "right", "to the right of": "right",
    "above": "above", "over": "above",
    "below": "below", "under": "below", "beneath": "below",
    "in front of": "front", "behind": "behind",
    "near": "near", "close to": "near", "far from": "far",
}
SORTED_RELATIONS = sorted(RELATION_KEYWORDS.keys(), key=len, reverse=True)

def _find_relation(text):
    for phrase in SORTED_RELATIONS:
        if phrase in text:
            return phrase, RELATION_KEYWORDS[phrase]
    return None, None

def _clean(text):
    text = text.strip()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    return text.strip(" .")

def parse_instruction(instruction):
    text = instruction.lower().strip().rstrip(".")
    pick_clause = text.split(" and place")[0]
    pick_clause = re.sub(r"^(pick up|put|place|move|set)\s+", "", pick_clause)

    relation_phrase, relation = _find_relation(pick_clause)
    if relation is None:
        return {"target": None, "reference": None, "relation": None, "raw": instruction}

    idx = pick_clause.find(relation_phrase)
    target = _clean(pick_clause[:idx])
    remainder = pick_clause[idx + len(relation_phrase):]
    remainder = re.sub(r"^\s*(of|to)\s+", "", remainder)

    if relation == "at" and not remainder.strip():
        reference = "table center"
    else:
        reference = _clean(remainder)

    if relation == "between" and " and " in reference:
        reference = [_clean(p) for p in reference.split(" and ")]

    return {"target": target or None, "reference": reference or None, "relation": relation, "raw": instruction}


# ----------------------------------------------------------------------------------------------------------------------
import torch, gc
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"   # was: grounding-dino-tiny

if "gdino_model" in globals():                          # free the old model from GPU memory
    del gdino_model
    gc.collect()
    torch.cuda.empty_cache()

gdino_processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(device).eval()
print("Grounding DINO loaded OK:", GDINO_MODEL_ID)


# ----------------------------------------------------------------------------------------------------------------------
from PIL import Image

def ground_object(image_pil, phrase, box_threshold=0.35, text_threshold=0.25):
    """
    image_pil: PIL Image
    phrase: object phrase, e.g. "black bowl"
    Returns: list of {bbox: [x1,y1,x2,y2], confidence: float, label: str}
    """
    text_prompt = phrase.lower().strip() + "."
    inputs = gdino_processor(images=image_pil, text=text_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = gdino_model(**inputs)

    results = gdino_processor.post_process_grounded_object_detection(
        outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[image_pil.size[::-1]],  # (height, width)
    )[0]

    detections = []
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        detections.append({
            "bbox": [round(v, 1) for v in box.tolist()],
            "confidence": round(score.item(), 3),
            "label": label,
        })
    return detections


# ----------------------------------------------------------------------------------------------------------------------
test_env = OffScreenRenderEnv(
    bddl_file_name=os.path.join(get_libero_path("bddl_files"), suite.get_task(0).problem_folder, suite.get_task(0).bddl_file),
    camera_heights=256, camera_widths=256, camera_depths=True,
)
test_env.seed(0)
test_obs = test_env.reset()


# ----------------------------------------------------------------------------------------------------------------------
def get_real_depth_map(env, depth_buffer):
    extent = env.sim.model.stat.extent
    near = env.sim.model.vis.map.znear * extent

    far = env.sim.model.vis.map.zfar * extent
    real_depth = near / (1.0 - depth_buffer * (1.0 - near / far))
    return real_depth[::-1, ::-1]   # match the RGB flip used for grounding

real_depth_map = get_real_depth_map(test_env, test_obs["agentview_depth"])
print("Real depth range (meters):", real_depth_map.min(), real_depth_map.max())
print("Depth at image center:", real_depth_map[128, 128])
