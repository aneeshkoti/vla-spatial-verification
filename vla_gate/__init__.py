"""vla_gate: a training-free spatial verification and action-gating layer for frozen vision-language-action policies.

Before the robot moves, the gate grounds the objects named in the instruction (G), tests the stated spatial relation in the
image (S) and in metric 3D space (D), combines them into one score C = 0.27 G + 0.40 S + 0.33 D, and lets the policy act
(C >= 0.70), re-grounds once, or stops the robot (C < 0.50). The policy itself is never changed.

Quick use (LIBERO):
    from vla_gate import SpatialGate
    from vla_gate.sim import make_env, setup_episode
    gate = SpatialGate.load("hybrid")
    env, task, instruction = make_env(0)
    obs = setup_episode(env, 0, 5)
    print(gate.check_libero(env, obs, instruction))
"""
from .config import engine_cfg, gate_cfg, frozen_hash
from .decision import decide_all, gate_decide, scene_fields, scene_vals, summarize
from .evidence import perceive_scene, scene_summary
from .gate import GateResult, SpatialGate, flipped_image
from .geometry import CameraModel, calibrate_camera, mujoco_depth_to_meters
from .parsing import parse_goal, parse_instruction

__version__ = "1.0.0"
__all__ = ["SpatialGate", "GateResult", "CameraModel", "calibrate_camera", "mujoco_depth_to_meters", "parse_instruction",
           "parse_goal", "perceive_scene", "scene_summary", "gate_decide", "decide_all", "scene_vals", "scene_fields",
           "summarize", "engine_cfg", "gate_cfg", "frozen_hash", "flipped_image"]
