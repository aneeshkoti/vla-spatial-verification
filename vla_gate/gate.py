"""SpatialGate: the verification layer in front of a robot policy.

    gate = SpatialGate.load("hybrid")                 # OWLv2 + Grounding DINO, frozen settings of the paper
    result = gate.check_libero(env, obs, instruction)  # before the robot moves
    if result.execute:
        ... let the policy act ...
    else:
        ... do not move the robot ...

The gate never changes the policy's actions. It only decides whether the policy may act.
"""
import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from .config import engine_cfg, gate_cfg
from .decision import _score, gate_decide, scene_fields, scene_vals
from .detectors import load_detector
from .evidence import perceive_scene
from .geometry import CameraModel, mujoco_depth_to_meters
from .parsing import parse_goal, parse_instruction


@dataclass
class GateResult:
    decision: str                 # "EXECUTE" or "ABSTAIN" (= stop: the robot does not move)
    C: float                      # evidence score of the grounding pass that was used
    G: float                      # grounding evidence
    S: float                      # 2D relation evidence (already weighted by grounding trust)
    D: float                      # 3D relation evidence (already weighted by grounding trust)
    S_rel: float
    D_rel: float
    ambiguity: float              # 1 = one bowl fits clearly; 0.5 = two bowls fit equally well
    C_first: float                # score of the first grounding pass
    regrounded: bool              # True if the re-grounding pass was needed
    scene: dict                   # boxes and measurements of the pass that was used
    scenes: dict = field(repr=False, default_factory=dict)
    row: dict = field(repr=False, default_factory=dict)
    seconds: float = 0.0

    @property
    def execute(self):
        return self.decision == "EXECUTE"

    def __str__(self):
        verdict = "EXECUTE (the policy may act)" if self.execute else "STOP (the robot does not move)"
        return (f"G = {self.G:.2f} | S = {self.S:.2f} | D = {self.D:.2f} | C = {self.C:.2f}"
                f"{' (after re-grounding)' if self.regrounded else ''} -> {verdict}")


class SpatialGate:
    """detector + perception settings + frozen decision settings"""

    def __init__(self, detector, kind="hybrid", engine=None, cfg=None):
        self.detector = detector
        self.kind = kind
        self.engine = engine if engine is not None else engine_cfg(kind)
        self.cfg = cfg if cfg is not None else gate_cfg(kind)

    @classmethod
    def load(cls, kind="hybrid", device=None):
        """kind = 'hybrid' (Study 2, recommended) or 'gdino' (the first gate, Study 1). Downloads the detector(s) on first use."""
        return cls(load_detector(kind, device), kind)

    def check(self, image_F, depth_F, cam, instruction, parsed=None, goal=None):
        """image_F: PIL image as the detector sees it; depth_F: metres, same layout; cam: CameraModel.
        Runs the first grounding pass, and the re-grounding pass only when the first score is below the 'execute' threshold."""
        t0 = time.time()
        parsed = parsed if parsed is not None else parse_instruction(instruction)
        goal = goal if goal is not None else parse_goal(instruction)
        row = {"instruction": instruction, "relation": parsed.get("relation"), "has_goal": int(bool(goal and goal.get("object")))}
        scenes = {"0": perceive_scene(self.detector, image_F, depth_F, cam, parsed, goal, 0, self.engine)}
        row.update(scene_fields(scenes["0"], 1.0, "0"))            # the action term A has weight 0 in the frozen gate
        ramp = tuple(self.cfg["ramp"])
        if _score(scene_vals(row, "0", ramp), self.cfg["weights"]) < self.cfg["high"] and self.cfg.get("reground", True):
            scenes["1"] = perceive_scene(self.detector, image_F, depth_F, cam, parsed, goal, 1, self.engine)
            row.update(scene_fields(scenes["1"], 1.0, "1"))
        decision, C, C0, rg, sfx = gate_decide(row, self.cfg)
        v = scene_vals(row, sfx, ramp)
        return GateResult(decision=decision, C=float(C), G=v["G"], S=v["S"], D=v["D"], S_rel=v["S_rel"], D_rel=v["D_rel"],
                          ambiguity=v["amb"], C_first=float(C0), regrounded=bool(rg), scene=scenes[sfx], scenes=scenes,
                          row=row, seconds=round(time.time() - t0, 2))

    def check_libero(self, env, obs, instruction, cam=None, **kw):
        """the same check on a LIBERO observation (camera image + MuJoCo depth of the 'agentview' camera)"""
        image_F = Image.fromarray(obs["agentview_image"][::-1, ::-1])
        depth_F = mujoco_depth_to_meters(env, obs["agentview_depth"])
        cam = cam if cam is not None else CameraModel.from_mujoco(env)
        return self.check(image_F, depth_F, cam, instruction, **kw)


def flipped_image(obs):
    """the camera image exactly as the detectors see it (numpy array)"""
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
