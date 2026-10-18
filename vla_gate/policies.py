"""The two frozen robot policies (SmolVLA and X-VLA) and the rollout loops used in the experiments.

The gate does not change the policies in any way: they are loaded from the Hugging Face hub and driven exactly as in the
experiments (SmolVLA: LeRobot processors, relative actions clipped to [-1, 1], 220 steps; X-VLA: LeRobot's own call
sequence for LIBERO, absolute end-effector targets, 800 steps).
"""
import math
import time

import numpy as np

from .config import SMOLVLA_MODEL_ID, XVLA_MODEL_ID
from .sim import MAX_STEPS, apply_obs_hook, object_keys, setup_episode, true_keys

XVLA_MAX_STEPS = 800                        # LeRobot's X-VLA LIBERO command uses --env.episode_length=800


def _device(device):
    if device is not None:
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


class SmolVLA:
    """HuggingFaceVLA/smolvla_libero. Call it with (obs, instruction) to get one 7-D action."""

    def __init__(self, model_id=SMOLVLA_MODEL_ID, device=None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        self.device = _device(device)
        self.policy = SmolVLAPolicy.from_pretrained(model_id).to(self.device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(self.policy.config, pretrained_path=model_id)

    def reset(self):
        self.policy.reset()

    @staticmethod
    def build_raw_batch(obs, instruction):
        import torch
        from robosuite.utils.transform_utils import quat2axisangle

        def to_img(x):
            return torch.from_numpy(x.copy()).permute(2, 0, 1).float() / 255.0
        img = obs["agentview_image"][::-1, ::-1]
        wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
        state = np.concatenate([obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"]]).astype(np.float32)
        return {"observation.images.image": to_img(img), "observation.images.image2": to_img(wrist),
                "observation.state": torch.from_numpy(state), "task": instruction}

    def __call__(self, obs, instruction):
        import torch
        batch = self.preprocessor(self.build_raw_batch(obs, instruction))
        with torch.no_grad():
            act = self.postprocessor(self.policy.select_action(batch))
        return np.clip(act.squeeze(0).float().cpu().numpy(), -1.0, 1.0)     # same clipping as the baseline


def failure_reason(r):
    """why did an episode fail? (graded from simulator truth, analysis only)"""
    if int(r["success"]) == 1:
        return "success"
    if r.get("other_lifted") and not int(r.get("target_lifted") or 0):
        return "wrong_object_grasped"
    me = r.get("min_eef_to_target_cm")
    if me is None or (isinstance(me, float) and math.isnan(me)) or me > 8.0:
        return "never_reached_target"
    if not int(r.get("target_lifted") or 0):
        return "reached_but_no_grasp"
    return "grasped_but_not_placed"


def _finish_row(obs, tkey, gkey, movable, lift, min_eef, task_id, state, perturb, instruction, success, step, t0):
    t_lift = int(tkey is not None and lift.get(tkey, 0.0) > 0.04)
    other = [k[:-4] for k in movable if k != tkey and lift[k] > 0.04]
    fdist = (round(100 * float(np.linalg.norm(np.asarray(obs[tkey])[:2] - np.asarray(obs[gkey])[:2])), 1) if tkey and gkey else None)
    row = {"task_id": task_id, "state": state, "condition": perturb or "clean", "instruction": instruction,
           "success": int(success), "episode_length": step + 1, "execution_time": round(time.time() - t0, 2),
           "min_eef_to_target_cm": round(100 * min_eef, 1) if np.isfinite(min_eef) else None,
           "target_lifted": t_lift, "other_lifted": "|".join(other), "final_target_to_goal_cm": fdist}
    row["failure_reason"] = failure_reason(row)
    return row


def rollout_smolvla(env, policy, task_id, state, instruction, perturb=None, max_steps=MAX_STEPS, on_frame=None):
    """one SmolVLA episode on the scene of (task, state, perturbation) with failure diagnostics.
    on_frame(obs): optional callback for every observation (e.g. to record a video)."""
    obs = setup_episode(env, task_id, state, perturb=perturb, instruction=instruction, policy=policy)
    tkey, gkey = true_keys(env, obs)
    movable = [k for k in object_keys(obs) if any(w in k for w in ("bowl", "plate", "ramekin", "cookies"))]
    z0 = {k: float(np.asarray(obs[k])[2]) for k in movable}
    lift = {k: 0.0 for k in movable}
    min_eef = float("inf")
    success, t0, step = False, time.time(), 0
    if on_frame is not None:
        on_frame(obs)
    for step in range(max_steps):
        act = policy(obs, instruction)
        obs, reward, done, info = env.step(act.tolist())
        obs = apply_obs_hook(env, obs)
        if on_frame is not None:
            on_frame(obs)
        for k in movable:
            lift[k] = max(lift[k], float(np.asarray(obs[k])[2]) - z0[k])
        if tkey:
            min_eef = min(min_eef, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], float) - np.asarray(obs[tkey], float))))
        if done:
            success = True
            break
    return _finish_row(obs, tkey, gkey, movable, lift, min_eef, task_id, state, perturb, instruction, success, step, t0)


class XVLA:
    """lerobot/xvla-libero, driven with LeRobot's own processors (the same call sequence as lerobot_eval)."""

    def __init__(self, model_id=XVLA_MODEL_ID, device=None):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
        from lerobot.policies.xvla.processor_xvla import make_xvla_libero_pre_post_processors
        self.device = _device(device)
        self.policy = XVLAPolicy.from_pretrained(model_id).to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=model_id,
                                                       preprocessor_overrides={"device_processor": {"device": self.device}})
        self.env_pre, self.env_post = make_xvla_libero_pre_post_processors()

    def reset(self):
        self.policy.reset()
        for p in (self.pre, self.post, self.env_pre, self.env_post):
            if hasattr(p, "reset"):
                p.reset()

    @staticmethod
    def _obs(env, obs):
        """raw simulator observation -> the observation LeRobot's LiberoEnv hands to a policy (batch of 1)"""
        ctrl = env.env.robots[0].controller
        b = lambda x: np.asarray(x)[None]
        return {"pixels": {"image": np.ascontiguousarray(obs["agentview_image"])[None],
                           "image2": np.ascontiguousarray(obs["robot0_eye_in_hand_image"])[None]},
                "robot_state": {"eef": {"pos": b(obs["robot0_eef_pos"]), "quat": b(obs["robot0_eef_quat"]), "mat": b(ctrl.ee_ori_mat)},
                                "gripper": {"qpos": b(obs["robot0_gripper_qpos"]), "qvel": b(obs["robot0_gripper_qvel"])},
                                "joints": {"pos": b(obs["robot0_joint_pos"]), "vel": b(obs["robot0_joint_vel"])}}}

    def action(self, env, obs, instruction):
        """7 numbers: ABSOLUTE target position (3), orientation as axis-angle (3), gripper (-1 open / +1 close)"""
        import torch
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION
        o = preprocess_observation(self._obs(env, obs))
        o["task"] = [instruction]
        o = self.pre(self.env_pre(o))
        with torch.inference_mode():
            a = self.policy.select_action(o)
        a = self.env_post({ACTION: self.post(a)})[ACTION]
        a = a.to("cpu").float().numpy()[0]
        assert a.shape == (7,) and np.isfinite(a).all(), f"unexpected X-VLA action {a}"
        return a


def rollout_xvla(env, policy, task_id, state, instruction, perturb=None, max_steps=XVLA_MAX_STEPS, on_frame=None):
    """X-VLA episode on exactly the scene SmolVLA gets (same set-up, perturbation and seed) + the same diagnostics."""
    import torch
    obs = setup_episode(env, task_id, state, perturb=perturb, instruction=instruction)
    policy.reset()
    torch.manual_seed(1000 * task_id + state)
    env.env.horizon = max(int(env.env.horizon), max_steps + 200)       # the simulator refuses to step past its horizon
    tkey, gkey = true_keys(env, obs)
    movable = [k for k in object_keys(obs) if any(w in k for w in ("bowl", "plate", "ramekin", "cookies"))]
    z0 = {k: float(np.asarray(obs[k])[2]) for k in movable}
    lift = {k: 0.0 for k in movable}
    min_eef = float("inf")
    success, t0, step = False, time.time(), 0
    if on_frame is not None:
        on_frame(obs)
    try:
        for rb in env.env.robots:
            rb.controller.use_delta = False                            # absolute targets (as LeRobot's LiberoEnv does)
        for step in range(max_steps):
            act = policy.action(env, obs, instruction)
            obs, reward, done, info = env.step(act)
            obs = apply_obs_hook(env, obs)
            if on_frame is not None:
                on_frame(obs)
            for k in movable:
                lift[k] = max(lift[k], float(np.asarray(obs[k])[2]) - z0[k])
            if tkey:
                min_eef = min(min_eef, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], float) - np.asarray(obs[tkey], float))))
            if done:
                success = True
                break
    finally:
        for rb in env.env.robots:
            rb.controller.use_delta = True
    return _finish_row(obs, tkey, gkey, movable, lift, min_eef, task_id, state, perturb, instruction, success, step, t0)
