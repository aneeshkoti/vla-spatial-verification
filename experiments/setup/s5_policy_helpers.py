"""SmolVLA's action for one observation, the camera model guard, the action-heading term A
(tested in Study 1 and removed: weight 0 in the frozen gate) and the evidence picture.
The earlier step-by-step gating code of the same notebook cells is not used by the final method and is left out.
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# ============================================================================
# SmolVLA's action for one observation + the calibrated camera model
# ============================================================================
def predict_action(obs, instruction):
    raw = build_raw_batch(obs, instruction)
    batch = preprocessor(raw)
    with torch.no_grad():
        act = postprocessor(policy.select_action(batch))
    return np.clip(act.squeeze(0).float().cpu().numpy(), -1.0, 1.0)     # same clipping as the baseline

def get_camera_model(env, obs):
    if CAMERA_FLIPS is None:
        raise RuntimeError("Camera not calibrated: run the 'TEST the engine' cell (calibrate_camera) first.")
    return CameraModel(env)


# ----------------------------------------------------------------------------------------------------------------------
# ============================================================================
# TASK 17: ACTION CONSISTENCY  A  — graded, phase-aware, world frame
#   approach phase: is SmolVLA moving toward the verified target bowl?
#   carry phase   : is it moving toward the verified goal (plate)?
#   gripper       : closing far from the bowl / opening far from the plate is suspicious.
#   Action dims 0,1,2 = world x,y,z (verified empirically in Task 16).
# ============================================================================
def action_evidence(action, obs, astate, cfg=None):
    cfg = cfg or CFG
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    v = np.asarray(action[:3], dtype=float)
    grip = float(action[6])
    if astate["phase"] == "approach":
        goal_pt = astate.get("target_pos")
    else:
        goal_pt = None if astate.get("goal_pos") is None else np.asarray(astate["goal_pos"], float) + np.array([0.0, 0.0, 0.10])
    if goal_pt is None:
        return 1.0, "no reference point (neutral)"
    d = np.asarray(goal_pt, float) - eef
    dist, dist_xy, speed = float(np.linalg.norm(d)), _norm2(d[:2]), float(np.linalg.norm(v))
    if dist_xy < cfg["a_above_radius"]:
        a, why = 1.0, "above object"
    elif speed < cfg["a_small_motion"]:
        prev = astate.get("A_ema")
        a, why = (1.0 if prev is None else float(prev)), "small motion (no new evidence, A kept)"
    else:
        cos3 = float(np.dot(v, d) / (speed * dist + 1e-9))
        vxy = _norm2(v[:2])
        cosxy = float(np.dot(v[:2], d[:2]) / (vxy * dist_xy + 1e-9)) if vxy > cfg["a_small_motion"] else cos3
        cos = max(cos3, cosxy)
        a, why = float(np.clip(0.5 + cos, 0.0, 1.0)), f"cos={cos:.2f}"
    if astate["phase"] == "approach" and grip > 0 and dist > cfg["grasp_radius"]:
        a, why = a * 0.5, why + ", closing gripper far from target"
    if astate["phase"] == "carry" and grip < 0 and dist_xy > cfg["release_radius"]:
        a, why = a * 0.5, why + ", opening gripper far from goal"
    return float(a), why

def update_phase(astate, action, obs, cfg=None):
    """approach -> carry after the gripper closes near the target; carry -> approach if it re-opens."""
    cfg = cfg or CFG
    eef = np.asarray(obs["robot0_eef_pos"], dtype=float)
    grip = float(action[6])
    if astate["phase"] == "approach":
        astate["close_steps"] = astate["close_steps"] + 1 if grip > 0 else 0
        tp = astate.get("target_pos")
        near_target = tp is None or _norm2((eef - np.asarray(tp, float))[:2]) < cfg["grasp_radius"]
        if astate["close_steps"] >= 3 and near_target:
            astate["phase"], astate["open_steps"] = "carry", 0
    else:
        astate["open_steps"] = astate["open_steps"] + 1 if grip < 0 else 0
        if astate["open_steps"] >= 3:
            astate["phase"], astate["close_steps"] = "approach", 0
            astate["target_pos"] = eef.copy()     # object was released around here


# ----------------------------------------------------------------------------------------------------------------------
# ---------- evidence panel (qualitative figure, Task 32) ----------
def save_evidence_panel(frame_F, scene, title, text, path):
    fig, ax = plt.subplots(1, figsize=(4.6, 5.2))
    ax.imshow(frame_F)
    def rect(b, color, lw, label=None):
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=lw, edgecolor=color, facecolor="none"))
        if label:
            ax.text(b[0], max(b[1] - 3, 6), label, color=color, fontsize=7, weight="bold")
    for c in scene.get("candidates", [])[1:]:
        rect(c["box"], "yellow", 1)
    if scene.get("goal"):
        rect(scene["goal"]["box_F"], "lime", 2, "goal")
    if scene.get("target"):
        rect(scene["target"]["box_F"], "red", 2, "target")
    for r in scene.get("refs", []):
        if r:
            rect(r["box_F"], "cyan", 2, "reference")
    ax.set_title(title, fontsize=8)
    ax.text(0.01, -0.02, text, transform=ax.transAxes, fontsize=7.5, va="top", family="monospace")
    ax.axis("off")
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
