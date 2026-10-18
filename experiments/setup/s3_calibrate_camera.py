"""Calibrates the camera convention once per session (moves the gripper; the chosen flip is printed).
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# ============================================================================
# TEST the engine on test_env (task 0) — run once per session (calibrates the camera).
# In the projection plot every red '+' must sit ON its object.
# ============================================================================
test_obs = setup_episode(test_env, 0, 0)
calibrate_camera(test_env, test_obs)          # moves the gripper left/right once
test_obs = setup_episode(test_env, 0, 0)      # back to the official start state
show_projection_check(test_env, test_obs, save_path=os.path.join(V3_DIR, "projection_check.png"))

_instr0 = suite.get_task(0).language
_cam = CameraModel(test_env)
_scene = perceive_scene(test_env, test_obs, parse_instruction(_instr0), parse_goal(_instr0), _cam)
print(_instr0)
print(scene_summary(_scene))
print("bowl candidates (check the 'conf' values of", GDINO_MODEL_ID, "):")
for _c in _scene["candidates"]:
    print("  ", _c)
