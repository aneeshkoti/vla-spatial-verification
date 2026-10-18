"""Study 1 set-up: splits, result files, baseline runner, verification of one case, offline gate,
statistics and summaries.
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# ============================================================================
# UPGRADE v4 — SETUP (run once per session, after the engine / Phase 3 cells). Takes seconds.
# Implements the reviewer plan: Tasks 1-5 on the DEVELOPMENT set, freeze, then Tasks 6-10 on the TEST set.
#   DEVELOPMENT = start states 0,1  (already used while building the method)
#   TEST        = start states 2,3,4 (never used before; touched only after the config is frozen)
# Ground-truth simulator positions are used ONLY for grading/analysis, never by the gate.
# ============================================================================
import math, hashlib, itertools
V4_DIR = os.path.join(RESULTS_ROOT, "v4")
V4_TAB = os.path.join(V4_DIR, "tables")
V4_QUAL = os.path.join(V4_DIR, "qualitative")
for _d in (V4_DIR, V4_TAB, V4_QUAL):
    os.makedirs(_d, exist_ok=True)

DEV_STATES, TEST_STATES = [0, 1], [2, 3, 4]
NUM_TASKS = 10
KINDS = ["clean", "absent_reference", "absent_target", "relation_violation", "distractor", "position_change", "occlusion"]
VALID_KINDS = ["clean", "distractor", "position_change", "occlusion"]
INVALID_KINDS_V4 = ["absent_reference", "absent_target", "relation_violation"]
VALID_PERTURBED = ["distractor", "position_change", "occlusion"]

# engine settings that stay fixed in v4 (resampling off, exactly one re-grounding pass)
CFG["resample_k"] = 0
CFG["max_re_ground"] = 1

BASE_V4_CSV = os.path.join(V4_DIR, "baseline_clean_v4.csv")      # Task 1
VERIFY_CSV = os.path.join(V4_DIR, "verify_v4.csv")               # Tasks 2-8 (dev + test verification logs)
CAND_JSONL = os.path.join(V4_DIR, "verify_candidates_v4.jsonl")  # per-candidate grounding data (Task 3/4)
ROLL_CSV = os.path.join(V4_DIR, "rollouts_v4.csv")               # Task 9 (perturbed SmolVLA rollouts)
FROZEN_JSON = os.path.join(V4_DIR, "frozen_config.json")         # Task 5 output

ROLL_FIELDS = ["task_id", "state", "split", "condition", "instruction", "success", "episode_length", "execution_time",
               "failure_reason", "min_eef_to_target_cm", "target_lifted", "other_lifted", "final_target_to_goal_cm"]

def _split(state):
    return "dev" if state in DEV_STATES else "test"

def _append(path, fields, row):
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)

def _done_keys(path, cols):
    try:
        d = pd.read_csv(path)
        return set(map(tuple, d[cols].astype(str).values.tolist()))
    except Exception:
        return set()

# ---------- baseline runner with failure diagnostics (same action sequence as run_baseline_episode_v2) ----------
def run_baseline_v4(env, task_id, state, instruction, perturb=None, max_steps=MAX_STEPS):
    obs = setup_episode(env, task_id, state, perturb=perturb, instruction=instruction)
    tkey, gkey = _true_keys(env, obs)
    movable = [k for k in _object_keys(obs) if any(w in k for w in ("bowl", "plate", "ramekin", "cookies"))]
    z0 = {k: float(np.asarray(obs[k])[2]) for k in movable}
    lift = {k: 0.0 for k in movable}
    min_eef = float("inf")
    success, t0, step = False, time.time(), 0
    for step in range(max_steps):
        act = predict_action(obs, instruction)
        obs, reward, done, info = env.step(act.tolist())
        obs = apply_obs_hook(obs)
        for k in movable:
            lift[k] = max(lift[k], float(np.asarray(obs[k])[2]) - z0[k])
        if tkey:
            min_eef = min(min_eef, float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"], float) - np.asarray(obs[tkey], float))))
        if done:
            success = True
            break
    t_lift = int(tkey is not None and lift.get(tkey, 0.0) > 0.04)
    other = [k[:-4] for k in movable if k != tkey and lift[k] > 0.04]
    fdist = (round(100 * float(np.linalg.norm(np.asarray(obs[tkey])[:2] - np.asarray(obs[gkey])[:2])), 1)
             if tkey and gkey else None)
    row = {"task_id": task_id, "state": state, "split": _split(state), "condition": perturb or "clean",
           "instruction": instruction, "success": int(success), "episode_length": step + 1,
           "execution_time": round(time.time() - t0, 2),
           "min_eef_to_target_cm": round(100 * min_eef, 1) if np.isfinite(min_eef) else None,
           "target_lifted": t_lift, "other_lifted": "|".join(other), "final_target_to_goal_cm": fdist}
    row["failure_reason"] = failure_reason(row)
    return row

def failure_reason(r):
    """Task 1: why did SmolVLA fail? (graded from simulator truth, analysis only)"""
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

# ---------- verification of one case: logs everything needed to recompute ANY gate configuration offline ----------
def _proj(cam, p):
    return cam.world_to_f(np.asarray(p, float))

def _grade_scene(sc, obs, cam, parsed, kind, tkey, gkey):
    """analysis only: is the selected target / reference / goal the right object?"""
    bowls = [k for k in _object_keys(obs) if "bowl" in k]
    tgt = sc["target"]
    if tgt is None:
        sel = "none"
    elif tkey and _contains(tgt["box_F"], _proj(cam, obs[tkey])):
        sel = "target"
    elif any(_contains(tgt["box_F"], _proj(cam, obs[b])) for b in bowls if b != tkey):
        sel = "other_bowl"
    else:
        sel = "not_a_bowl"
    right_t = "" if kind == "absent_target" else int(sel == "target")
    ok = []
    for rp, r in zip(_as_list(parsed.get("reference")), sc["refs"]):
        rk = _ref_key(rp, obs)
        if rk is None:
            continue                                   # table centre / absent object: not gradable
        tr = np.asarray(obs[rk], float)
        ok.append(int(r is not None and (_contains(r["box_F"], _proj(cam, tr), pad=12)
                                         or float(np.linalg.norm(np.asarray(r["pos"])[:2] - tr[:2])) < 0.10)))
    right_r = "" if (not ok or kind == "absent_reference") else int(min(ok))
    g = sc["goal"]
    right_g = int(g is not None and gkey is not None and (_contains(g["box_F"], _proj(cam, obs[gkey]), pad=12)
                                                          or float(np.linalg.norm(np.asarray(g["pos"])[:2] - np.asarray(obs[gkey])[:2])) < 0.10))
    cands = []
    for c in sc.get("candidates", []):
        cands.append({"rel": c["rel"], "s": c["s"], "d": c["d"], "conf": c["conf"],
                      "right": int(tkey is not None and kind != "absent_target" and _contains(c["box"], _proj(cam, obs[tkey])))})
    return sel, right_t, right_r, right_g, cands

def _scene_fields(sc, A, sfx):
    refs = [None if r is None else round(float(r["conf"]), 4) for r in sc["refs"]]
    return {f"G_{sfx}": sc["G"], f"S_{sfx}": sc["S"], f"D_{sfx}": sc["D"], f"S_rel_{sfx}": sc["S_rel"],
            f"D_rel_{sfx}": sc["D_rel"], f"amb_{sfx}": sc.get("ambiguity", 1.0), f"pair_g_{sfx}": sc.get("pair_g"),
            f"A_{sfx}": round(float(A), 4),
            f"t_conf_{sfx}": None if sc["target"] is None else round(float(sc["target"]["conf"]), 4),
            f"ref_confs_{sfx}": json.dumps(refs),
            f"goal_conf_{sfx}": None if sc["goal"] is None else round(float(sc["goal"]["conf"]), 4)}

def _A_for(scene, act, obs):
    st = {"phase": "approach", "target_pos": scene["target"]["pos"] if scene["target"] else None,
          "goal_pos": scene["goal"]["pos"] if scene["goal"] else None, "A_ema": None}
    return action_evidence(act, obs, st, CFG)[0]

def verify_case(task_id, state, kind, save_panel=True):
    global OBS_HOOK
    t_all = time.time()
    env, _, clean_instr = get_task_env_and_instruction(task_id)
    instr = perturb_instruction(clean_instr, kind, state) if kind in ("absent_reference", "absent_target") else clean_instr
    obs = setup_episode(env, task_id, state, perturb=kind if kind in SCENE_PERTURBATIONS else None, instruction=instr)
    tkey, gkey = _true_keys(env, obs)
    parsed, goal = parse_instruction(instr), parse_goal(instr)
    t0 = time.time()
    act = predict_action(obs, instr)                     # SmolVLA's first action = the baseline's first action
    t_policy = time.time() - t0
    cam = get_camera_model(env, obs)
    t0 = time.time()
    sc0 = perceive_scene(env, obs, parsed, goal, cam, 0, CFG)
    A0 = _A_for(sc0, act, obs)
    t_p0 = time.time() - t0
    t0 = time.time()
    sc1 = perceive_scene(env, obs, parsed, goal, cam, 1, CFG)     # the one re-grounding pass (synonyms)
    A1 = _A_for(sc1, act, obs)
    t_p1 = time.time() - t0
    refs = _as_list(parsed.get("reference"))
    rel = parsed.get("relation")
    row = {"task_id": task_id, "state": state, "split": _split(state), "kind": kind,
           "valid": int(kind not in INVALID_KINDS_V4), "instruction": instr, "relation": rel,
           "has_goal": int(bool(goal and goal.get("object"))),
           "label_uncertain": int((kind == "position_change" and any("plate" in (r or "") for r in refs))
                                  or (kind == "distractor" and rel == "near")),
           "t_policy_s": round(t_policy, 3), "t_perceive0_s": round(t_p0, 3), "t_perceive1_s": round(t_p1, 3)}
    cand_rec = {"task_id": task_id, "state": state, "kind": kind}
    for sfx, sc, A in (("0", sc0, A0), ("1", sc1, A1)):
        row.update(_scene_fields(sc, A, sfx))
        sel, rt, rr, rg, cands = _grade_scene(sc, obs, cam, parsed, kind, tkey, gkey)
        row.update({f"sel_{sfx}": sel, f"right_target_{sfx}": rt, f"right_ref_{sfx}": rr, f"right_goal_{sfx}": rg})
        cand_rec[f"cands_{sfx}"] = cands
    if save_panel:
        best = sc1 if (0.2 * sc1["G"] + 0.3 * sc1["S"] + 0.25 * sc1["D"]) > (0.2 * sc0["G"] + 0.3 * sc0["S"] + 0.25 * sc0["D"]) else sc0
        save_evidence_panel(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), best,
                            f"task {task_id} state {state} [{kind}]\n{instr}",
                            f"G={best['G']}  S={best['S']}  D={best['D']}  S_rel={best['S_rel']}  D_rel={best['D_rel']}",
                            os.path.join(V4_QUAL, f"t{task_id}_s{state}_{kind}.png"))
    env.close()
    OBS_HOOK = None
    row["seconds_total"] = round(time.time() - t_all, 2)
    return row, cand_rec

VERIFY_FIELDS = (["task_id", "state", "split", "kind", "valid", "instruction", "relation", "has_goal", "label_uncertain",
                  "t_policy_s", "t_perceive0_s", "t_perceive1_s", "seconds_total"]
                 + [f"{k}_{s}" for s in ("0", "1") for k in ("G", "S", "D", "S_rel", "D_rel", "amb", "pair_g", "A", "t_conf",
                                                           "ref_confs", "goal_conf", "sel", "right_target", "right_ref", "right_goal")]
                 + ["frozen_hash"])

def run_verification(states, frozen_hash=""):
    done = _done_keys(VERIFY_CSV, ["task_id", "state", "kind"])
    todo = [(t, s, k) for s in states for t in range(NUM_TASKS) for k in KINDS if (str(t), str(s), k) not in done]
    print(f"verification: {len(todo)} cases to run ({len(done)} already saved)")
    t_start = time.time()
    for i, (t, s, k) in enumerate(todo):
        row, cand = verify_case(t, s, k)
        row["frozen_hash"] = frozen_hash
        _append(VERIFY_CSV, VERIFY_FIELDS, row)
        with open(CAND_JSONL, "a") as f:
            f.write(json.dumps(cand) + "\n")
        eta = (time.time() - t_start) / (i + 1) * (len(todo) - i - 1) / 60
        print(f"[{i+1}/{len(todo)}] t{t} s{s} {k:18s} G={row['G_0']:.2f} S_rel={row['S_rel_0']:.2f} D_rel={row['D_rel_0']:.2f} "
              f"sel={row['sel_0']:10s} ({row['seconds_total']}s, ~{eta:.0f} min left)")

# ---------- offline gate: recompute the decision for ANY configuration from the logs ----------
def _num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None

def _gcal(conf, ramp):
    conf = _num(conf)
    return 0.0 if conf is None else ramp_up(conf, ramp[0], ramp[1])

def scene_vals(r, sfx, ramp):
    """G, S, D recomputed for a calibration ramp (the grounding choice itself is kept from the log)."""
    tc = _gcal(r[f"t_conf_{sfx}"], ramp)
    rc = [_gcal(c, ramp) for c in json.loads(r[f"ref_confs_{sfx}"])]
    gs = [tc] + rc + ([_gcal(r[f"goal_conf_{sfx}"], ramp)] if int(r["has_goal"]) else [])
    G = float(np.prod(gs) ** (1.0 / len(gs))) if gs else 0.0
    pair = min([tc] + rc) if rc else 0.0
    trust = math.sqrt(max(pair, 0.0))
    s_rel, d_rel = float(r[f"S_rel_{sfx}"]), float(r[f"D_rel_{sfx}"])
    return {"G": G, "S": s_rel * trust, "D": d_rel * trust, "S_rel": s_rel, "D_rel": d_rel,
            "amb": float(r[f"amb_{sfx}"]), "A": float(r[f"A_{sfx}"])}

def _score(v, w):
    return w[0] * v["G"] + w[1] * v["S"] + w[2] * v["D"] + w[3] * v["A"]

def gate_decide(r, cfg):
    """returns (decision, C_final, C_initial, reground_used, scene_used) with exactly one re-grounding pass."""
    w, hi, lo, ramp = cfg["weights"], cfg["high"], cfg["low"], tuple(cfg["ramp"])
    v0 = scene_vals(r, "0", ramp)
    C0 = _score(v0, w)
    if C0 >= hi:
        return "EXECUTE", C0, C0, 0, "0"
    v, sfx = v0, "0"
    if cfg.get("reground", True):
        v1 = scene_vals(r, "1", ramp)
        part = lambda x: w[0] * x["G"] + w[1] * x["S"] + w[2] * x["D"]
        if part(v1) > part(v0):
            v, sfx = v1, "1"
    C = _score(v, w)
    if C >= hi:
        return "EXECUTE", C, C0, 1, sfx
    if C < lo:
        return "ABSTAIN", C, C0, 1, sfx
    if cfg.get("mid_rule", True):
        ok = min(v["S_rel"], v["D_rel"]) >= cfg.get("mid_min_rel", 0.3) and v["amb"] >= cfg.get("mid_min_unamb", 0.75)
        return ("EXECUTE" if ok else "ABSTAIN"), C, C0, 1, sfx
    return "EXECUTE", C, C0, 1, sfx

CURRENT_CFG = {"name": "v3 (paper)", "weights": (0.20, 0.30, 0.25, 0.25), "high": 0.75, "low": 0.50,
               "ramp": (0.22, 0.40), "reground": True, "mid_rule": True, "mid_min_rel": 0.3, "mid_min_unamb": 0.75}

# ---------- statistics ----------
def wilson(k, n, z=1.959964):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))

def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)

def auroc(pos, neg):
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))

def pct(x):
    return round(100.0 * float(x), 1) if x == x else float("nan")

def load_verify(split):
    d = pd.read_csv(VERIFY_CSV)
    d = d[d.split == split].drop_duplicates(["task_id", "state", "kind"], keep="last").reset_index(drop=True)
    return d

def load_cands():
    out = {}
    if os.path.exists(CAND_JSONL):
        for line in open(CAND_JSONL):
            c = json.loads(line)
            out[(c["task_id"], c["state"], c["kind"])] = c
    return out

def decide_all(df, cfg):
    res = [gate_decide(r, cfg) for r in df.to_dict("records")]
    out = df.copy()
    out["decision"] = [x[0] for x in res]
    out["C"] = [x[1] for x in res]
    out["C_init"] = [x[2] for x in res]
    out["reground_used"] = [x[3] for x in res]
    out["scene_used"] = [x[4] for x in res]
    out["executed"] = (out.decision == "EXECUTE").astype(int)
    out["initial_band"] = ((out.C_init >= cfg["low"]) & (out.C_init < cfg["high"])).astype(int)
    return out

def summarize(dec, base_success=None):
    """IER = invalid executed / invalid; valid acceptance; clean TSR (needs paired baseline outcomes)."""
    inv, val = dec[dec.valid == 0], dec[dec.valid == 1]
    out = {"N_invalid": len(inv), "IER %": pct(inv.executed.mean()) if len(inv) else float("nan"),
           "N_valid": len(val), "valid_accept %": pct(val.executed.mean()) if len(val) else float("nan"),
           "valid_abstain %": pct(1 - val.executed.mean()) if len(val) else float("nan"),
           "AUROC(C)": round(auroc(dec[dec.valid == 1].C, dec[dec.valid == 0].C), 3)}
    if base_success is not None:
        cl = dec[dec.kind == "clean"]
        s = [base_success.get((int(r.task_id), int(r.state))) for r in cl.itertuples()]
        pairs = [(e, x) for e, x in zip(cl.executed, s) if x is not None]
        if pairs:
            out["clean_TSR %"] = pct(np.mean([e * x for e, x in pairs]))
            out["clean_TSR_smolvla %"] = pct(np.mean([x for _, x in pairs]))
    out["J"] = round((1 - inv.executed.mean() if len(inv) else 0) + (val.executed.mean() if len(val) else 0) - 1, 4)
    return out

def base_success_map():
    b = pd.read_csv(BASE_V4_CSV).drop_duplicates(["task_id", "state"], keep="last")
    return {(int(r.task_id), int(r.state)): int(r.success) for r in b.itertuples()}

print("v4 setup loaded | dev states", DEV_STATES, "| test states", TEST_STATES, "| results ->", V4_DIR)
