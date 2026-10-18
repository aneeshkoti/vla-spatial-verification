"""Study 2 set-up: OWLv2, the hybrid detector, the gate runner and the comparison tables.
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# %% CELL v6_A_setup — run once per session, AFTER the usual setup cells (top of notebook ... "v4 setup"). Seconds.
# Defines the hybrid detector (OWLv2 + Grounding DINO), the gate runner and the comparison table used by cells B-G.
import copy, hashlib
V6_DIR = os.path.join(RESULTS_ROOT, "v6")
os.makedirs(V6_DIR, exist_ok=True)
DEV6_STATES = [0, 1, 2]                 # development scenes (already used to choose the detector)
FRESH_STATES = [5, 6, 7, 8, 9]          # fresh test scenes (never used before) -> 350 cases
OWL_RAMP = (0.10, 0.25)                 # OWLv2 confidence scale: 0.10 = "seen", 0.25 = "sure"
GD_PHRASES = {"wooden cabinet", "cabinet", "drawer", "stove", "cooktop", "burner", "cookie box", "box", "carton"}
GD_FALLBACK = {"plate", "white plate", "dish"}
HYB_DEV_CSV = os.path.join(V6_DIR, "verify_owl_hybrid.csv")          # hybrid gate, development scenes
FRESH_GD_CSV = os.path.join(V6_DIR, "fresh_verify_gdino.csv")        # old gate, fresh scenes
FRESH_HY_CSV = os.path.join(V6_DIR, "fresh_verify_hybrid.csv")       # hybrid gate, fresh scenes
FRESH_ORC_CSV = os.path.join(V6_DIR, "fresh_verify_hybrid_oracle.csv")
ROLL6_CSV = os.path.join(V6_DIR, "fresh_rollouts.csv")               # SmolVLA rollouts, fresh scenes
FROZEN6_JSON = os.path.join(V6_DIR, "frozen_config_v6.json")
DEV_CHOICE_JSON = os.path.join(V6_DIR, "dev_choice.json")           # which hybrid version was chosen on development
_CFG_KEYS6 = ("min_conf", "ground_min_conf", "conf_floor", "conf_full", "scene_vocab", "max_area_frac")
# contrast vocabulary used by the hybrid gate: [] = contrast check off. Restored from the frozen file / dev choice if present.
HYB_SCENE_VOCAB = []
if os.path.exists(FROZEN6_JSON):
    HYB_SCENE_VOCAB = list(json.load(open(FROZEN6_JSON))["detector"]["scene_vocab"])
elif os.path.exists(DEV_CHOICE_JSON):
    HYB_SCENE_VOCAB = list(json.load(open(DEV_CHOICE_JSON))["scene_vocab"])

# --- load OWLv2 ---
from transformers import Owlv2Processor, Owlv2ForObjectDetection
if "_owl_mdl" not in globals():
    _owl_proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    _owl_mdl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()
    print("OWLv2 loaded")
# --- end load ---
if "_GO_REAL" not in globals():
    _GO_REAL = ground_object                      # the real Grounding DINO function (kept once)
if "_POOL_REAL6" not in globals():
    _POOL_REAL6 = globals().get("_POOL_REAL", _pool)

def ground_object_owl(image_pil, phrase, box_threshold=0.10, text_threshold=None):
    """same output format as ground_object(), OWLv2 inside"""
    inp = _owl_proc(text=[["a photo of a " + phrase.lower().strip()]], images=image_pil, return_tensors="pt",
                    truncation=True, max_length=16).to(device)
    with torch.no_grad():
        out = _owl_mdl(**inp)
    ts = torch.tensor([[max(image_pil.size), max(image_pil.size)]], device=device)
    try:
        r = _owl_proc.post_process_grounded_object_detection(outputs=out, target_sizes=ts, threshold=box_threshold)[0]
    except Exception:
        r = _owl_proc.post_process_object_detection(outputs=out, target_sizes=ts, threshold=box_threshold)[0]
    return [{"bbox": [round(v, 1) for v in b.tolist()], "confidence": round(float(s), 3), "label": phrase}
            for b, s in zip(r["boxes"], r["scores"])]

def _gd_rescaled(image_pil, phrase, box_threshold):
    """Grounding DINO boxes with their confidence moved onto OWLv2's scale (0.22-0.40 -> OWL_RAMP)."""
    out = []
    for d in _GO_REAL(image_pil, phrase, box_threshold=0.22, text_threshold=0.20):
        c = OWL_RAMP[0] + (float(d["confidence"]) - 0.22) * (OWL_RAMP[1] - OWL_RAMP[0]) / (0.40 - 0.22)
        if c >= box_threshold:
            out.append({**d, "confidence": round(float(c), 3)})
    return out

def ground_object_hybrid(image_pil, phrase, box_threshold=0.10, text_threshold=None):
    """OWLv2 for bowls / ramekin / plate; Grounding DINO for cabinet, stove, cookie box (and the plate if OWLv2 finds none)."""
    p = phrase.lower().strip()
    if p in GD_PHRASES:
        return _gd_rescaled(image_pil, phrase, box_threshold)
    dets = ground_object_owl(image_pil, phrase, box_threshold)
    if not dets and p in GD_FALLBACK:
        dets = _gd_rescaled(image_pil, phrase, box_threshold)
    return dets

# ---------- oracle object identity (analysis only), now covering the cabinet and the stove ----------
_ORC6 = {}
def _fixture_region6(env, cam, word):
    """image region of a fixed object (cabinet / stove) that has no position in obs: box around its projected parts"""
    m, d = env.sim.model, env.sim.data
    pts = []
    for gi in range(m.ngeom):
        bname = m.body_id2name(int(m.geom_bodyid[gi])) or ""
        if word in bname:
            r = cam.world_to_f(np.asarray(d.geom_xpos[gi], float))
            if r is not None:
                pts.append([r[0], r[1]])
    if not pts:
        return None
    pts = np.clip(np.array(pts, float), 0, IMG_W)
    x1, y1 = pts.min(0); x2, y2 = pts.max(0)
    return [float(x1), float(y1), float(x2), float(y2)] if (x2 - x1 > 4 and y2 - y1 > 4) else None

def _pool_oracle6(img, phrase, attempt, cfg, cache):
    dets = _POOL_REAL6(img, phrase, attempt, cfg, cache)
    obs, cam = _ORC6["obs"], _ORC6["cam"]
    p = phrase.lower()
    if "bowl" in p:
        k = _ORC6["tkey"]
        if k is None or k not in obs:
            return []
        uv = cam.world_to_f(np.asarray(obs[k], float))
        return [d for d in dets if _contains(d["bbox"], uv, pad=0)]
    rk = _ref_key(phrase, obs)
    if rk is not None and rk in obs:
        gt = np.asarray(obs[rk], float)
        uv = cam.world_to_f(gt)
        keep = []
        for d in dets:
            ok = _contains(d["bbox"], uv, pad=12)
            if not ok:
                m = measure_detection(d, _ORC6["depth"], cam)
                ok = float(np.linalg.norm(np.asarray(m["pos"])[:2] - gt[:2])) < 0.10
            if ok:
                keep.append(d)
        return keep
    word = "cabinet" if ("cabinet" in p or "drawer" in p) else ("stove" if ("stove" in p or "cooktop" in p or "burner" in p) else None)
    if word is None:
        return []                                          # the object does not exist in the scene
    reg = _ORC6["regions"].get(word)
    if reg is None:
        return dets                                        # identity cannot be checked -> leave the detector's answer
    keep = []
    for d in dets:
        b = d["bbox"]
        c = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        rc = ((reg[0] + reg[2]) / 2.0, (reg[1] + reg[3]) / 2.0)
        if _contains(reg, c, pad=10) or _contains(b, rc, pad=0):
            keep.append(d)
    return keep

# ---------- one scene -> one log row (same columns as verify_v4.csv, so decide_all() works on it) ----------
def gate_case6(env, task_id, state, kind, clean_instr):
    instr = perturb_instruction(clean_instr, kind, state) if kind in ("absent_reference", "absent_target") else clean_instr
    obs = setup_episode(env, task_id, state, perturb=kind if kind in SCENE_PERTURBATIONS else None, instruction=instr)
    tkey, gkey = _true_keys(env, obs)
    parsed, goal = parse_instruction(instr), parse_goal(instr)
    cam = get_camera_model(env, obs)
    _ORC6.update({"obs": obs, "cam": cam, "tkey": tkey, "depth": get_real_depth_map(env, obs["agentview_depth"]),
                  "regions": {w: _fixture_region6(env, cam, w) for w in ("cabinet", "stove")}})
    refs = _as_list(parsed.get("reference"))
    rel = parsed.get("relation")
    row = {"task_id": task_id, "state": state, "split": "dev" if state in DEV6_STATES else "fresh", "kind": kind,
           "valid": int(kind not in INVALID_KINDS_V4), "instruction": instr, "relation": rel,
           "has_goal": int(bool(goal and goal.get("object"))),
           "label_uncertain": int((kind == "position_change" and any("plate" in (r or "") for r in refs))
                                  or (kind == "distractor" and rel == "near")),
           "t_policy_s": 0.0, "seconds_total": 0.0}
    for sfx, att in (("0", 0), ("1", 1)):
        t0 = time.time()
        sc = perceive_scene(env, obs, parsed, goal, cam, att, CFG)
        row[f"t_perceive{sfx}_s"] = round(time.time() - t0, 3)
        row.update(_scene_fields(sc, 1.0, sfx))                      # the action term has weight 0 in the frozen gate
        sel, rt, rr, rg, _ = _grade_scene(sc, obs, cam, parsed, kind, tkey, gkey)
        row.update({f"sel_{sfx}": sel, f"right_target_{sfx}": rt, f"right_ref_{sfx}": rr, f"right_goal_{sfx}": rg})
    return row

def run_gate_sweep(states, mode, path, oracle=False):
    """mode 'gdino' = the gate of the first paper; 'hybrid' = OWLv2 + Grounding DINO. Resumable; always restores settings."""
    global ground_object, _pool, OBS_HOOK
    assert mode in ("gdino", "hybrid")
    save = {k: copy.deepcopy(CFG[k]) for k in _CFG_KEYS6}
    done = _done_keys(path, ["task_id", "state", "kind"])
    n_todo = sum((str(t), str(s), k) not in done for t in range(NUM_TASKS) for s in states for k in KINDS)
    print(f"\n=== {mode}{' + oracle' if oracle else ''}: {n_todo} scenes to run ({len(done)} already saved) -> {os.path.basename(path)}")
    try:
        if mode == "hybrid":
            ground_object = ground_object_hybrid
            CFG.update({"min_conf": [OWL_RAMP[0]] * 3, "ground_min_conf": OWL_RAMP[0], "conf_floor": OWL_RAMP[0],
                        "conf_full": OWL_RAMP[1], "scene_vocab": list(HYB_SCENE_VOCAB)})
        else:
            ground_object = _GO_REAL
        _pool = _pool_oracle6 if oracle else _POOL_REAL6
        i, t_start = 0, time.time()
        for task_id in range(NUM_TASKS):
            todo = [(s, k) for s in states for k in KINDS if (str(task_id), str(s), k) not in done]
            if not todo:
                continue
            env, _, clean_instr = get_task_env_and_instruction(task_id)
            for s, k in todo:
                row = gate_case6(env, task_id, s, k, clean_instr)
                row["frozen_hash"] = mode + ("_oracle" if oracle else "")
                _append(path, VERIFY_FIELDS, row)
                i += 1
                print(f"[{i}/{n_todo}] t{task_id} s{s} {k:18s} G={row['G_0']:.2f} S_rel={row['S_rel_0']:.2f} D_rel={row['D_rel_0']:.2f} "
                      f"sel={row['sel_0']:10s} (~{(time.time() - t_start) / i * (n_todo - i) / 60:.1f} min left)")
            env.close()
    finally:
        ground_object = _GO_REAL                           # always put Grounding DINO and the old settings back
        _pool = _POOL_REAL6
        CFG.update(save)
        OBS_HOOK = None

# ---------- loading + deciding + comparing ----------
def load6(path, states):
    d = pd.read_csv(path).drop_duplicates(["task_id", "state", "kind"], keep="last")
    return d[d.state.isin(states)].reset_index(drop=True)

def gate_cfg6(mode):
    f = json.load(open(FROZEN_JSON))                       # weights / thresholds of the first paper (unchanged)
    c = {**f, "weights": tuple(f["weights"]), "ramp": tuple(f["ramp"])}
    return {**c, "ramp": OWL_RAMP} if mode == "hybrid" else c

def compare_gates6(res, success_map=None, title=""):
    """res: {name: decided dataframe}. Prints the comparison and returns the summary table."""
    rows = []
    for name, dec in res.items():
        s = summarize(dec, success_map)
        val = dec[dec.kind.isin(VALID_KINDS)]
        inv = dec[dec.valid == 0]
        k = int(inv.executed.sum())
        v1 = dec[dec.valid == 1]
        ok_bowl = [int(r.executed == 1 and getattr(r, f"sel_{r.scene_used}") == "target") for r in v1.itertuples()]
        rows.append({"gate": name, "scenes": len(dec),
                     "right bowl % (valid)": pct(pd.to_numeric(val.right_target_0, errors="coerce").mean()),
                     "right reference %": pct(pd.to_numeric(dec.right_ref_0, errors="coerce").mean()),
                     "RIGHT instructions accepted %": s["valid_accept %"],
                     "  ...accepted on the RIGHT bowl %": pct(np.mean(ok_bowl)) if ok_bowl else float("nan"),
                     "WRONG instructions acted on % (lower=better)": s["IER %"],
                     "  95% CI": str(wilson(k, len(inv))),
                     "nominal success %": s.get("clean_TSR %"), "AUROC": s["AUROC(C)"], "J": s["J"],
                     "seconds per scene": round(float((dec.t_perceive0_s + dec.t_perceive1_s * dec.reground_used).mean()), 2)})
    summ = pd.DataFrame(rows)
    print("\n" + title)
    print(summ.set_index("gate").T.to_string())
    print("\nrobot allowed to act, per condition (%):")
    print(pd.DataFrame({n: d.groupby("kind").executed.mean().mul(100).round(0) for n, d in res.items()}).reindex(KINDS).to_string())
    print("\nRIGHT instructions accepted, per task (%):")
    print(pd.DataFrame({n: d[d.valid == 1].groupby("task_id").executed.mean().mul(100).round(0) for n, d in res.items()}).to_string())
    print("\nright bowl on valid scenes, per task (%):")
    print(pd.DataFrame({n: pd.to_numeric(d[d.kind.isin(VALID_KINDS)].right_target_0, errors="coerce").groupby(d.task_id).mean().mul(100).round(0)
                        for n, d in res.items()}).to_string())
    return summ

def paired6(a, b, mask_col_value):
    """exact McNemar between two gates on the same scenes. returns (only_a_acts, only_b_acts, p)"""
    m = a[["task_id", "state", "kind", "valid", "executed"]].merge(b[["task_id", "state", "kind", "executed"]],
                                                                    on=["task_id", "state", "kind"], suffixes=("_a", "_b"))
    m = m[m.valid == mask_col_value]
    x = int(((m.executed_a == 1) & (m.executed_b == 0)).sum()); y = int(((m.executed_a == 0) & (m.executed_b == 1)).sum())
    return x, y, mcnemar_exact(x, y)

print("hybrid contrast vocabulary:", HYB_SCENE_VOCAB if HYB_SCENE_VOCAB else "[] (contrast check off)")
print("v6 setup loaded | development states", DEV6_STATES, "| fresh test states", FRESH_STATES, "| results ->", V6_DIR)
