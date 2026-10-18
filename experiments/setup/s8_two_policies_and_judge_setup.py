"""Set-up for the second policy, the VLM judge and the attached run: one runner for every gate variant,
GPU memory planning and the analysis tools.
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# %% CELL v7_A_setup — run once per session, right AFTER the cell "v6_A_setup". Seconds. Only defines tools; runs nothing heavy.
# Stage B: every gate variant is run through ONE runner (run_sweep7) so all variants see exactly the same scenes.
#   detector : how objects are found          ("gdino" = first paper, "hybrid" = v6 list routing, list-free cascades are added in Part 5)
#   depth    : where the depth comes from     ("sim" = simulator depth; estimated depth is added in Part 6)
#   parser   : how the sentence is read       ("rule" = the parser of the first paper; "rule7" / "llm" are added in Part 5)
#   phrasing : which wording the gate is given ("canonical" = LIBERO wording; "para" = reworded, added in Part 5)
import gc, copy, hashlib
V7_DIR = os.path.join(RESULTS_ROOT, "v7")
V7_FIG = os.path.join(V7_DIR, "figures")
for _d in (V7_DIR, V7_FIG):
    os.makedirs(_d, exist_ok=True)

DEV7_STATES = [0, 1, 2]                 # development scenes (already used while building the method)
TEST7_STATES = [10, 11, 12, 13, 14]     # NEW scenes, never used before: touched only after the v7 freeze
FROZEN7_JSON = os.path.join(V7_DIR, "frozen_config_v7.json")
DEV_CHOICES7_JSON = os.path.join(V7_DIR, "dev_choices_v7.json")
assert not (set(TEST7_STATES) & set(DEV7_STATES + FRESH_STATES + DEV_STATES + TEST_STATES)), "test states must be unused"

# ---------- development choices are stored on disk so a kernel restart cannot lose (or silently change) them ----------
def choices7():
    return json.load(open(DEV_CHOICES7_JSON)) if os.path.exists(DEV_CHOICES7_JSON) else {}

def set_choice7(key, value):
    """stores a development choice; once v7 is frozen nothing can change any more (the stored value is kept and shown)"""
    c = choices7()
    if os.path.exists(FROZEN7_JSON):
        print(f"   (v7 is frozen: the stored choice for '{key}' stays ->", json.dumps(c.get(key), default=float)[:200], ")")
        return
    c[key] = value
    json.dump(c, open(DEV_CHOICES7_JSON, "w"), indent=1, default=float)
    print(f"   development choice saved: {key} ->", json.dumps(value, default=float)[:300])

# ---------- GPU memory (8 GB): the detectors always stay on the GPU; only ONE robot policy (or the VLM) is there with them ----------
def _move7(name, dev):
    m = globals().get(name)
    if m is None or not hasattr(m, "to"):
        return
    try:
        m.to(dev)
    except Exception as e:
        print(f"   (could not move {name} to {dev}: {type(e).__name__}: {str(e)[:80]})")

def gpu_plan(mode):
    """'sweep'  : no robot policy on the GPU (gate-only work; the VLM may be loaded)
       'smolvla': SmolVLA on the GPU, X-VLA on the CPU, VLM unloaded (it reloads by itself when needed)
       'xvla'   : X-VLA on the GPU, SmolVLA on the CPU, VLM unloaded"""
    assert mode in ("sweep", "smolvla", "xvla")
    if device == "cpu":
        return
    if mode == "sweep":
        _move7("policy", "cpu"); _move7("xpolicy", "cpu")
    else:
        if globals().get("vlm") is not None:
            globals()["vlm"] = None                     # a 4-bit model cannot be moved; it is dropped and reloaded on demand
        _move7("xpolicy" if mode == "smolvla" else "policy", "cpu")
        _move7("policy" if mode == "smolvla" else "xpolicy", device)
    gc.collect()
    torch.cuda.empty_cache()

def gpu_report():
    if device != "cpu":
        print(f"   GPU memory in use: {torch.cuda.memory_allocated() / 1e9:.1f} GB (reserved {torch.cuda.memory_reserved() / 1e9:.1f} GB)")

def _on_cpu7(m):
    try:
        return next(m.parameters()).device.type == "cpu"
    except Exception:
        return False

# SmolVLA is put back on the GPU whenever it is asked for an action (every old cell that calls predict_action keeps working)
if "_predict_action_v6" not in globals():
    _predict_action_v6 = predict_action
def predict_action(obs, instruction):
    if device != "cpu" and _on_cpu7(policy):
        gpu_plan("smolvla")
    return _predict_action_v6(obs, instruction)

# the old sweep function parks the robot policies first (needed once the VLM shares the GPU)
if "_run_gate_sweep_v6" not in globals():
    _run_gate_sweep_v6 = run_gate_sweep
def run_gate_sweep(states, mode, path, oracle=False):
    gpu_plan("sweep")
    return _run_gate_sweep_v6(states, mode, path, oracle=oracle)

# ---------- registries (later cells add entries) ----------
if "_GRD_REAL7" not in globals():
    _GRD_REAL7 = get_real_depth_map                     # the simulator-depth function, kept so it can always be restored
DETECTORS7 = {"gdino": {"fn": _GO_REAL, "ramp": "gdino"},            # Grounding DINO for everything (first paper)
              "hybrid": {"fn": ground_object_hybrid, "ramp": "owl"}}  # v6: OWLv2 + Grounding DINO, hand-written routing list
DEPTHS7 = {"sim": None}                                 # name -> function(obs) returning a depth map in metres (flipped-image layout)
PARSERS7 = {"rule": lambda s: (parse_instruction(s), parse_goal(s))}
PHRASINGS7 = {"canonical": None}                        # name -> function(canonical_instruction, task_id, state, kind) -> wording
SCENE7 = {}                                             # scratch space for the scene being processed (caches, depth map)
ROUTE7 = {}                                             # which detector answered which phrase (filled by the cascade detectors)

V7_FIELDS = VERIFY_FIELDS + ["variant", "canonical_instruction", "parse_same", "t_depth_s", "reused"]

def _norm_parse7(parsed, goal):
    """comparable form of a parse: (target, relation, sorted references, goal object)"""
    refs = sorted(str(r).strip().lower() for r in _as_list(parsed.get("reference")))
    return (str(parsed.get("target") or "").strip().lower(), parsed.get("relation"), tuple(refs),
            str((goal or {}).get("object") or "").strip().lower())

def _depth_hook7(env, depth_buffer):
    return SCENE7["depth_F"]

# ---------- one scene -> one log row (same columns as the v4 / v6 logs, so decide_all() works on it) ----------
def gate_case7(env, task_id, state, kind, clean_instr, parser="rule", phrasing="canonical", depth="sim", split="dev"):
    global get_real_depth_map
    canon = perturb_instruction(clean_instr, kind, state) if kind in ("absent_reference", "absent_target") else clean_instr
    # the SCENE is always built from the canonical instruction, so every variant sees the identical scene
    obs = setup_episode(env, task_id, state, perturb=kind if kind in SCENE_PERTURBATIONS else None, instruction=canon)
    tkey, gkey = _true_keys(env, obs)
    p_can, g_can = parse_instruction(canon), parse_goal(canon)
    instr = canon if phrasing == "canonical" else PHRASINGS7[phrasing](canon, task_id, state, kind)
    parsed, goal = PARSERS7[parser](instr)
    cam = get_camera_model(env, obs)
    SCENE7.clear()
    SCENE7.update({"task_id": task_id, "state": state, "kind": kind, "vlm_cache": {}})
    _ORC6.update({"obs": obs, "cam": cam, "tkey": tkey, "depth": _GRD_REAL7(env, obs["agentview_depth"]),
                  "regions": {w: _fixture_region6(env, cam, w) for w in ("cabinet", "stove")}})
    refs_c, rel_c = _as_list(p_can.get("reference")), p_can.get("relation")
    row = {"task_id": task_id, "state": state, "split": split, "kind": kind,
           "valid": int(kind not in INVALID_KINDS_V4), "instruction": instr, "relation": parsed.get("relation"),
           "has_goal": int(bool(goal and goal.get("object"))),
           "label_uncertain": int((kind == "position_change" and any("plate" in (r or "") for r in refs_c))
                                  or (kind == "distractor" and rel_c == "near")),
           "t_policy_s": 0.0, "seconds_total": 0.0, "t_depth_s": 0.0, "canonical_instruction": canon,
           "parse_same": int(_norm_parse7(parsed, goal) == _norm_parse7(p_can, g_can))}
    try:
        if DEPTHS7[depth] is not None:                     # estimated depth: ONLY the gate sees it (the scene was built with the simulator)
            t0 = time.time()
            SCENE7["depth_F"] = DEPTHS7[depth](obs)
            row["t_depth_s"] = round(time.time() - t0, 3)
            get_real_depth_map = _depth_hook7
        for sfx, att in (("0", 0), ("1", 1)):
            t0 = time.time()
            sc = perceive_scene(env, obs, parsed, goal, cam, att, CFG)
            row[f"t_perceive{sfx}_s"] = round(time.time() - t0, 3)
            row.update(_scene_fields(sc, 1.0, sfx))                  # the action term has weight 0 in the frozen gate
            sel, rt, rr, rg, _ = _grade_scene(sc, obs, cam, parsed, kind, tkey, gkey)
            row.update({f"sel_{sfx}": sel, f"right_target_{sfx}": rt, f"right_ref_{sfx}": rr, f"right_goal_{sfx}": rg})
    finally:
        get_real_depth_map = _GRD_REAL7
    return row

def sweep_path7(tag, name):
    return os.path.join(V7_DIR, f"{tag}_{name}.csv")

def _exact_parse7(parsed, goal):
    """everything the gate reads from a parse, in order (used to decide whether a saved canonical result can be reused)"""
    return (str(parsed.get("target") or "").strip().lower(), parsed.get("relation"),
            tuple(str(r).strip().lower() for r in _as_list(parsed.get("reference"))), str((goal or {}).get("object") or "").strip().lower())

def run_sweep7(states, name, det="hybrid", depth="sim", parser="rule", phrasing="canonical", tag="dev", oracle=False, limit=None,
               base=None, only=None):
    """One gate variant over states x 10 tasks x 7 conditions -> results/v7/<tag>_<name>.csv. Resumable; always restores settings.
    base = name of the SAME gate's sweep on the canonical wording: when a reworded sentence is read exactly like the canonical one,
    the gate's inputs are identical, so that saved row is reused (column reused = 1) instead of simulating the scene again."""
    global ground_object, _pool, OBS_HOOK, get_real_depth_map
    for kind_, reg, val in (("detector", DETECTORS7, det), ("depth", DEPTHS7, depth), ("parser", PARSERS7, parser), ("phrasing", PHRASINGS7, phrasing)):
        assert val in reg, f"{kind_} '{val}' is not defined yet - run the cell that defines it first (known: {list(reg)})"
    path = sweep_path7(tag, name)
    meta = {"name": name, "det": det, "ramp": DETECTORS7[det]["ramp"], "depth": depth, "parser": parser, "phrasing": phrasing, "tag": tag}
    mpath = path.replace(".csv", ".meta.json")
    if os.path.exists(mpath):
        old = json.load(open(mpath))
        assert {k: old.get(k) for k in meta} == meta, f"{os.path.basename(path)} was started with other settings: {old}"
    else:
        json.dump(meta, open(mpath, "w"), indent=1)
    base_rows = {}
    if base is not None and phrasing != "canonical" and os.path.exists(sweep_path7(tag, base)):
        bm = json.load(open(sweep_path7(tag, base).replace(".csv", ".meta.json")))
        assert (bm["det"], bm["depth"], bm["phrasing"]) == (det, depth, "canonical"), f"'{base}' is not the same gate on the canonical wording"
        base_rows = {(int(r["task_id"]), int(r["state"]), r["kind"]): r for r in load6(sweep_path7(tag, base), states).to_dict("records")}
    save = {k: copy.deepcopy(CFG[k]) for k in _CFG_KEYS6}
    done = _done_keys(path, ["task_id", "state", "kind"])
    queue = [(t, s, k) for t in range(NUM_TASKS) for s in states for k in KINDS if (str(t), str(s), k) not in done]
    if only is not None:
        queue = [q for q in queue if q in set(only)]
    if limit is not None:
        queue = queue[:limit]
    print(f"\n=== {name} [{tag}] det={det} depth={depth} parser={parser} phrasing={phrasing}: "
          f"{len(queue)} scenes to run ({len(done)} already saved) -> {os.path.basename(path)}")
    if not queue:
        return path
    gpu_plan("sweep")
    ROUTE7.clear()
    n_reused = 0
    try:
        ground_object = DETECTORS7[det]["fn"]
        if DETECTORS7[det]["ramp"] == "owl":
            CFG.update({"min_conf": [OWL_RAMP[0]] * 3, "ground_min_conf": OWL_RAMP[0], "conf_floor": OWL_RAMP[0],
                        "conf_full": OWL_RAMP[1], "scene_vocab": list(HYB_SCENE_VOCAB)})
        _pool = _pool_oracle6 if oracle else _POOL_REAL6
        i, t_start, env, cur = 0, time.time(), None, None
        for (t, s, k) in queue:
            clean_instr = suite.get_task(t).language
            row = None
            if (t, s, k) in base_rows:
                canon = perturb_instruction(clean_instr, k, s) if k in ("absent_reference", "absent_target") else clean_instr
                instr = PHRASINGS7[phrasing](canon, t, s, k)
                if _exact_parse7(*PARSERS7[parser](instr)) == _exact_parse7(parse_instruction(canon), parse_goal(canon)):
                    row = {**base_rows[(t, s, k)], "instruction": instr, "canonical_instruction": canon, "parse_same": 1, "reused": 1}
                    n_reused += 1
            if row is None:
                if t != cur:
                    if env is not None:
                        env.close()
                    env, _, _ = get_task_env_and_instruction(t)
                    cur = t
                row = gate_case7(env, t, s, k, clean_instr, parser=parser, phrasing=phrasing, depth=depth, split=tag)
                row["reused"] = 0
            row["frozen_hash"], row["variant"] = name + ("_oracle" if oracle else ""), name
            _append(path, V7_FIELDS, row)
            i += 1
            print(f"[{i}/{len(queue)}] t{t} s{s} {k:18s} G={float(row['G_0']):.2f} S_rel={float(row['S_rel_0']):.2f} D_rel={float(row['D_rel_0']):.2f} "
                  f"sel={str(row['sel_0']):10s}{' (reused)' if row['reused'] else ''} (~{(time.time() - t_start) / i * (len(queue) - i) / 60:.1f} min left)")
        if env is not None:
            env.close()
    finally:
        ground_object = _GO_REAL                           # always put Grounding DINO, the simulator depth and the old settings back
        _pool = _POOL_REAL6
        get_real_depth_map = _GRD_REAL7
        CFG.update(save)
        OBS_HOOK = None
    if n_reused:
        print(f"   {n_reused} of {len(queue)} scenes reused the saved result of '{base}' (the sentence was read exactly like the canonical one)")
    if ROUTE7:
        rp = path.replace(".csv", ".routes.json")
        prev = json.load(open(rp)) if os.path.exists(rp) else {}
        for ph, c in ROUTE7.items():
            for kk, v in c.items():
                prev.setdefault(ph, {})[kk] = prev.get(ph, {}).get(kk, 0) + v
        json.dump(prev, open(rp, "w"), indent=1)
    return path

def verify_reuse7(name, tag, states, n=12, seed=0):
    """recomputes n reused scenes from scratch and compares them with the reused rows (the check that reusing is exact)"""
    meta = json.load(open(sweep_path7(tag, name).replace(".csv", ".meta.json")))
    d = load6(sweep_path7(tag, name), states)
    r = d[pd.to_numeric(d.reused, errors="coerce") == 1]
    if not len(r):
        print(f"   [{name}] no reused scenes -> nothing to verify")
        return True
    pick = r.sample(min(n, len(r)), random_state=seed)
    keys = [(int(x.task_id), int(x.state), x.kind) for x in pick.itertuples()]
    chk = name + "__check"
    for f in (sweep_path7(tag, chk), sweep_path7(tag, chk).replace(".csv", ".meta.json")):
        if os.path.exists(f):
            os.remove(f)
    run_sweep7(states, chk, det=meta["det"], depth=meta["depth"], parser=meta["parser"], phrasing=meta["phrasing"], tag=tag, only=keys)
    c = load6(sweep_path7(tag, chk), states)
    m = pick.merge(c, on=["task_id", "state", "kind"], suffixes=("", "_new"))
    cfg = gate_cfg7(meta["ramp"], True)
    same_dec = sum(gate_decide(a, cfg)[0] == gate_decide(b, cfg)[0] for a, b in zip(pick.sort_values(["task_id", "state", "kind"]).to_dict("records"),
                                                                                  c.sort_values(["task_id", "state", "kind"]).to_dict("records")))
    dmax = max(float((m[col].astype(float) - m[col + "_new"].astype(float)).abs().max()) for col in ("G_0", "S_rel_0", "D_rel_0", "G_1", "S_rel_1", "D_rel_1"))
    ok = same_dec == len(c) and dmax < 0.02
    print(f"   [{name}] reuse check on {len(c)} recomputed scenes: same decision in {same_dec}/{len(c)}, largest difference in G/S/D = {dmax:.4f} -> "
          + ("OK" if ok else "MISMATCH - check before using the reused rows"))
    return ok

def n_done7(tag, name, states):
    p = sweep_path7(tag, name)
    return len(load6(p, states)) if os.path.exists(p) else 0

def gate_cfg7(ramp="owl", reground=True):
    """the frozen gate of the first paper (weights 0.27/0.40/0.33, thresholds 0.70/0.50); only the confidence ramp follows the detector"""
    return {**gate_cfg6("hybrid" if ramp == "owl" else "gdino"), "reground": bool(reground)}

def decide7(tag, name, states, reground=True):
    meta = json.load(open(sweep_path7(tag, name).replace(".csv", ".meta.json")))
    d = decide_all(load6(sweep_path7(tag, name), states), gate_cfg7(meta["ramp"], reground))
    if not reground:
        d["reground_used"] = 0                           # no second grounding pass -> its time is not counted either
    return d

def youden7(dec):
    inv, val = dec[dec.valid == 0], dec[dec.valid == 1]
    return round(float(val.executed.mean() - inv.executed.mean()) * 100, 1)

def brief7(dec):
    """the four numbers used by every pre-set selection rule"""
    inv, val = dec[dec.valid == 0], dec[dec.valid == 1]
    return {"N": len(dec), "valid accepted %": pct(val.executed.mean()), "wrong acted %": pct(inv.executed.mean()),
            "J": youden7(dec), "AUROC": round(auroc(val.C, inv.C), 3)}

# ---------- progress overview ----------
def status7():
    """prints what exists on disk, so you always know where you are"""
    def n(path, cols=("task_id", "state")):
        try:
            return len(pd.read_csv(path).drop_duplicates(list(cols)))
        except Exception:
            return 0
    print("STAGE A (v6 test, states 5-9)")
    print(f"   gate decisions old / hybrid : {n(FRESH_GD_CSV, ('task_id','state','kind'))} / {n(FRESH_HY_CSV, ('task_id','state','kind'))} of 350")
    print(f"   SmolVLA rollouts            : {n(ROLL6_CSV, ('task_id','state','condition'))} of 200")
    print("STAGE B (results/v7)")
    for f in sorted(os.listdir(V7_DIR)):
        if f.endswith(".csv"):
            try:
                print(f"   {f:44s} {len(pd.read_csv(os.path.join(V7_DIR, f))):5d} rows")
            except Exception:
                print(f"   {f:44s} (empty)")
    c = choices7()
    print("development choices so far:", json.dumps(c, default=float) if c else "none yet")
    print("v7 frozen:", "YES - " + json.load(open(FROZEN7_JSON))["frozen_at"] if os.path.exists(FROZEN7_JSON) else "not yet")

print("v7 setup loaded | development states", DEV7_STATES, "| v6 test states", FRESH_STATES, "| NEW test states", TEST7_STATES, "| results ->", V7_DIR)
status7()


# ----------------------------------------------------------------------------------------------------------------------
# %% CELL v7_B_tools — analysis helpers used by every results cell. Run once per session after "v7_A_setup". Seconds.
def success_map7(path):
    """rollout file -> {(task, state, condition): 0/1}"""
    if not os.path.exists(path):
        return {}
    d = pd.read_csv(path).drop_duplicates(["task_id", "state", "condition"], keep="last")
    return {(int(r.task_id), int(r.state), r.condition): int(r.success) for r in d.itertuples()}

def fisher_exact7(a, b, c, d):
    """two-sided Fisher exact test for the table [[a, b], [c, d]] (no scipy needed)"""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0
    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))

def _attach7(dec, smap):
    out = dec.copy()
    out["success"] = [smap.get((int(r.task_id), int(r.state), r.kind)) for r in out.itertuples()]
    return out

def outcome7(valid, executed, s):
    """per scene: 'stop' (gate refused), 'bad' (acted on a wrong instruction, or acted and failed), 'success'; None if no rollout exists"""
    if not executed:
        return "stop"
    if valid == 0:
        return "bad"
    return None if s is None or s != s else ("success" if s == 1 else "bad")

def always_act7(dec):
    """the policy WITHOUT any gate, written as a 'gate' that accepts everything (so it goes through the same code)"""
    a = dec.copy()
    a["executed"] = 1
    a["decision"] = "EXECUTE"
    return a

def policy_gate_table7(gates, policies, title, save_as=None):
    """gates: {name: decided dataframe (same scenes)}, policies: {name: success map from rollouts}. One row per policy x gate."""
    rows = []
    for pname, smap in policies.items():
        if not smap:
            continue
        first = next(iter(gates.values()))
        for gname, dec in [("no gate (always acts)", always_act7(first))] + list(gates.items()):
            d = _attach7(dec, smap)
            inv, val = d[d.valid == 0], d[d.valid == 1]
            known = val[val.success.notna()]
            acc, ref = known[known.executed == 1], known[known.executed == 0]
            k_inv = int(inv.executed.sum())
            oc = [outcome7(r.valid, r.executed, r.success) for r in d.itertuples()]
            oc = [o for o in oc if o is not None]
            rows.append({"policy": pname, "gate": gname,
                         "wrong instr. acted on %": pct(k_inv / len(inv)) if len(inv) else float("nan"),
                         "  95% CI": str(wilson(k_inv, len(inv))),
                         "right instr. accepted %": pct(val.executed.mean()) if len(val) else float("nan"),
                         "task success on right instr. %": pct((known.executed * known.success).mean()) if len(known) else float("nan"),
                         "success when acting %": pct(acc.success.mean()) if len(acc) else float("nan"),
                         "success of the REFUSED %": pct(ref.success.mean()) if len(ref) else float("nan"),
                         "selection p (Fisher)": (round(fisher_exact7(int(acc.success.sum()), int(len(acc) - acc.success.sum()),
                                                                     int(ref.success.sum()), int(len(ref) - ref.success.sum())), 4)
                                                  if len(acc) and len(ref) else float("nan")),
                         "rolled out": f"{len(known)}/{len(val)}",
                         "all scenes: success %": pct(np.mean([o == "success" for o in oc])) if oc else float("nan"),
                         "all scenes: stopped %": pct(np.mean([o == "stop" for o in oc])) if oc else float("nan"),
                         "all scenes: bad %": pct(np.mean([o == "bad" for o in oc])) if oc else float("nan")})
    tab = pd.DataFrame(rows)
    print("\n" + title)
    if len(tab):
        print(tab.set_index(["policy", "gate"]).T.to_string())
        if save_as:
            tab.to_csv(os.path.join(V7_DIR, save_as), index=False)
    else:
        print("   (no rollouts found yet)")
    return tab

def head_to_head7(dec_a, dec_b, smap, name_a, name_b):
    """scene-by-scene comparison of two gates in front of the same policy (success > stop > bad); exact sign test"""
    rank = {"success": 2, "stop": 1, "bad": 0}
    a, b = _attach7(dec_a, smap), _attach7(dec_b, smap)
    m = a[["task_id", "state", "kind", "valid", "executed", "success"]].merge(
        b[["task_id", "state", "kind", "executed"]], on=["task_id", "state", "kind"], suffixes=("_a", "_b"))
    wa = wb = n = 0
    for r in m.itertuples():
        oa, ob = outcome7(r.valid, r.executed_a, r.success), outcome7(r.valid, r.executed_b, r.success)
        if oa is None or ob is None:
            continue
        n += 1
        wa += rank[oa] > rank[ob]
        wb += rank[ob] > rank[oa]
    return f"{name_a} better on {wa}, same on {n - wa - wb}, {name_b} better on {wb} of {n} scenes (exact sign test p = {mcnemar_exact(wa, wb):.2e})"

def per_condition7(gates, policies, title):
    """acted % per condition for every gate, plus each policy's success there"""
    rows = []
    for kind in KINDS:
        first = next(iter(gates.values()))
        row = {"condition": kind, "N": int((first.kind == kind).sum())}
        for gname, dec in gates.items():
            g = dec[dec.kind == kind]
            row[f"{gname}: acted %"] = pct(g.executed.mean()) if len(g) else float("nan")
        if kind in VALID_KINDS:
            for pname, smap in policies.items():
                s = [smap.get((int(r.task_id), int(r.state), r.kind)) for r in first[first.kind == kind].itertuples()]
                s = [x for x in s if x is not None]
                row[f"{pname} success % (n)"] = f"{pct(np.mean(s))} ({len(s)})" if s else "-"
        rows.append(row)
    tab = pd.DataFrame(rows)
    print("\n" + title)
    print(tab.to_string(index=False))
    return tab

def no_depth7(df):
    """offline ablation: the depth evidence D is replaced by the 2D evidence S (the object choice in the log is unchanged)"""
    d = df.copy()
    for sfx in ("0", "1"):
        d[f"D_rel_{sfx}"], d[f"D_{sfx}"] = d[f"S_rel_{sfx}"], d[f"S_{sfx}"]
    return d

def summary7(res, title, save_as=None):
    """one row per variant; works for gate logs and for the VLM judge (which has no boxes, so some columns stay empty)"""
    rows = []
    for name, d in res.items():
        inv, val = d[d.valid == 0], d[d.valid == 1]
        k = int(inv.executed.sum())
        row = {"variant": name, "scenes": len(d), "right instr. accepted %": pct(val.executed.mean()),
               "wrong instr. acted on %": pct(k / len(inv)) if len(inv) else float("nan"), "  95% CI": str(wilson(k, len(inv))),
               "J": youden7(d), "AUROC": round(auroc(val.C, inv.C), 3)}
        if "right_target_0" in d.columns:
            row["right bowl % (valid)"] = pct(pd.to_numeric(d[d.kind.isin(VALID_KINDS)].right_target_0, errors="coerce").mean())
            ok = [int(r.executed == 1 and getattr(r, f"sel_{r.scene_used}") == "target") for r in val.itertuples()]
            row["accepted on the right bowl %"] = pct(np.mean(ok)) if ok else float("nan")
        if "parse_same" in d.columns and d.parse_same.notna().any():
            row["sentence read correctly %"] = pct(pd.to_numeric(d.parse_same, errors="coerce").mean())
        if "t_perceive0_s" in d.columns:
            sec = d.t_perceive0_s + d.t_perceive1_s * d.reground_used
            if "t_depth_s" in d.columns:
                sec = sec + pd.to_numeric(d.t_depth_s, errors="coerce").fillna(0)
            row["seconds per scene"] = round(float(sec.mean()), 2)
        elif "seconds" in d.columns:
            row["seconds per scene"] = round(float(d.seconds.mean()), 2)
        rows.append(row)
    tab = pd.DataFrame(rows)
    print("\n" + title)
    print(tab.set_index("variant").T.to_string())
    if save_as:
        tab.to_csv(os.path.join(V7_DIR, save_as), index=False)
    return tab

def both7(dec_a, dec_b):
    """a combined checker: the robot acts only if BOTH checkers say go (score = the smaller of the two scores)"""
    m = dec_a[["task_id", "state", "kind", "valid", "executed", "C"]].merge(
        dec_b[["task_id", "state", "kind", "executed", "C"]], on=["task_id", "state", "kind"], suffixes=("_a", "_b"))
    m["executed"] = (m.executed_a & m.executed_b).astype(int)
    m["C"] = m[["C_a", "C_b"]].min(axis=1)
    m["decision"] = np.where(m.executed == 1, "EXECUTE", "ABSTAIN")
    return m

def mcnemar7(a, b, name_a, name_b):
    for v, what in ((1, "right instructions"), (0, "wrong instructions")):
        x, y, p = paired6(a, b, v)
        print(f"   {what}: only {name_a} lets the robot act in {x}; only {name_b} in {y}; exact McNemar p = {p:.4f}")

# ---------- the VLM judge's saved answers as a decision table (needs no VLM: it only reads the saved file) ----------
VLM_JUDGE_JSON = os.path.join(V7_DIR, "vlm_judge_choice.json")      # the judge's setting, fixed once on development scenes

def vlm_judge_choice7():
    return json.load(open(VLM_JUDGE_JSON)) if os.path.exists(VLM_JUDGE_JSON) else None

def vlm_decisions7(tag, states, score, thr):
    """the judge's log as a decision table with the same columns the gate tables use (executed, C = score)"""
    d = pd.read_csv(os.path.join(V7_DIR, f"vlm_judge_{tag}.csv")).drop_duplicates(["task_id", "state", "kind"], keep="last")
    d = d[d.state.isin(states)].reset_index(drop=True)
    d["C"] = d[f"p_{score}"].astype(float)
    d["executed"] = (d.C >= thr).astype(int)
    d["decision"] = np.where(d.executed == 1, "EXECUTE", "ABSTAIN")
    return d
print("v7 analysis tools loaded")
