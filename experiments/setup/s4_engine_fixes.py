"""Fixes v3.1 and v3.2: the final perceive_scene used in every experiment of the paper.
Code exactly as used for the paper; only the result paths were made relative (see s0_paths.py)."""
# FIXES v3.1: resampling OFF + Task 26 crash fix
CFG["resample_k"] = 0

if "_free_spot_v30" not in globals():
    _free_spot_v30 = free_spot

def free_spot(obs, exclude_key, far_from=None, far_min=0.20, env=None, z=None):
    for fm, use_vis in ((far_min, True), (0.75 * far_min, True), (0.75 * far_min, False), (0.10, False)):
        p = _free_spot_v30(obs, exclude_key, far_from, fm, env if use_vis else None, z if use_vis else None)
        if p is not None:
            return p
    return _free_spot_v30(obs, exclude_key, None, 0.0, None, None)

print("fixes v3.1 applied: resample_k =", CFG["resample_k"])


# ----------------------------------------------------------------------------------------------------------------------
# FIXES v3.2 — from the real DINO diagnostic
CFG["resample_k"] = 0
CFG["use_head_noun"] = False
CFG["scene_vocab"] = []
CFG["ref_max_above_table"] = 0.30

def perceive_scene(env, obs, parsed, goal, cam, attempt=0, cfg=None):
    cfg = cfg or CFG
    img = Image.fromarray(obs["agentview_image"][::-1, ::-1])
    depth_F = get_real_depth_map(env, obs["agentview_depth"])
    rel = parsed.get("relation")
    tphrase = parsed.get("target")
    ref_phrases = _as_list(parsed.get("reference"))
    gphrase = goal.get("object") if goal else None
    stacked = rel in STACK_RELATIONS
    cache = {}
    obj_phrases = [p for p in [tphrase] + ref_phrases + [gphrase] if p and not is_table_center(p)]
    vocab = list(dict.fromkeys(obj_phrases + list(cfg.get("scene_vocab", []))))

    def competitors(p):
        out = []
        for q in vocab:
            if q == p or (_words(p) & _words(q)):
                continue
            if stacked and {p, q} <= set([tphrase] + ref_phrases):
                continue
            out.append(q)
        return out

    def pool_for(p, with_head_noun=False):
        dets = list(_pool(img, p, attempt, cfg, cache))
        head = p.split()[-1]
        if with_head_noun and head != p:
            dets += [{**d, "confidence": d["confidence"] * cfg["head_noun_scale"], "phrase_used": head}
                     for d in _pool(img, head, attempt, cfg, cache)]
            dets = clean_detections(dets, 0.0, allow_large=False, cfg=cfg)
        dets = _contrast_filter(dets, p, competitors(p), img, cfg, cache)
        return [measure_detection(d, depth_F, cam) for d in dets]

    t_recs = pool_for(tphrase, with_head_noun=cfg.get("use_head_noun", False)) if tphrase else []
    ref_recs = {}
    for rp in ref_phrases:
        if is_table_center(rp):
            ref_recs[rp] = [measure_detection(d, depth_F, cam) for d in _table_center_candidates(img, attempt, cfg)]
        else:
            ref_recs[rp] = sorted(pool_for(rp), key=lambda r: r["conf"], reverse=True)[: cfg["max_ref_candidates"]]
    goal_recs = pool_for(gphrase) if gphrase else []

    # a reference must be a table-top object (the robot body is not a 'stove')
    zs = [float(g["pos"][2]) for g in goal_recs] or [float(t["pos"][2]) for t in t_recs]
    if zs:
        table_z = min(zs)
        for rp in ref_recs:
            if not is_table_center(rp):
                ref_recs[rp] = [r for r in ref_recs[rp] if float(r["pos"][2]) <= table_z + cfg.get("ref_max_above_table", 0.30)]

    scored = []
    for t in t_recs:
        options = []
        for rp in ref_phrases:
            opts = [r for r in ref_recs[rp] if _pair_ok(t, r, stacked, cfg)]
            options.append(opts if opts else [None])
        best = None
        for combo in itertools.product(*options):
            combo = list(combo)
            if len(combo) == 2 and None not in combo and box_iou(combo[0]["box_F"], combo[1]["box_F"]) > cfg["exclusive_iou"]:
                continue
            s_rel = relation_score_2d(rel, t, combo) if rel else 0.0
            d_rel = relation_score_3d(rel, t, combo, cam) if rel else 0.0
            w = float(np.mean([0.7 + 0.3 * r["g"] if r is not None else 0.0 for r in combo])) if combo else 0.0
            sc = 0.5 * (s_rel + d_rel) * w
            if best is None or sc > best[0]:
                best = (sc, combo, s_rel, d_rel)
        if best is None:
            best = (0.0, [None] * len(ref_phrases), 0.0, 0.0)
        scored.append((best[0], t, best[1], best[2], best[3]))
    scored.sort(key=lambda x: x[0] + 0.05 * x[1]["g"], reverse=True)

    ambig = 1.0
    if scored:
        best_rel, target, refs, S_rel, D_rel = scored[0]
        second = scored[1][0] if len(scored) > 1 else None
        if second is not None:
            ambig = 0.5 + 0.5 * ramp_up(best_rel - second, cfg["ambig_lo"], cfg["ambig_hi"])
        S_rel, D_rel = S_rel * ambig, D_rel * ambig
    else:
        target, S_rel, D_rel, second = None, 0.0, 0.0, None
        refs = [ref_recs[rp][0] if ref_recs[rp] else None for rp in ref_phrases]

    goal_rec = None
    gc = [g for g in goal_recs if target is None or box_iou(g["box_F"], target["box_F"]) < cfg["exclusive_iou"]]
    if gc:
        goal_rec = max(gc, key=lambda g: g["conf"])

    gs = [target["g"] if target else 0.0] + [r["g"] if r else 0.0 for r in refs]
    if gphrase:
        gs.append(goal_rec["g"] if goal_rec else 0.0)
    G = float(np.prod(gs) ** (1.0 / len(gs))) if gs else 0.0
    pair_g = min([target["g"] if target else 0.0] + [r["g"] if r else 0.0 for r in refs]) if refs else 0.0
    rel_trust = math.sqrt(max(pair_g, 0.0))
    S, D = S_rel * rel_trust, D_rel * rel_trust
    return {
        "G": round(G, 4), "S": round(S, 4), "D": round(D, 4),
        "S_rel": round(S_rel, 4), "D_rel": round(D_rel, 4), "pair_g": round(pair_g, 4),
        "target": target, "refs": refs, "goal": goal_rec, "relation": rel,
        "n_target_candidates": len(t_recs), "second_best": second, "ambiguity": round(ambig, 3),
        "candidates": [{"rel": round(x[0], 3), "s": round(x[3], 3), "d": round(x[4], 3), "conf": round(x[1]["conf"], 3),
                        "phrase": x[1]["phrase"], "box": [round(v) for v in x[1]["box_F"]],
                        "world": np.round(x[1]["pos"], 3).tolist()} for x in scored],
        "attempt": attempt,
    }
