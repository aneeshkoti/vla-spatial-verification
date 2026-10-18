"""Scene verification: ground the named objects and measure the evidence G (grounding), S (2D relation), D (3D relation).

This is the final version of the experiment code's perceive_scene (engine v3 with fixes v3.1 / v3.2):
  * every bowl candidate is scored against the relation; (bowl, reference) pairs are chosen jointly
  * the evidence is measured once, before the robot moves
  * relation evidence is weighted by how sure we are that both objects were found (sqrt of the weaker g)
  * an ambiguity penalty when two bowls fit the relation almost equally well
  * a box only counts as object X if no other object of the instruction is detected more confidently on the same box
"""
import itertools
import math

import numpy as np

from .config import ENGINE_CFG, IMG_H, IMG_W
from .detectors import ground_phrase, clean_detections
from .geometry import box_iou, box_wh, measure_detection, ramp_up
from .parsing import as_list, is_table_center
from .relations import relation_score_2d, relation_score_3d

STACK_RELATIONS = ("on", "in")


def _table_center_candidates(detector, image_pil, attempt, cfg):
    min_conf = cfg["min_conf"][min(attempt, len(cfg["min_conf"]) - 1)]
    dets = [d for d in detector(image_pil, "table", box_threshold=min_conf, text_threshold=cfg["text_threshold"])]
    dets = clean_detections(dets, min_conf, allow_large=True, cfg=cfg)
    if not dets:   # fallback: image centre, low confidence
        return [{"bbox": [IMG_W / 2 - 4, IMG_H / 2 - 4, IMG_W / 2 + 4, IMG_H / 2 + 4],
                 "confidence": cfg["conf_floor"] + 0.5 * (cfg["conf_full"] - cfg["conf_floor"]), "phrase_used": "image center"}]
    x1, y1, x2, y2 = dets[0]["bbox"]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    return [{"bbox": [cx - 4, cy - 4, cx + 4, cy + 4], "confidence": dets[0]["confidence"], "phrase_used": "table"}]


def _words(p):
    return set(p.lower().split())


def _pool(detector, img, phrase, attempt, cfg, cache):
    """cleaned detections of one phrase (cached per image + attempt)."""
    key = (phrase, attempt)
    if key not in cache:
        cache[key] = ground_phrase(detector, img, phrase, attempt, cfg, min_conf=cfg["ground_min_conf"])
    return cache[key]


def _contrast_filter(detector, dets, phrase, competitors, img, cfg, cache):
    """drop a box if another object is detected on the SAME box more confidently."""
    keep = []
    for d in dets:
        rival = 0.0
        for q in competitors:
            for e in _pool(detector, img, q, 0, cfg, cache):
                if box_iou(d["bbox"], e["bbox"]) > cfg["contrast_iou"]:
                    rival = max(rival, float(e["confidence"]))
        if rival <= float(d["confidence"]) + cfg["contrast_margin"]:
            keep.append(d)
    return keep


def _pair_ok(t, r, stacked, cfg):
    """may bowl t be paired with reference box r?"""
    iou = box_iou(r["box_F"], t["box_F"])
    if not stacked:
        return iou < cfg["exclusive_iou"]                            # next-to / between: different objects
    if iou < cfg["stack_iou"]:
        return True
    # (almost) the same box: only OK if the support is visible BELOW the bowl (hidden ramekin under a bowl)
    return (r["box"][3] - t["box"][3]) >= max(2.0, 0.05 * box_wh(t["box"])[1])


def perceive_scene(detector, image_F, depth_F, cam, parsed, goal, attempt=0, cfg=None):
    """image_F: PIL image as the detector sees it (camera image flipped [::-1, ::-1]); depth_F: depth in metres, same layout.
    parsed / goal: from parse_instruction / parse_goal. attempt 1 = the re-grounding pass (adds synonyms).
    Returns G, S, D (and S_rel, D_rel, the chosen boxes, the ambiguity factor and every bowl candidate)."""
    cfg = cfg or ENGINE_CFG
    img = image_F
    rel = parsed.get("relation")
    tphrase = parsed.get("target")
    ref_phrases = as_list(parsed.get("reference"))
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
        dets = list(_pool(detector, img, p, attempt, cfg, cache))
        head = p.split()[-1]
        if with_head_noun and head != p:
            dets += [{**d, "confidence": d["confidence"] * cfg["head_noun_scale"], "phrase_used": head}
                     for d in _pool(detector, img, head, attempt, cfg, cache)]
            dets = clean_detections(dets, 0.0, allow_large=False, cfg=cfg)
        dets = _contrast_filter(detector, dets, p, competitors(p), img, cfg, cache)
        return [measure_detection(d, depth_F, cam, cfg) for d in dets]

    t_recs = pool_for(tphrase, with_head_noun=cfg.get("use_head_noun", False)) if tphrase else []
    ref_recs = {}
    for rp in ref_phrases:
        if is_table_center(rp):
            ref_recs[rp] = [measure_detection(d, depth_F, cam, cfg) for d in _table_center_candidates(detector, img, attempt, cfg)]
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

    # every bowl candidate x every reference combination: how well does it satisfy the relation?
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
                continue                                             # 'between A and B': A and B must differ
            s_rel = relation_score_2d(rel, t, combo) if rel else 0.0
            d_rel = relation_score_3d(rel, t, combo, cam, cfg) if rel else 0.0
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
        if second is not None:                                        # two bowls fit almost equally well -> less evidence
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


def scene_summary(scene):
    def fmt(r):
        if r is None:
            return "NOT FOUND"
        return f"conf={r['conf']:.2f} box={[round(v) for v in r['box_F']]} world={np.round(r['pos'], 3).tolist()}"
    lines = [f"G={scene['G']:.3f}  S={scene['S']:.3f} (rel {scene['S_rel']:.2f})  D={scene['D']:.3f} (rel {scene['D_rel']:.2f})  "
             f"relation={scene['relation']}  attempt={scene['attempt']}",
             f"  target : {fmt(scene['target'])}   ({scene['n_target_candidates']} candidates)"]
    for i, r in enumerate(scene["refs"]):
        lines.append(f"  ref[{i}] : {fmt(r)}")
    lines.append(f"  goal   : {fmt(scene['goal'])}")
    return "\n".join(lines)
