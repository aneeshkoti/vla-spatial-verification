"""The decision rule:  C = wG*G + wS*S + wD*D  ->  EXECUTE / one re-grounding pass / STOP.

The gate decides from a "row": the evidence of the first grounding pass (suffix "0") and, if needed, of the re-grounding pass
(suffix "1"). The same row format is written to the result CSV files, so every decision in the paper can be recomputed
offline from those files (scripts/reproduce_numbers.py), and the live gate uses exactly the same function.
"""
import json
import math

import numpy as np

from .geometry import ramp_up
from .stats import auroc, pct


def scene_fields(sc, A, sfx):
    """the numbers of one grounding pass that the decision needs (rounded exactly as in the result files)"""
    refs = [None if r is None else round(float(r["conf"]), 4) for r in sc["refs"]]
    return {f"G_{sfx}": sc["G"], f"S_{sfx}": sc["S"], f"D_{sfx}": sc["D"], f"S_rel_{sfx}": sc["S_rel"],
            f"D_rel_{sfx}": sc["D_rel"], f"amb_{sfx}": sc.get("ambiguity", 1.0), f"pair_g_{sfx}": sc.get("pair_g"),
            f"A_{sfx}": round(float(A), 4),
            f"t_conf_{sfx}": None if sc["target"] is None else round(float(sc["target"]["conf"]), 4),
            f"ref_confs_{sfx}": json.dumps(refs),
            f"goal_conf_{sfx}": None if sc["goal"] is None else round(float(sc["goal"]["conf"]), 4)}


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
    """G, S, D of one grounding pass, recomputed from the logged confidences for a calibration ramp."""
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
    """returns (decision, C_final, C_initial, reground_used, scene_used) with exactly one re-grounding pass.
    C >= high -> EXECUTE. Otherwise the re-grounded scene is used if its evidence is higher; then C >= high -> EXECUTE,
    C < low -> ABSTAIN (stop), and in between EXECUTE only if the 2D AND the 3D relation evidence are >= 0.3 and the
    instruction is not ambiguous (ambiguity factor >= 0.75)."""
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


def decide_all(df, cfg):
    """the gate decision for every row of a result table (pandas DataFrame)"""
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
    """IER = invalid executed / invalid; valid acceptance; AUROC of C; clean task success (needs paired rollouts)."""
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
