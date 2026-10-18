"""Recomputes the main numbers of the paper from the saved result files, in a few seconds, without a GPU, a simulator or models.

    python scripts/reproduce_numbers.py                     # reads results/
    python scripts/reproduce_numbers.py --results ~/smolvla_project/results

Every gate decision is recomputed from the logged evidence with the frozen settings (configs/), using the same decision code as
the experiments. Each line shows the value printed in the paper and the recomputed value.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vla_gate.config import FROZEN_STUDY1_JSON, frozen_hash, gate_cfg  # noqa: E402
from vla_gate.decision import _num, decide_all, scene_vals, summarize  # noqa: E402
from vla_gate.sim import INVALID_KINDS, KINDS, VALID_KINDS  # noqa: E402
from vla_gate.stats import auroc, mcnemar_exact, pct  # noqa: E402

ROWS = []


def _py(v):
    """numpy numbers -> plain Python numbers (for comparing and printing)"""
    if isinstance(v, (tuple, list)):
        return tuple(_py(x) for x in v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def check(section, what, paper, value, tol=1e-6):
    """paper: the number as printed in the paper; value: recomputed and rounded the same way. They must be equal."""
    value = _py(value)
    if value is None:
        ok = None
    elif isinstance(paper, (int, float)) and isinstance(value, (int, float)):
        ok = abs(float(paper) - float(value)) <= tol
    elif isinstance(paper, tuple) and isinstance(value, tuple) and len(paper) == len(value):
        ok = all((abs(float(a) - float(b)) <= tol) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b
                 for a, b in zip(_flat(paper), _flat(value))) and len(_flat(paper)) == len(_flat(value))
    else:
        ok = paper == value
    ROWS.append((section, what, paper, value, ok))


def _flat(t):
    out = []
    for x in t:
        out.extend(_flat(x) if isinstance(x, tuple) else [x])
    return out


def skip(section, what):
    ROWS.append((section, what, "-", None, None))


def read_dedup(path, keys):
    return pd.read_csv(path).drop_duplicates(keys, keep="last")


# ======================================================================================================== Study 1
def study1(R):
    S = "Study 1 (first gate)"
    v4, v5 = os.path.join(R, "v4"), os.path.join(R, "v5")
    verify_csv = os.path.join(v4, "verify_v4.csv")
    if not os.path.exists(verify_csv):
        skip(S, f"all numbers (missing {verify_csv})")
        return
    fz = os.path.join(v4, "frozen_config.json")
    if os.path.exists(fz):
        check(S, "frozen configuration (hash)", "a4a4d6846a", frozen_hash(fz))
    F = json.load(open(fz if os.path.exists(fz) else FROZEN_STUDY1_JSON))
    FCFG = {**F, "weights": tuple(F["weights"]), "ramp": tuple(F["ramp"])}

    def load_verify(split):
        d = pd.read_csv(verify_csv)
        return d[d.split == split].drop_duplicates(["task_id", "state", "kind"], keep="last").reset_index(drop=True)
    dv, tv = load_verify("dev"), load_verify("test")
    check(S, "test cases", 210, len(tv), 0)
    test = decide_all(tv, FCFG)
    inv, val = test[test.valid == 0], test[test.valid == 1]
    check(S, "impossible instructions released (of 90)", 11, int(inv.executed.sum()), 0)
    check(S, "IER, invalid execution rate (%)", 12.2, pct(inv.executed.mean()))
    check(S, "valid instructions accepted (of 120)", 52, int(val.executed.sum()), 0)
    check(S, "VAR, valid acceptance rate (%)", 43.3, pct(val.executed.mean()))
    check(S, "AUROC of the score C", 0.741, round(auroc(val.C, inv.C), 3), 0.0005)
    check(S, "re-grounding pass used (of 210)", 152, int(test.reground_used.sum()), 0)
    check(S, "gate time per case (s)", 3.12, round(float((test.t_perceive0_s + test.t_perceive1_s * test.reground_used).mean()), 2), 0.005)
    check(S, "instructed bowl selected, valid cases (%)",
          37.5, pct(pd.to_numeric(tv[tv.kind.isin(VALID_KINDS)].right_target_0, errors="coerce").mean()))

    def renorm(w):
        s = sum(w)
        return tuple(x / s for x in w)
    for name, w, p_ier, p_var, p_auc in [("G only", (1, 0, 0, 0), 58.9, 80.0, 0.671),
                                         ("G + S", renorm((0.20, 0.30, 0, 0)), 18.9, 52.5, 0.733),
                                         ("G + S + D", renorm((0.20, 0.30, 0.25, 0)), 12.2, 43.3, 0.741)]:
        s = summarize(decide_all(tv, {**FCFG, "weights": w}))
        check(S, f"ablation {name}: IER / VAR / AUROC", (p_ier, p_var, p_auc), (s["IER %"], s["valid_accept %"], s["AUROC(C)"]))

    # random checker with the same acceptance rate (2000 draws, seed 0) and the detector-confidence checker (tuned on dev)
    a = float(val.executed.mean())
    rng = np.random.default_rng(0)
    vflag = test.valid.values
    r_ier, r_var = [], []
    for _ in range(2000):
        ex = (rng.random(len(test)) < a).astype(int)
        r_ier.append(pct(ex[vflag == 0].mean()))
        r_var.append(pct(ex[vflag == 1].mean()))
    check(S, "random checker: IER / VAR (%)", (43.2, 43.4), (round(float(np.mean(r_ier)), 1), round(float(np.mean(r_var)), 1)))

    def g_only(t):
        return {**FCFG, "weights": (1.0, 0.0, 0.0, 0.0), "high": t, "low": t, "mid_rule": False}
    best_t, best_j = None, -9
    for t in np.round(np.arange(0.05, 0.96, 0.025), 3):
        j = summarize(decide_all(dv, g_only(t)))["J"]
        if j > best_j + 1e-9:
            best_t, best_j = float(t), j
    det = decide_all(tv, g_only(best_t))
    sd = summarize(det)
    check(S, "detector-confidence checker: threshold / IER / VAR / AUROC", (0.95, 27.8, 56.7, 0.68),
          (round(best_t, 3), sd["IER %"], sd["valid_accept %"], sd["AUROC(C)"]))

    orc_csv = os.path.join(v5, "oracle_identity_test.csv")
    if os.path.exists(orc_csv):
        orc = decide_all(read_dedup(orc_csv, ["task_id", "state", "kind"]), FCFG)
        so = summarize(orc)
        check(S, "oracle identity: IER / VAR / AUROC", (0.0, 29.2, 0.654), (so["IER %"], so["valid_accept %"], so["AUROC(C)"]))
    else:
        skip(S, "oracle identity (missing v5/oracle_identity_test.csv)")

    base_csv, roll_csv = os.path.join(v4, "baseline_clean_v4.csv"), os.path.join(v4, "rollouts_v4.csv")
    if not (os.path.exists(base_csv) and os.path.exists(roll_csv)):
        skip(S, "task success and head-to-head (missing baseline_clean_v4.csv / rollouts_v4.csv)")
        return
    b = read_dedup(base_csv, ["task_id", "state"])
    check(S, "SmolVLA baseline, 50 nominal episodes (%)", 66.0, pct(b.success.mean()))
    bsm = {(int(r.task_id), int(r.state)): int(r.success) for r in b.itertuples()}
    roll = read_dedup(roll_csv, ["task_id", "state", "condition"])
    rsm = {(int(r.task_id), int(r.state), r.condition): int(r.success) for r in roll.itertuples()}
    test["s"] = [bsm.get((int(r.task_id), int(r.state))) if r.kind == "clean" else rsm.get((int(r.task_id), int(r.state), r.kind))
                 for r in test.itertuples()]
    cl = test[(test.kind == "clean") & test.s.notna()]
    check(S, "nominal task success: SmolVLA / gated (%)", (66.7, 43.3), (pct(cl.s.mean()), pct((cl.s * cl.executed).mean())))
    vk = test[test.valid == 1]
    check(S, "valid tasks solved (of 120): SmolVLA / gated", (46, 20), (int(vk.s.sum()), int((vk.s * vk.executed).sum())))

    rank = {"success": 2, "stop": 1, "bad": 0}

    def outcome(valid, executed, s):
        if not executed:
            return "stop"
        if valid == 0:
            return "bad"
        return None if s is None or s != s else ("success" if s == 1 else "bad")
    better = [0, 0, 0]
    for r in test.itertuples():
        so, oo = outcome(r.valid, 1, r.s), outcome(r.valid, r.executed, r.s)
        if so is None or oo is None:
            continue
        better[0 if rank[oo] > rank[so] else (2 if rank[oo] < rank[so] else 1)] += 1
    check(S, "head-to-head with SmolVLA: gated better / same / worse", (121, 63, 26), tuple(better))

    def classify(r):
        sfx = r.scene_used
        v = scene_vals(r._asdict(), sfx, tuple(FCFG["ramp"]))
        rt, rr, sel = _num(getattr(r, f"right_target_{sfx}")), _num(getattr(r, f"right_ref_{sfx}")), getattr(r, f"sel_{sfx}")
        grounded_ok = (rt in (1.0, None)) and (rr in (1.0, None)) and sel in ("target", "other_bowl")
        if r.valid == 0 and r.executed == 1:
            if r.kind in ("absent_reference", "absent_target"):
                return "grounding (absent object detected)"
            if not ((rr in (1.0, None)) and sel in ("target", "other_bowl")):
                return "grounding"
            if v["D_rel"] >= 0.5:
                return "depth / 3D"
            if v["S_rel"] >= 0.5:
                return "spatial relation"
            return "threshold / calibration"
        if r.valid == 1 and r.executed == 0:
            if r.label_uncertain == 1:
                return "label uncertain (not counted)"
            if not grounded_ok or sel != "target":
                return "grounding"
            if v["S_rel"] < 0.3:
                return "spatial relation"
            if v["D_rel"] < 0.3:
                return "depth / 3D"
            return "threshold / calibration"
        if r.valid == 1 and r.executed == 1 and r.s == 0:
            return "action / execution"
        return None
    fails = pd.Series([classify(r) for r in test.itertuples()]).dropna()
    check(S, "errors of the gated system / of which grounding", (111, 56), (int(len(fails)), int((fails == "grounding").sum())))

    off = os.path.join(v5, "official_eval_summary.json")
    if os.path.exists(off):
        check(S, "official LeRobot evaluation of SmolVLA (%)", 62.0, float(json.load(open(off))["official_pc_success"]))


# ======================================================================================================== Study 2
FRESH = [5, 6, 7, 8, 9]


def load6(path, states):
    d = pd.read_csv(path).drop_duplicates(["task_id", "state", "kind"], keep="last")
    return d[d.state.isin(states)].reset_index(drop=True)


def success_map(path):
    if not os.path.exists(path):
        return {}
    d = read_dedup(path, ["task_id", "state", "condition"])
    return {(int(r.task_id), int(r.state), r.condition): int(r.success) for r in d.itertuples()}


def vts(dec, smap):
    """valid-task success: share of the valid scenes that ended in success (a stopped scene counts as a failure)"""
    d = dec.copy()
    d["success"] = [smap.get((int(r.task_id), int(r.state), r.kind)) for r in d.itertuples()]
    known = d[(d.valid == 1) & d.success.notna()]
    return pct((known.executed * known.success).mean())


def study2(R):
    S = "Study 2 (hybrid gate)"
    v6, v7 = os.path.join(R, "v6"), os.path.join(R, "v7")
    gd_csv, hy_csv = os.path.join(v6, "fresh_verify_gdino.csv"), os.path.join(v6, "fresh_verify_hybrid.csv")
    if not (os.path.exists(gd_csv) and os.path.exists(hy_csv)):
        skip(S, f"all numbers (missing {gd_csv} / {hy_csv})")
        return
    fz = os.path.join(v6, "frozen_config_v6.json")
    if os.path.exists(fz):
        check(S, "frozen hybrid configuration (hash)", "a5f5591cec", frozen_hash(fz))
    gates = {"first gate": decide_all(load6(gd_csv, FRESH), gate_cfg("gdino")),
             "hybrid gate": decide_all(load6(hy_csv, FRESH), gate_cfg("hybrid"))}
    paper = {"first gate": (18, 84, 12.0, 42.0, 35.0, 0.738, 4.72), "hybrid gate": (8, 105, 5.3, 52.5, 67.5, 0.875, 2.66)}
    for name, d in gates.items():
        inv, val = d[d.valid == 0], d[d.valid == 1]
        p = paper[name]
        check(S, f"{name}: scenes", 350, len(d), 0)
        check(S, f"{name}: impossible released (of 150) / valid accepted (of 200)", p[:2], (int(inv.executed.sum()), int(val.executed.sum())))
        check(S, f"{name}: IER / VAR (%)", p[2:4], (pct(inv.executed.mean()), pct(val.executed.mean())))
        check(S, f"{name}: instructed bowl selected, valid scenes (%)", p[4],
              pct(pd.to_numeric(d[d.kind.isin(VALID_KINDS)].right_target_0, errors="coerce").mean()))
        check(S, f"{name}: AUROC", p[5], round(auroc(val.C, inv.C), 3), 0.0005)
        check(S, f"{name}: seconds per scene", p[6], round(float((d.t_perceive0_s + d.t_perceive1_s * d.reground_used).mean()), 2), 0.005)
    old, new = gates["first gate"], gates["hybrid gate"]
    m = old[["task_id", "state", "kind", "valid", "executed"]].merge(new[["task_id", "state", "kind", "executed"]],
                                                                     on=["task_id", "state", "kind"], suffixes=("_a", "_b"))
    for v, name, p in ((0, "impossible", (11, 1, 0.0063)), (1, "valid", (38, 59, 0.0417))):
        mm = m[m.valid == v]
        x = int(((mm.executed_a == 1) & (mm.executed_b == 0)).sum())
        y = int(((mm.executed_a == 0) & (mm.executed_b == 1)).sum())
        check(S, f"{name}: only first gate acts / only hybrid acts / McNemar p", p, (x, y, round(mcnemar_exact(x, y), 4)), 0.00005)

    orc_csv = os.path.join(v6, "fresh_verify_hybrid_oracle.csv")
    if os.path.exists(orc_csv):
        o = decide_all(load6(orc_csv, FRESH), gate_cfg("hybrid"))
        so = summarize(o)
        check(S, "hybrid + oracle identity: IER / VAR / AUROC", (0.0, 51.5, 0.764), (so["IER %"], so["valid_accept %"], so["AUROC(C)"]))
        check(S, "hybrid + oracle identity: instructed bowl found (%)", 96.5, pct((o[o.kind.isin(VALID_KINDS)].sel_0 == "target").mean()))

    smol, xvla = success_map(os.path.join(v6, "fresh_rollouts.csv")), success_map(os.path.join(v7, "xvla_rollouts.csv"))
    pols = {"SmolVLA": smol, "X-VLA": xvla}
    for pname, smap, p in (("SmolVLA", smol, 42.5), ("X-VLA", xvla, 62.5)):
        v = [s for (t, st, k), s in smap.items() if k in VALID_KINDS and st in FRESH]
        if v:
            check(S, f"{pname} alone: success on the 200 valid scenes (%)", p, pct(np.mean(v)))
        else:
            skip(S, f"{pname} rollouts (file missing)")

    judge_json, judge_csv = os.path.join(v7, "vlm_judge_choice.json"), os.path.join(v7, "vlm_judge_fresh.csv")
    if os.path.exists(judge_json) and os.path.exists(judge_csv):
        J = json.load(open(judge_json))
        check(S, "VLM judge setting fixed on development: score / threshold", ("decomp", 0.05), (J["score"], float(J["thr"])))
        vj = pd.read_csv(judge_csv).drop_duplicates(["task_id", "state", "kind"], keep="last")
        vj = vj[vj.state.isin(FRESH)].reset_index(drop=True)
        vj["C"] = vj[f"p_{J['score']}"].astype(float)
        vj["executed"] = (vj.C >= J["thr"]).astype(int)
        both = new[["task_id", "state", "kind", "valid", "executed", "C"]].merge(
            vj[["task_id", "state", "kind", "executed", "C"]], on=["task_id", "state", "kind"], suffixes=("_a", "_b"))
        both["executed"] = (both.executed_a & both.executed_b).astype(int)
        both["C"] = both[["C_a", "C_b"]].min(axis=1)
        for name, d, p in (("VLM judge", vj, (23.3, 69.5, 0.833)), ("hybrid AND VLM judge", both, (3.3, 38.0, 0.798))):
            inv, val = d[d.valid == 0], d[d.valid == 1]
            check(S, f"{name}: IER / VAR / AUROC", p, (pct(inv.executed.mean()), pct(val.executed.mean()), round(auroc(val.C, inv.C), 3)))
        gates.update({"VLM judge": vj, "hybrid AND VLM judge": both})
    else:
        skip(S, "VLM judge (missing v7/vlm_judge_choice.json / vlm_judge_fresh.csv)")

    vts_paper = {"first gate": (16.0, 23.5), "hybrid gate": (23.5, 35.0), "VLM judge": (31.5, 44.0), "hybrid AND VLM judge": (18.0, 25.0)}
    if smol and xvla:
        for name, d in gates.items():
            check(S, f"valid-task success behind the {name}: SmolVLA / X-VLA (%)", vts_paper[name], (vts(d, smol), vts(d, xvla)))


# ======================================================================================================== Table IV
def table4(R):
    S = "Detector choice (Table IV, development scenes)"
    v4, v6 = os.path.join(R, "v4"), os.path.join(R, "v6")
    verify_csv = os.path.join(v4, "verify_v4.csv")
    fz = os.path.join(v4, "frozen_config.json")
    F = json.load(open(fz if os.path.exists(fz) else FROZEN_STUDY1_JSON))
    FCFG = {**F, "weights": tuple(F["weights"]), "ramp": tuple(F["ramp"])}
    OWL = {**FCFG, "ramp": (0.10, 0.25)}
    rows = [("first gate, 70 scenes of state 0", verify_csv, [0], FCFG, (13.3, 42.5, 32.5, 0.673, 0.292)),
            ("OWLv2 alone, 70 scenes of state 0", os.path.join(v6, "verify_owl.csv"), [0], OWL, (0.0, 32.5, 62.5, 0.752, 0.325)),
            ("hybrid, 70 scenes of state 0", os.path.join(v6, "verify_owl_hybrid.csv"), [0], OWL, (3.3, 47.5, 62.5, 0.849, 0.442)),
            ("first gate, 210 scenes of states 0-2", verify_csv, [0, 1, 2], FCFG, (10.0, 40.8, 31.7, 0.754, 0.308)),
            ("hybrid gate, 210 scenes of states 0-2", os.path.join(v6, "verify_owl_hybrid.csv"), [0, 1, 2], OWL, (6.7, 57.5, 70.0, 0.874, 0.508))]
    for name, path, states, cfg, p in rows:
        if not os.path.exists(path):
            skip(S, f"{name} (missing {os.path.relpath(path, R)})")
            continue
        d = decide_all(load6(path, states), cfg)
        s = summarize(d)
        rb = pct(pd.to_numeric(d[d.kind.isin(VALID_KINDS)].right_target_0, errors="coerce").mean())
        check(S, f"{name}: IER / VAR / right bowl / AUROC / J", p, (s["IER %"], s["valid_accept %"], rb, s["AUROC(C)"], round(s["J"], 3)))


# ======================================================================================================== attached run
def attached(R):
    S = "Attached run (state 5)"
    v6, v7 = os.path.join(R, "v6"), os.path.join(R, "v7")
    saved = {"gdino": os.path.join(v6, "fresh_verify_gdino.csv"), "hybrid": os.path.join(v6, "fresh_verify_hybrid.csv")}
    paper = {("hybrid", "smolvla"): ((3, 30), (23, 40), (9, 40)), ("hybrid", "xvla"): ((3, 30), (23, 40), (15, 40)),
             ("gdino", "smolvla"): ((5, 30), (22, 40), (7, 40))}
    for (g, p), exp in paper.items():
        path = os.path.join(v7, f"live_{g}_{p}.csv")
        if not os.path.exists(path):
            skip(S, f"{g} gate + {p} (missing {path})")
            continue
        d = read_dedup(path, ["task_id", "state", "kind"])
        d = d[d.state.isin([5])]
        inv, val = d[d.valid == 0], d[d.valid == 1]
        got = ((int(inv.robot_moved.sum()), len(inv)), (int(val.robot_moved.sum()), len(val)), (int(val.success.sum()), len(val)))
        name = f"{'hybrid' if g == 'hybrid' else 'first'} gate + {'SmolVLA' if p == 'smolvla' else 'X-VLA'}"
        check(S, f"{name}: robot moved on impossible / on valid / successes", exp, got)
        if os.path.exists(saved[g]):
            sd = decide_all(load6(saved[g], [5]), gate_cfg(g))[["task_id", "state", "kind", "decision"]]
            mm = d.merge(sd, on=["task_id", "state", "kind"], suffixes=("_live", "_saved"))
            check(S, f"{name}: live decision = decision of the separate sweep", (69, 70),
                  (int((mm.decision_live == mm.decision_saved).sum()), len(mm)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=os.environ.get("VLA_GATE_RESULTS", os.path.join(ROOT, "results")),
                    help="folder with the v4 ... v7 result folders (default: results/)")
    args = ap.parse_args()
    R = os.path.expanduser(args.results)
    print(f"result files: {R}\n")
    for f in (study1, table4, study2, attached):
        try:
            f(R)
        except Exception as e:                       # report and go on with the next study
            ROWS.append((f.__name__, f"stopped with {type(e).__name__}: {e}", "-", None, False))
    sec = None
    n_ok = n_bad = n_skip = 0
    for section, what, paper, value, ok in ROWS:
        if section != sec:
            print(f"\n{section}")
            sec = section
        mark = "  ok  " if ok else ("  --  " if ok is None else "  DIFF")
        n_ok, n_bad, n_skip = n_ok + (ok is True), n_bad + (ok is False), n_skip + (ok is None)
        print(f"{mark} {what:72s} paper {str(paper):34s} recomputed {value if value is not None else 'skipped'}")
    print(f"\n{n_ok} numbers reproduced, {n_bad} different, {n_skip} skipped (result file not present)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
