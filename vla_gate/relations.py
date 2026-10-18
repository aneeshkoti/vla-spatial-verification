"""Graded spatial relations, in the image (2D) and in the world frame (3D, metres).

Every relation returns a truth degree in [0, 1] instead of a yes / no answer, so one pixel over a threshold cannot flip
the evidence. 2D distances are scaled by box size, 3D distances by the objects' measured sizes.
"""
import math

import numpy as np

from .config import ENGINE_CFG
from .geometry import ramp_down, ramp_up, box_diag, box_wh, frac_inside


def _norm2(a):
    return float(math.hypot(a[0], a[1]))


def _between_score(p, a, b, perp_lo=0.25, perp_hi=0.60):
    """p between a and b (2D points / arrays): inside the segment AND close to the line."""
    p, a, b = np.asarray(p, float)[:2], np.asarray(a, float)[:2], np.asarray(b, float)[:2]
    ab = b - a
    L = float(np.linalg.norm(ab))
    if L < 1e-6:
        return 0.0
    t = float(np.dot(p - a, ab) / (L * L))
    perp = float(abs(ab[0] * (p - a)[1] - ab[1] * (p - a)[0]) / L) / L
    inside = ramp_down(max(-t, t - 1.0), 0.0, 0.2)
    return inside * ramp_down(perp, perp_lo, perp_hi)


def relation_score_2d(rel, t, refs):
    """t, refs[i]: measured detections with 'box' (true-view pixel box). Returns a degree in [0, 1]."""
    if t is None or not refs or any(r is None for r in refs):
        return 0.0
    r = refs[0]
    ct, cr = np.array(t["center"]), np.array(r["center"])
    scale = 0.5 * (box_diag(t["box"]) + box_diag(r["box"]))
    gap = float(np.linalg.norm(ct - cr)) / scale
    wt, ht = box_wh(t["box"])
    wr, hr = box_wh(r["box"])
    if rel == "between":
        return _between_score(ct, refs[0]["center"], refs[1]["center"]) if len(refs) >= 2 else 0.0
    if rel == "near":
        return ramp_down(gap, 1.0, 2.2)
    if rel == "on":
        horiz = ramp_down(abs(ct[0] - cr[0]) / (0.5 * wr), 0.7, 1.1)
        bottom = t["box"][3]                                   # bowl's bottom edge (image y grows down)
        out = max(r["box"][1] - bottom, bottom - r["box"][3], 0.0) / hr
        return horiz * ramp_down(out, 0.05, 0.35)
    if rel == "in":
        return max(ramp_up(frac_inside(t["box"], r["box"]), 0.3, 0.7), 0.8 * ramp_down(gap, 0.8, 2.0))
    if rel == "at":
        return ramp_down(float(np.linalg.norm(ct - cr)) / box_diag(t["box"]), 1.0, 3.0)
    if rel == "far":
        return ramp_up(gap, 2.5, 4.0)
    signed = {
        "left":   (cr[0] - ct[0]) / scale,
        "right":  (ct[0] - cr[0]) / scale,
        "above":  (cr[1] - ct[1]) / scale,
        "below":  (ct[1] - cr[1]) / scale,
        "front":  (ct[1] - cr[1]) / scale,
        "behind": (cr[1] - ct[1]) / scale,
    }.get(rel)
    return ramp_up(signed, 0.2, 0.8) if signed is not None else 0.0


def relation_score_3d(rel, t, refs, cam, cfg=None):
    """The same relations in the WORLD frame (metres, z up)."""
    cfg = cfg or ENGINE_CFG
    if t is None or not refs or any(r is None for r in refs):
        return 0.0
    r = refs[0]
    pt, pr = np.asarray(t["pos"], float), np.asarray(r["pos"], float)
    h = _norm2(pt[:2] - pr[:2])
    dz = float(pt[2] - pr[2])
    st, sr = max(t["size"], 0.02), max(r["size"], 0.02)
    if rel == "between":
        return _between_score(pt, refs[0]["pos"], refs[1]["pos"]) if len(refs) >= 2 else 0.0
    if rel == "near":                                          # 'next to' = (almost) touching
        return ramp_down(h - 0.5 * (st + sr), 0.05, 0.14)
    if rel == "on":                                            # inside the support's footprint AND resting on its top
        inside = ramp_down(h, 0.5 * sr, 0.5 * sr + 0.04)
        margin, width = cfg["on_dz_max"]
        lo = 0.5 * max(r.get("height", 0.05), 0.02) + margin           # tall support (cabinet): its measured point is mid-front
        resting = ramp_up(dz, -0.02, 0.01) * ramp_down(dz, lo, lo + width)
        return inside * resting
    if rel == "in":
        return ramp_down(h, 0.5 * sr, 0.5 * sr + 0.10)
    if rel == "at":
        return ramp_down(h, 0.08, 0.25)
    if rel == "far":
        return ramp_up(h, 0.25, 0.40)
    scale = 0.5 * (st + sr) + 0.02
    right, fwd = cam.horizontal_axes()
    signed = {
        "left":   float(np.dot(pr - pt, right)) / scale,
        "right":  float(np.dot(pt - pr, right)) / scale,
        "front":  float(np.dot(pr - pt, fwd)) / scale,
        "behind": float(np.dot(pt - pr, fwd)) / scale,
        "above":  dz / scale,
        "below":  -dz / scale,
    }.get(rel)
    return ramp_up(signed, 0.2, 0.8) if signed is not None else 0.0
