"""Statistics used in the paper (no scipy needed)."""
import math

import numpy as np


def wilson(k, n, z=1.959964):
    """95% Wilson interval of k / n, in percent"""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(100 * (c - h), 1), round(100 * (c + h), 1))


def mcnemar_exact(b, c):
    """exact two-sided McNemar test (= exact sign test) on the discordant counts b and c"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def fisher_exact(a, b, c, d):
    """two-sided Fisher exact test for the table [[a, b], [c, d]]"""
    n, r1, c1 = a + b + c + d, a + b, a + c
    if n == 0:
        return 1.0

    def p(x):
        return math.comb(r1, x) * math.comb(n - r1, c1 - x) / math.comb(n, c1)
    p0 = p(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(p(x) for x in range(lo, hi + 1) if p(x) <= p0 * (1 + 1e-9)))


def auroc(pos, neg):
    """probability that a valid case scores higher than an invalid one (ties count half)"""
    pos, neg = list(pos), list(neg)
    if not pos or not neg:
        return float("nan")
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def pct(x):
    return round(100.0 * float(x), 1) if x == x else float("nan")
