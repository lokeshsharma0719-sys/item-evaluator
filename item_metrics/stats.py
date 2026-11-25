"""Statistics: Wilson intervals, Cohen's / Fleiss' kappa, Krippendorff's alpha (nominal), seeded bootstrap."""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Callable, Hashable, Optional, Sequence

import numpy as np
from scipy.stats import norm

Z95 = float(norm.ppf(0.975))
SMALL_N = 10
NDIGITS = 4


def r(x: Optional[float], nd: int = NDIGITS) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return round(float(x), nd)


def wilson(k: float, n: int, z: float = Z95) -> tuple[Optional[float], Optional[float]]:
    if n <= 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def est(value: Optional[float], lo: Optional[float], hi: Optional[float], n: int) -> dict[str, Any]:
    d: dict[str, Any] = {"value": r(value), "ci_low": r(lo), "ci_high": r(hi), "n": int(n)}
    if n < SMALL_N:
        d["small_n"] = True
    return d


def prop(k: float, n: int) -> dict[str, Any]:
    """A proportion with its Wilson 95% interval."""
    if n <= 0:
        return est(None, None, None, 0)
    lo, hi = wilson(k, n)
    return est(k / n, lo, hi, n)


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> Optional[float]:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return None
    po = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(x for x, _ in pairs), Counter(y for _, y in pairs)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[Hashable]]) -> Optional[float]:
    """ratings[i] = labels given to item i by each rater (all items need the same number of raters >= 2)."""
    rows = [list(x) for x in ratings if len(x) >= 2]
    if not rows:
        return None
    m = len(rows[0])
    rows = [x for x in rows if len(x) == m]
    cats = sorted({c for x in rows for c in x}, key=str)
    if not rows or len(cats) == 0:
        return None
    counts = np.array([[x.count(c) for c in cats] for x in rows], dtype=float)
    n_items = counts.shape[0]
    p_j = counts.sum(axis=0) / (n_items * m)
    p_i = ((counts * counts).sum(axis=1) - m) / (m * (m - 1))
    pbar, pe = p_i.mean(), float((p_j * p_j).sum())
    if pe >= 1.0:
        return 1.0 if pbar == 1.0 else 0.0
    return float((pbar - pe) / (1 - pe))


def krippendorff_alpha_nominal(units: Sequence[Sequence[Optional[Hashable]]]) -> Optional[float]:
    """units[u] = values from each coder for unit u; None = missing. Units with < 2 values are skipped."""
    coinc: Counter = Counter()
    for vals in units:
        v = [x for x in vals if x is not None]
        m = len(v)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    coinc[(v[i], v[j])] += 1.0 / (m - 1)
    if not coinc:
        return None
    cats = sorted({c for c, _ in coinc}, key=str)
    n_c = {c: sum(coinc[(c, k)] for k in cats) for c in cats}
    n = sum(n_c.values())
    d_o = sum(coinc[(c, k)] for c in cats for k in cats if c != k)
    d_e = sum(n_c[c] * n_c[k] for c in cats for k in cats if c != k)
    if d_e == 0:
        return 1.0 if d_o == 0 else 0.0
    return 1.0 - (n - 1) * d_o / d_e


def bootstrap(units: Sequence[Any], stat: Callable[[list[Any]], Optional[float]], seed: int,
              n_boot: int = 1000) -> tuple[Optional[float], Optional[float]]:
    """95% percentile interval of `stat` over item resamples."""
    units = list(units)
    if not units:
        return None, None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(units), len(units))
        v = stat([units[i] for i in idx])
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return None, None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_est(units: Sequence[Any], stat: Callable[[list[Any]], Optional[float]], seed: int,
             n_boot: int = 1000) -> dict[str, Any]:
    units = list(units)
    value = stat(units) if units else None
    lo, hi = bootstrap(units, stat, seed, n_boot) if value is not None else (None, None)
    return est(value, lo, hi, len(units))


def mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
