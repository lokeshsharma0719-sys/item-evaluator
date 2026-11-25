"""Normalised scores (spec section 8a): every sub-score in [0, 1], 1 best, and weighted geometric composites."""
from __future__ import annotations

import math
from typing import Any, Optional


def higher_better(p: Optional[float]) -> Optional[float]:
    return None if p is None else min(1.0, max(0.0, p))


def lower_better(rate: Optional[float]) -> Optional[float]:
    return None if rate is None else min(1.0, max(0.0, 1.0 - rate))


def agreement(k: Optional[float]) -> Optional[float]:
    """kappa / alpha are clipped at 0: chance-level agreement is 0, not 0.5."""
    return None if k is None else min(1.0, max(0.0, k))


def chance_adjusted(x: Optional[float]) -> Optional[float]:
    return None if x is None else min(1.0, max(0.0, x))


def error_score(e: Optional[float]) -> Optional[float]:
    return None if e is None else min(1.0, max(0.0, 1.0 - e))


def composite(components: dict[str, Optional[float]], weights: dict[str, float], required: set[str],
              floor: float = 0.01) -> tuple[Optional[float], dict[str, float]]:
    """Weighted geometric mean over available components; weights rescaled to sum to 1.

    Missing optional components are dropped; a missing required component makes the composite None."""
    avail = {k: v for k, v in components.items() if v is not None and k in weights and weights[k] > 0}
    if any(components.get(k) is None for k in required):
        return None, {}
    total = sum(weights[k] for k in avail)
    if not avail or total <= 0:
        return None, {}
    used = {k: weights[k] / total for k in avail}
    log_s = sum(w * math.log(max(floor, min(1.0, avail[k]))) for k, w in used.items())
    return math.exp(log_s), {k: round(w, 4) for k, w in used.items()}


def bar(v: Optional[float], width: int = 20) -> str:
    if v is None:
        return "n/a".ljust(width + 2)
    n = int(round(max(0.0, min(1.0, v)) * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def block(value: Optional[float], ci: tuple[Optional[float], Optional[float]], n: int,
          components: dict[str, Optional[float]], weights_used: dict[str, float], **extra: Any) -> dict[str, Any]:
    def r(x: Optional[float]) -> Optional[float]:
        return None if x is None else round(x, 4)
    d = {"value": r(value), "ci_low": r(ci[0]), "ci_high": r(ci[1]), "n": n,
         "components": {k: r(v) for k, v in components.items() if v is not None}, "weights_used": weights_used}
    d.update(extra)
    return d
