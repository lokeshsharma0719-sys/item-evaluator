"""Shared pieces for the metric modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass
class MetricResult:
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame]
    normalised: dict[str, Any] = field(default_factory=dict)
    sub_scores: dict[str, Optional[float]] = field(default_factory=dict)
    flags: list[dict[str, Any]] = field(default_factory=list)
    worst: list[dict[str, Any]] = field(default_factory=list)   # 10 worst items for report.md


def record(item_id: str, model: str, meta: dict[str, Any], parsed: Any, **extra: Any) -> dict[str, Any]:
    return {"item_id": item_id, "model": model, "status": meta.get("status", "ok"),
            "parsed": parsed.model_dump(mode="json") if parsed is not None else None, **extra}


def is_ok(rec: Optional[dict[str, Any]]) -> bool:
    return rec is not None and rec.get("status") == "ok" and rec.get("parsed") is not None


def budget_skipped(rec: Optional[dict[str, Any]]) -> bool:
    """The call was not answered: budget stop, or a call error (keys exhausted, quota, network).

    Such items are left out as 'not evaluated' rather than scored as wrong. A parse failure is different: the model
    answered with unusable output, and that counts against it."""
    return rec is not None and rec.get("status") in ("budget_skipped", "error")


def flag(metric: str, value: Optional[float], threshold: float, higher_is_better: bool = True) -> dict[str, Any]:
    if value is None:
        status = "n/a"
    elif higher_is_better:
        status = "ok" if value >= threshold else "below"
    else:
        status = "ok" if value <= threshold else "above"
    return {"metric": metric, "value": value, "threshold": threshold, "status": status}


def df(rows: list[dict[str, Any]], columns: Optional[list[str]] = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns or [])
    d = pd.DataFrame(rows)
    if columns:
        for c in columns:
            if c not in d.columns:
                d[c] = None
        d = d[columns + [c for c in d.columns if c not in columns]]
    return d
