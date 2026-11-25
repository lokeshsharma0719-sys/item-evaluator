"""CSV loading: column mapping, JSON-cell parsing, per-format validation -> list[Item] + invalid rows."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .answers import parse_number, to_quantity
from .config import Config
from .schemas import FORMATS, Criterion, Item, NumericKey, Option, Stage

log = logging.getLogger(__name__)

SCHEMA_COLUMNS = {
    "item_id", "format", "stem", "options", "answer_key", "stages", "rubric", "model_answer", "topic",
    "course_code", "course_title", "syllabus_topics", "skill", "bloom_level", "difficulty_band", "source",
    "explanation", "hints", "textbook_ref", "human_key_correct",
}


class RowError(ValueError):
    pass


@dataclass
class LoadResult:
    items: list[Item]
    invalid: list[dict[str, Any]]
    n_rows: int
    tolerance_capped: list[str] = field(default_factory=list)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and pd.isna(v)) or (isinstance(v, str) and not v.strip())


def _json_cell(v: Any, what: str) -> Any:
    if _blank(v):
        return None
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except json.JSONDecodeError as e:
        raise RowError(f"{what}: JSON does not parse ({e.msg})") from None


def _bool(v: Any) -> Optional[bool]:
    if _blank(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise RowError(f"human_key_correct: '{v}' is not true/false")


def numeric_key(v: Any, what: str) -> NumericKey:
    """Accepts {"value":..,"unit":..,"tolerance":..}, a bare number, or a string such as '1000 W/m^2'."""
    if isinstance(v, dict):
        raw = v.get("value", v.get("val"))
        val, _ = parse_number(raw)
        if val is None:
            raise RowError(f"{what}: no numeric value")
        tol = v.get("tolerance")
        if tol is not None:
            t, _ = parse_number(tol)
            if t is None or t < 0:
                raise RowError(f"{what}: tolerance is not a non-negative number")
            tol = t
        return NumericKey(value=val, unit=str(v.get("unit") or ""), tolerance=tol)
    val, rest = parse_number(v)
    if val is None:
        raise RowError(f"{what}: no numeric value")
    return NumericKey(value=val, unit=rest)


def _stage_key(v: Any) -> Any:
    if _blank(v):
        return None
    if isinstance(v, dict):
        return numeric_key(v, "stage answer_key")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return NumericKey(value=float(v))
    s = str(v).strip()
    # "about 4.85", "~1.1e6 W/m^2", "+6240 J", "2446.5 kmol/h (about 2446)": a single value with a unit
    core = re.sub(r"^(?:about|approx(?:imately)?\.?|roughly|~|≈|\+)\s*", "", s, flags=re.I)
    core = re.sub(r"\s*\((?:about|approx(?:imately)?\.?|roughly|~|≈|i\.e\.)[^)]*\)\s*$", "", core, flags=re.I)
    val, rest = parse_number(core)
    if val is not None and (not rest or (len(rest) <= 24 and to_quantity(1.0, rest) is not None)):
        return NumericKey(value=val, unit=rest)
    return s


def _options(row: dict[str, Any], has_options_col: bool) -> list[Option]:
    raw = _json_cell(row.get("options"), "options") if has_options_col else None
    if raw is None:
        raw = [{"id": lab, "text": row.get(f"option_{lab}")} for lab in "ABCD"
               if not _blank(row.get(f"option_{lab}"))]
    if isinstance(raw, dict):  # {"A": "text", ...}
        raw = [{"id": k, "text": v} for k, v in raw.items()]
    if not isinstance(raw, list):
        raise RowError("options: must be a JSON list")
    out = []
    for o in raw:
        if not isinstance(o, dict):
            raise RowError("options: every option must be an object")
        oid = o.get("id", o.get("option_id"))
        if _blank(oid) or _blank(o.get("text")):
            raise RowError("options: every option needs an id and text")
        out.append(Option(id=str(oid).strip().upper(), text=str(o["text"]),
                          misconception_id=str(o.get("misconception_id") or ""),
                          failed_layer=str(o.get("failed_layer") or ""),
                          rationale=str(o.get("rationale") or ""),
                          correct=o.get("correct") if isinstance(o.get("correct"), bool) else None))
    return out


def _rubric(v: Any) -> list[Criterion]:
    raw = _json_cell(v, "rubric")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RowError("rubric: must be a JSON list")
    out = []
    for i, c in enumerate(raw, 1):
        if isinstance(c, str):
            c = {"text": c}
        if not isinstance(c, dict):
            raise RowError("rubric: every criterion must be an object")
        text = str(c.get("text") or "").strip()
        if not text:
            raise RowError(f"rubric: criterion {i} has empty text")
        w = c.get("weight", 1)
        wv, rest = parse_number(w) if w is not None else (1.0, "")
        if wv is None or rest or wv <= 0:
            raise RowError(f"rubric: criterion {i} weight '{w}' is not a positive number")
        out.append(Criterion(criterion_id=str(c.get("criterion_id") or c.get("id") or f"c{i}"), text=text,
                             weight=wv, layer=str(c.get("layer") or ""),
                             anchor_pass=str(c.get("anchor_pass") or ""),
                             anchor_fail=str(c.get("anchor_fail") or "")))
    if len({c.criterion_id for c in out}) != len(out):
        raise RowError("rubric: criterion ids are not unique")
    return out


def _stages(v: Any) -> list[Stage]:
    raw = _json_cell(v, "stages")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise RowError("stages: must be a JSON list")
    out = []
    for i, s in enumerate(raw, 1):
        if not isinstance(s, dict):
            raise RowError("stages: every stage must be an object")
        prompt = s.get("prompt")
        if _blank(prompt):
            raise RowError(f"stages: stage {i} has no prompt")
        key = s.get("answer_key", s.get("key"))
        out.append(Stage(stage_id=str(s.get("stage_id") or s.get("id") or f"S{i}"), prompt=str(prompt),
                         answer_key=_stage_key(key)))
    return out


def parse_row(row: dict[str, Any], row_number: int, cfg: Config, has_options_col: bool) -> tuple[Item, bool]:
    """Returns (item, tolerance_was_capped). Raises RowError with a reason."""
    def s(name: str) -> str:
        v = row.get(name)
        return "" if _blank(v) else str(v).strip()

    item_id, fmt, stem = s("item_id"), s("format"), s("stem")
    if not item_id:
        raise RowError("item_id: missing")
    if fmt not in FORMATS:
        raise RowError(f"format: '{fmt}' is not one of {', '.join(FORMATS)}")
    if not stem:
        raise RowError("stem: missing")

    options = _options(row, has_options_col) if fmt == "mcq_4" or has_options_col else []
    stages = _stages(row.get("stages"))
    rubric = _rubric(row.get("rubric"))
    capped = False
    key: Any = None
    raw_key = row.get("answer_key")

    if fmt == "mcq_4":
        if len(options) != 4 or len({o.id for o in options}) != 4:
            raise RowError("options: MCQ needs exactly 4 distinct option ids")
        k = _json_cell(raw_key, "answer_key") if isinstance(raw_key, str) and raw_key.strip()[:1] in '"{[' else raw_key
        if _blank(k):
            flagged = [o.id for o in options if o.correct]
            k = flagged[0] if len(flagged) == 1 else None
        if _blank(k) or str(k).strip().upper() not in {o.id for o in options}:
            raise RowError(f"answer_key: '{'' if _blank(k) else k}' is not one of the option ids")
        key = str(k).strip().upper()
    elif fmt in ("numeric", "staged"):
        if _blank(raw_key):
            raise RowError("answer_key: missing numeric key")
        parsed = _json_cell(raw_key, "answer_key") if str(raw_key).strip()[:1] in "{[" else raw_key
        key = numeric_key(parsed, "answer_key")
        if fmt == "staged" and len(stages) < 2:
            raise RowError("stages: staged items need at least 2 stages")
    elif not _blank(raw_key):
        key = str(raw_key)

    if fmt in ("short_text", "rubric_task") and not rubric:
        raise RowError("rubric: required for short_text and rubric_task")

    tol_default, tol_cap = cfg.numeric.default_tolerance, cfg.numeric.tolerance_cap
    for k in [key] + [st.answer_key for st in stages]:
        if isinstance(k, NumericKey):
            if k.tolerance is None:
                k.tolerance = tol_default
            if k.tolerance > tol_cap:
                k.tolerance = tol_cap
                capped = True

    bloom = None
    if s("bloom_level"):
        b, _ = parse_number(s("bloom_level"))
        bloom = int(b) if b is not None else None
    hints = row.get("hints")
    if isinstance(hints, str) and hints.strip()[:1] in "[{":
        try:
            hints = json.loads(hints)
        except json.JSONDecodeError:
            pass
    syl = [t.strip() for t in s("syllabus_topics").split("|") if t.strip()]
    extra = {k: v for k, v in row.items() if k not in SCHEMA_COLUMNS and not k.startswith("option_")
             and not _blank(v)}
    item = Item(item_id=item_id, format=fmt, stem=stem, options=options, answer_key=key, stages=stages,
                rubric=rubric, model_answer=s("model_answer"), topic=s("topic"),
                course_code=s("course_code") or "_ALL", course_title=s("course_title"), syllabus_topics=syl,
                skill=s("skill"), bloom_level=bloom, difficulty_band=s("difficulty_band"), source=s("source"),
                explanation=s("explanation"), hints=None if _blank(hints) else hints,
                textbook_ref=s("textbook_ref"), human_key_correct=_bool(row.get("human_key_correct")),
                row_number=row_number, extra={k: str(v) for k, v in extra.items()})
    return item, capped


def load_items(path: str | Path, cfg: Config) -> LoadResult:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if cfg.column_map:
        df = df.rename(columns=cfg.column_map)
        dup = sorted({c for c in df.columns if list(df.columns).count(c) > 1})
        if dup:
            raise ValueError(f"column_map produces duplicate columns: {', '.join(dup)}; drop or rename the "
                             "original column first")
    has_options_col = "options" in df.columns
    items: list[Item] = []
    invalid: list[dict[str, Any]] = []
    capped: list[str] = []
    seen: set[str] = set()
    for i, row in enumerate(df.to_dict(orient="records"), start=2):  # row 1 is the header
        try:
            item, was_capped = parse_row(row, i, cfg, has_options_col)
            if item.item_id in seen:
                raise RowError(f"item_id: '{item.item_id}' is duplicated")
        except RowError as e:
            invalid.append({"row_number": i, "item_id": str(row.get("item_id", "")), "reason": str(e)})
            continue
        seen.add(item.item_id)
        if was_capped:
            capped.append(item.item_id)
            log.warning("tolerance capped at %s for item %s", cfg.numeric.tolerance_cap, item.item_id)
        items.append(item)
    return LoadResult(items=items, invalid=invalid, n_rows=len(df), tolerance_capped=capped)


def write_invalid(invalid: list[dict[str, Any]], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / "invalid_rows.csv"
    pd.DataFrame(invalid, columns=["row_number", "item_id", "reason"]).to_csv(p, index=False)
    return p


def eligibility_table(items: list[Item]) -> pd.DataFrame:
    rows = []
    for fmt in sorted({it.format for it in items}):
        its = [it for it in items if it.format == fmt]
        rows.append({"format": fmt, "rows": len(its), "m1": sum(it.eligible("m1") for it in its),
                     "m2": sum(it.eligible("m2") for it in its), "m3": sum(it.eligible("m3") for it in its)})
    rows.append({"format": "TOTAL", "rows": len(items), "m1": sum(r["m1"] for r in rows),
                 "m2": sum(r["m2"] for r in rows), "m3": sum(r["m3"] for r in rows)})
    return pd.DataFrame(rows)
