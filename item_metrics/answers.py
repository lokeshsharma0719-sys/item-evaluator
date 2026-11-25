"""Answer normalisation and comparison: MCQ option ids (with shuffle remap), numbers, units (pint), tolerance."""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional, Sequence

from .schemas import Item, NumericKey, Option, SolverOut

MATCH, WRONG, UNIT, UNPARSEABLE, MISSING = "match", "wrong_value", "unit_mismatch", "unparseable", "missing"
LABELS = ("A", "B", "C", "D")


def stable_rng(*parts: Any) -> random.Random:
    """Seeded RNG independent of PYTHONHASHSEED."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return random.Random(int(h[:16], 16))


# ---------------------------------------------------------------- MCQ

def shuffle_options(options: Sequence[Option], seed: int, item_id: str, model: str, shuffle: bool = True
                    ) -> tuple[list[tuple[str, Option]], dict[str, str]]:
    """Returns ([(shown_label, option)], {shown_label: original_id})."""
    order = list(options)
    if shuffle:
        stable_rng(seed, item_id, model, "options").shuffle(order)
    shown = [(LABELS[i] if i < len(LABELS) else str(i + 1), o) for i, o in enumerate(order)]
    return shown, {lab: o.id for lab, o in shown}


_LETTER = re.compile(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])")


def parse_option_label(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    s = str(answer).strip()
    if not s:
        return None
    if len(s) == 1 and s.upper() in LABELS:
        return s.upper()
    m = re.match(r"^\(?\s*(?:option\s+)?([A-Da-d])\s*[\).:]?(?:\s|$)", s, re.I)
    if m:
        return m.group(1).upper()
    m = _LETTER.search(s)
    return m.group(1).upper() if m else None


# ---------------------------------------------------------------- numbers

_NUM = re.compile(
    r"^\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?)"
    r"(?:\s*(?:x|×|\*|\\times|·)\s*10\s*(?:\^|\*\*)\s*\{?\s*([-+]?\d+)\s*\}?)?"
    r"(.*)$", re.S)


def parse_number(x: Any) -> tuple[Optional[float], str]:
    """Returns (value, trailing text). Handles 1,200 / 1.2e3 / 1.2 x 10^3 / unicode minus / a/b fractions."""
    if x is None or isinstance(x, bool):
        return None, ""
    if isinstance(x, (int, float)):
        return (float(x), "") if math.isfinite(float(x)) else (None, "")
    s = str(x).strip().replace("−", "-").replace("–", "-")
    s = s.replace("$", "").strip()
    frac = re.match(r"^\s*([-+]?\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$", s)
    if frac and float(frac.group(2)) != 0:
        return float(frac.group(1)) / float(frac.group(2)), ""
    m = _NUM.match(s)
    if not m:
        return None, s
    v = float(m.group(1).replace(",", ""))
    if m.group(2):
        v *= 10 ** int(m.group(2))
    return (v if math.isfinite(v) else None), m.group(3).strip()


# ---------------------------------------------------------------- units

@lru_cache(maxsize=1)
def _ureg():
    import pint
    return pint.UnitRegistry(autoconvert_offset_to_baseunit=True)


_EMPTY_UNITS = {"", "-", "none", "dimensionless", "unitless", "n/a", "na", "1"}
_SUPERSCRIPTS = {"²": "**2", "³": "**3", "¹": "**1", "⁻¹": "**-1", "⁻²": "**-2", "⁻³": "**-3"}


def normalise_unit(u: Any) -> str:
    s = "" if u is None else str(u).strip()
    if s.lower() in _EMPTY_UNITS:
        return ""
    s = s.replace("$", "").replace("\\,", " ").replace("{", "(").replace("}", ")")
    s = s.replace("\\cdot", "*").replace("·", "*").replace("⋅", "*").replace("×", "*")
    for k in ("⁻¹", "⁻²", "⁻³"):
        s = s.replace(k, _SUPERSCRIPTS[k])
    for k in ("²", "³", "¹"):
        s = s.replace(k, _SUPERSCRIPTS[k])
    s = s.replace("^", "**").replace("°C", "degC").replace("°F", "degF").replace("°", "deg")
    s = s.replace("µ", "u").replace("μ", "u").replace("%", "percent")
    s = re.sub(r"\bhrs?\b", "hour", s)
    s = re.sub(r"\bsecs?\b", "second", s)
    s = re.sub(r"\bmins?\b", "minute", s)
    s = re.sub(r"\blit(?:re|er)s?\b", "liter", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_quantity(value: float, unit: Any):
    """pint Quantity, or None when the unit cannot be parsed."""
    ureg = _ureg()
    u = normalise_unit(unit)
    if not u:
        return ureg.Quantity(value, "dimensionless")
    try:
        return ureg.Quantity(value, ureg.parse_units(u))
    except Exception:
        pass
    try:
        expr = ureg.parse_expression(u)
        if hasattr(expr, "units"):
            return ureg.Quantity(value * float(expr.magnitude), expr.units)
        return ureg.Quantity(value * float(expr), "dimensionless")
    except Exception:
        return None


def convert(value: float, from_unit: Any, to_unit: Any) -> Optional[float]:
    """Value expressed in `to_unit`; None when either unit is unparseable or they are not convertible."""
    if normalise_unit(from_unit) == normalise_unit(to_unit):
        return value
    a = to_quantity(value, from_unit)
    b = to_quantity(1.0, to_unit)
    if a is None or b is None:
        return None
    try:
        return float(a.to(b.units).magnitude) / float(b.magnitude)
    except Exception:
        return None


def within_tolerance(a: float, k: float, tol: float) -> bool:
    slack = 1e-9 * max(1.0, abs(k))  # floating-point slack so exactly-at-tolerance values match
    if k == 0:
        return abs(a) <= tol + slack
    return abs(a - k) <= tol * abs(k) + slack


def effective_tolerance(key: NumericKey, default: float, cap: float) -> float:
    t = key.tolerance if key.tolerance is not None else default
    return min(t, cap)


def compare_numeric(value: Any, unit: Any, key: NumericKey, tol: float) -> tuple[str, Optional[float]]:
    """Returns (match_reason, answer value converted to the key's unit)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return MISSING, None
    v, rest = parse_number(value)
    if v is None:
        return UNPARSEABLE, None
    u = unit if (unit is not None and str(unit).strip()) else rest
    conv = convert(v, u, key.unit)
    if conv is None:
        return UNIT, None
    return (MATCH if within_tolerance(conv, key.value, tol) else WRONG), conv


def norm_text(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s if s is not None else "")).strip().lower()


def compare_key(value: Any, unit: Any, key: Any, default_tol: float, cap: float) -> tuple[str, Any]:
    """Compare one answer to a numeric or text key."""
    if isinstance(key, NumericKey):
        return compare_numeric(value, unit, key, effective_tolerance(key, default_tol, cap))
    if value is None or not str(value).strip():
        return MISSING, None
    return (MATCH if norm_text(value) == norm_text(key) else WRONG), norm_text(value)


# ---------------------------------------------------------------- item-level matching

@dataclass
class ItemMatch:
    match: bool
    reason: str
    canonical: Any            # comparable form of the solver's answer (orig option id / value in key unit)
    display: str              # what the solver answered, for CSVs


def _stage_answer(out: SolverOut, stage_id: str) -> tuple[Any, str]:
    sa = out.stage_answers.get(stage_id)
    if sa is None:
        for k, v in out.stage_answers.items():
            if k.strip().lower() == stage_id.strip().lower():
                sa = v
                break
    return (None, "") if sa is None else (sa.value, sa.unit)


def match_item(item: Item, out: Optional[SolverOut], label_map: dict[str, str], key: Any,
               stage_keys: Optional[dict[str, Any]], default_tol: float, cap: float) -> ItemMatch:
    """`key` and `stage_keys` are passed explicitly so planted keys can be re-checked against stored answers."""
    if out is None:
        return ItemMatch(False, MISSING, None, "")
    if item.format == "mcq_4":
        lab = parse_option_label(out.answer)
        if lab is None:
            reason = MISSING if out.answer in (None, "") else UNPARSEABLE
            return ItemMatch(False, reason, None, str(out.answer or ""))
        orig = label_map.get(lab)
        if orig is None:
            return ItemMatch(False, UNPARSEABLE, None, lab)
        ok = orig.upper() == str(key).strip().upper()
        return ItemMatch(ok, MATCH if ok else WRONG, orig.upper(), orig.upper())

    display = f"{out.answer} {out.unit}".strip()
    reason, conv = compare_key(out.answer, out.unit, key, default_tol, cap)
    if item.format == "numeric":
        return ItemMatch(reason == MATCH, reason, conv, display)
    # staged: final value AND every keyed stage must match
    reasons = [reason]
    parts = [display]
    for st in item.stages:
        sk = (stage_keys or {}).get(st.stage_id, st.answer_key)
        val, unit = _stage_answer(out, st.stage_id)
        parts.append(f"{st.stage_id}={val} {unit}".strip())
        if sk is None or not checkable_stage_key(sk):
            continue                    # free-text stage keys cannot be compared deterministically
        r, _ = compare_key(val, unit, sk, default_tol, cap)
        reasons.append(r)
    bad = [r for r in reasons if r != MATCH]
    return ItemMatch(not bad, bad[0] if bad else MATCH, conv, "; ".join(parts))


MAX_TEXT_KEY_WORDS = 3


def checkable_stage_key(k: Any) -> bool:
    """Numeric keys and short text keys (a word or label) are compared; sentence-length explanations are not."""
    return isinstance(k, NumericKey) or len(str(k).split()) <= MAX_TEXT_KEY_WORDS


def same_answer(item: Item, a: Any, b: Any, key: Any, default_tol: float, cap: float) -> bool:
    """Do two solvers give the same (non-key) answer? MCQ: same id; numeric: within tolerance of each other."""
    if a is None or b is None:
        return False
    if item.format == "mcq_4" or not isinstance(key, NumericKey):
        return a == b
    tol = effective_tolerance(key, default_tol, cap)
    return within_tolerance(float(a), float(b), tol) and within_tolerance(float(b), float(a), tol)
