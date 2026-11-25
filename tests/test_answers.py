from __future__ import annotations

from item_metrics.answers import (MATCH, MISSING, UNIT, UNPARSEABLE, WRONG, compare_numeric, match_item,
                                  parse_number, shuffle_options)
from item_metrics.schemas import Item, NumericKey, Option, SolverOut, Stage


def _mcq() -> Item:
    return Item(item_id="q", format="mcq_4", stem="s", answer_key="B",
                options=[Option(id=x, text=f"text {x}") for x in "ABCD"])


def test_shuffle_is_seeded_and_remaps():
    it = _mcq()
    shown1, map1 = shuffle_options(it.options, 1234, "q", "m1")
    shown2, map2 = shuffle_options(it.options, 1234, "q", "m1")
    assert map1 == map2
    assert sorted(map1.values()) == ["A", "B", "C", "D"] and list(map1) == ["A", "B", "C", "D"]
    _, other = shuffle_options(it.options, 1234, "q", "m2")
    assert any(shuffle_options(it.options, 1234, "q", f"m{i}")[1] != map1 for i in range(2, 8)) or other != map1
    # the solver answers with the shown label of the original "B"
    label = next(lab for lab, orig in map1.items() if orig == "B")
    m = match_item(it, SolverOut(answer=f"({label})"), map1, "B", None, 0.02, 0.02)
    assert m.match and m.reason == MATCH and m.canonical == "B"
    wrong = next(lab for lab, orig in map1.items() if orig != "B")
    assert not match_item(it, SolverOut(answer=wrong), map1, "B", None, 0.02, 0.02).match
    assert match_item(it, SolverOut(answer=""), map1, "B", None, 0.02, 0.02).reason == MISSING
    assert match_item(it, SolverOut(answer="none of them"), map1, "B", None, 0.02, 0.02).reason == UNPARSEABLE


def test_tolerance_boundary():
    k = NumericKey(value=100, unit="", tolerance=0.02)
    assert compare_numeric(102, "", k, 0.02)[0] == MATCH          # exactly 2%
    assert compare_numeric(98, "", k, 0.02)[0] == MATCH
    assert compare_numeric(102.01, "", k, 0.02)[0] == WRONG       # 2.01%
    assert compare_numeric(97.99, "", k, 0.02)[0] == WRONG


def test_units():
    k = NumericKey(value=1000, unit="W/m^2")
    assert compare_numeric(1, "kW/m^2", k, 0.02)[0] == MATCH
    assert compare_numeric(1.0, "kW/m²", k, 0.02)[0] == MATCH
    assert compare_numeric("1000 W/m^2", "", k, 0.02)[0] == MATCH
    assert compare_numeric(1000, "W m^-2", k, 0.02)[0] == MATCH
    assert compare_numeric(1000, "kg", k, 0.02)[0] == UNIT
    assert compare_numeric(1000, "blarg/zz", k, 0.02)[0] == UNIT
    assert compare_numeric("about a thousand", "W/m^2", k, 0.02)[0] == UNPARSEABLE
    assert compare_numeric(None, "", k, 0.02)[0] == MISSING
    assert compare_numeric(50, "%", NumericKey(value=0.5), 0.02)[0] == MATCH
    assert compare_numeric(373.15, "K", NumericKey(value=100, unit="degC"), 0.02)[0] == MATCH


def test_zero_key():
    k = NumericKey(value=0, unit="")
    assert compare_numeric(0.01, "", k, 0.02)[0] == MATCH
    assert compare_numeric(-0.02, "", k, 0.02)[0] == MATCH
    assert compare_numeric(0.03, "", k, 0.02)[0] == WRONG


def test_staged_fails_on_one_stage():
    it = Item(item_id="s", format="staged", stem="s", answer_key=NumericKey(value=10, unit="kW"),
              stages=[Stage(stage_id="S1", prompt="p", answer_key=NumericKey(value=5, unit="kW")),
                      Stage(stage_id="S2", prompt="p", answer_key=NumericKey(value=10, unit="kW"))])
    good = SolverOut(answer=10, unit="kW", stage_answers={"S1": {"value": 5000, "unit": "W"}, "S2": {"value": 10, "unit": "kW"}})
    assert match_item(it, good, {}, it.answer_key, None, 0.02, 0.02).match
    bad = SolverOut(answer=10, unit="kW", stage_answers={"S1": {"value": 6, "unit": "kW"}, "S2": {"value": 10, "unit": "kW"}})
    m = match_item(it, bad, {}, it.answer_key, None, 0.02, 0.02)
    assert not m.match and m.reason == WRONG
    missing = SolverOut(answer=10, unit="kW", stage_answers={"S2": {"value": 10, "unit": "kW"}})
    assert match_item(it, missing, {}, it.answer_key, None, 0.02, 0.02).reason == MISSING


def test_parse_number_forms():
    assert parse_number("1,200") == (1200.0, "")
    assert parse_number("1.2e3 W")[0] == 1200.0
    assert parse_number("1.2 x 10^3")[0] == 1200.0
    assert parse_number("1.2 × 10^-3 m")[0] == 1.2e-3
    assert parse_number("−5")[0] == -5.0
    assert parse_number("3/4")[0] == 0.75
    assert parse_number("abc")[0] is None
    assert parse_number(True)[0] is None
