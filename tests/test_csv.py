from __future__ import annotations

import csv
import json
from pathlib import Path

from item_metrics.config import Config
from item_metrics.io_csv import load_items
from item_metrics.schemas import NumericKey


def _write(path: Path, rows: list[dict]) -> Path:
    cols = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


MCQ_OPTS = json.dumps([{"id": x, "text": f"opt {x}"} for x in "ABCD"])


def test_fixture_parses_every_format(fixture_csv):
    res = load_items(fixture_csv, Config())
    assert res.invalid == []
    by = {it.item_id: it for it in res.items}
    assert {it.format for it in res.items} == {"mcq_4", "numeric", "staged", "short_text", "rubric_task"}
    assert [o.id for o in by["mcq_ok"].options] == ["A", "B", "C", "D"]
    assert by["mcq_ok"].options[0].misconception_id == "MISC_HALF_DT"
    assert by["mcq_ok"].answer_key == "B"
    assert by["mcq_ok"].human_key_correct is True and by["mcq_badkey"].human_key_correct is False
    k = by["num_conv"].answer_key
    assert isinstance(k, NumericKey) and k.value == 1000 and k.unit == "W/m^2" and k.tolerance == 0.02
    st = by["staged_rad"].stages
    assert [s.stage_id for s in st] == ["S1", "S2"] and st[0].answer_key.value == 3543.75
    assert [c.weight for c in by["st_fins"].rubric] == [2, 1, 1]
    # defaults: criterion ids c1..cn, weight 1
    assert [(c.criterion_id, c.weight) for c in by["st_biot"].rubric] == [("c1", 1.0), ("c2", 1.0)]
    assert len(by["mcq_ok"].syllabus_topics) == 8
    assert by["mcq_ok"].course_code == "HT101"


def test_option_columns(tmp_path):
    p = _write(tmp_path / "a.csv", [{"item_id": "q1", "format": "mcq_4", "stem": "s", "topic": "t",
                                     "option_A": "a", "option_B": "b", "option_C": "c", "option_D": "d",
                                     "answer_key": "c"}])
    res = load_items(p, Config())
    assert res.invalid == []
    it = res.items[0]
    assert [o.text for o in it.options] == ["a", "b", "c", "d"] and it.answer_key == "C"


def test_column_map(tmp_path):
    p = _write(tmp_path / "a.csv", [{"id": "q1", "kind": "mcq_4", "question": "s", "options": MCQ_OPTS,
                                     "correct_option": "A", "syllabus_topic": "t"}])
    cfg = Config(column_map={"id": "item_id", "kind": "format", "question": "stem", "correct_option": "answer_key",
                             "syllabus_topic": "topic"})
    res = load_items(p, cfg)
    assert res.invalid == [] and res.items[0].answer_key == "A" and res.items[0].topic == "t"


def test_rejections(tmp_path):
    good_mcq = {"format": "mcq_4", "stem": "s", "options": MCQ_OPTS, "answer_key": "A"}
    rows = [
        {"item_id": "no_stem", "format": "mcq_4", "stem": "", "options": MCQ_OPTS, "answer_key": "A"},
        {"item_id": "bad_format", "format": "essay", "stem": "s"},
        {"item_id": "bad_json", "format": "mcq_4", "stem": "s", "options": "[{bad", "answer_key": "A"},
        {"item_id": "three_opts", "format": "mcq_4", "stem": "s",
         "options": json.dumps([{"id": x, "text": x} for x in "ABC"]), "answer_key": "A"},
        {"item_id": "dup_opts", "format": "mcq_4", "stem": "s",
         "options": json.dumps([{"id": x, "text": x} for x in "AABC"]), "answer_key": "A"},
        {"item_id": "key_not_option", **{**good_mcq, "answer_key": "E"}},
        {"item_id": "num_no_value", "format": "numeric", "stem": "s", "answer_key": json.dumps({"unit": "m"})},
        {"item_id": "staged_one", "format": "staged", "stem": "s", "answer_key": "5",
         "stages": json.dumps([{"stage_id": "S1", "prompt": "p", "answer_key": 5}])},
        {"item_id": "rubric_empty", "format": "short_text", "stem": "s", "rubric": json.dumps([{"text": " "}])},
        {"item_id": "rubric_weight", "format": "short_text", "stem": "s",
         "rubric": json.dumps([{"text": "x", "weight": -1}])},
        {"item_id": "rubric_missing", "format": "rubric_task", "stem": "s"},
        {"item_id": "ok", **good_mcq},
        {"item_id": "ok", **good_mcq},
    ]
    res = load_items(_write(tmp_path / "bad.csv", rows), Config())
    reasons = {x["item_id"]: x["reason"] for x in res.invalid}
    assert [it.item_id for it in res.items] == ["ok"]
    expect = {"no_stem": "stem", "bad_format": "format", "bad_json": "JSON does not parse",
              "three_opts": "exactly 4", "dup_opts": "exactly 4", "key_not_option": "not one of the option ids",
              "num_no_value": "no numeric value", "staged_one": "at least 2 stages", "rubric_empty": "empty text",
              "rubric_weight": "positive number", "rubric_missing": "rubric: required", "ok": "duplicated"}
    for iid, frag in expect.items():
        assert frag in reasons[iid], (iid, reasons.get(iid))
    assert all(x["row_number"] >= 2 for x in res.invalid)


def test_tolerance_cap(tmp_path):
    p = _write(tmp_path / "a.csv", [{"item_id": "n", "format": "numeric", "stem": "s",
                                     "answer_key": json.dumps({"value": 3, "unit": "m", "tolerance": 0.1})},
                                    {"item_id": "m", "format": "numeric", "stem": "s", "answer_key": "4.5 kg"}])
    res = load_items(p, Config())
    by = {it.item_id: it for it in res.items}
    assert by["n"].answer_key.tolerance == 0.02 and res.tolerance_capped == ["n"]
    assert by["m"].answer_key.value == 4.5 and by["m"].answer_key.unit == "kg"
    assert by["m"].answer_key.tolerance == 0.02
