from __future__ import annotations

import asyncio

import pytest
from conftest import config_dict
from fake_llm import smart_fake

from item_metrics.config import Config
from item_metrics.io_csv import load_items
from item_metrics.llm import LLMClient
from item_metrics.metrics import topic_blind as m3
from item_metrics.text_match import topic_named_in

CANDS = ["Conduction", "Convection", "Transient conduction", "Heat exchangers"]


def test_map_topic():
    assert m3.map_topic("conduction ", CANDS) == "Conduction"                      # exact after normalisation
    assert m3.map_topic("Transient conduction analysis", CANDS) == "Transient conduction"   # Jaccard 2/3
    assert m3.map_topic("Thermodynamic cycles", CANDS) == m3.UNMAPPED
    assert m3.map_topic("none", CANDS) == m3.NONE


def test_majority_ties():
    assert m3._majority(["A", "B"]) is None
    assert m3._majority(["A", "A", "B"]) == "A"
    assert m3._majority([None, "B"]) == "B"
    assert m3._majority([None, None]) is None


def test_leak_check():
    assert topic_named_in("Fourier's law", "According to Fourier's law, the flux ...")
    assert topic_named_in("Heat exchangers", "Size the heat exchangers in series")
    assert topic_named_in("Fin efficiency", "What is the efficiency of a fin")   # every content word present
    assert not topic_named_in("Convection", "The convective coefficient is 25")
    assert not topic_named_in("Extended surfaces", "fins added to surfaces")


def test_mrr_chance():
    assert m3.mrr_chance(3, 3) == pytest.approx((1 / 4) * (1 + 1 / 2 + 1 / 3))
    assert m3.mrr_chance(1, 3) == pytest.approx((1 / 2) * (1 + 1 / 2))   # only K + 1 ranks exist


def test_end_to_end(tmp_path, fixture_csv):
    cfg = Config.model_validate(config_dict(tmp_path))
    items = load_items(fixture_csv, cfg).items
    fake = smart_fake()
    raw = asyncio.run(m3.collect(items, cfg, LLMClient(cfg.llm, tmp_path / "out", backend=fake)))
    assert len(fake.calls) == 8 * 2
    res = m3.score(items, raw, cfg)
    t = res.tables["m3_topic_classification"].set_index("item_id")
    assert t["K"].iloc[0] == 8
    assert bool(t.loc["mcq_named", "topic_named_in_stem"]) and t["topic_named_in_stem"].sum() == 1
    # rt_hx: t1 right, t2 wrong -> tie -> no majority -> miss
    assert t.loc["rt_hx", "majority"] is None or t.loc["rt_hx", "majority"] != t.loc["rt_hx", "majority"]
    assert not t.loc["rt_hx", "correct"]
    s = res.summary
    assert s["blind_topic_accuracy"]["value"] == pytest.approx(7 / 8)
    chance = 1 / 9
    expect = (7 * 1.0 + (0 - chance) / (1 - chance)) / 8
    assert s["chance_adjusted_accuracy"]["value"] == pytest.approx(expect, abs=1e-4)
    assert t.loc["rt_hx", "item_score"] == pytest.approx((0.5 - chance) / (1 - chance), abs=1e-4)
    assert s["blind_topic_accuracy_excluding_named"]["n"] == 7
    assert s["top_k_accuracy"]["value"] == 1.0          # t2 ranked the truth second
    assert s["low_topic_count_courses"] == []
    assert res.normalised["M3"]["value"] is not None
    assert res.normalised["M3_strict"]["n"] == 7
