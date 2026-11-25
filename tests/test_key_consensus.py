from __future__ import annotations

import asyncio

import pytest
from conftest import config_dict
from fake_llm import smart_fake

from item_metrics.config import Config
from item_metrics.io_csv import load_items
from item_metrics.llm import LLMClient
from item_metrics.metrics import key_consensus as m1
from item_metrics.schemas import Item, NumericKey, Option

SOLVERS = ["openrouter/va/s1", "openrouter/vb/s2", "openrouter/vc/s3"]
IDENTITY = {x: x for x in "ABCD"}


def _cfg(tmp_path, **over) -> Config:
    return Config.model_validate(config_dict(tmp_path, **over))


def _mcq(iid: str, key: str = "B") -> Item:
    return Item(item_id=iid, format="mcq_4", stem="s", answer_key=key,
                options=[Option(id=x, text=f"t{x}") for x in "ABCD"])


def _rec(iid: str, model: str, parsed: dict | None, status: str = "ok") -> dict:
    return {"item_id": iid, "model": model, "status": status, "parsed": parsed, "label_map": IDENTITY}


def _mcq_out(ans: str, correct: tuple[str, ...] | None = None, ambiguous: bool = False) -> dict:
    correct = correct if correct is not None else (ans,)
    return {"answer": ans, "option_verdicts": [{"id": x, "correct": x in correct} for x in "ABCD"],
            "ambiguous": ambiguous}


def _raw(answers: dict[str, list[dict | None]]) -> dict:
    return {"m1_solver": [_rec(iid, SOLVERS[i], a) for iid, outs in answers.items() for i, a in enumerate(outs)],
            "m1_verifier": []}


def test_every_status(tmp_path):
    cfg = _cfg(tmp_path)
    items = [_mcq(x) for x in ("full", "major", "split", "alldis", "other")]
    raw = _raw({
        "full": [_mcq_out("B")] * 3,
        "major": [_mcq_out("B"), _mcq_out("B"), _mcq_out("A")],
        "split": [_mcq_out("B"), _mcq_out("A"), _mcq_out("C")],
        "alldis": [_mcq_out("A"), _mcq_out("C"), _mcq_out("D")],
        "other": [_mcq_out("A"), _mcq_out("A"), _mcq_out("C")],
    })
    res = m1.score(items, raw, cfg)
    t = res.tables["m1_key_consensus"].set_index("item_id")
    assert t.loc["full", "status"] == "full_agree"
    assert t.loc["major", "status"] == "majority_agree"
    assert t.loc["split", "status"] == "split"
    assert t.loc["alldis", "status"] == "all_disagree"
    assert t.loc["other", "status"] == "solvers_agree_on_other"
    assert t.loc["other", "most_common_other_answer"] == "A"
    assert bool(t.loc["full", "accepted"]) and not bool(t.loc["major", "accepted"])
    assert res.summary["key_consensus_rate"]["value"] == 0.2
    # majority gate accepts the majority item
    res2 = m1.score(items, raw, _cfg(tmp_path, key_consensus={"gate_rule": "majority"}))
    assert res2.tables["m1_key_consensus"].set_index("item_id").loc["major", "accepted"]
    review = res.tables["m1_keys_to_review"]
    assert list(review["item_id"])[:2] == ["other", "alldis"]   # sorted by severity


def test_planted_recall_hand_example(tmp_path):
    cfg = _cfg(tmp_path, numeric={"planted_perturbations": [0.03]})
    mcq = _mcq("q1")
    num = Item(item_id="n1", format="numeric", stem="s", answer_key=NumericKey(value=100, unit="", tolerance=0.02))
    raw = _raw({"q1": [_mcq_out("B")] * 3,
                "n1": [{"answer": 101.5, "unit": ""}] * 3})     # within 2% of 100, and within 2% of 103
    res = m1.score([mcq, num], raw, cfg)
    audit = res.tables["m1_planted_audit"]
    assert len(audit) == 5                                       # 3 distractors + 100 x (1 +/- 0.03)
    assert audit["caught"].sum() == 4                            # 103 is not caught: 101.5 is within 2% of it
    per = res.summary["planted_error_recall"]
    assert per["overall"]["value"] == 0.8
    assert per["by_format"]["mcq_4"]["value"] == 1.0 and per["by_format"]["numeric"]["value"] == 0.5
    assert all(v["value"] == 0.8 for v in per["by_solver"].values())
    uncaught = audit[~audit["caught"]].iloc[0]
    assert uncaught["planted_kind"] == "final+" and uncaught["delta"] == 0.03


def test_untrusted_when_human_says_wrong(tmp_path):
    cfg = _cfg(tmp_path)
    it = _mcq("q1")
    it.human_key_correct = False
    res = m1.score([it], _raw({"q1": [_mcq_out("B")] * 3}), cfg)
    assert res.summary["planted_error_recall"]["overall"]["n"] == 0
    g = res.summary["ground_truth"]
    assert g["confusion"]["accepted_but_key_wrong"] == 1


def test_ambiguity_rules(tmp_path):
    cfg = _cfg(tmp_path)
    items = [_mcq(x) for x in ("r1", "r2", "r3", "clean")]
    raw = _raw({
        "r1": [_mcq_out("B", ("B", "C"))] * 3,                                    # two options marked correct
        "r2": [_mcq_out("B"), _mcq_out("B"), _mcq_out("B", ("B",))],
        "r3": [_mcq_out("B"), _mcq_out("B"), _mcq_out("B", ambiguous=True)],
        "clean": [_mcq_out("B")] * 3,
    })
    # r2: solvers disagree about option C without anyone marking != 1 correct
    raw["m1_solver"][4]["parsed"]["option_verdicts"] = [{"id": "A", "correct": False}, {"id": "B", "correct": False},
                                                        {"id": "C", "correct": True}, {"id": "D", "correct": False}]
    t = m1.score(items, raw, cfg).tables["m1_key_consensus"].set_index("item_id")
    assert "1_not_one_correct" in t.loc["r1", "ambiguity_rule"]
    assert t.loc["r2", "ambiguity_rule"] == "2_option_disagreement"
    assert t.loc["r3", "ambiguity_rule"] == "3_flagged_ambiguous"
    assert not t.loc["clean", "ambiguous"]
    assert t.loc["r3", "item_score"] == 0.5


def test_missing_and_budget(tmp_path):
    cfg = _cfg(tmp_path)
    raw = _raw({"q1": [_mcq_out("B"), _mcq_out("B"), None], "q2": [_mcq_out("B")] * 3})
    raw["m1_solver"][2]["status"] = "parse_failure"
    raw["m1_solver"][5]["status"] = "budget_skipped"
    res = m1.score([_mcq("q1"), _mcq("q2")], raw, cfg)
    t = res.tables["m1_key_consensus"].set_index("item_id")
    assert t.loc["q1", "status"] == "majority_agree"
    assert t.loc["q1", f"match_reason__{SOLVERS[2]}"] == "missing"
    assert "q2" not in t.index and res.summary["incomplete_items"] == 1


def test_collect_with_shuffle_end_to_end(tmp_path, fixture_csv):
    cfg = _cfg(tmp_path)
    items = load_items(fixture_csv, cfg).items
    fake = smart_fake()
    llm = LLMClient(cfg.llm, tmp_path / "out", backend=fake)
    raw = asyncio.run(m1.collect(items, cfg, llm))
    assert len(fake.calls) == 5 * 3
    res = m1.score(items, raw, cfg)
    t = res.tables["m1_key_consensus"].set_index("item_id")
    assert t.loc["mcq_ok", "status"] == "full_agree"
    assert t.loc["num_conv", "status"] == "full_agree"
    assert t.loc["staged_rad", "status"] == "full_agree"
    assert t.loc["mcq_badkey", "status"] == "solvers_agree_on_other"   # the deliberately wrong key
    assert t.loc["mcq_badkey", "most_common_other_answer"] == "A"
    g = res.summary["ground_truth"]
    assert g["confusion"]["flagged_and_wrong_key"] == 1 and g["recall"]["value"] == 1.0


def test_verifier(tmp_path, fixture_csv):
    cfg = _cfg(tmp_path, models={**config_dict(tmp_path)["models"], "verifier": ["openrouter/vd/ver"]})
    items = load_items(fixture_csv, cfg).items
    fake = smart_fake()
    # sycophantic verifier: accepts everything
    llm = LLMClient(cfg.llm, tmp_path / "out", backend=fake)
    raw = asyncio.run(m1.collect(items, cfg, llm))
    res = m1.score(items, raw, cfg)
    v = res.summary["verifier"]
    assert v["recall"]["value"] == 0.0 and v["false_reject_rate"]["value"] == 0.0
    assert v["balanced_accuracy"] == 0.5
    assert "verifier" in res.normalised["M1"]["weights_used"]
    assert abs(sum(res.normalised["M1"]["weights_used"].values()) - 1) < 1e-3
