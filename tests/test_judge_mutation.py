from __future__ import annotations

import pytest
from conftest import config_dict

from item_metrics.config import Config
from item_metrics.metrics import judge_mutation as m2
from item_metrics.schemas import Criterion, Item

J1, J2 = "openrouter/va/j1", "openrouter/vb/j2"
TEXT = "The answer text says alpha, beta and gamma."


def _cfg(tmp_path, judges=(J1,), runs=1, **over) -> Config:
    d = config_dict(tmp_path, **over)
    d["models"]["judges"] = list(judges)
    d["judge_mutation"] = {**d["judge_mutation"], "judge_runs": runs}
    return Config.model_validate(d)


def _item() -> Item:
    return Item(item_id="it", format="short_text", stem="Explain.", topic="t",
                rubric=[Criterion(criterion_id=c, text=f"criterion {c}") for c in ("c1", "c2", "c3")])


def _judge(item_id: str, variant: str, model: str, fails: set[str], run: int = 0, quote: str = "answer text",
           drop: set[str] = frozenset()) -> dict:
    vs = [{"criterion_id": c, "verdict": "fail" if c in fails else "pass", "quote": "" if c in fails else quote}
          for c in ("c1", "c2", "c3") if c not in drop]
    return {"item_id": item_id, "variant_id": variant, "model": model, "run": run, "status": "ok",
            "parsed": {"verdicts": vs, "injection_detected": False}}


def _raw(judge_fails: dict[str, set[str]], invalid: set[str] = frozenset(), model: str = J1) -> dict:
    variants = [{"item_id": "it", "variant_id": f"mut:{c}", "kind": "mutation", "criterion_id": c, "text": TEXT,
                 "what_was_broken": c, "edit_size": "small", "status": "ok"} for c in ("c1", "c2", "c3")]
    variants.append({"item_id": "it", "variant_id": "para:1", "kind": "paraphrase", "criterion_id": "", "text": TEXT,
                     "what_was_broken": "", "edit_size": "", "status": "ok"})
    val = []
    for v in variants:
        ok = v["variant_id"] not in invalid
        p = ({"target_violated": ok, "others_preserved": True, "fluent": True} if v["kind"] == "mutation"
             else {"meaning_preserved": ok, "all_criteria_still_met": True})
        val.append({"item_id": "it", "variant_id": v["variant_id"], "model": "openrouter/vd/val", "status": "ok",
                    "parsed": p})
    judges = [_judge("it", vid, model, fails) for vid, fails in judge_fails.items()]
    return {"m2_reference": [{"item_id": "it", "reference": TEXT, "generated": False, "status": "ok"}],
            "m2_variants": variants, "m2_validation": val, "m2_judge": judges}


HAND = {"ref": set(), "mut:c1": {"c1"}, "mut:c2": set(), "mut:c3": {"c3", "c1"}, "para:1": set()}


def test_quote_rule():
    it = _item()
    ans = "Heat flows from  HOT to cold, $q = k \\Delta T / L$."
    parsed = {"verdicts": [
        {"criterion_id": "c1", "verdict": "pass", "quote": "heat flows from hot to cold,"},   # case/space/punct
        {"criterion_id": "c2", "verdict": "pass", "quote": "heat moves from hot to cold"},    # not verbatim
        {"criterion_id": "C3", "verdict": "PASS", "quote": "q = k \\Delta T / L"}]}           # $ stripped, id case
    rv, _ = m2.run_verdicts(it, ans, parsed)
    assert rv["c1"]["verdict"] == "pass" and rv["c1"]["quote_ok"]
    assert rv["c2"]["verdict"] == "fail" and rv["c2"]["quote_rejected"]
    assert rv["c3"]["verdict"] == "pass"
    rv, _ = m2.run_verdicts(it, ans, {"verdicts": [{"criterion_id": "c1", "verdict": "pass", "quote": ""}]})
    assert rv["c1"]["verdict"] == "fail" and rv["c1"]["quote_rejected"]
    assert rv["c2"]["missing"] and rv["c2"]["verdict"] == "fail"


def test_expected_labels():
    ref = {"kind": "reference", "criterion_id": ""}
    mut = {"kind": "mutation", "criterion_id": "c2"}
    para = {"kind": "paraphrase", "criterion_id": ""}
    assert [m2._expected(ref, c) for c in ("c1", "c2")] == ["pass", "pass"]
    assert [m2._expected(mut, c) for c in ("c1", "c2", "c3")] == ["pass", "fail", "pass"]
    assert m2._expected(para, "c2") == "pass"


def test_hand_computed_scores(tmp_path):
    cfg = _cfg(tmp_path)
    res = m2.score([_item()], _raw(HAND), cfg)
    j = res.summary["judges"][J1]
    crit = res.tables["m2_criterion_scores"]
    c = crit[crit["judge"] == J1].set_index("criterion_id")
    assert (c.loc["c1", "sensitivity"], c.loc["c1", "specificity"]) == (1.0, 0.75)
    assert (c.loc["c2", "sensitivity"], c.loc["c2", "specificity"]) == (0.0, 1.0)
    assert (c.loc["c3", "sensitivity"], c.loc["c3", "specificity"]) == (1.0, 1.0)
    assert j["balanced_accuracy"]["value"] == pytest.approx((0.875 + 0.5 + 1.0) / 3, abs=1e-4)
    assert j["sensitivity"]["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert j["specificity"]["value"] == pytest.approx(11 / 12, abs=1e-4)
    assert j["collateral_damage_rate"]["value"] == pytest.approx(1 / 3, abs=1e-4)
    assert j["score_error"] == pytest.approx((1 / 3 + 1 / 3) / 5, abs=1e-4)
    assert j["paraphrase_flip_rate"]["value"] == 0.0
    assert res.summary["rubric_defects"] == 0


def test_invalid_mutations_excluded(tmp_path):
    cfg = _cfg(tmp_path)
    res = m2.score([_item()], _raw(HAND, invalid={"mut:c2"}), cfg)
    crit = res.tables["m2_criterion_scores"]
    c = crit[crit["judge"] == J1].set_index("criterion_id")
    assert c.loc["c2", "n_mutations"] == 0 and c.loc["c2", "balanced_accuracy"] is None or \
        c.loc["c2", "balanced_accuracy"] != c.loc["c2", "balanced_accuracy"]     # None / NaN
    assert c.loc["c1", "specificity"] == pytest.approx(2 / 3, abs=1e-4)
    assert res.summary["mutation_validity_rate"]["value"] == 0.75
    j = res.summary["judges"][J1]
    assert j["balanced_accuracy"]["value"] == pytest.approx(((1 + 2 / 3) / 2 + 1.0) / 2, abs=1e-4)


def test_human_labels_override(tmp_path):
    cfg = _cfg(tmp_path)
    raw = _raw(HAND)
    base = m2.score([_item()], raw, cfg).summary["judges"][J1]["balanced_accuracy"]["value"]
    after = m2.score([_item()], raw, cfg, human={("it", "mut:c2"): False}).summary["judges"][J1]
    assert after["balanced_accuracy"]["value"] != base


def test_reference_defect(tmp_path):
    cfg = _cfg(tmp_path, judges=(J1, J2))
    raw = _raw({k: v | {"c3"} if k == "ref" else v for k, v in HAND.items()})
    raw["m2_judge"] += [_judge("it", vid, J2, fails | ({"c3"} if vid == "ref" else set()))
                        for vid, fails in HAND.items()]
    res = m2.score([_item()], raw, cfg)
    assert res.summary["rubric_defects"] == 1
    assert list(res.tables["m2_rubric_defects"]["criterion_id"]) == ["c3"]
    crit = res.tables["m2_criterion_scores"]
    assert "c3" not in set(crit["criterion_id"])


def test_run_split_and_missing(tmp_path):
    cfg = _cfg(tmp_path, runs=2)
    raw = _raw(HAND)
    # second run on the paraphrase fails c2: split -> counts as fail, credit 0.5
    # (a split on the reference would make c2 a reference defect instead)
    extra = [_judge("it", vid, J1, fails, run=1) for vid, fails in HAND.items() if vid != "para:1"]
    extra.append(_judge("it", "para:1", J1, {"c2"}, run=1))
    raw["m2_judge"] += extra
    res = m2.score([_item()], raw, cfg)
    j = res.summary["judges"][J1]
    assert j["run_split_rate"]["value"] == pytest.approx(1 / 15, abs=1e-4)
    assert j["paraphrase_flip_rate"]["value"] == pytest.approx(1 / 3, abs=1e-4)
    assert j["score_error"] == pytest.approx((1 / 3 + 1 / 3 + 0.5 / 3) / 5, abs=1e-4)
    raw2 = _raw(HAND)
    raw2["m2_judge"][4] = _judge("it", "para:1", J1, set(), drop={"c3"})
    j2 = m2.score([_item()], raw2, _cfg(tmp_path)).summary["judges"][J1]
    assert j2["missing_verdict_rate"]["value"] == pytest.approx(1 / 15, abs=1e-4)
