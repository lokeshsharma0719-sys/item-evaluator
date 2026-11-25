"""M2 Consistency: judge accuracy on an answer-mutation set (spec section 5).

Answers with known-by-construction verdicts (reference, one minimal mutation per criterion, paraphrases) are
graded by every judge; sensitivity / specificity per criterion are computed in code from the verdicts.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .. import normalise as N
from ..answers import stable_rng
from ..config import Config
from ..llm import LLMClient, gather_progress
from ..prompts import (JUDGE, MUTATION_VALIDATOR, MUTATOR_CRITERION, MUTATOR_PARAPHRASE, PARAPHRASE_VALIDATOR,
                       REFERENCE_WRITER, build_judge, build_mutation_validator, build_mutator_criterion,
                       build_mutator_paraphrase, build_paraphrase_validator, build_reference_writer)
from ..schemas import (Criterion, Item, JudgeOut, MutationOut, MutationValidOut, ParaphraseOut, ParaphraseValidOut,
                       ReferenceOut)
from ..stats import boot_est, cohen_kappa, krippendorff_alpha_nominal, mean, prop, r
from ..text_match import quote_in
from .common import MetricResult, budget_skipped, df, flag, is_ok, record

log = logging.getLogger(__name__)
REF = "ref"
ENSEMBLE = "ensemble"
PASS, FAIL, SPLIT = "pass", "fail", "split"


def _cid(s: Any) -> str:
    return str(s or "").strip().lower()


# ---------------------------------------------------------------- per-run verdicts (quote rule, missing rule)

def run_verdicts(item: Item, answer: str, parsed: Optional[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], bool]:
    """Effective pass/fail per criterion for one judge run, plus injection flag. A pass whose quote is not found
    verbatim (after normalisation) becomes fail; a missing criterion is fail."""
    given = {}
    for v in (parsed or {}).get("verdicts", []) or []:
        cid = _cid(v.get("criterion_id"))
        if cid and cid not in given:
            given[cid] = v
    out: dict[str, dict[str, Any]] = {}
    for c in item.rubric:
        v = given.get(_cid(c.criterion_id))
        if v is None:
            out[c.criterion_id] = {"verdict": FAIL, "raw": "", "quote": "", "quote_ok": None, "missing": True,
                                   "quote_rejected": False}
            continue
        raw = str(v.get("verdict", "")).strip().lower()
        quote = str(v.get("quote") or "")
        if raw == PASS:
            ok = quote_in(quote, answer)
            out[c.criterion_id] = {"verdict": PASS if ok else FAIL, "raw": raw, "quote": quote, "quote_ok": ok,
                                   "missing": False, "quote_rejected": not ok}
        else:
            out[c.criterion_id] = {"verdict": FAIL, "raw": raw, "quote": quote, "quote_ok": None, "missing": False,
                                   "quote_rejected": False}
    return out, bool((parsed or {}).get("injection_detected"))


def combine_runs(runs: list[dict[str, dict[str, Any]]], cid: str) -> tuple[str, float]:
    passes = sum(r_[cid]["verdict"] == PASS for r_ in runs)
    if passes == len(runs):
        return PASS, 1.0
    if passes == 0:
        return FAIL, 0.0
    return SPLIT, 0.5


def judge_variant(item: Item, answer: str, recs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """All runs of one judge on one variant -> per-criterion verdict/credit and run-level counts."""
    recs = [x for x in recs if not budget_skipped(x)]
    if not recs:
        return None
    runs, inj = [], 0
    for x in sorted(recs, key=lambda y: y.get("run", 0)):
        rv, injected = run_verdicts(item, answer, x["parsed"] if is_ok(x) else None)
        runs.append(rv)
        inj += injected
    out = {"verdict": {}, "credit": {}, "runs": runs, "n_runs": len(runs), "injection": inj}
    for c in item.rubric:
        v, cr = combine_runs(runs, c.criterion_id)
        out["verdict"][c.criterion_id] = v
        out["credit"][c.criterion_id] = cr
    return out


def find_defects(item: Item, ref_by_judge: dict[str, Optional[dict[str, Any]]]) -> dict[str, list[str]]:
    """criterion_id -> judges failing it on the reference, for criteria failed by a majority of judges."""
    avail = {j: v for j, v in ref_by_judge.items() if v is not None}
    out = {}
    for c in item.rubric:
        failing = [j for j, v in avail.items() if v["verdict"][c.criterion_id] != PASS]
        if avail and len(failing) > len(avail) / 2:
            out[c.criterion_id] = failing
    return out


# ---------------------------------------------------------------- collect

def _order(item: Item, cfg: Config, variant_id: str, judge: str, run: int) -> list[Criterion]:
    crit = list(item.rubric)
    stable_rng(cfg.seed, item.item_id, variant_id, judge, run, "criteria").shuffle(crit)
    return crit


def _judge_all(item: Item, variant_id: str, answer: str, cfg: Config, llm: LLMClient) -> list:
    async def one(judge: str, run: int) -> dict[str, Any]:
        order = _order(item, cfg, variant_id, judge, run)
        system, user = build_judge(item, answer, order)
        parsed, meta = await llm.call_json("judge", judge, system, user, JudgeOut, prompt_version=JUDGE.version,
                                           cache_key_extra=f"run{run}", item_id=item.item_id, variant_id=variant_id)
        return record(item.item_id, judge, meta, parsed, variant_id=variant_id, run=run,
                      criteria_order=[c.criterion_id for c in order])
    return [one(j, rn) for j in cfg.models.judges for rn in range(cfg.judge_mutation.judge_runs)]


async def collect(items: list[Item], cfg: Config, llm: LLMClient) -> dict[str, list[dict[str, Any]]]:
    elig = [it for it in items if it.eligible("m2")]
    mutator, validator = cfg.models.mutator, cfg.models.mutation_validator

    async def reference(it: Item) -> dict[str, Any]:
        if it.model_answer.strip():
            return {"item_id": it.item_id, "reference": it.model_answer, "generated": False, "status": "ok"}
        system, user = build_reference_writer(it)
        parsed, meta = await llm.call_json("reference_writer", mutator, system, user, ReferenceOut,
                                           prompt_version=REFERENCE_WRITER.version, item_id=it.item_id,
                                           variant_id=REF)
        return {"item_id": it.item_id, "reference": parsed.answer if parsed else "", "generated": True,
                "status": meta["status"] if parsed else (meta["status"] if meta["status"] != "ok" else "parse_failure")}

    refs = await gather_progress((reference(it) for it in elig), "M2 references")
    ref_text = {x["item_id"]: x["reference"] for x in refs if x["status"] == "ok" and x["reference"].strip()}
    with_ref = [it for it in elig if it.item_id in ref_text]

    jobs = [c for it in with_ref for c in _judge_all(it, REF, ref_text[it.item_id], cfg, llm)]
    judge_recs = await gather_progress(jobs, "M2 reference check")

    by_iv = _group_judge(judge_recs)
    variant_jobs = []
    for it in with_ref:
        ref_by_judge = {j: judge_variant(it, ref_text[it.item_id], by_iv.get((it.item_id, REF, j), []))
                        for j in cfg.models.judges}
        defects = find_defects(it, ref_by_judge)
        live = [c for c in it.rubric if c.criterion_id not in defects]
        if not live:
            continue
        for c in live:
            variant_jobs.append(_mutate(it, ref_text[it.item_id], c, mutator, llm))
        for k in range(1, cfg.judge_mutation.paraphrase_controls + 1):
            variant_jobs.append(_paraphrase(it, ref_text[it.item_id], k, mutator, llm))
    variants = await gather_progress(variant_jobs, "M2 mutations")
    good = [v for v in variants if v["status"] == "ok" and v["text"].strip()]

    item_by_id = {it.item_id: it for it in with_ref}

    async def validate(v: dict[str, Any]) -> dict[str, Any]:
        it = item_by_id[v["item_id"]]
        ref = ref_text[it.item_id]
        if v["kind"] == "mutation":
            crit = next(c for c in it.rubric if c.criterion_id == v["criterion_id"])
            system, user = build_mutation_validator(it, ref, v["text"], crit)
            schema, pv = MutationValidOut, MUTATION_VALIDATOR.version
        else:
            system, user = build_paraphrase_validator(it, ref, v["text"])
            schema, pv = ParaphraseValidOut, PARAPHRASE_VALIDATOR.version
        parsed, meta = await llm.call_json("mutation_validator", validator, system, user, schema, prompt_version=pv,
                                           item_id=it.item_id, variant_id=v["variant_id"])
        return record(it.item_id, validator, meta, parsed, variant_id=v["variant_id"])

    validations = await gather_progress((validate(v) for v in good), "M2 validation")

    # Every generated variant is judged (not only valid ones) so that `rescore --human-labels` can change which
    # variants count without making new calls.
    jobs = [c for v in good for c in _judge_all(item_by_id[v["item_id"]], v["variant_id"], v["text"], cfg, llm)]
    judge_recs += await gather_progress(jobs, "M2 judging")
    return {"m2_reference": refs, "m2_variants": variants, "m2_validation": validations, "m2_judge": judge_recs}


async def _mutate(it: Item, ref: str, c: Criterion, mutator: str, llm: LLMClient) -> dict[str, Any]:
    system, user = build_mutator_criterion(it, ref, c)
    vid = f"mut:{c.criterion_id}"
    parsed, meta = await llm.call_json("mutator", mutator, system, user, MutationOut,
                                       prompt_version=MUTATOR_CRITERION.version, item_id=it.item_id, variant_id=vid)
    return {"item_id": it.item_id, "variant_id": vid, "kind": "mutation", "criterion_id": c.criterion_id,
            "text": parsed.mutated_answer if parsed else "", "what_was_broken": parsed.what_was_broken if parsed else "",
            "edit_size": (parsed.edit_size or "").lower() if parsed else "", "status": meta["status"]}


async def _paraphrase(it: Item, ref: str, k: int, mutator: str, llm: LLMClient) -> dict[str, Any]:
    system, user = build_mutator_paraphrase(it, ref, k)
    vid = f"para:{k}"
    parsed, meta = await llm.call_json("mutator", mutator, system, user, ParaphraseOut,
                                       prompt_version=MUTATOR_PARAPHRASE.version, item_id=it.item_id, variant_id=vid)
    return {"item_id": it.item_id, "variant_id": vid, "kind": "paraphrase", "criterion_id": "",
            "text": parsed.paraphrased_answer if parsed else "", "what_was_broken": "", "edit_size": "",
            "status": meta["status"]}


def _group_judge(recs: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    g: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for x in recs:
        g[(x["item_id"], x["variant_id"], x["model"])].append(x)
    return g


# ---------------------------------------------------------------- scoring

def load_human_labels(path: str | Path) -> dict[tuple[str, str], bool]:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for row in d.to_dict(orient="records"):
        v = str(row.get("human_valid", "")).strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            out[(row["item_id"], row["variant_id"])] = True
        elif v in ("false", "0", "no", "n", "f"):
            out[(row["item_id"], row["variant_id"])] = False
    return out


def _validator_ok(v: Optional[dict[str, Any]], kind: str) -> Optional[bool]:
    if not is_ok(v):
        return None
    p = v["parsed"]
    keys = ("target_violated", "others_preserved", "fluent") if kind == "mutation" else (
        "meaning_preserved", "all_criteria_still_met")
    return all(bool(p.get(k)) for k in keys)


def build_units(items: list[Item], raw: dict[str, list[dict[str, Any]]], cfg: Config,
                human: Optional[dict[tuple[str, str], bool]] = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                                                             list[dict[str, Any]]]:
    """Per-item units for scoring, plus rows for the defects and review CSVs."""
    human = human or {}
    require_human = cfg.judge_mutation.require_human_validation
    judges = list(cfg.models.judges)
    refs = {x["item_id"]: x for x in raw.get("m2_reference", [])}
    variants_by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for v in raw.get("m2_variants", []):
        variants_by_item[v["item_id"]].append(v)
    val = {(x["item_id"], x["variant_id"]): x for x in raw.get("m2_validation", [])}
    jg = _group_judge(raw.get("m2_judge", []))

    units, defect_rows, review_rows = [], [], []
    for it in [i for i in items if i.eligible("m2")]:
        ref = refs.get(it.item_id)
        if not ref or ref.get("status") != "ok" or not ref.get("reference", "").strip():
            continue
        ref_text = ref["reference"]
        ref_j = {j: judge_variant(it, ref_text, jg.get((it.item_id, REF, j), [])) for j in judges}
        defects = find_defects(it, ref_j)
        for cid, failing in defects.items():
            c = next(x for x in it.rubric if x.criterion_id == cid)
            defect_rows.append({"item_id": it.item_id, "criterion_id": cid, "criterion_text": c.text,
                                "layer": c.layer, "judges_failing": "|".join(failing),
                                "reference_generated": ref.get("generated", False), "reference": ref_text})
        variants = [{"variant_id": REF, "kind": "reference", "criterion_id": "", "edit_size": "", "valid": True,
                     "text": ref_text, "judged": ref_j}]
        n_generated = n_valid = 0
        for v in variants_by_item.get(it.item_id, []):
            if v.get("status") != "ok" or not v.get("text", "").strip():
                continue
            vrec = val.get((it.item_id, v["variant_id"]))
            if (vrec is None or budget_skipped(vrec)) and (it.item_id, v["variant_id"]) not in human:
                continue                # never validated (keys or budget ran out): not evaluated, not invalid
            n_generated += 1
            auto = _validator_ok(vrec, v["kind"])
            h = human.get((it.item_id, v["variant_id"]))
            valid = h if h is not None else (False if require_human else bool(auto))
            n_valid += valid
            crit = next((c for c in it.rubric if c.criterion_id == v["criterion_id"]), None)
            review_rows.append({
                "item_id": it.item_id, "variant_id": v["variant_id"], "kind": v["kind"],
                "criterion_id": v["criterion_id"], "criterion_text": crit.text if crit else "",
                "reference": ref_text, "variant_text": v["text"], "what_was_broken": v.get("what_was_broken", ""),
                "validator_verdict": "" if auto is None else ("valid" if auto else "invalid"),
                "validator_note": (vrec or {}).get("parsed", {}).get("note", "") if is_ok(vrec) else "",
                "human_valid": "" if h is None else str(h).lower()})
            variants.append({"variant_id": v["variant_id"], "kind": v["kind"], "criterion_id": v["criterion_id"],
                             "edit_size": v.get("edit_size", ""), "valid": valid, "text": v["text"],
                             "judged": {j: judge_variant(it, v["text"], jg.get((it.item_id, v["variant_id"], j), []))
                                        for j in judges}})
        crits = [{"criterion_id": c.criterion_id, "text": c.text, "weight": c.weight, "layer": c.layer,
                  "defect": c.criterion_id in defects} for c in it.rubric]
        units.append({"item_id": it.item_id, "format": it.format, "criteria": crits, "variants": variants,
                      "n_generated": n_generated, "n_valid": n_valid, "reference_generated": ref.get("generated")})
    return units, defect_rows, review_rows


def _expected(variant: dict[str, Any], cid: str) -> str:
    if variant["kind"] == "mutation" and variant["criterion_id"] == cid:
        return FAIL
    return PASS


def _verdicts_for(unit: dict[str, Any], variant: dict[str, Any], judge: str, judges: list[str]
                  ) -> Optional[dict[str, tuple[str, float]]]:
    """(verdict, credit) per criterion for a judge, or for the majority-vote ensemble."""
    if judge != ENSEMBLE:
        j = variant["judged"].get(judge)
        if j is None:
            return None
        return {cid: (j["verdict"][cid], j["credit"][cid]) for cid in j["verdict"]}
    avail = [variant["judged"][x] for x in judges if variant["judged"].get(x) is not None]
    if not avail:
        return None
    out = {}
    for c in unit["criteria"]:
        cid = c["criterion_id"]
        n_pass = sum(a["verdict"][cid] == PASS for a in avail)
        ok = n_pass > len(avail) / 2
        out[cid] = (PASS if ok else FAIL, 1.0 if ok else 0.0)
    return out


def judge_stats(units: list[dict[str, Any]], judge: str, judges: list[str]) -> dict[str, Any]:
    sens_c: dict[tuple[int, str, str], list[int]] = defaultdict(list)   # (item, cid) -> 1 if judge said fail
    spec_c: dict[tuple[int, str, str], list[int]] = defaultdict(list)   # (item, cid) -> 1 if judge said pass
    meta_c: dict[tuple[int, str, str], dict[str, Any]] = {}
    collateral, para_flips, score_err, k_judge, k_exp = [], [], [], [], []
    raw_pass = quote_rej = slots = missing = pairs = splits = calls = inj = 0
    by_edit: dict[str, list[int]] = defaultdict(list)
    for ui, u in enumerate(units):
        live = [c for c in u["criteria"] if not c["defect"]]
        if not live:
            continue
        wsum = sum(c["weight"] for c in live)
        ref_v = _verdicts_for(u, u["variants"][0], judge, judges)
        for c in live:
            meta_c[(ui, u["item_id"], c["criterion_id"])] = {"layer": c["layer"], "format": u["format"]}
        for v in u["variants"]:
            if not v["valid"]:
                continue
            vd = _verdicts_for(u, v, judge, judges)
            if vd is None:
                continue
            if judge != ENSEMBLE:
                jr = v["judged"][judge]
                calls += jr["n_runs"]
                inj += jr["injection"]
                for run in jr["runs"]:
                    for c in live:
                        x = run[c["criterion_id"]]
                        slots += 1
                        missing += x["missing"]
                        raw_pass += x["raw"] == PASS
                        quote_rej += x["quote_rejected"]
            other_failed = False
            for c in live:
                cid = c["criterion_id"]
                verdict, _ = vd[cid]
                exp = _expected(v, cid)
                pairs += 1
                splits += verdict == SPLIT
                said_pass = verdict == PASS
                k_judge.append(said_pass)
                k_exp.append(exp == PASS)
                if exp == FAIL:
                    sens_c[(ui, u["item_id"], cid)].append(0 if said_pass else 1)
                    by_edit[v.get("edit_size") or "unknown"].append(0 if said_pass else 1)
                else:
                    spec_c[(ui, u["item_id"], cid)].append(1 if said_pass else 0)
                    if v["kind"] == "mutation" and not said_pass:
                        other_failed = True
                if v["kind"] == "paraphrase" and ref_v is not None:
                    para_flips.append(int(verdict != ref_v[cid][0]))
            if v["kind"] == "mutation":
                collateral.append(int(other_failed))
            score = sum(c["weight"] * vd[c["criterion_id"]][1] for c in live) / wsum
            if v["kind"] == "mutation":
                wc = next((c["weight"] for c in live if c["criterion_id"] == v["criterion_id"]), 0.0)
                expected = 1 - wc / wsum
            else:
                expected = 1.0
            score_err.append(abs(score - expected))

    per_crit = []
    for key in sorted(set(sens_c) | set(spec_c)):
        s, p = sens_c.get(key, []), spec_c.get(key, [])
        sv = sum(s) / len(s) if s else None
        pv = sum(p) / len(p) if p else None
        ba = (sv + pv) / 2 if sv is not None and pv is not None else None
        per_crit.append({"_key": key, "item_id": key[1], "criterion_id": key[2], **meta_c.get(key, {}),
                         "sensitivity": sv, "specificity": pv, "balanced_accuracy": ba,
                         "n_mutations": len(s), "n_should_pass": len(p)})
    bas = [x["balanced_accuracy"] for x in per_crit if x["balanced_accuracy"] is not None]
    tot_s = [y for v in sens_c.values() for y in v]
    tot_p = [y for v in spec_c.values() for y in v]

    def group(field: str) -> dict[str, Any]:
        g: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for x in per_crit:
            g[str(x.get(field) or "unknown")].append(x)
        out = {}
        for k, xs in sorted(g.items()):
            s = [y for x in xs for y in sens_c.get(x["_key"], [])]
            p = [y for x in xs for y in spec_c.get(x["_key"], [])]
            sv, pv = (sum(s) / len(s) if s else None), (sum(p) / len(p) if p else None)
            out[k] = {"sensitivity": r(sv), "specificity": r(pv), "n_mutations": len(s), "n_should_pass": len(p),
                      "balanced_accuracy": r((sv + pv) / 2) if sv is not None and pv is not None else None}
        return out

    is_ens = judge == ENSEMBLE
    return {
        "per_criterion": per_crit,
        "jba": mean(bas),
        "balanced_accuracy": r(mean(bas)), "criteria_scored": len(bas),
        "sensitivity": prop(sum(tot_s), len(tot_s)), "specificity": prop(sum(tot_p), len(tot_p)),
        "collateral_damage_rate": prop(sum(collateral), len(collateral)),
        "paraphrase_flip_rate": prop(sum(para_flips), len(para_flips)),
        "score_error": r(mean(score_err)), "_score_error": mean(score_err),
        "kappa": r(cohen_kappa(k_judge, k_exp)) if k_judge else None,
        "quote_rejection_rate": None if is_ens else prop(quote_rej, raw_pass),
        "missing_verdict_rate": None if is_ens else prop(missing, slots),
        "run_split_rate": None if is_ens else prop(splits, pairs),
        "injection_false_positive_rate": None if is_ens else prop(inj, calls),
        "by_layer": group("layer"), "by_format": group("format"),
        "by_edit_size": {k: {"sensitivity": r(sum(v) / len(v)), "n": len(v)} for k, v in sorted(by_edit.items())},
    }


def score(items: list[Item], raw: dict[str, list[dict[str, Any]]], cfg: Config,
          human: Optional[dict[tuple[str, str], bool]] = None) -> MetricResult:
    judges = [j for j in cfg.models.judges]
    units, defect_rows, review_rows = build_units(items, raw, cfg, human)
    w, floor = cfg.normalised.m2_weights, cfg.normalised.floor

    def comps(st: dict[str, Any]) -> dict[str, Optional[float]]:
        return {"balanced_accuracy": N.higher_better(st["jba"]), "score_accuracy": N.error_score(st["_score_error"])}

    def comp_value_for(judge: str):
        def f(us: list[dict[str, Any]]) -> Optional[float]:
            return N.composite(comps(judge_stats(us, judge, judges)), w, {"balanced_accuracy"}, floor)[0]
        return f

    judge_blocks: dict[str, Any] = {}
    normalised_per_judge: dict[str, Any] = {}
    crit_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    stats_by_judge = {}
    for i, j in enumerate(judges + [ENSEMBLE]):
        st = judge_stats(units, j, judges)
        stats_by_judge[j] = st
        c = comps(st)
        v, used = N.composite(c, w, {"balanced_accuracy"}, floor)
        b = boot_est(units, comp_value_for(j), cfg.seed + 21 + i, n_boot=1000) if v is not None else \
            {"ci_low": None, "ci_high": None}
        jba_ci = boot_est(units, lambda us, jj=j: judge_stats(us, jj, judges)["jba"], cfg.seed + 51 + i)
        normalised_per_judge[j] = N.block(v, (b["ci_low"], b["ci_high"]), len(units), c, used)
        blk = {k: st[k] for k in ("sensitivity", "specificity", "collateral_damage_rate", "paraphrase_flip_rate",
                                  "score_error", "kappa", "quote_rejection_rate", "missing_verdict_rate",
                                  "run_split_rate", "injection_false_positive_rate", "criteria_scored",
                                  "by_layer", "by_format", "by_edit_size")}
        blk["balanced_accuracy"] = jba_ci
        blk["composite"] = r(v)
        judge_blocks[j] = blk
        jr = {"judge": j, "composite": r(v), "balanced_accuracy": st["balanced_accuracy"],
              "criteria_scored": st["criteria_scored"]}
        for k in ("sensitivity", "specificity", "collateral_damage_rate", "paraphrase_flip_rate",
                  "quote_rejection_rate", "missing_verdict_rate", "run_split_rate", "injection_false_positive_rate"):
            jr[k] = (st[k] or {}).get("value") if isinstance(st[k], dict) else st[k]
        jr["score_error"], jr["kappa"] = st["score_error"], st["kappa"]
        judge_rows.append(jr)

    ens_ba = {(x["item_id"], x["criterion_id"]): x["balanced_accuracy"]
              for x in stats_by_judge[ENSEMBLE]["per_criterion"]}
    for j, st in stats_by_judge.items():
        for x in st["per_criterion"]:
            crit_rows.append({"judge": j, "item_id": x["item_id"], "criterion_id": x["criterion_id"],
                              "layer": x.get("layer", ""), "sensitivity": r(x["sensitivity"]),
                              "specificity": r(x["specificity"]), "balanced_accuracy": r(x["balanced_accuracy"]),
                              "n_mutations": x["n_mutations"], "n_should_pass": x["n_should_pass"],
                              "criterion_score": r(ens_ba.get((x["item_id"], x["criterion_id"])))})

    # verdict rows and Krippendorff alpha across judges
    verdict_rows: list[dict[str, Any]] = []
    alpha_units: list[list[Optional[bool]]] = []
    for u in units:
        defect = {c["criterion_id"]: c["defect"] for c in u["criteria"]}
        for v in u["variants"]:
            for j in judges:
                jr = v["judged"].get(j)
                if jr is None:
                    continue
                for run_no, run in enumerate(jr["runs"]):
                    for cid, x in run.items():
                        verdict_rows.append({"judge": j, "run": run_no, "item_id": u["item_id"],
                                             "variant_id": v["variant_id"], "kind": v["kind"], "criterion_id": cid,
                                             "expected": _expected(v, cid), "verdict_raw": x["raw"],
                                             "verdict": x["verdict"], "quote": x["quote"], "quote_ok": x["quote_ok"],
                                             "missing": x["missing"], "variant_valid": v["valid"],
                                             "criterion_defect": defect[cid]})
            if not v["valid"]:
                continue
            for c in u["criteria"]:
                if c["defect"]:
                    continue
                vals = []
                for j in judges:
                    jr = v["judged"].get(j)
                    vals.append(None if jr is None else jr["verdict"][c["criterion_id"]] == PASS)
                alpha_units.append(vals)
    alpha = krippendorff_alpha_nominal(alpha_units) if len(judges) >= 2 else None

    n_gen = sum(u["n_generated"] for u in units)
    n_valid = sum(u["n_valid"] for u in units)
    summary = {
        "rubric_items": len(units),
        "items_without_reference": sum(1 for it in items if it.eligible("m2")) - len(units),
        "references_generated": sum(bool(u["reference_generated"]) for u in units),
        "criteria_total": sum(len(u["criteria"]) for u in units),
        "criteria_scored": stats_by_judge[ENSEMBLE]["criteria_scored"],
        "rubric_defects": len(defect_rows),
        "mutation_validity_rate": prop(n_valid, n_gen),
        "mutation_validity_by_item": {u["item_id"]: prop(u["n_valid"], u["n_generated"]) for u in units},
        "human_validation_used": bool(human) or cfg.judge_mutation.require_human_validation,
        "judges": judge_blocks,
        "krippendorff_alpha": r(alpha),
    }
    ens = normalised_per_judge[ENSEMBLE]
    m2 = dict(ens)
    m2["per_judge"] = {j: normalised_per_judge[j]["value"] for j in judges}
    sub: dict[str, Optional[float]] = {"m2.mutation_validity": N.higher_better(summary["mutation_validity_rate"]["value"]),
                                       "m2.judge_alpha": N.agreement(alpha)}
    for j in judges + [ENSEMBLE]:
        st = stats_by_judge[j]
        sub[f"m2.{j}.balanced_accuracy"] = N.higher_better(st["jba"])
        sub[f"m2.{j}.sensitivity"] = N.higher_better(st["sensitivity"]["value"])
        sub[f"m2.{j}.specificity"] = N.higher_better(st["specificity"]["value"])
        sub[f"m2.{j}.no_collateral_damage"] = N.lower_better(st["collateral_damage_rate"]["value"])
        sub[f"m2.{j}.no_paraphrase_flip"] = N.lower_better(st["paraphrase_flip_rate"]["value"])
        sub[f"m2.{j}.score_accuracy"] = N.error_score(st["_score_error"])
        sub[f"m2.{j}.kappa"] = N.agreement(st["kappa"])
        if j != ENSEMBLE:
            for k, name in (("quote_rejection_rate", "no_quote_rejection"), ("run_split_rate", "no_run_split"),
                            ("missing_verdict_rate", "no_missing_verdict"),
                            ("injection_false_positive_rate", "no_injection_false_positive")):
                sub[f"m2.{j}.{name}"] = N.lower_better(st[k]["value"])

    th = cfg.thresholds
    flags = [flag("consistency_m2.ensemble.balanced_accuracy", stats_by_judge[ENSEMBLE]["balanced_accuracy"],
                  th.judge_balanced_accuracy),
             flag("consistency_m2.ensemble.sensitivity", stats_by_judge[ENSEMBLE]["sensitivity"]["value"],
                  th.judge_sensitivity)]
    for j in judges:
        flags.append(flag(f"consistency_m2.{j}.balanced_accuracy", stats_by_judge[j]["balanced_accuracy"],
                          th.judge_balanced_accuracy))

    worst = sorted([x for x in crit_rows if x["judge"] == ENSEMBLE and x["criterion_score"] is not None],
                   key=lambda x: (x["criterion_score"], x["item_id"], x["criterion_id"]))[:10]
    review_cols = ["item_id", "variant_id", "kind", "criterion_id", "criterion_text", "reference", "variant_text",
                   "what_was_broken", "validator_verdict", "human_valid"]
    return MetricResult(
        summary=summary,
        tables={"m2_rubric_defects": df(defect_rows, ["item_id", "criterion_id", "criterion_text", "layer",
                                                      "judges_failing", "reference_generated", "reference"]),
                "m2_mutations_for_review": df(review_rows, review_cols),
                "m2_verdicts": df(verdict_rows, ["judge", "run", "item_id", "variant_id", "kind", "criterion_id",
                                                 "expected", "verdict_raw", "verdict", "quote", "quote_ok"]),
                "m2_criterion_scores": df(crit_rows, ["judge", "item_id", "criterion_id", "layer", "sensitivity",
                                                      "specificity", "balanced_accuracy", "n_mutations",
                                                      "n_should_pass", "criterion_score"]),
                "m2_judge_scores": df(judge_rows, ["judge", "composite", "balanced_accuracy", "sensitivity",
                                                   "specificity"])},
        normalised={"M2": m2}, sub_scores=sub, flags=flags,
        worst=[{"item_id": x["item_id"], "criterion_id": x["criterion_id"], "criterion_score": x["criterion_score"]}
               for x in worst],
    )
