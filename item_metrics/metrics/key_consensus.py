"""M1 Accuracy: pre-gate multi-solver key consensus, planted-error audit, option-level ambiguity (spec section 4)."""
from __future__ import annotations

import itertools
import logging
from collections import Counter, defaultdict
from typing import Any, Optional

from .. import normalise as N
from ..answers import ItemMatch, match_item, same_answer, shuffle_options, stable_rng
from ..config import Config
from ..llm import LLMClient, gather_progress
from ..prompts import SOLVER, VERIFIER, build_solver, build_verifier
from ..schemas import Item, NumericKey, SolverOut, VerifierOut
from ..stats import boot_est, cohen_kappa, prop, r
from .common import MetricResult, budget_skipped, df, flag, is_ok, record

log = logging.getLogger(__name__)

SEVERITY = {"solvers_agree_on_other": 5, "all_disagree": 4, "split": 3, "majority_agree": 2, "ambiguous": 1}


# ---------------------------------------------------------------- planted keys

def planted_keys(item: Item, deltas: list[float]) -> list[dict[str, Any]]:
    """Wrong keys to plant into a trusted item. Each: kind, delta, key, stage_keys."""
    out: list[dict[str, Any]] = []
    stage_keys = {s.stage_id: s.answer_key for s in item.stages}
    if item.format == "mcq_4":
        for o in item.options:
            if o.id != item.answer_key:
                out.append({"planted_kind": f"distractor:{o.id}", "delta": None, "key": o.id, "stage_keys": None})
        return out
    k = item.answer_key
    if not isinstance(k, NumericKey):
        return out
    if k.value == 0:
        log.info("item %s: key is 0, relative perturbations are not defined; no planted keys", item.item_id)
    for d in deltas:
        signs = (+1, -1) if item.format == "numeric" else (+1,)
        for sgn in signs:
            if k.value == 0:
                break
            pk = k.model_copy(update={"value": k.value * (1 + sgn * d)})
            out.append({"planted_kind": f"final{'+' if sgn > 0 else '-'}", "delta": d, "key": pk,
                        "stage_keys": stage_keys})
        if item.format == "staged":
            for st in item.stages:
                sk = st.answer_key
                if isinstance(sk, NumericKey) and sk.value != 0:
                    ps = dict(stage_keys)
                    ps[st.stage_id] = sk.model_copy(update={"value": sk.value * (1 + d)})
                    out.append({"planted_kind": f"stage:{st.stage_id}+", "delta": d, "key": k, "stage_keys": ps})
    return out


# ---------------------------------------------------------------- collect

async def collect(items: list[Item], cfg: Config, llm: LLMClient) -> dict[str, list[dict[str, Any]]]:
    elig = [it for it in items if it.eligible("m1")]

    async def solve(it: Item, model: str) -> dict[str, Any]:
        shown, label_map = shuffle_options(it.options, cfg.seed, it.item_id, model,
                                           cfg.key_consensus.shuffle_options)
        system, user = build_solver(it, shown)
        parsed, meta = await llm.call_json("solver", model, system, user, SolverOut, prompt_version=SOLVER.version,
                                           item_id=it.item_id)
        return record(it.item_id, model, meta, parsed, label_map=label_map)

    solver_recs = await gather_progress((solve(it, m) for it in elig for m in cfg.models.solvers), "M1 solvers")
    raw: dict[str, list[dict[str, Any]]] = {"m1_solver": solver_recs, "m1_verifier": []}
    if not cfg.models.verifier:
        return raw

    per_item = _per_item(elig, raw, cfg)
    trusted = [it for it in elig if per_item.get(it.item_id, {}).get("trusted")]

    async def verify(it: Item, model: str, kind: str, pk: dict[str, Any]) -> dict[str, Any]:
        system, user = build_verifier(it, pk["key"], _stage_keys_for_prompt(it, pk.get("stage_keys")))
        parsed, meta = await llm.call_json("verifier", model, system, user, VerifierOut,
                                           prompt_version=VERIFIER.version, item_id=it.item_id, variant_id=kind)
        return record(it.item_id, model, meta, parsed, key_kind=kind, planted_kind=pk.get("planted_kind", ""))

    jobs = []
    for it in trusted:
        planted = planted_keys(it, cfg.numeric.planted_perturbations)
        if not planted:
            continue
        rng = stable_rng(cfg.seed, it.item_id, "verifier")
        pick = planted[rng.randrange(len(planted))]
        original = {"key": it.answer_key, "stage_keys": {s.stage_id: s.answer_key for s in it.stages},
                    "planted_kind": "original"}
        for model in cfg.models.verifier:
            pair = [("original", original), ("planted", pick)]
            rng.shuffle(pair)
            jobs.extend(verify(it, model, kind, pk) for kind, pk in pair)
    raw["m1_verifier"] = await gather_progress(jobs, "M1 verifier")
    return raw


def _stage_keys_for_prompt(item: Item, stage_keys: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if item.format != "staged" or not stage_keys:
        return None
    return {k: v for k, v in stage_keys.items() if v is not None}


# ---------------------------------------------------------------- scoring

def _solver_out(rec: Optional[dict[str, Any]]) -> Optional[SolverOut]:
    return SolverOut.model_validate(rec["parsed"]) if is_ok(rec) else None


def _matches(item: Item, outs: dict[str, tuple[Optional[SolverOut], dict[str, str]]], key: Any,
             stage_keys: Optional[dict[str, Any]], cfg: Config) -> dict[str, ItemMatch]:
    t, cap = cfg.numeric.default_tolerance, cfg.numeric.tolerance_cap
    return {m: match_item(item, o, lm, key, stage_keys, t, cap) for m, (o, lm) in outs.items()}


def _accepted(m: int, n: int, rule: str) -> bool:
    return m == n if rule == "all" else m > n / 2


def _status(item: Item, matches: dict[str, ItemMatch], cfg: Config) -> tuple[str, Optional[str]]:
    n, m = len(matches), sum(x.match for x in matches.values())
    others = [x.canonical for x in matches.values() if not x.match and x.canonical is not None]
    t, cap = cfg.numeric.default_tolerance, cfg.numeric.tolerance_cap
    best, best_n = None, 0
    for a in others:
        c = sum(1 for b in others if same_answer(item, a, b, item.answer_key, t, cap))
        if c > best_n:
            best, best_n = a, c
    most_common_other = None if best is None else (best if isinstance(best, str) else r(best, 6))
    if m == n:
        return "full_agree", most_common_other
    if m == 0:
        return ("solvers_agree_on_other" if best_n >= 2 else "all_disagree"), most_common_other
    if m > n / 2:
        return "majority_agree", most_common_other
    return "split", most_common_other


def _ambiguity(item: Item, outs: dict[str, tuple[Optional[SolverOut], dict[str, str]]]) -> list[str]:
    fired: list[str] = []
    per_option: dict[str, set[bool]] = defaultdict(set)
    rule1 = False
    for o, lm in outs.values():
        if o is None:
            continue
        if item.format == "mcq_4" and o.option_verdicts:
            n_correct = 0
            for v in o.option_verdicts:
                orig = lm.get(v.id.strip().upper()[:1])
                if orig is None:
                    continue
                per_option[orig].add(bool(v.correct))
                n_correct += bool(v.correct)
            if n_correct != 1:
                rule1 = True
    if rule1:
        fired.append("1_not_one_correct")
    if any(len(s) > 1 for s in per_option.values()):
        fired.append("2_option_disagreement")
    if any(o is not None and o.ambiguous for o, _ in outs.values()):
        fired.append("3_flagged_ambiguous")
    return fired


def _per_item(elig: list[Item], raw: dict[str, list[dict[str, Any]]], cfg: Config,
              key_labels: Optional[dict[str, bool]] = None) -> dict[str, dict[str, Any]]:
    solvers = list(cfg.models.solvers)
    by = {(x["item_id"], x["model"]): x for x in raw.get("m1_solver", [])}
    solvers = [s for s in solvers if any(k[1] == s for k in by)] or solvers
    out: dict[str, dict[str, Any]] = {}
    for it in elig:
        recs = {s: by.get((it.item_id, s)) for s in solvers}
        if any(v is None or budget_skipped(v) for v in recs.values()):
            out[it.item_id] = {"incomplete": True}
            continue
        outs = {s: (_solver_out(rec), rec.get("label_map") or {}) for s, rec in recs.items()}
        matches = _matches(it, outs, it.answer_key, None, cfg)
        status, other = _status(it, matches, cfg)
        n, m = len(matches), sum(x.match for x in matches.values())
        amb = _ambiguity(it, outs)
        human = key_labels.get(it.item_id, it.human_key_correct) if key_labels else it.human_key_correct
        trusted = status == "full_agree" and human is not False
        out[it.item_id] = {"incomplete": False, "outs": outs, "matches": matches, "status": status,
                           "most_common_other": other, "n": n, "m": m, "ambiguity": amb, "human": human,
                           "accepted": _accepted(m, n, cfg.key_consensus.gate_rule), "trusted": trusted}
    return out


def score(items: list[Item], raw: dict[str, list[dict[str, Any]]], cfg: Config,
          key_labels: Optional[dict[str, bool]] = None) -> MetricResult:
    elig = [it for it in items if it.eligible("m1")]
    per = _per_item(elig, raw, cfg, key_labels)
    complete = [it for it in elig if not per[it.item_id]["incomplete"]]
    solvers = [s for s in cfg.models.solvers if any(x["model"] == s for x in raw.get("m1_solver", []))] \
        or list(cfg.models.solvers)
    rule = cfg.key_consensus.gate_rule

    # ---- per-item table, planted audit
    rows, review, planted_rows, units = [], [], [], []
    for it in complete:
        p = per[it.item_id]
        amb = bool(p["ambiguity"])
        row: dict[str, Any] = {"item_id": it.item_id, "format": it.format, "n_solvers": p["n"], "n_match": p["m"],
                               "status": p["status"], "accepted": p["accepted"], "ambiguous": amb,
                               "ambiguity_rule": "|".join(p["ambiguity"])}
        for s in solvers:
            row[f"answer__{s}"] = p["matches"][s].display
            row[f"match_reason__{s}"] = p["matches"][s].reason
        row["most_common_other_answer"] = p["most_common_other"]
        row["item_score"] = round((p["m"] / p["n"]) * (0.5 if amb else 1.0), 4) if p["n"] else 0.0
        rows.append(row)
        if p["status"] != "full_agree" or amb:
            sev_key = p["status"] if p["status"] != "full_agree" else "ambiguous"
            rv = {"severity": SEVERITY[sev_key], "item_id": it.item_id, "format": it.format, "status": p["status"],
                  "ambiguity_rule": row["ambiguity_rule"], "key": _key_str(it.answer_key),
                  "most_common_other_answer": p["most_common_other"]}
            for s in solvers:
                o = p["outs"][s][0]
                rv[f"answer__{s}"] = p["matches"][s].display
                rv[f"working__{s}"] = o.working if o else ""
                rv[f"ambiguity_reason__{s}"] = o.ambiguity_reason if o else ""
            review.append(rv)

        unit = {"full": p["status"] == "full_agree", "accepted": p["accepted"], "amb": amb, "planted": 0,
                "caught": 0, "ver": Counter()}
        if p["trusted"]:
            for pk in planted_keys(it, cfg.numeric.planted_perturbations):
                pm = _matches(it, p["outs"], pk["key"], pk["stage_keys"], cfg)
                mm = sum(x.match for x in pm.values())
                caught = not _accepted(mm, len(pm), rule)
                caught_by = [s for s in solvers if not pm[s].match]
                planted_rows.append({"item_id": it.item_id, "format": it.format, "planted_kind": pk["planted_kind"],
                                     "delta": pk["delta"], "caught": caught, "caught_by": "|".join(caught_by),
                                     **{f"_by__{s}": not pm[s].match for s in solvers}})
                unit["planted"] += 1
                unit["caught"] += caught
        units.append(unit)
    review.sort(key=lambda x: (-x["severity"], x["item_id"]))

    # ---- verifier
    verifier = None
    if cfg.models.verifier and raw.get("m1_verifier"):
        vstats: dict[str, Counter] = defaultdict(Counter)
        unit_by_id = {it.item_id: u for it, u in zip(complete, units)}
        for v in raw["m1_verifier"]:
            if not is_ok(v):
                continue
            kind = v["key_kind"]
            rejected = not bool(v["parsed"].get("key_correct"))
            c = vstats[v["model"]]
            c[f"{kind}_n"] += 1
            c[f"{kind}_rejected"] += rejected
            if v["item_id"] in unit_by_id:
                unit_by_id[v["item_id"]]["ver"][f"{kind}_n"] += 1
                unit_by_id[v["item_id"]]["ver"][f"{kind}_rejected"] += rejected
        tot = sum(vstats.values(), Counter())
        verifier = {"recall": prop(tot["planted_rejected"], tot["planted_n"]),
                    "false_reject_rate": prop(tot["original_rejected"], tot["original_n"]),
                    "per_model": {m: {"recall": prop(c["planted_rejected"], c["planted_n"]),
                                      "false_reject_rate": prop(c["original_rejected"], c["original_n"])}
                                  for m, c in sorted(vstats.items())}}
        rc, fr = verifier["recall"]["value"], verifier["false_reject_rate"]["value"]
        verifier["balanced_accuracy"] = r((rc + (1 - fr)) / 2) if rc is not None and fr is not None else None

    # ---- ground truth
    ground = None
    labelled = [(it, per[it.item_id]) for it in complete if per[it.item_id]["human"] is not None]
    if labelled:
        tp = sum(1 for _, p in labelled if not p["accepted"] and p["human"] is False)
        fp = sum(1 for _, p in labelled if not p["accepted"] and p["human"] is True)
        fn = sum(1 for _, p in labelled if p["accepted"] and p["human"] is False)
        tn = sum(1 for _, p in labelled if p["accepted"] and p["human"] is True)
        pr, rc = prop(tp, tp + fp), prop(tp, tp + fn)
        f1 = (2 * pr["value"] * rc["value"] / (pr["value"] + rc["value"])
              if pr["value"] and rc["value"] else (0.0 if pr["value"] is not None and rc["value"] is not None else None))
        ground = {"n": len(labelled), "precision": pr, "recall": rc, "f1": r(f1),
                  "confusion": {"flagged_and_wrong_key": tp, "flagged_but_key_correct": fp,
                                "accepted_but_key_wrong": fn, "accepted_and_key_correct": tn}}

    # ---- summary numbers
    n = len(complete)
    counts = Counter(x["status"] for x in rows)
    status_counts = {s: counts.get(s, 0) for s in
                     ("full_agree", "majority_agree", "split", "all_disagree", "solvers_agree_on_other")}
    n_planted = len(planted_rows)

    def per_group(key: str) -> dict[str, Any]:
        g: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for x in planted_rows:
            g["distractor" if x[key] is None else str(x[key])].append(x)
        return {k: prop(sum(y["caught"] for y in v), len(v)) for k, v in sorted(g.items())}

    per_summary = {"overall": prop(sum(x["caught"] for x in planted_rows), n_planted),
                   "by_format": per_group("format"), "by_delta": per_group("delta"),
                   "by_solver": {s: prop(sum(x[f"_by__{s}"] for x in planted_rows), n_planted) for s in solvers},
                   "note": "upper bound: planted keys go only into items all solvers already agree on"}

    kappas = {}
    for a, b in itertools.combinations(solvers, 2):
        k = cohen_kappa([per[it.item_id]["matches"][a].match for it in complete],
                        [per[it.item_id]["matches"][b].match for it in complete])
        kappas[f"{a}|{b}"] = r(k)

    summary = {
        "eligible_items": n,
        "incomplete_items": len(elig) - n,
        "solvers": solvers, "gate_rule": rule,
        "key_consensus_rate": prop(status_counts["full_agree"], n),
        "status_counts": status_counts,
        "acceptance_rate": prop(sum(x["accepted"] for x in rows), n),
        "planted_error_recall": per_summary,
        "ambiguity_rate": prop(sum(x["ambiguous"] for x in rows), n),
        "ambiguity_rule_counts": dict(Counter(rl for x in rows for rl in x["ambiguity_rule"].split("|") if rl)),
        "solver_pairwise_kappa": kappas,
        "verifier": verifier,
        "ground_truth": ground,
    }

    # ---- normalised
    w = dict(cfg.normalised.m1_weights)
    verifier_on = verifier is not None
    if not verifier_on:
        w.pop("verifier", None)

    def comps(us: list[dict[str, Any]]) -> dict[str, Optional[float]]:
        if not us:
            return {"kcr": None, "per": None, "no_ambiguity": None}
        planted = sum(u["planted"] for u in us)
        c: dict[str, Optional[float]] = {
            "kcr": sum(u["full"] for u in us) / len(us),
            "per": (sum(u["caught"] for u in us) / planted) if planted else None,
            "no_ambiguity": 1 - sum(u["amb"] for u in us) / len(us),
        }
        if verifier_on:
            v = sum((u["ver"] for u in us), Counter())
            rec = v["planted_rejected"] / v["planted_n"] if v["planted_n"] else None
            frr = v["original_rejected"] / v["original_n"] if v["original_n"] else None
            c["verifier"] = (rec + (1 - frr)) / 2 if rec is not None and frr is not None else None
        return c

    floor = cfg.normalised.floor

    def comp_value(us: list[dict[str, Any]]) -> Optional[float]:
        return N.composite(comps(us), w, {"kcr"}, floor)[0]

    c_all = comps(units)
    value, used = N.composite(c_all, w, {"kcr"}, floor)
    b = boot_est(units, comp_value, cfg.seed + 11)
    m1 = N.block(value, (b["ci_low"], b["ci_high"]), n, c_all, used)

    sub = {
        "m1.kcr": N.higher_better(summary["key_consensus_rate"]["value"]),
        "m1.acceptance": N.higher_better(summary["acceptance_rate"]["value"]),
        "m1.per": N.higher_better(per_summary["overall"]["value"]),
        "m1.no_ambiguity": N.lower_better(summary["ambiguity_rate"]["value"]),
        **{f"m1.solver_kappa.{k}": N.agreement(v) for k, v in kappas.items()},
    }
    if verifier:
        sub["m1.verifier_recall"] = N.higher_better(verifier["recall"]["value"])
        sub["m1.verifier_no_false_reject"] = N.lower_better(verifier["false_reject_rate"]["value"])

    th = cfg.thresholds
    flags = [flag("accuracy_m1.key_consensus_rate", summary["key_consensus_rate"]["value"], th.key_consensus_rate),
             flag("accuracy_m1.planted_error_recall", per_summary["overall"]["value"], th.planted_error_recall),
             flag("accuracy_m1.ambiguity_rate", summary["ambiguity_rate"]["value"], th.ambiguity_rate_max, False)]

    table_cols = list(rows[0].keys()) if rows else ["item_id", "format", "n_solvers", "n_match", "status",
                                                     "accepted", "ambiguous", "ambiguity_rule",
                                                     "most_common_other_answer", "item_score"]
    planted_cols = ["item_id", "format", "planted_kind", "delta", "caught", "caught_by"]
    worst = sorted(rows, key=lambda x: (x["item_score"], x["item_id"]))[:10]
    return MetricResult(
        summary=summary,
        tables={"m1_key_consensus": df(rows, table_cols),
                "m1_planted_audit": df([{c: x[c] for c in planted_cols} for x in planted_rows], planted_cols),
                "m1_keys_to_review": df(review, ["severity", "item_id", "format", "status", "ambiguity_rule", "key",
                                                 "most_common_other_answer"])},
        normalised={"M1": m1}, sub_scores=sub, flags=flags,
        worst=[{"item_id": x["item_id"], "item_score": x["item_score"], "status": x["status"],
                "ambiguity_rule": x["ambiguity_rule"]} for x in worst],
    )


def _key_str(k: Any) -> str:
    if isinstance(k, NumericKey):
        return f"{k.value:g} {k.unit}".strip()
    return str(k)
