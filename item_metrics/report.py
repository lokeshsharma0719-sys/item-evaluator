"""Outputs: summary.json, report.md and per-metric CSVs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .config import Config
from .metrics.common import MetricResult
from .normalise import bar

SECTION = {"m1": "accuracy_m1", "m2": "consistency_m2", "m3": "faithfulness_m3"}


def build_summary(results: dict[str, MetricResult], cfg: Config) -> dict[str, Any]:
    s: dict[str, Any] = {SECTION[m]: (results[m].summary if m in results else None) for m in ("m1", "m2", "m3")}
    norm: dict[str, Any] = {}
    sub: dict[str, Any] = {}
    flags: list[dict[str, Any]] = []
    for m in ("m1", "m2", "m3"):
        if m in results:
            norm.update(results[m].normalised)
            sub.update(results[m].sub_scores)
            flags.extend(results[m].flags)
    norm["sub_scores"] = {k: (None if v is None else round(v, 4)) for k, v in sorted(sub.items())}
    s["normalised"] = norm
    s["flags"] = flags
    return s


def write_tables(results: dict[str, MetricResult], out: Path) -> None:
    for res in results.values():
        for name, table in res.tables.items():
            table.to_csv(out / f"{name}.csv", index=False)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False, default=str), encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def _ci(d: Optional[dict[str, Any]]) -> tuple[str, str, str]:
    if not d:
        return "n/a", "", ""
    ci = f"[{_fmt(d.get('ci_low'))}, {_fmt(d.get('ci_high'))}]" if d.get("ci_low") is not None else ""
    n = str(d.get("n", "")) + (" (small n)" if d.get("small_n") else "")
    return _fmt(d.get("value")), ci, n


def _flag_status(flags: list[dict[str, Any]], metric: str) -> tuple[str, str]:
    for f in flags:
        if f["metric"] == metric:
            return _fmt(f["threshold"]), f["status"]
    return "", ""


def render_report(summary: dict[str, Any], results: dict[str, MetricResult], cfg: Config,
                  manifest: dict[str, Any]) -> str:
    L: list[str] = ["# Item quality report", ""]
    if manifest.get("budget_stopped"):
        L += ["> **Budget stop:** the run reached `max_cost_usd` and results are partial.", ""]
    nt = (manifest.get("llm") or {}).get("models_without_temperature") or []
    if nt:
        L += [f"> **Not at temperature 0:** {', '.join('`' + m + '`' for m in nt)} reject the temperature setting, "
              "so their outputs may vary between uncached runs.", ""]
    if manifest.get("single_vendor"):
        L += ["> **Single vendor:** some roles use models from one vendor (`allow_single_vendor: true`). Their "
              "mistakes are more likely to coincide, so agreement-based numbers (key consensus, ensemble judge, "
              "majority topic) are less trustworthy than with mixed vendors.", ""]
    L += [f"Input: `{manifest.get('input_file')}` (sha256 `{str(manifest.get('input_sha256', ''))[:12]}`), "
          f"{manifest.get('rows', {}).get('valid', '?')} valid rows, "
          f"{manifest.get('rows', {}).get('invalid', '?')} invalid.", ""]

    norm = summary["normalised"]
    L += ["## Composite scores (0 to 1, higher is better)", "", "| Score | Value | 95% CI | n | Components |",
          "|---|---|---|---|---|"]
    for key, label in (("M1", "M1 Accuracy"), ("M2", "M2 Consistency (ensemble judge)"),
                       ("M3", "M3 Faithfulness"), ("M3_strict", "M3 strict (topic not named in stem)")):
        b = norm.get(key)
        if not b:
            continue
        comps = ", ".join(f"{k} {_fmt(v)}" for k, v in b.get("components", {}).items())
        v, ci, n = _ci(b)
        L.append(f"| {label} | {v} `{bar(b.get('value'))}` | {ci} | {n} | {comps} |")
    if norm.get("M2", {}).get("per_judge"):
        L += ["", "Per-judge M2 composites: " + ", ".join(f"`{j}` {_fmt(v)}"
                                                         for j, v in norm["M2"]["per_judge"].items())]
    L.append("")

    flags = summary["flags"]
    L += ["## Headline metrics", "", "| Quality | Metric | Value | 95% CI | n | Threshold | Status |",
          "|---|---|---|---|---|---|---|"]
    m1, m2, m3 = summary.get("accuracy_m1"), summary.get("consistency_m2"), summary.get("faithfulness_m3")
    if m1:
        for label, d, fk in (("Key Consensus Rate", m1["key_consensus_rate"], "accuracy_m1.key_consensus_rate"),
                             ("Planted-Error Recall", m1["planted_error_recall"]["overall"],
                              "accuracy_m1.planted_error_recall"),
                             ("Ambiguity Rate", m1["ambiguity_rate"], "accuracy_m1.ambiguity_rate")):
            th, st = _flag_status(flags, fk)
            L.append(f"| Accuracy | {label} | {' | '.join(_ci(d))} | {th} | {st} |")
    if m2:
        ens = m2["judges"]["ensemble"]
        th, st = _flag_status(flags, "consistency_m2.ensemble.balanced_accuracy")
        L.append(f"| Consistency | Judge Balanced Accuracy (ensemble) | {' | '.join(_ci(ens['balanced_accuracy']))} "
                 f"| {th} | {st} |")
    if m3:
        th, st = _flag_status(flags, "faithfulness_m3.blind_topic_accuracy")
        L.append(f"| Faithfulness | Blind Topic Accuracy | {' | '.join(_ci(m3['blind_topic_accuracy']))} | {th} | {st} |")
        L.append(f"| Faithfulness | Chance-adjusted accuracy | {' | '.join(_ci(m3['chance_adjusted_accuracy']))} | | |")
    L.append("")

    if m1:
        L += _m1_section(m1, results["m1"])
    if m2:
        L += _m2_section(m2, results["m2"])
    if m3:
        L += _m3_section(m3, results["m3"])
    return "\n".join(L) + "\n"


def _worst_table(rows: list[dict[str, Any]], csv_name: str) -> list[str]:
    if not rows:
        return ["(no items)", ""]
    cols = list(rows[0].keys())
    out = [f"Worst items (full rows in [{csv_name}]({csv_name})):", "", "| " + " | ".join(cols) + " |",
           "|" + "---|" * len(cols)]
    for r_ in rows:
        out.append("| " + " | ".join(_fmt(r_[c]).replace("|", "/") for c in cols) + " |")
    return out + [""]


def _m1_section(m1: dict[str, Any], res: MetricResult) -> list[str]:
    L = ["## M1 Accuracy: multi-solver key consensus", "",
         "Each keyed item is solved blind by every solver (no key shown). The Key Consensus Rate is the share of "
         "items where every solver's answer matches the key. Planted-Error Recall plants wrong keys into trusted "
         "items and counts how many the solver check would catch.", "",
         f"Eligible items: {m1['eligible_items']} (incomplete: {m1['incomplete_items']}); gate rule "
         f"`{m1['gate_rule']}`; acceptance rate {_fmt(m1['acceptance_rate']['value'])}.", "",
         "Status counts: " + ", ".join(f"{k} {v}" for k, v in m1["status_counts"].items()), ""]
    per = m1["planted_error_recall"]
    L += ["Planted-Error Recall by format: " + ", ".join(f"{k} {_fmt(v['value'])} (n={v['n']})"
                                                        for k, v in per["by_format"].items()),
          "", "By perturbation: " + ", ".join(f"{k} {_fmt(v['value'])} (n={v['n']})" for k, v in per["by_delta"].items()),
          "", "Single-solver recall: " + ", ".join(f"`{k}` {_fmt(v['value'])}" for k, v in per["by_solver"].items()),
          "", "Pairwise solver kappa: " + ", ".join(f"`{k}` {_fmt(v)}" for k, v in m1["solver_pairwise_kappa"].items()),
          ""]
    if m1.get("verifier"):
        v = m1["verifier"]
        L += [f"Key-visible verifier: recall {_fmt(v['recall']['value'])}, false-reject rate "
              f"{_fmt(v['false_reject_rate']['value'])}, balanced accuracy {_fmt(v['balanced_accuracy'])}.", ""]
    if m1.get("ground_truth"):
        g = m1["ground_truth"]
        L += [f"Against human key labels (n={g['n']}): precision {_fmt(g['precision']['value'])}, recall "
              f"{_fmt(g['recall']['value'])}, F1 {_fmt(g['f1'])}. Confusion: "
              + ", ".join(f"{k} {v}" for k, v in g["confusion"].items()), ""]
    L += _worst_table(res.worst, "m1_key_consensus.csv")
    L += ["Keys to review are in [m1_keys_to_review.csv](m1_keys_to_review.csv).", "",
          "**Limitations.** Solver agreement is not proof of correctness: several models can share a mistake. "
          "Planted-Error Recall is computed only on items the solvers already agree on (the easy ones), so it is an "
          "upper bound. Blind solving cannot measure sycophancy; the optional verifier does.", ""]
    return L


def _m2_section(m2: dict[str, Any], res: MetricResult) -> list[str]:
    L = ["## M2 Consistency: judge accuracy on an answer-mutation set", "",
         "Each rubric item gets a reference answer, one minimal mutation per criterion (that criterion should fail) "
         "and paraphrases (everything should pass). Sensitivity is how often a judge fails the broken criterion; "
         "specificity is how often it passes criteria that should pass. Judge Balanced Accuracy is the macro "
         "average over criteria.", "",
         f"Rubric items: {m2['rubric_items']}, criteria scored: {m2['criteria_scored']}, rubric defects: "
         f"{m2['rubric_defects']} (see [m2_rubric_defects.csv](m2_rubric_defects.csv)). Mutation validity rate "
         f"{_fmt(m2['mutation_validity_rate']['value'])} (n={m2['mutation_validity_rate']['n']}). Krippendorff alpha "
         f"across judges {_fmt(m2['krippendorff_alpha'])}.", "",
         "| Judge | Composite | JBA | Sensitivity | Specificity | Collateral | Paraphrase flip | Score error | kappa "
         "| Quote rejected | Run split |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for j, b in m2["judges"].items():
        def v(k: str) -> str:
            x = b.get(k)
            return _fmt(x.get("value") if isinstance(x, dict) else x)
        L.append(f"| `{j}` | {_fmt(b['composite'])} | {v('balanced_accuracy')} | {v('sensitivity')} | "
                 f"{v('specificity')} | {v('collateral_damage_rate')} | {v('paraphrase_flip_rate')} | "
                 f"{v('score_error')} | {v('kappa')} | {v('quote_rejection_rate')} | {v('run_split_rate')} |")
    ens = m2["judges"]["ensemble"]
    if ens.get("by_layer"):
        L += ["", "Ensemble by criterion layer: " + ", ".join(
            f"{k} BA {_fmt(x['balanced_accuracy'])}" for k, x in ens["by_layer"].items())]
    if ens.get("by_edit_size"):
        L += ["", "Ensemble sensitivity by edit size: " + ", ".join(
            f"{k} {_fmt(x['sensitivity'])} (n={x['n']})" for k, x in ens["by_edit_size"].items())]
    L.append("")
    L += _worst_table(res.worst, "m2_criterion_scores.csv")
    L += ["Mutations for human spot-checks: [m2_mutations_for_review.csv](m2_mutations_for_review.csv) (fill "
          "`human_valid`, then `rescore --human-labels`).", "",
          "**Limitations.** Mutations are written by an LLM and validated by another LLM until a person reviews them; "
          "an invalid mutation that slips through counts against the judge.", ""]
    return L


def _m3_section(m3: dict[str, Any], res: MetricResult) -> list[str]:
    L = ["## M3 Faithfulness: blind topic classification", "",
         "Classifiers see the item content and the course's topic list (shuffled, plus NONE) but not the assigned "
         "topic. Blind Topic Accuracy is the share of items whose majority-vote top-1 is the assigned topic.", "",
         f"Items: {m3['items']}; excluding items that name their topic in the stem: "
         f"{_fmt(m3['blind_topic_accuracy_excluding_named']['value'])} "
         f"(n={m3['blind_topic_accuracy_excluding_named']['n']}). Top-{m3['top_k']} accuracy "
         f"{_fmt(m3['top_k_accuracy']['value'])}, MRR {_fmt(m3['mrr'])}, off-syllabus rate "
         f"{_fmt(m3['off_syllabus_rate']['value'])}, classifier kappa ({m3['classifier_kappa_kind']}) "
         f"{_fmt(m3['classifier_kappa'])}.", "",
         "Per classifier top-1: " + ", ".join(f"`{k}` {_fmt(v['value'])}" for k, v in m3["per_model_top1"].items()), "",
         "By course: " + ", ".join(f"{k} {_fmt(v['value'])} (n={v['n']})" for k, v in m3["by_course"].items()), ""]
    if m3["low_topic_count_courses"]:
        L += [f"Courses with few candidate topics (accuracy inflated): {', '.join(m3['low_topic_count_courses'])}", ""]
    if m3.get("embedding"):
        e = m3["embedding"]
        L += [f"Embedding rank (`{e['model']}`): assigned topic ranked first for "
              f"{_fmt(e['rank1_share']['value'])} of items; mean rank {_fmt(e['mean_rank'])}.", ""]
    L += _worst_table(res.worst, "m3_topic_classification.csv")
    L += ["Confusion pairs (>= 2 occurrences) are in [m3_confusions.csv](m3_confusions.csv); they usually show "
          "overlapping syllabus topics rather than bad items.", "",
          "**Limitations.** Accuracy depends on how many candidate topics a course has and how much they overlap. "
          "Items that name their topic in the stem are trivially classified; see the strict figures.", ""]
    return L
