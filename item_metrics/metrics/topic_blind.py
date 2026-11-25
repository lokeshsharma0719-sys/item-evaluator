"""M3 Faithfulness: blind topic classification (spec section 6)."""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from typing import Any, Optional

from .. import normalise as N
from ..answers import stable_rng
from ..config import Config
from ..llm import LLMClient, gather_progress
from ..prompts import TOPIC_CLASSIFIER, build_topic_classifier
from ..schemas import Item, TopicOut
from ..stats import boot_est, cohen_kappa, fleiss_kappa, mean, prop, r
from ..text_match import jaccard, normalise, topic_named_in
from .common import MetricResult, budget_skipped, df, flag, is_ok, record

log = logging.getLogger(__name__)
NONE, UNMAPPED, MISSING = "NONE", "unmapped", None
JACCARD_MIN = 0.6


def candidate_topics(items: list[Item]) -> tuple[dict[str, list[str]], list[str]]:
    """Per course: union of syllabus_topics over its rows, else its distinct topics; the assigned topic of every
    item is always added. Returns (course -> topics, log of added topics)."""
    by_course: dict[str, list[Item]] = defaultdict(list)
    for it in items:
        by_course[it.course_code].append(it)
    out: dict[str, list[str]] = {}
    added: list[str] = []
    for course, its in by_course.items():
        topics: list[str] = []
        seen: set[str] = set()

        def add(t: str) -> bool:
            n = normalise(t)
            if n and n not in seen:
                seen.add(n)
                topics.append(t.strip())
                return True
            return False

        syl = [t for it in its for t in it.syllabus_topics]
        for t in (syl or [it.topic for it in its]):
            add(t)
        for it in its:
            if it.topic and add(it.topic) and syl:
                added.append(f"{course}: '{it.topic}' (item {it.item_id}) was not in syllabus_topics; added")
        out[course] = topics
    for a in added:
        log.info(a)
    return out, added


def map_topic(t: Any, candidates: list[str]) -> str:
    s = "" if t is None else str(t)
    n = normalise(s)
    if n == "none":
        return NONE
    for c in candidates:
        if normalise(c) == n:
            return c
    best, best_j = UNMAPPED, 0.0
    for c in candidates:
        j = jaccard(s, c)
        if j > best_j:
            best, best_j = c, j
    return best if best_j >= JACCARD_MIN else UNMAPPED


def _item_text(item: Item) -> str:
    parts = [item.stem] + [o.text for o in item.options] + [s.prompt for s in item.stages]
    return "\n".join(parts)


async def collect(items: list[Item], cfg: Config, llm: LLMClient) -> dict[str, list[dict[str, Any]]]:
    elig = [it for it in items if it.eligible("m3")]
    cands, _ = candidate_topics(elig)
    k = cfg.topic_blind.top_k

    async def one(it: Item, model: str) -> dict[str, Any]:
        shown = list(cands[it.course_code])
        stable_rng(cfg.seed, it.item_id, model, "topics").shuffle(shown)
        system, user = build_topic_classifier(it, shown, k)
        parsed, meta = await llm.call_json("topic_classifier", model, system, user, TopicOut,
                                           prompt_version=TOPIC_CLASSIFIER.version, item_id=it.item_id)
        return record(it.item_id, model, meta, parsed, candidates_shown=shown)

    recs = await gather_progress((one(it, m) for it in elig for m in cfg.models.topic_classifiers), "M3 topics")
    raw: dict[str, list[dict[str, Any]]] = {"m3_topic": recs}

    if cfg.models.embedding:
        emb = []
        for it in elig:
            cl = cands[it.course_code]
            vecs = await llm.embed(cfg.models.embedding, [_item_text(it)] + cl, item_id=it.item_id)
            if vecs is None:
                log.warning("embedding rank skipped (provider returned nothing)")
                emb = []
                break
            sims = [_cos(vecs[0], v) for v in vecs[1:]]
            order = sorted(range(len(cl)), key=lambda i: -sims[i])
            rank = next(i + 1 for i, j in enumerate(order) if normalise(cl[j]) == normalise(it.topic))
            emb.append({"item_id": it.item_id, "model": cfg.models.embedding, "rank": rank, "K": len(cl),
                        "similarity_assigned": round(sims[order[rank - 1]], 5)})
        raw["m3_embedding"] = emb
    return raw


def _cos(a: list[float], b: list[float]) -> float:
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def mrr_chance(K: int, k: int) -> float:
    return (1.0 / (K + 1)) * sum(1.0 / r_ for r_ in range(1, min(k, K + 1) + 1))


def _majority(labels: list[Optional[str]]) -> Optional[str]:
    c = Counter(x for x in labels if x is not None)
    if not c:
        return None
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def score(items: list[Item], raw: dict[str, list[dict[str, Any]]], cfg: Config) -> MetricResult:
    elig = [it for it in items if it.eligible("m3")]
    cands, added = candidate_topics(elig)
    models = list(cfg.models.topic_classifiers)
    k = cfg.topic_blind.top_k
    by: dict[tuple[str, str], dict[str, Any]] = {(x["item_id"], x["model"]): x for x in raw.get("m3_topic", [])}
    models = [m for m in models if any(key[1] == m for key in by)] or models

    rows: list[dict[str, Any]] = []
    incomplete = 0
    for it in elig:
        recs = {m: by.get((it.item_id, m)) for m in models}
        if any(v is None or budget_skipped(v) for v in recs.values()):
            incomplete += 1
            continue
        cl = cands[it.course_code]
        K = len(cl)
        top1: dict[str, Optional[str]] = {}
        ranks: dict[str, Optional[int]] = {}
        unmapped = 0
        for m, rec in recs.items():
            if not is_ok(rec):
                top1[m], ranks[m] = MISSING, None
                continue
            mapped = [map_topic(x.get("topic"), cl) for x in rec["parsed"].get("ranked", [])]
            unmapped += sum(x == UNMAPPED for x in mapped)
            top1[m] = mapped[0] if mapped else UNMAPPED
            ranks[m] = next((i + 1 for i, x in enumerate(mapped[:k]) if x == it.topic.strip()
                             or normalise(x) == normalise(it.topic)), None)
        maj = _majority(list(top1.values()))
        assigned = next(c for c in cl if normalise(c) == normalise(it.topic))
        correct = maj is not None and normalise(maj) == normalise(assigned)
        chance = 1.0 / (K + 1)
        q = sum(1 for m in models if top1[m] is not None and normalise(top1[m]) == normalise(assigned)) / len(models)
        rr_item = mean([1.0 / ranks[m] if ranks[m] else 0.0 for m in models]) or 0.0
        mc = mrr_chance(K, k)
        row = {"item_id": it.item_id, "course_code": it.course_code, "format": it.format,
               "bloom_level": it.bloom_level, "assigned_topic": assigned, "K": K}
        for m in models:
            row[f"top1__{m}"] = top1[m]
            row[f"rank_assigned__{m}"] = ranks[m]
        row.update({
            "majority": maj, "correct": correct,
            "topic_named_in_stem": topic_named_in(assigned, it.stem),
            "low_topic_count": K < cfg.topic_blind.min_topics_warning,
            "item_score": round(max(0.0, (q - chance) / (1 - chance)), 4),
            "_chance": chance, "_q": q, "_rr": rr_item, "_mrr_adj": max(0.0, (rr_item - mc) / (1 - mc)),
            "_topk": [ranks[m] is not None for m in models], "_rr_models": [1.0 / ranks[m] if ranks[m] else 0.0
                                                                            for m in models],
            "_top1": [top1[m] for m in models], "_unmapped": unmapped,
        })
        rows.append(row)

    def acc_adj(rs: list[dict[str, Any]]) -> Optional[float]:
        return mean([((1.0 if x["correct"] else 0.0) - x["_chance"]) / (1 - x["_chance"]) for x in rs])

    def mrr_adj(rs: list[dict[str, Any]]) -> Optional[float]:
        return mean([x["_mrr_adj"] for x in rs])

    def comps(rs: list[dict[str, Any]]) -> dict[str, Optional[float]]:
        return {"chance_adj_accuracy": N.chance_adjusted(acc_adj(rs)), "chance_adj_mrr": N.chance_adjusted(mrr_adj(rs))}

    w, floor = cfg.normalised.m3_weights, cfg.normalised.floor

    def comp_value(rs: list[dict[str, Any]]) -> Optional[float]:
        return N.composite(comps(rs), w, {"chance_adj_accuracy"}, floor)[0] if rs else None

    def composite_block(rs: list[dict[str, Any]], seed_off: int) -> dict[str, Any]:
        c = comps(rs) if rs else {}
        v, used = N.composite(c, w, {"chance_adj_accuracy"}, floor) if rs else (None, {})
        b = boot_est(rs, comp_value, cfg.seed + seed_off)
        return N.block(v, (b["ci_low"], b["ci_high"]), len(rs), c, used)

    n = len(rows)
    n_correct = sum(x["correct"] for x in rows)
    not_named = [x for x in rows if not x["topic_named_in_stem"]]
    topk_flat = [t for x in rows for t in x["_topk"]]
    rr_flat = [t for x in rows for t in x["_rr_models"]]
    per_model = {m: prop(sum(1 for x in rows if x[f"top1__{m}"] is not None
                             and normalise(x[f"top1__{m}"]) == normalise(x["assigned_topic"])), n) for m in models}
    full = [x["_top1"] for x in rows if all(t is not None for t in x["_top1"])]
    if len(models) == 2:
        kappa = cohen_kappa([t[0] for t in full], [t[1] for t in full])
    elif len(models) > 2:
        kappa = fleiss_kappa(full)
    else:
        kappa = None

    def by_group(key: str) -> dict[str, Any]:
        g: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for x in rows:
            g[str(x[key])].append(x)
        return {kk: prop(sum(y["correct"] for y in v), len(v)) for kk, v in sorted(g.items())}

    conf = Counter((x["assigned_topic"], x["majority"]) for x in rows if not x["correct"] and x["majority"])
    confusions = [{"assigned_topic": a, "predicted_topic": b, "count": c} for (a, b), c in conf.most_common() if c >= 2]
    off = [x for x in rows if x["majority"] == NONE]
    low_courses = sorted({x["course_code"] for x in rows if x["low_topic_count"]})

    summary: dict[str, Any] = {
        "items": n, "incomplete_items": incomplete, "classifiers": models,
        "blind_topic_accuracy": prop(n_correct, n),
        "blind_topic_accuracy_excluding_named": prop(sum(x["correct"] for x in not_named), len(not_named)),
        "chance_adjusted_accuracy": boot_est(rows, acc_adj, cfg.seed + 31),
        "chance_adjusted_mrr": boot_est(rows, mrr_adj, cfg.seed + 32),
        "top_k": k, "top_k_accuracy": prop(sum(topk_flat), len(topk_flat)),
        "off_syllabus_rate": prop(len(off), n),
        "mrr": r(mean(rr_flat)),
        "per_model_top1": per_model,
        "classifier_kappa": r(kappa), "classifier_kappa_kind": "cohen" if len(models) == 2 else "fleiss",
        "unmapped_topics": sum(x["_unmapped"] for x in rows),
        "topic_named_in_stem_items": n - len(not_named),
        "low_topic_count_courses": low_courses,
        "topics_added_to_candidates": added,
        "by_course": by_group("course_code"), "by_format": by_group("format"),
        "by_bloom_level": by_group("bloom_level"), "by_topic_named_in_stem": by_group("topic_named_in_stem"),
        "embedding": None,
    }
    emb = raw.get("m3_embedding") or []
    if emb:
        summary["embedding"] = {"model": emb[0]["model"], "rank1_share": prop(sum(e["rank"] == 1 for e in emb), len(emb)),
                                "mean_rank": r(mean([e["rank"] for e in emb]))}

    m3 = composite_block(rows, 41)
    m3_strict = composite_block(not_named, 42)
    sub = {
        "m3.top1_majority": N.higher_better(summary["blind_topic_accuracy"]["value"]),
        "m3.top1_majority_excluding_named": N.higher_better(summary["blind_topic_accuracy_excluding_named"]["value"]),
        "m3.top_k": N.higher_better(summary["top_k_accuracy"]["value"]),
        "m3.no_off_syllabus": N.lower_better(summary["off_syllabus_rate"]["value"]),
        "m3.chance_adj_accuracy": N.chance_adjusted(summary["chance_adjusted_accuracy"]["value"]),
        "m3.chance_adj_mrr": N.chance_adjusted(summary["chance_adjusted_mrr"]["value"]),
        "m3.classifier_kappa": N.agreement(kappa),
        **{f"m3.top1.{m}": N.higher_better(v["value"]) for m, v in per_model.items()},
    }
    table_cols = [c for c in (rows[0].keys() if rows else []) if not c.startswith("_")]
    table = df([{c: x[c] for c in table_cols} for x in rows], table_cols)
    worst = sorted(rows, key=lambda x: (x["item_score"], x["item_id"]))[:10]
    return MetricResult(
        summary=summary,
        tables={"m3_topic_classification": table,
                "m3_confusions": df(confusions, ["assigned_topic", "predicted_topic", "count"]),
                "m3_off_syllabus": df([{c: x[c] for c in table_cols} for x in off], table_cols)},
        normalised={"M3": m3, "M3_strict": m3_strict},
        sub_scores=sub,
        flags=[flag("faithfulness_m3.blind_topic_accuracy", summary["blind_topic_accuracy"]["value"],
                    cfg.thresholds.topic_accuracy)],
        worst=[{"item_id": x["item_id"], "item_score": x["item_score"], "assigned": x["assigned_topic"],
                "majority": x["majority"]} for x in worst],
    )
