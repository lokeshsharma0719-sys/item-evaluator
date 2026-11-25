"""Commands: validate, estimate, run, rescore."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shutil
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import Config, config_for_manifest, load_config, load_env, missing_keys, validate_config
from .io_csv import eligibility_table, file_sha256, load_items, write_invalid
from .llm import Backend, LLMClient
from .metrics import judge_mutation, key_consensus, topic_blind
from .metrics.common import MetricResult
from .prompts import build_solver, build_topic_classifier, prompt_hashes
from .answers import shuffle_options
from .report import build_summary, render_report, write_json, write_tables
from .schemas import Item

log = logging.getLogger(__name__)
EXIT_OK, EXIT_CONFIG, EXIT_BUDGET = 0, 1, 2
MODULES = {"m1": key_consensus, "m2": judge_mutation, "m3": topic_blind}
COLLECT_ORDER = ("m3", "m1", "m2")


def parse_metrics(s: str) -> set[str]:
    ms = {x.strip().lower() for x in s.split(",") if x.strip()}
    bad = ms - set(MODULES)
    if bad:
        raise SystemExit(f"unknown metrics: {', '.join(sorted(bad))} (use m1,m2,m3)")
    return ms


# ---------------------------------------------------------------- validate

def cmd_validate(input_path: str, config_path: Optional[str], out_dir: str) -> int:
    cfg = load_config(config_path)
    res = load_items(input_path, cfg)
    p = write_invalid(res.invalid, out_dir)
    print(f"{res.n_rows} rows read, {len(res.items)} valid, {len(res.invalid)} invalid (listed in {p})")
    if res.tolerance_capped:
        print(f"tolerance capped at {cfg.numeric.tolerance_cap} for: {', '.join(res.tolerance_capped)}")
    for x in res.invalid[:20]:
        print(f"  row {x['row_number']} ({x['item_id'] or '?'}): {x['reason']}")
    print()
    print("Rows by format and metric eligibility:")
    print(eligibility_table(res.items).to_string(index=False))
    return EXIT_OK


# ---------------------------------------------------------------- estimate

OUT_TOKENS = {"solver": 400, "verifier": 150, "reference_writer": 400, "mutator": 0, "mutation_validator": 60,
              "judge": 0, "topic_classifier": 120}


def _price(model: str, tin: int, tout: int) -> Optional[float]:
    from .llm import price_tokens
    return price_tokens(model, tin, tout)


def cmd_estimate(input_path: str, config_path: Optional[str], metrics: set[str]) -> int:
    cfg = load_config(config_path)
    items = load_items(input_path, cfg).items
    rows: list[dict[str, Any]] = []

    def add(role: str, model: str, calls: int, chars_in: int, tokens_out: int, formula: str) -> None:
        tin, tout = chars_in // 4, tokens_out
        rows.append({"role": role, "model": model or "(unset)", "calls": calls, "formula": formula,
                     "est_in_tokens": tin, "est_out_tokens": tout,
                     "est_cost_usd": _price(model, tin, tout) if model else None})

    if "m1" in metrics:
        elig = [it for it in items if it.eligible("m1")]
        for m in cfg.models.solvers:
            chars = sum(sum(map(len, build_solver(it, shuffle_options(it.options, cfg.seed, it.item_id, m)[0])))
                        for it in elig)
            add("solver", m, len(elig), chars, OUT_TOKENS["solver"] * len(elig), "eligible x 1")
        for m in cfg.models.verifier:
            add("verifier", m, 2 * len(elig), 2 * sum(len(it.stem) + 900 for it in elig),
                2 * OUT_TOKENS["verifier"] * len(elig), "trusted x 2 (upper bound: trusted <= eligible)")
    if "m2" in metrics:
        elig = [it for it in items if it.eligible("m2")]
        J, R, P = len(cfg.models.judges), cfg.judge_mutation.judge_runs, cfg.judge_mutation.paraphrase_controls
        mut_calls = mut_chars = mut_out = val_calls = val_chars = 0
        jr_calls = jr_chars = jr_out = 0
        ref_calls = ref_chars = 0
        for it in elig:
            C = len(it.rubric)
            ref_len = len(it.model_answer) or 1500
            base = len(it.stem) + sum(len(c.text) + 40 for c in it.rubric) + 900
            gen_ref = 0 if it.model_answer.strip() else 1
            ref_calls += J * R
            ref_chars += J * R * (base + ref_len)
            mut_calls += C + P + gen_ref
            mut_chars += (C + P) * (base + ref_len) + gen_ref * base
            mut_out += (C + P) * (ref_len // 4 + 60) + gen_ref * 400
            val_calls += C + P
            val_chars += (C + P) * (base + 2 * ref_len)
            jr_calls += (C + P + 1) * J * R
            jr_chars += (C + P + 1) * J * R * (base + ref_len)
            jr_out += (C + P + 1) * J * R * (60 * C + 40)
        for j in cfg.models.judges:
            add("judge (reference check)", j, ref_calls // max(1, J), ref_chars // max(1, J), 0, "J x R per item")
            add("judge", j, jr_calls // max(1, J), jr_chars // max(1, J), jr_out // max(1, J),
                "(C + P + 1) x J x R per item; includes the reference, served from cache")
        add("mutator", cfg.models.mutator or "", mut_calls, mut_chars, mut_out, "C + P (+1 if reference generated)")
        add("mutation_validator", cfg.models.mutation_validator or "", val_calls, val_chars,
            OUT_TOKENS["mutation_validator"] * val_calls, "C + P per item")
    if "m3" in metrics:
        elig = [it for it in items if it.eligible("m3")]
        cands, _ = topic_blind.candidate_topics(elig)
        for m in cfg.models.topic_classifiers:
            chars = sum(sum(map(len, build_topic_classifier(it, cands[it.course_code], cfg.topic_blind.top_k)))
                        for it in elig)
            add("topic_classifier", m, len(elig), chars, OUT_TOKENS["topic_classifier"] * len(elig), "items x 1")
        if cfg.models.embedding:
            add("embedding", cfg.models.embedding, len(elig), sum(len(it.stem) for it in elig), 0, "items x 1")

    d = pd.DataFrame(rows)
    if d.empty:
        print("No calls: no eligible items, or no models configured for the selected metrics.")
        return EXIT_OK
    print(d.to_string(index=False))
    uniq = d[d["role"] != "judge (reference check)"]
    known = uniq["est_cost_usd"].dropna()
    print()
    print(f"Network calls (reference-check calls are reused by judging): {int(uniq['calls'].sum())}")
    if len(known):
        print(f"Rough cost: ${known.sum():.4f}" + ("" if len(known) == len(uniq) else
                                                   " (some models have no price in LiteLLM's table)"))
    else:
        print("Rough cost: unknown (no prices in LiteLLM's table for these models)")
    print("Token estimate: 4 characters ~ 1 token; output sizes are rough per-role guesses.")
    return EXIT_OK


# ---------------------------------------------------------------- run / rescore

def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r_ in rows:
            f.write(json.dumps(r_, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_all(items: list[Item], raw: dict[str, list[dict[str, Any]]], cfg: Config, metrics: set[str],
              human: Optional[dict] = None, key_labels: Optional[dict[str, bool]] = None) -> dict[str, MetricResult]:
    out: dict[str, MetricResult] = {}
    if "m1" in metrics:
        out["m1"] = key_consensus.score(items, raw, cfg, key_labels)
    if "m2" in metrics:
        out["m2"] = judge_mutation.score(items, raw, cfg, human)
    if "m3" in metrics:
        out["m3"] = topic_blind.score(items, raw, cfg)
    return out


def _versions() -> dict[str, str]:
    v = {"python": platform.python_version()}
    for p in ("litellm", "pydantic", "pandas", "numpy", "scipy", "pint", "json-repair", "tenacity"):
        try:
            v[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            v[p] = "?"
    return v


def _write_outputs(out: Path, results: dict[str, MetricResult], cfg: Config, manifest: dict[str, Any]) -> dict:
    summary = build_summary(results, cfg)
    write_tables(results, out)
    write_json(out / "summary.json", summary)
    (out / "report.md").write_text(render_report(summary, results, cfg, manifest), encoding="utf-8")
    return summary


def cmd_run(input_path: str, config_path: Optional[str], out_dir: str, metrics: set[str], limit: Optional[int] = None,
            no_cache: bool = False, strict: bool = False, backend: Optional[Backend] = None,
            embed_backend: Any = None) -> int:
    cfg = load_config(config_path)
    errors, warnings = validate_config(cfg, metrics, strict)
    for w in warnings:
        print(f"warning: {w}")
    if errors:
        for e in errors:
            print(f"config error: {e}")
        return EXIT_CONFIG
    load_env()
    missing = missing_keys(cfg, metrics) if backend is None else []
    if missing:
        for k in missing:
            print(f"config error: {k} is not set (environment or item_metrics/.env)")
        return EXIT_CONFIG

    out = Path(out_dir)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    for f in ("llm_calls.jsonl", "parse_failures.jsonl"):
        (out / f).unlink(missing_ok=True)
    (out / "parse_failures.jsonl").touch()
    loaded = load_items(input_path, cfg)
    write_invalid(loaded.invalid, out)
    items = loaded.items[:limit] if limit else loaded.items
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    llm = LLMClient(cfg.llm, out, use_cache=not no_cache, backend=backend, embed_backend=embed_backend)

    async def collect_all() -> dict[str, list[dict[str, Any]]]:
        raw: dict[str, list[dict[str, Any]]] = {}
        for m in COLLECT_ORDER:
            if m in metrics:
                raw.update(await MODULES[m].collect(items, cfg, llm))
        return raw

    raw = asyncio.run(collect_all())
    _write_jsonl(out / "raw" / "items.jsonl", [it.model_dump(mode="json") for it in items])
    for name, rows in raw.items():
        _write_jsonl(out / "raw" / f"{name}.jsonl", rows)

    results = score_all(items, raw, cfg, metrics)
    manifest = {
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": str(input_path), "input_sha256": file_sha256(input_path),
        "rows": {"read": loaded.n_rows, "valid": len(loaded.items), "invalid": len(loaded.invalid),
                 "used": len(items), "limit": limit, "tolerance_capped": loaded.tolerance_capped},
        "metrics": sorted(metrics), "config": config_for_manifest(cfg),
        "models": {"generator_model": cfg.generator_model, **cfg.models.model_dump()},
        "prompts": prompt_hashes(), "libraries": _versions(), "seed": cfg.seed,
        "llm": {"network_calls": llm.n_network_calls, "cache_hits": llm.n_cache_hits,
                "parse_failures": llm.n_parse_failures, "cost_unknown_calls": llm.cost_unknown_calls,
                "total_cost_usd": round(llm.spent, 6), "cache_used": not no_cache,
                "api_keys": llm.key_status(),
                "models_without_temperature": sorted(llm.no_temperature)},
        "budget_stopped": llm.budget_stopped, "warnings": warnings,
        "single_vendor": bool(cfg.allow_single_vendor and any("one vendor" in w for w in warnings)),
    }
    write_json(out / "run_manifest.json", manifest)
    _write_outputs(out, results, cfg, manifest)
    print(f"wrote {out}: {llm.n_network_calls} calls, {llm.n_cache_hits} cache hits, ${llm.spent:.4f}"
          + (" (BUDGET STOPPED: partial results)" if llm.budget_stopped else ""))
    for prov, st in llm.key_status().items():
        for k in st["per_key"]:
            if k["disabled"] or k["rate_limits"] or k["models_unavailable"]:
                print(f"{prov} key #{k['number']}: {k['calls']} calls, {k['rate_limits']} rate limits"
                      + (f", disabled ({k['reason']})" if k["disabled"] else "")
                      + "".join(f"; not available for {m}" for m in k["models_unavailable"]))
    if llm.cost_unknown_calls:
        print(f"warning: {llm.cost_unknown_calls} calls had no known price; spent cost and the max_cost_usd "
              "stop undercount them")
    return EXIT_BUDGET if llm.budget_stopped else EXIT_OK


def _key_labels(path: str) -> dict[str, bool]:
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    out = {}
    for row in d.to_dict(orient="records"):
        v = str(row.get("human_key_correct", "")).strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            out[row["item_id"]] = True
        elif v in ("false", "0", "no", "n", "f"):
            out[row["item_id"]] = False
    return out


def cmd_rescore(out_dir: str, human_labels: Optional[str] = None, key_labels: Optional[str] = None,
                config_path: Optional[str] = None) -> int:
    """Recompute every metric from raw/*.jsonl; no network calls."""
    out = Path(out_dir)
    manifest = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
    cfg = load_config(config_path) if config_path else Config.model_validate(manifest["config"])
    metrics = set(manifest["metrics"])
    items = [Item.model_validate(x) for x in _read_jsonl(out / "raw" / "items.jsonl")]
    raw = {p.stem: _read_jsonl(p) for p in sorted((out / "raw").glob("*.jsonl")) if p.stem != "items"}
    human = None
    if human_labels:
        human = judge_mutation.load_human_labels(human_labels)
    elif cfg.judge_mutation.require_human_validation and (out / "m2_mutations_for_review.csv").exists():
        human = judge_mutation.load_human_labels(out / "m2_mutations_for_review.csv")
    keys = _key_labels(key_labels) if key_labels else None
    results = score_all(items, raw, cfg, metrics, human, keys)
    manifest.setdefault("rescores", []).append({
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "human_labels": human_labels,
        "key_labels": key_labels, "config": config_path})
    write_json(out / "run_manifest.json", manifest)
    _write_outputs(out, results, cfg, manifest)
    print(f"rescored {out} from raw/ (no LLM calls)")
    return EXIT_OK
