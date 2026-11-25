"""Gemini (gemini/<model>) alongside OpenRouter: model strings, vendor rule, keys, prices, a full fake run."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml
from conftest import ROOT, config_dict
from fake_llm import smart_fake

from item_metrics.__main__ import main
from item_metrics.config import Config, api_key_for, load_config, missing_keys, validate_config, vendor_of
from item_metrics.llm import LLMClient, price_tokens

ALL = {"m1", "m2", "m3"}
GEMINI_ONLY = {
    "solvers": ["gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro", "gemini/gemini-3.5-flash"],
    "judges": ["gemini/gemini-2.5-flash", "gemini/gemini-3.5-flash"],
    "mutator": "gemini/gemini-2.5-pro", "mutation_validator": "gemini/gemini-2.5-flash-lite",
    "topic_classifiers": ["gemini/gemini-2.5-flash", "gemini/gemini-3.5-flash"],
}


def test_vendor_and_keys():
    assert vendor_of("gemini/gemini-2.5-flash") == "google"
    assert vendor_of("openrouter/openai/gpt-4o-mini") == "openai"
    assert vendor_of("gemini/") == "" and vendor_of("openai/gpt-4o") == ""
    env = {"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "o"}
    assert api_key_for("gemini/gemini-2.5-flash", env) == "g"
    assert api_key_for("gemini/gemini-2.5-flash", {"GOOGLE_API_KEY": "x"}) == "x"
    assert api_key_for("openrouter/openai/gpt-4o-mini", env) == "o"


def test_gemini_only_needs_the_switch(tmp_path):
    cfg = Config.model_validate(config_dict(tmp_path, models=GEMINI_ONLY))
    errors, _ = validate_config(cfg, ALL)
    assert any("one vendor" in e and "allow_single_vendor" in e for e in errors)
    cfg = Config.model_validate(config_dict(tmp_path, models=GEMINI_ONLY, allow_single_vendor=True))
    errors, warnings = validate_config(cfg, ALL)
    assert errors == [] and any("one vendor" in w for w in warnings)


def test_mixed_providers_satisfy_vendor_rule(tmp_path):
    models = {"solvers": ["gemini/gemini-2.5-flash", "openrouter/openai/gpt-4o-mini"],
              "topic_classifiers": ["gemini/gemini-2.5-flash", "openrouter/anthropic/claude-x"]}
    cfg = Config.model_validate(config_dict(tmp_path, models=models))
    errors, warnings = validate_config(cfg, {"m1", "m3"})
    assert errors == [] and warnings == []


def test_bad_model_strings_and_duplicates(tmp_path):
    models = {"solvers": ["openai/gpt-4o", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-flash"],
              "topic_classifiers": ["gemini/a", "openrouter/x/b"]}
    cfg = Config.model_validate(config_dict(tmp_path, models=models, allow_single_vendor=True))
    errors, _ = validate_config(cfg, {"m1"})
    assert any("not a supported model string" in e for e in errors)
    assert any("listed twice" in e for e in errors)


def test_missing_keys_per_provider(tmp_path):
    models = {"solvers": ["gemini/gemini-2.5-flash", "openrouter/openai/gpt-4o-mini"], "topic_classifiers": []}
    cfg = Config.model_validate(config_dict(tmp_path, models=models))
    assert missing_keys(cfg, {"m1"}, {"GEMINI_API_KEY": "g", "OPENROUTER_API_KEY": "o"}) == []
    miss = missing_keys(cfg, {"m1"}, {"OPENROUTER_API_KEY": "o"})
    assert len(miss) == 1 and "GEMINI_API_KEY" in miss[0]
    assert missing_keys(cfg, {"m1"}, {"GOOGLE_API_KEY": "g", "OPENROUTER_API_KEY": "o"}) == []


def test_run_refuses_without_gemini_key(tmp_path, fixture_csv, monkeypatch):
    import os
    for v in [k for k in os.environ if k.startswith(("GEMINI_API_KEY", "GOOGLE_API_KEY"))]:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr("item_metrics.runner.load_env", lambda: None)   # ignore a real item_metrics/.env
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(config_dict(tmp_path, models=GEMINI_ONLY, allow_single_vendor=True)))
    assert main(["run", "--input", str(fixture_csv), "--config", str(p), "--out", str(tmp_path / "o")]) == 1


def test_gemini_prices_known():
    assert price_tokens("gemini/gemini-2.5-flash", 1000, 100) > 0


def test_gemini_example_config_is_valid():
    cfg = load_config(ROOT / "config.gemini.example.yaml")
    errors, warnings = validate_config(cfg, ALL)
    assert errors == [] and warnings


def test_backend_gets_gemini_key(tmp_path, monkeypatch):
    """The real backend passes the Gemini key (not the OpenRouter one) and no OpenRouter-only options."""
    monkeypatch.setenv("GEMINI_API_KEY", "gem-key")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    seen = {}

    class Msg:
        content = '{"answer": "x"}'

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]
        usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()

    async def fake_acompletion(**kw):
        seen.update(kw)
        return Resp()

    import litellm
    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    cfg = Config.model_validate(config_dict(tmp_path))
    client = LLMClient(cfg.llm, tmp_path / "o")
    res = asyncio.run(client._litellm_backend("gemini/gemini-2.5-flash", [{"role": "user", "content": "hi"}]))
    assert seen["api_key"] == "gem-key" and "extra_body" not in seen
    assert seen["model"] == "gemini/gemini-2.5-flash" and seen["temperature"] == 0
    assert res.cost > 0 and not res.cost_unknown        # priced from LiteLLM's table


def test_full_fake_run_gemini_only(tmp_path, fixture_csv):
    # the fake backend keys some behaviour on model-name suffixes; map them onto Gemini-style names
    models = {**GEMINI_ONLY, "judges": ["gemini/judge-a-j1", "gemini/judge-b-j2"],
              "topic_classifiers": ["gemini/topic-a-t1", "gemini/topic-b-t2"]}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(config_dict(tmp_path, models=models, allow_single_vendor=True)))
    out = tmp_path / "o"
    assert main(["run", "--input", str(fixture_csv), "--config", str(p), "--out", str(out)], backend=smart_fake()) == 0
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["single_vendor"] is True
    assert "Single vendor" in (out / "report.md").read_text(encoding="utf-8")
    s = json.loads((out / "summary.json").read_text())
    assert s["faithfulness_m3"]["blind_topic_accuracy"]["value"] == pytest.approx(7 / 8)
