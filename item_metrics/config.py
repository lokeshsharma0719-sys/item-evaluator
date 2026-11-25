"""Configuration: pydantic model loaded from YAML, plus role / vendor-diversity validation."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHT_TOL = 1e-6


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelsCfg(_M):
    solvers: list[str] = Field(default_factory=list)
    verifier: list[str] = Field(default_factory=list)
    judges: list[str] = Field(default_factory=list)
    mutator: Optional[str] = None
    mutation_validator: Optional[str] = None
    topic_classifiers: list[str] = Field(default_factory=list)
    embedding: Optional[str] = None


class LLMCfg(_M):
    temperature: float = 0.0
    max_concurrency: int = 8
    timeout_s: float = 120
    max_retries: int = 4
    retry_max_wait_s: float = 30        # cap on one backoff wait; raise it for low per-minute quotas (Gemini free tier)
    key_rotation: Literal["failover", "round_robin"] = "failover"   # with several keys per provider
    key_cooldown_s: float = 5.0         # first rest for a rate-limited key; doubles per strike up to retry_max_wait_s
    json_mode: bool = True
    cache_dir: str = ".cache/llm"
    max_cost_usd: float = 10.0


class NumericCfg(_M):
    default_tolerance: float = 0.02
    tolerance_cap: float = 0.02
    planted_perturbations: list[float] = Field(default_factory=lambda: [0.03, 0.05, 0.10])


class KeyConsensusCfg(_M):
    gate_rule: Literal["all", "majority"] = "all"
    shuffle_options: bool = True


class JudgeMutationCfg(_M):
    judge_runs: int = 2
    paraphrase_controls: int = 2
    require_human_validation: bool = False


class TopicBlindCfg(_M):
    top_k: int = 3
    min_topics_warning: int = 4


class ThresholdsCfg(_M):
    key_consensus_rate: float = 0.90
    planted_error_recall: float = 0.90
    ambiguity_rate_max: float = 0.05
    judge_balanced_accuracy: float = 0.85
    judge_sensitivity: float = 0.80
    topic_accuracy: float = 0.80


class NormalisedCfg(_M):
    floor: float = 0.01
    m1_weights: dict[str, float] = Field(
        default_factory=lambda: {"kcr": 0.50, "per": 0.25, "no_ambiguity": 0.25, "verifier": 0.15})
    m2_weights: dict[str, float] = Field(default_factory=lambda: {"balanced_accuracy": 0.70, "score_accuracy": 0.30})
    m3_weights: dict[str, float] = Field(
        default_factory=lambda: {"chance_adj_accuracy": 0.70, "chance_adj_mrr": 0.30})


class Config(_M):
    generator_model: Optional[str] = None
    # The spec requires >= 2 vendors per role. With only one provider key (e.g. Gemini alone) that cannot be met;
    # this switch turns the vendor rule into a warning that is recorded in the manifest and the report.
    allow_single_vendor: bool = False
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    llm: LLMCfg = Field(default_factory=LLMCfg)
    seed: int = 1234
    numeric: NumericCfg = Field(default_factory=NumericCfg)
    key_consensus: KeyConsensusCfg = Field(default_factory=KeyConsensusCfg)
    judge_mutation: JudgeMutationCfg = Field(default_factory=JudgeMutationCfg)
    topic_blind: TopicBlindCfg = Field(default_factory=TopicBlindCfg)
    thresholds: ThresholdsCfg = Field(default_factory=ThresholdsCfg)
    normalised: NormalisedCfg = Field(default_factory=NormalisedCfg)
    column_map: dict[str, str] = Field(default_factory=dict)


def load_config(path: str | os.PathLike | None) -> Config:
    if path is None:
        return Config()
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(data)


def load_env() -> None:
    """Read provider keys from the environment or item_metrics/.env. Values are never logged."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)


# LiteLLM provider prefix -> main key variables (numbered _2.._9 and comma-separated *_API_KEYS also work).
PROVIDER_KEYS = {"openrouter": ("OPENROUTER_API_KEY",), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}
MODEL_FORMS = "openrouter/<vendor>/<model> or gemini/<model>"


def provider_of(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def vendor_of(model: str) -> str:
    """Model family owner: the segment after openrouter/, or google for gemini/ models."""
    parts = model.split("/")
    if parts[0] == "openrouter":
        return parts[1] if len(parts) >= 3 and parts[1] and parts[2] else ""
    if parts[0] == "gemini":
        return "google" if len(parts) >= 2 and parts[1] else ""
    return ""


def api_key_for(model: str, env: Optional[dict[str, str]] = None) -> Optional[str]:
    """The first configured key for the model's provider."""
    from .keys import keys_from_env
    keys = keys_from_env(provider_of(model), env)
    return keys[0] if keys else None


def missing_keys(cfg: "Config", metrics: set[str], env: Optional[dict[str, str]] = None) -> list[str]:
    """Providers used by the configured roles whose key is not set, as 'VAR (provider)' strings."""
    used = {provider_of(m) for ms in roles_for(cfg, metrics).values() for m in ms}
    from .keys import keys_from_env
    return [f"{' or '.join(PROVIDER_KEYS[p])} (for {p}/ models)" for p in sorted(used)
            if p in PROVIDER_KEYS and not keys_from_env(p, env)]


def roles_for(cfg: Config, metrics: set[str]) -> dict[str, list[str]]:
    m = cfg.models
    roles: dict[str, list[str]] = {}
    if "m1" in metrics:
        roles["solvers"] = list(m.solvers)
        if m.verifier:
            roles["verifier"] = list(m.verifier)
    if "m2" in metrics:
        roles["judges"] = list(m.judges)
        roles["mutator"] = [m.mutator] if m.mutator else []
        roles["mutation_validator"] = [m.mutation_validator] if m.mutation_validator else []
    if "m3" in metrics:
        roles["topic_classifiers"] = list(m.topic_classifiers)
        if m.embedding:
            roles["embedding"] = [m.embedding]
    return roles


def validate_config(cfg: Config, metrics: set[str], strict: bool = False) -> tuple[list[str], list[str]]:
    """Returns (errors, warnings). Errors make `run` exit with code 1."""
    errors: list[str] = []
    warnings: list[str] = []
    roles = roles_for(cfg, metrics)

    vendor_problems = errors if not cfg.allow_single_vendor else warnings
    for role, models in roles.items():
        for mdl in models:
            if provider_of(mdl) not in PROVIDER_KEYS or not vendor_of(mdl):
                errors.append(f"{role}: '{mdl}' is not a supported model string ({MODEL_FORMS})")
            if "<" in mdl:
                errors.append(f"{role}: '{mdl}' is still a placeholder")
        if len(set(models)) != len(models):
            errors.append(f"{role}: the same model is listed twice")
        if role != "embedding" and len(models) >= 2 and len({vendor_of(x) for x in models}) < 2:
            vendor_problems.append(f"{role}: all models share one vendor ({vendor_of(models[0])}); errors may be "
                                   "correlated" + ("" if cfg.allow_single_vendor else
                                                   "; use at least 2 vendors or set allow_single_vendor: true"))

    if "m1" in metrics and len(cfg.models.solvers) < 2:
        errors.append("solvers: M1 needs at least 2 solvers (3 recommended) from at least 2 vendors")
    if "m2" in metrics:
        m = cfg.models
        if not m.judges:
            errors.append("judges: M2 needs at least 1 judge")
        if not m.mutator:
            errors.append("mutator: M2 needs a mutator model")
        if not m.mutation_validator:
            errors.append("mutation_validator: M2 needs a mutation validator model")
        elif m.mutation_validator == m.mutator or m.mutation_validator in m.judges:
            errors.append("mutation_validator must differ from the mutator and from every judge")
        m2_models = set(m.judges) | ({m.mutator} if m.mutator else set()) | (
            {m.mutation_validator} if m.mutation_validator else set())
        if len(m2_models) >= 2 and len({vendor_of(x) for x in m2_models}) < 2:
            vendor_problems.append("M2 uses several models but all from one vendor" + (
                "" if cfg.allow_single_vendor else "; use at least 2 vendors or set allow_single_vendor: true"))
    if "m3" in metrics and len(cfg.models.topic_classifiers) < 2:
        errors.append("topic_classifiers: M3 needs at least 2 classifiers from at least 2 vendors")

    gen = cfg.generator_model
    if gen:
        checker_roles = {r: ms for r, ms in roles.items() if r not in ("embedding",)}
        for role, models in checker_roles.items():
            if gen in models:
                msg = f"{role}: generator_model '{gen}' must not check its own items"
                (errors if strict else warnings).append(msg)

    w = cfg.normalised
    m1_core = {k: v for k, v in w.m1_weights.items() if k != "verifier"}
    for name, weights, expected in (("m1_weights (without verifier)", m1_core, {"kcr", "per", "no_ambiguity"}),
                                    ("m2_weights", w.m2_weights, {"balanced_accuracy", "score_accuracy"}),
                                    ("m3_weights", w.m3_weights, {"chance_adj_accuracy", "chance_adj_mrr"})):
        if set(weights) != expected:
            errors.append(f"normalised.{name}: keys must be {sorted(expected)}")
        elif abs(sum(weights.values()) - 1.0) > WEIGHT_TOL:
            errors.append(f"normalised.{name}: weights must sum to 1 (got {sum(weights.values()):.4f})")
        if any(v < 0 for v in weights.values()):
            errors.append(f"normalised.{name}: weights must be non-negative")
    if not 0 < cfg.numeric.default_tolerance:
        errors.append("numeric.default_tolerance must be > 0")
    if any(d <= 0 for d in cfg.numeric.planted_perturbations):
        errors.append("numeric.planted_perturbations must be positive")
    if cfg.judge_mutation.judge_runs < 1:
        errors.append("judge_mutation.judge_runs must be >= 1")
    if cfg.topic_blind.top_k < 1:
        errors.append("topic_blind.top_k must be >= 1")
    return errors, warnings


def config_for_manifest(cfg: Config) -> dict:
    """The config contains no secrets (the key only ever comes from the environment)."""
    return cfg.model_dump(mode="json")
