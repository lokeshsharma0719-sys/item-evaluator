"""LiteLLM wrapper: async, one semaphore, retries, JSON parse + repair, disk cache, cost tracking, budget stop.

All traffic goes through `litellm.acompletion` / `litellm.aembedding` (OpenRouter or Gemini model strings). The network
backend is `LLMClient.backend`; tests replace it with a scripted fake (tests/fake_llm.py). The cache sits above the
backend, so a cached call never reaches it.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional, Type

from pydantic import BaseModel, ValidationError

from .config import LLMCfg, provider_of
from .keys import KeyPool, classify_error, redact

log = logging.getLogger(__name__)

# litellm otherwise downloads its price table on import; the bundled copy keeps the only outbound traffic the
# model calls themselves.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_TELEMETRY", "False")

Backend = Callable[..., Awaitable["BackendResult"]]


@dataclass
class BackendResult:
    text: str
    cost: float = 0.0
    cost_unknown: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BudgetStopped(RuntimeError):
    pass


def _litellm():
    import litellm
    litellm.telemetry = False
    litellm.suppress_debug_info = True
    return litellm


def _transient_errors() -> tuple[type, ...]:
    lt = _litellm()
    # Rate limits are not retried here: the key pool rotates to another key or rests the key instead.
    names = ("Timeout", "APIConnectionError", "InternalServerError", "ServiceUnavailableError", "APIError")
    errs = tuple(getattr(lt, n) for n in names if isinstance(getattr(lt, n, None), type))
    return errs + (asyncio.TimeoutError,)


def strip_fences(s: str) -> str:
    s = (s or "").strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.S)
    return m.group(1) if m else s


def parse_json_text(text: str) -> Any:
    s = strip_fences(text)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        import json_repair
        v = json_repair.loads(s)
        return v if v not in ("", None) else None
    except Exception:
        return None


def _validate(obj: Any, schema: Type[BaseModel]) -> Optional[BaseModel]:
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return None
    try:
        return schema.model_validate(obj)
    except ValidationError:
        return None


class LLMClient:
    def __init__(self, cfg: LLMCfg, out_dir: str | Path | None = None, use_cache: bool = True,
                 backend: Optional[Backend] = None, embed_backend: Optional[Callable[..., Awaitable[Any]]] = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.use_cache = use_cache
        self.cache_dir = Path(cfg.cache_dir)
        self.backend: Backend = backend or self._litellm_backend
        self.embed_backend = embed_backend or self._litellm_embed
        self.spent = 0.0
        self.budget_stopped = False
        self.n_network_calls = 0
        self.n_cache_hits = 0
        self.n_parse_failures = 0
        self.cost_unknown_calls = 0
        self._no_json_mode: set[str] = set()
        self.no_temperature: set[str] = set()
        self._sem: Optional[asyncio.Semaphore] = None
        self._pools: dict[str, KeyPool] = {}

    # ------------------------------------------------------------ plumbing
    @property
    def sem(self) -> asyncio.Semaphore:
        if self._sem is None:
            self._sem = asyncio.Semaphore(max(1, self.cfg.max_concurrency))
        return self._sem

    def _log(self, name: str, row: dict[str, Any]) -> None:
        if self.out_dir is None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with (self.out_dir / name).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def cache_key(self, model: str, system: str, user: str, prompt_version: str, extra: str) -> str:
        payload = json.dumps([model, system, user, self.cfg.temperature, prompt_version, extra], ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_get(self, key: str) -> Optional[dict[str, Any]]:
        if not self.use_cache:
            return None
        p = self._cache_path(key)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def _cache_put(self, key: str, data: dict[str, Any]) -> None:
        p = self._cache_path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)

    def over_budget(self) -> bool:
        if self.spent >= self.cfg.max_cost_usd:
            self.budget_stopped = True
        return self.budget_stopped

    # ------------------------------------------------------------ keys
    def pool(self, model: str) -> KeyPool:
        p = provider_of(model)
        if p not in self._pools:
            self._pools[p] = KeyPool.from_env(p, mode=self.cfg.key_rotation, cooldown_s=self.cfg.key_cooldown_s,
                                              max_cooldown_s=max(self.cfg.key_cooldown_s, self.cfg.retry_max_wait_s))
            if len(self._pools[p].keys) > 1:
                log.info("%d %s keys loaded (%s)", len(self._pools[p].keys), p, self.cfg.key_rotation)
        return self._pools[p]

    def key_status(self) -> dict[str, Any]:
        """Per provider: number of keys, calls, rate limits and disabled keys. Never key values."""
        return {p: pool.status() for p, pool in sorted(self._pools.items())}

    async def _with_keys(self, model: str, fn: Callable[[str], Awaitable[Any]]) -> Any:
        """fn(api_key) with key rotation; transient non-key errors are retried on the same key."""
        pool = self.pool(model)
        n = max(1, len(pool.keys))
        result, _ = await pool.run(lambda key: self._with_retries(lambda: fn(key)),
                                   max_rate_limits=max(1, self.cfg.max_retries) * n, model=model)
        return result

    # ------------------------------------------------------------ real backend
    async def _litellm_backend(self, model: str, messages: list[dict[str, str]], **_ctx: Any) -> BackendResult:
        lt = _litellm()
        base: dict[str, Any] = dict(model=model, messages=messages, temperature=self.cfg.temperature,
                                    timeout=self.cfg.timeout_s)
        if model.startswith("openrouter/"):
            base["extra_body"] = {"usage": {"include": True}}   # ask OpenRouter to report the call's cost

        async def attempt(key: str) -> Any:
            kwargs = dict(base, api_key=key)
            if model in self.no_temperature:
                kwargs.pop("temperature", None)
            use_json = self.cfg.json_mode and model not in self._no_json_mode
            if use_json:
                kwargs["response_format"] = {"type": "json_object"}
            for _ in range(3):
                try:
                    return await lt.acompletion(**kwargs)
                except Exception as e:
                    if classify_error(e) is not None:
                        raise
                    msg = str(e).lower()
                    # Some models reject temperature: drop it, remember it, and record it in the manifest
                    # (those models' outputs are not at temperature 0).
                    if "temperature" in kwargs and "unsupportedparams" in type(e).__name__.lower() \
                            and "temperature" in msg:
                        self.no_temperature.add(model)
                        kwargs.pop("temperature", None)
                        log.warning("%s does not accept temperature; calling it without", model)
                        continue
                    # The model rejected response_format: retry without it and remember it.
                    if "response_format" in kwargs and _looks_like_param_error(e):
                        self._no_json_mode.add(model)
                        kwargs.pop("response_format", None)
                        continue
                    raise
            return await lt.acompletion(**kwargs)

        resp = await self._with_keys(model, attempt)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        tout = int(getattr(usage, "completion_tokens", 0) or 0)
        cost = response_cost(resp, model, tin, tout)
        return BackendResult(text=text, cost=cost or 0.0, cost_unknown=cost is None,
                             prompt_tokens=tin, completion_tokens=tout)

    async def _with_retries(self, fn: Callable[[], Awaitable[Any]]) -> Any:
        from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
        async for attempt in AsyncRetrying(stop=stop_after_attempt(max(1, self.cfg.max_retries)),
                                           wait=wait_exponential(multiplier=1, min=1, max=self.cfg.retry_max_wait_s),
                                           retry=retry_if_exception_type(_transient_errors()), reraise=True):
            with attempt:
                return await fn()

    async def _litellm_embed(self, model: str, texts: list[str]) -> tuple[list[list[float]], float]:
        lt = _litellm()
        resp = await self._with_keys(model, lambda key: lt.aembedding(model=model, input=texts,
                                                                      timeout=self.cfg.timeout_s, api_key=key))
        vecs = [d["embedding"] if isinstance(d, dict) else d.embedding for d in resp.data]
        try:
            cost = float(lt.completion_cost(completion_response=resp) or 0.0)
        except Exception:
            cost = 0.0
        return vecs, cost

    # ------------------------------------------------------------ public
    async def call_json(self, role: str, model: str, system: str, user: str, schema: Type[BaseModel],
                        prompt_version: str = "", cache_key_extra: str = "", item_id: str = "",
                        variant_id: str = "") -> tuple[Optional[BaseModel], dict[str, Any]]:
        """Returns (parsed or None, meta). meta['status'] is ok | parse_failure | budget_skipped | error."""
        key = self.cache_key(model, system, user, prompt_version, cache_key_extra)
        meta: dict[str, Any] = {"role": role, "model": model, "item_id": item_id, "variant_id": variant_id,
                                "cache_hit": False, "cost": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                                "latency_s": 0.0, "status": "ok"}
        cached = self._cache_get(key)
        if cached is not None:
            parsed = _validate(parse_json_text(cached["text"]), schema)
            if parsed is not None:
                self.n_cache_hits += 1
                meta["cache_hit"] = True
                self._log("llm_calls.jsonl", meta)
                return parsed, meta

        async with self.sem:
            if self.over_budget():
                meta["status"] = "budget_skipped"
                self._log("llm_calls.jsonl", meta)
                return None, meta
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            t0 = time.monotonic()
            texts: list[str] = []
            parsed = None
            try:
                for attempt in range(2):
                    res = await self.backend(model, messages, role=role, item_id=item_id, variant_id=variant_id)
                    self.n_network_calls += 1
                    self._account(meta, res)
                    texts.append(res.text)
                    parsed = _validate(parse_json_text(res.text), schema)
                    if parsed is not None or attempt == 1 or self.over_budget():
                        break
                    messages = messages + [
                        {"role": "assistant", "content": res.text},
                        {"role": "user", "content": "Return only valid JSON matching this schema, no markdown, "
                                                    "no prose outside the JSON: "
                                                    + json.dumps(schema.model_json_schema())}]
            except Exception as e:
                meta["status"] = "error"
                meta["error"] = f"{type(e).__name__}: {redact(e)[:300]}"
                log.warning("LLM call failed (%s %s %s): %s", role, model, item_id, meta["error"])
            meta["latency_s"] = round(time.monotonic() - t0, 3)

        if parsed is None:
            if meta["status"] == "ok":
                meta["status"] = "parse_failure"
            self.n_parse_failures += 1
            self._log("parse_failures.jsonl", {**meta, "raw": texts})
        else:
            self._cache_put(key, {"text": texts[-1], "model": model, "role": role})
        self._log("llm_calls.jsonl", meta)
        return parsed, meta

    def _account(self, meta: dict[str, Any], res: BackendResult) -> None:
        self.spent += res.cost
        meta["cost"] = round(meta["cost"] + res.cost, 8)
        meta["prompt_tokens"] += res.prompt_tokens
        meta["completion_tokens"] += res.completion_tokens
        if res.cost_unknown:
            meta["cost_unknown"] = True
            self.cost_unknown_calls += 1

    async def embed(self, model: str, texts: list[str], item_id: str = "") -> Optional[list[list[float]]]:
        """Embeddings with the same cache and budget. Returns None (never raises) when unsupported."""
        key = hashlib.sha256(json.dumps(["embed", model, texts]).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            self.n_cache_hits += 1
            return cached["vectors"]
        async with self.sem:
            if self.over_budget():
                return None
            try:
                vecs, cost = await self.embed_backend(model, texts)
            except Exception as e:
                log.warning("embedding failed for %s (%s); skipping embedding rank", model, redact(e)[:200])
                return None
            self.n_network_calls += 1
            self.spent += cost
        self._log("llm_calls.jsonl", {"role": "embedding", "model": model, "item_id": item_id, "cost": cost,
                                      "status": "ok", "cache_hit": False})
        self._cache_put(key, {"vectors": vecs})
        return vecs


def price_tokens(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Price from LiteLLM's bundled table. The table may not list OpenRouter routes, so fall back to the
    vendor's own entry for the bare model name (OpenRouter passes vendor prices through)."""
    lt = _litellm()
    names = [model]
    if model.startswith("openrouter/"):
        rest = model.split("/", 1)[1]                       # vendor/model
        names += [rest, rest.split("/", 1)[-1]]             # vendor/model, model
    for name in names:
        try:
            a, b = lt.cost_per_token(model=name, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            return float(a) + float(b)
        except Exception:
            continue
    return None


def response_cost(resp: Any, model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """Cost of one response; None when no source knows it (recorded as cost_unknown)."""
    lt = _litellm()
    try:
        c = lt.completion_cost(completion_response=resp)
        if c:
            return float(c)
    except Exception:
        pass
    hidden = getattr(resp, "_hidden_params", None) or {}
    if hidden.get("response_cost"):
        return float(hidden["response_cost"])
    usage_cost = getattr(getattr(resp, "usage", None), "cost", None)   # OpenRouter reports it when available
    if usage_cost:
        return float(usage_cost)
    return price_tokens(model, prompt_tokens, completion_tokens)


def _looks_like_param_error(e: Exception) -> bool:
    s = f"{type(e).__name__} {e}".lower()
    return any(t in s for t in ("response_format", "unsupportedparams", "json_object", "badrequest"))


async def gather_progress(coros: Iterable[Awaitable[Any]], desc: str) -> list[Any]:
    """asyncio.gather with a tqdm bar; results keep input order."""
    from tqdm import tqdm
    coros = list(coros)
    if not coros:
        return []
    bar = tqdm(total=len(coros), desc=desc, leave=False, disable=not coros)

    async def wrap(c: Awaitable[Any]) -> Any:
        try:
            return await c
        finally:
            bar.update(1)

    try:
        return await asyncio.gather(*(wrap(c) for c in coros))
    finally:
        bar.close()
