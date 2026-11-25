"""Several API keys per provider, with failover.

Keys come from the environment / item_metrics/.env:
    GEMINI_API_KEY, GEMINI_API_KEY_2 ... GEMINI_API_KEY_9      (GOOGLE_API_KEY is also accepted)
    GEMINI_API_KEYS=key1,key2,...                              (comma-separated, optional)
and the same with OPENROUTER_ instead of GEMINI_. Duplicates are dropped; order is kept.

On an error the key is classified:
- dead (invalid, expired, revoked, no permission, daily quota used up): never used again in this run;
- rate limited (429 / per-minute quota): rested for a cooldown that doubles per consecutive strike, and the next
  key is tried at once;
- anything else (timeouts, 5xx, bad requests): not the key's fault; the caller retries or fails as before.

Keys are referred to by provider and number only ("gemini key #2"); their values are never logged.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

log = logging.getLogger(__name__)

PROVIDER_PREFIXES = {"openrouter": ("OPENROUTER",), "gemini": ("GEMINI", "GOOGLE")}
MAX_NUMBERED = 9

DEAD, MODEL_OFF, RATE = "dead", "model_unavailable", "rate_limited"
# Whole key unusable: auth, expiry, revocation, account credit or billing.
_DEAD_MARKERS = ("api key not valid", "api_key_invalid", "invalid api key", "invalid_api_key", "incorrect api key",
                 "api key expired", "key expired", "token expired", "access_token_expired", "expired api key",
                 "invalid authentication", "unauthenticated", "permission denied", "permission_denied",
                 "api key was reported as leaked", "has been suspended", "account is disabled", "key revoked",
                 "no auth credentials", "user not found", "insufficient credits", "insufficient_quota", "billing")
# This key cannot serve this model for the rest of the run (other models on the key still work):
# the model is not offered to the key, or its daily quota on this key is used up or zero.
_MODEL_OFF_MARKERS = ("perday", "per day", "daily limit", "limit: 0", "is not found for api version",
                      "not supported for generatecontent", "model not found", "no longer available")
_RATE_MARKERS = ("rate limit", "ratelimit", "resource_exhausted", "resource exhausted", "too many requests",
                 "quota", "429")


class AllKeysFailed(RuntimeError):
    pass


_SECRETS: set[str] = set()
_URL_KEY = re.compile(r"((?:[?&]key=)|(?:api[_-]?key[\"']?\s*[:=]\s*[\"']?)|(?:bearer\s+))[^\s&\"',}]+", re.I)


def register_secret(value: str) -> None:
    if value and len(value) >= 8:
        _SECRETS.add(value)


def redact(text: Any) -> str:
    """Mask every known key value and anything that looks like a key in a URL or header."""
    s = str(text)
    for v in sorted(_SECRETS, key=len, reverse=True):
        s = s.replace(v, "***")
    return _URL_KEY.sub(lambda m: m.group(1) + "***", s)


def keys_from_env(provider: str, env: Optional[Mapping[str, str]] = None) -> list[str]:
    env = os.environ if env is None else env
    out: list[str] = []
    for prefix in PROVIDER_PREFIXES.get(provider, ()):
        names = [f"{prefix}_API_KEY"] + [f"{prefix}_API_KEY_{i}" for i in range(2, MAX_NUMBERED + 1)]
        vals = [env.get(n, "") for n in names]
        vals += (env.get(f"{prefix}_API_KEYS", "") or "").split(",")
        for v in vals:
            v = (v or "").strip().strip('"').strip("'")
            if v and v not in out:
                out.append(v)
    return out


def classify_error(e: BaseException) -> Optional[str]:
    """DEAD (whole key), MODEL_OFF (this key, this model), RATE (rest and retry) or None (not a key problem)."""
    name = type(e).__name__.lower()
    status = getattr(e, "status_code", None)
    msg = f"{name} {e}".lower()
    if "authentication" in name or "permissiondenied" in name or status in (401, 403):
        return DEAD
    # Checked before the dead markers: Gemini's per-day, per-model quota error also says "check your plan and
    # billing details", but only this model on this key is used up.
    if "notfound" in name or status == 404 or any(m in msg for m in _MODEL_OFF_MARKERS):
        return MODEL_OFF
    if any(m in msg for m in _DEAD_MARKERS):
        return DEAD
    if "ratelimit" in name or status == 429 or any(m in msg for m in _RATE_MARKERS):
        return RATE
    return None


@dataclass
class ModelState:
    cooldown_until: float = 0.0
    strikes: int = 0
    off_reason: str = ""


@dataclass
class KeyState:
    key: str
    number: int
    dead_reason: str = ""
    calls: int = 0
    rate_limits: int = 0
    failures: int = 0
    models: dict = field(default_factory=dict)

    def m(self, model: str) -> ModelState:
        return self.models.setdefault(model, ModelState())


@dataclass
class KeyPool:
    provider: str
    keys: list
    mode: str = "failover"                 # failover: stick to one key until it fails; round_robin: spread calls
    cooldown_s: float = 5.0
    max_cooldown_s: float = 60.0
    _next: dict = field(default_factory=dict)
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _sleep: Callable[[float], Awaitable[Any]] = field(default=asyncio.sleep, repr=False)

    @classmethod
    def from_env(cls, provider: str, env: Optional[Mapping[str, str]] = None, **kw: Any) -> "KeyPool":
        keys = keys_from_env(provider, env)
        for k in keys:
            register_secret(k)
        return cls(provider, [KeyState(k, i + 1) for i, k in enumerate(keys)], **kw)

    def label(self, s: KeyState, model: str = "") -> str:
        return f"{self.provider} key #{s.number}" + (f" ({model})" if model else "")

    def usable(self, model: str) -> list:
        return [s for s in self.keys if not s.dead_reason and not s.m(model).off_reason]

    async def acquire(self, model: str = "") -> KeyState:
        """A usable key for this model; waits for the earliest cooldown when every usable key is resting."""
        while True:
            live = self.usable(model)
            if not live:
                raise AllKeysFailed(self.describe_failure(model))
            now = self._clock()
            n = len(self.keys)
            start = self._next.get(model, 0)
            order = [self.keys[(start + i) % n] for i in range(n)]
            ready = [s for s in order if s in live and s.m(model).cooldown_until <= now]
            if ready:
                s = ready[0]
                idx = self.keys.index(s)
                self._next[model] = (idx + 1) % n if self.mode == "round_robin" else idx
                s.calls += 1
                return s
            wait = min(s.m(model).cooldown_until for s in live) - now
            log.warning("all %d usable %s keys are rate limited for %s; waiting %.1fs", len(live), self.provider,
                        model or "this model", wait)
            await self._sleep(max(0.0, wait))

    def report_success(self, s: KeyState, model: str = "") -> None:
        s.m(model).strikes = 0

    def _advance(self, s: KeyState, model: str) -> None:
        if self.mode == "failover" and self.keys.index(s) == self._next.get(model, 0):
            self._next[model] = (self.keys.index(s) + 1) % len(self.keys)

    def report_failure(self, s: KeyState, kind: str, e: BaseException, model: str = "") -> None:
        s.failures += 1
        ms = s.m(model)
        if kind == DEAD:
            s.dead_reason = _short(e)
            log.warning("%s disabled for this run: %s", self.label(s), s.dead_reason)
        elif kind == MODEL_OFF:
            ms.off_reason = _short(e)
            log.warning("%s cannot serve this model for the rest of the run: %s", self.label(s, model),
                        ms.off_reason)
        elif kind == RATE:
            s.rate_limits += 1
            ms.strikes += 1
            rest = min(self.max_cooldown_s, self.cooldown_s * (2 ** (ms.strikes - 1)))
            ms.cooldown_until = self._clock() + rest
            log.info("%s rate limited; resting %.0fs", self.label(s, model), rest)
        self._advance(s, model)

    def describe_failure(self, model: str = "") -> str:
        if not self.keys:
            return f"no {self.provider} API key is set"
        parts = [f"#{s.number}: {s.dead_reason or s.m(model).off_reason}" for s in self.keys
                 if s.dead_reason or s.m(model).off_reason]
        where = f" for {model}" if model else ""
        return f"all {len(self.keys)} {self.provider} keys failed{where} ({'; '.join(parts)})"

    def status(self) -> dict:
        """For the manifest: numbers and counts only, never key values."""
        return {"keys": len(self.keys), "mode": self.mode,
                "per_key": [{"number": s.number, "calls": s.calls, "rate_limits": s.rate_limits,
                             "failures": s.failures, "disabled": bool(s.dead_reason), "reason": s.dead_reason,
                             "models_unavailable": {m: st.off_reason for m, st in sorted(s.models.items())
                                                    if st.off_reason}} for s in self.keys]}

    async def run(self, fn: Callable[[str], Awaitable[Any]], max_rate_limits: int, model: str = "") -> tuple:
        """Call fn(api_key), rotating keys on key errors. Returns (result, key number)."""
        rate_hits = 0
        while True:
            s = await self.acquire(model)
            try:
                result = await fn(s.key)
            except Exception as e:
                kind = classify_error(e)
                if kind is None:
                    raise
                self.report_failure(s, kind, e, model)
                if kind == RATE:
                    rate_hits += 1
                    if rate_hits > max_rate_limits:
                        raise
                continue
            self.report_success(s, model)
            return result, s.number


def _short(e: BaseException) -> str:
    s = " ".join(redact(e).split())
    return f"{type(e).__name__}: {s[:160]}"
