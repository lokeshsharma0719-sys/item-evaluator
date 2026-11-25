"""Several API keys per provider: parsing, failover on dead keys, rest on rate limits, redaction."""
from __future__ import annotations

import asyncio
import json

import litellm
import pytest
from conftest import config_dict

from item_metrics.config import Config, missing_keys
from item_metrics.keys import DEAD, MODEL_OFF, RATE, KeyPool, classify_error, keys_from_env, redact
from item_metrics.llm import LLMClient
from item_metrics.schemas import ReferenceOut

MODEL = "gemini/gemini-2.5-flash"
KEYS = {"GEMINI_API_KEY": "key-one-aaaa", "GEMINI_API_KEY_2": "key-two-bbbb", "GEMINI_API_KEY_3": "key-three-cc",
        "GEMINI_API_KEY_4": "key-four-ddd"}


def _auth(msg="API key not valid. Please pass a valid API key."):
    return litellm.AuthenticationError(message=msg, llm_provider="gemini", model=MODEL)


def _rate(msg="Resource has been exhausted (e.g. check quota). GenerateRequestsPerMinutePerProjectPerModel"):
    return litellm.RateLimitError(message=msg, llm_provider="gemini", model=MODEL)


def _bad_request(msg):
    return litellm.BadRequestError(message=msg, llm_provider="gemini", model=MODEL)


class Msg:
    content = '{"answer": "ok"}'


class Resp:
    choices = [type("C", (), {"message": Msg()})()]
    usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 5})()


def _client(tmp_path, monkeypatch, behaviour, env=KEYS, **llm_over):
    """behaviour(api_key, n) -> None (success) or an exception to raise. Records the keys used."""
    for k in list(KEYS) + ["GEMINI_API_KEYS", "GOOGLE_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    used: list[str] = []

    async def fake(**kw):
        used.append(kw["api_key"])
        e = behaviour(kw["api_key"], len(used))
        if e is not None:
            raise e
        return Resp()

    monkeypatch.setattr(litellm, "acompletion", fake)
    cfg = Config.model_validate(config_dict(tmp_path, llm={"key_cooldown_s": 0.01, "retry_max_wait_s": 0.05,
                                                           "max_retries": 2, **llm_over}))
    client = LLMClient(cfg.llm, tmp_path / "out", use_cache=False)
    return client, used


def _call(client, i=0):
    return asyncio.run(client.call_json("test", MODEL, "s", f"u{i}", ReferenceOut))


def test_keys_from_env_forms():
    env = {"GEMINI_API_KEY": "a", "GEMINI_API_KEY_2": "b", "GEMINI_API_KEY_4": "d", "GEMINI_API_KEYS": "e, a ,f",
           "GOOGLE_API_KEY": "g"}
    assert keys_from_env("gemini", env) == ["a", "b", "d", "e", "f", "g"]
    assert keys_from_env("gemini", {"GEMINI_API_KEY_3": '"x"'}) == ["x"]       # quotes stripped
    assert keys_from_env("openrouter", {"OPENROUTER_API_KEY_2": "o"}) == ["o"]
    cfg = Config.model_validate({"models": {"solvers": [MODEL, "gemini/b"]}, "allow_single_vendor": True})
    assert missing_keys(cfg, {"m1"}, {"GEMINI_API_KEY_3": "x"}) == []


def test_classify():
    assert classify_error(_auth()) == DEAD
    assert classify_error(_bad_request("API key expired. Please renew the API key.")) == DEAD
    assert classify_error(_bad_request("Request had invalid authentication credentials. ACCESS_TOKEN_EXPIRED")) == DEAD
    assert classify_error(_rate()) == RATE
    assert classify_error(_rate("Quota exceeded for metric GenerateRequestsPerDayPerProjectPerModel-FreeTier"))         == MODEL_OFF
    nf = litellm.NotFoundError(message="models/gemini-x is not found for API version v1beta", llm_provider="gemini",
                               model=MODEL)
    assert classify_error(nf) == MODEL_OFF
    assert classify_error(_bad_request("Invalid JSON payload: unknown field")) is None
    real = ('{"error": {"code": 429, "message": "You exceeded your current quota, please check your plan and billing '
            'details. Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, '
            'limit: 20, model: gemini-3.6-flash", "details": [{"quotaId": '
            '"GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}}')
    assert classify_error(_rate(real)) == MODEL_OFF            # per model per day, not the whole key
    assert classify_error(_bad_request("Your account has insufficient credits; add billing")) == DEAD
    assert classify_error(litellm.Timeout(message="timed out", model=MODEL, llm_provider="gemini")) is None


def test_dead_key_is_skipped_for_the_rest_of_the_run(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: _auth() if k == "key-one-aaaa" else None)
    p, meta = _call(client, 0)
    assert p is not None and meta["status"] == "ok"
    assert used == ["key-one-aaaa", "key-two-bbbb"]
    _call(client, 1)
    assert used[-1] == "key-two-bbbb" and used.count("key-one-aaaa") == 1    # never retried
    st = client.key_status()["gemini"]["per_key"]
    assert st[0]["disabled"] and "API key not valid" in st[0]["reason"] and not st[1]["disabled"]


def test_rate_limited_key_rests_and_next_key_takes_over(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: _rate() if (k == "key-one-aaaa" and n == 1) else None)
    p, _ = _call(client)
    assert p is not None and used == ["key-one-aaaa", "key-two-bbbb"]
    st = client.key_status()["gemini"]["per_key"]
    assert st[0]["rate_limits"] == 1 and not st[0]["disabled"]


def test_single_key_waits_out_its_cooldown(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: _rate() if n <= 2 else None,
                           env={"GEMINI_API_KEY": "only-key-zzz"})
    p, _ = _call(client)
    assert p is not None and used == ["only-key-zzz"] * 3


def test_all_keys_dead_fails_the_call_not_the_run(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: _auth())
    p, meta = _call(client)
    assert p is None and meta["status"] == "error"
    assert "all 4 gemini keys failed" in meta["error"]
    assert len(used) == 4
    p, meta = _call(client, 1)                  # later calls fail fast, no new requests
    assert p is None and len(used) == 4


def test_endless_rate_limits_give_up(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: _rate())
    p, meta = _call(client)
    assert p is None and meta["status"] == "error"
    assert len(used) == 2 * 4 + 1               # max_retries x keys, then give up


def test_round_robin_spreads_calls(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: None, key_rotation="round_robin")
    for i in range(4):
        _call(client, i)
    assert used == ["key-one-aaaa", "key-two-bbbb", "key-three-cc", "key-four-ddd"]


def test_failover_sticks_to_one_key(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: None)
    for i in range(3):
        _call(client, i)
    assert used == ["key-one-aaaa"] * 3


def test_transient_errors_retry_the_same_key(tmp_path, monkeypatch):
    def b(k, n):
        return litellm.InternalServerError(message="backend error", llm_provider="gemini", model=MODEL) if n == 1 \
            else None
    client, used = _client(tmp_path, monkeypatch, b)
    p, _ = _call(client)
    assert p is not None and used == ["key-one-aaaa", "key-one-aaaa"]


def test_key_values_never_logged(tmp_path, monkeypatch):
    def b(k, n):
        return _auth(f"API key not valid: https://generativelanguage.googleapis.com/v1?key={k}")
    client, _ = _client(tmp_path, monkeypatch, b)
    _, meta = _call(client)
    text = (tmp_path / "out" / "llm_calls.jsonl").read_text() + json.dumps(client.key_status()) + meta["error"]
    for v in KEYS.values():
        assert v not in text
    assert redact("x?key=abcdefghij&y=1") == "x?key=***&y=1"
    assert redact("Authorization: Bearer sk-or-v1-123456") == "Authorization: Bearer ***"


def _not_found():
    return litellm.NotFoundError(message="models/x is not found for API version v1beta", llm_provider="gemini",
                                 model=MODEL)


def test_model_unavailable_on_one_key_only(tmp_path, monkeypatch):
    """Key 1 cannot serve model A (but can serve B): A moves to key 2, B stays on key 1."""
    seen = []

    async def fake(**kw):
        seen.append((kw["api_key"], kw["model"]))
        if kw["api_key"] == "key-one-aaaa" and kw["model"] == "gemini/model-a":
            raise _not_found()
        return Resp()

    for k in KEYS:
        monkeypatch.setenv(k, KEYS[k])
    monkeypatch.setattr(litellm, "acompletion", fake)
    cfg = Config.model_validate(config_dict(tmp_path, llm={"key_cooldown_s": 0.01, "retry_max_wait_s": 0.05}))
    client = LLMClient(cfg.llm, tmp_path / "out", use_cache=False)
    for i, m in enumerate(["gemini/model-a", "gemini/model-b", "gemini/model-a", "gemini/model-b"]):
        p, meta = asyncio.run(client.call_json("t", m, "s", f"u{i}", ReferenceOut))
        assert p is not None
    assert seen == [("key-one-aaaa", "gemini/model-a"), ("key-two-bbbb", "gemini/model-a"),
                    ("key-one-aaaa", "gemini/model-b"), ("key-two-bbbb", "gemini/model-a"),
                    ("key-one-aaaa", "gemini/model-b")]
    st = client.key_status()["gemini"]["per_key"][0]
    assert not st["disabled"] and list(st["models_unavailable"]) == ["gemini/model-a"]


def test_model_unavailable_everywhere_fails_cleanly(tmp_path, monkeypatch):
    client, used = _client(tmp_path, monkeypatch, lambda k, n: _not_found())
    p, meta = _call(client)
    assert p is None and meta["status"] == "error" and "keys failed for" in meta["error"]
    assert len(used) == 4
