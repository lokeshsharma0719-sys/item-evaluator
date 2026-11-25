"""The three metrics. Each module has `collect` (async, makes LLM calls, returns raw records) and `score`
(deterministic, from raw records only, so `rescore` needs no network)."""
