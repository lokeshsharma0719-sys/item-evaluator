"""CLI: python -m item_metrics validate | estimate | run | rescore

Exit codes: 0 success (even if quality is poor), 1 invalid config, 2 budget stop.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Optional

from . import runner


def main(argv: Optional[list[str]] = None, backend: Any = None, embed_backend: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="item_metrics", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="check the CSV; no LLM calls")
    v.add_argument("--input", required=True)
    v.add_argument("--config")
    v.add_argument("--out", default="out", help="where invalid_rows.csv is written")

    e = sub.add_parser("estimate", help="count calls per role and a rough cost; no LLM calls")
    e.add_argument("--input", required=True)
    e.add_argument("--config", required=True)
    e.add_argument("--metrics", default="m1,m2,m3")

    r = sub.add_parser("run", help="run the metrics")
    r.add_argument("--input", required=True)
    r.add_argument("--config", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--metrics", default="m1,m2,m3")
    r.add_argument("--limit", type=int, default=None, help="first N valid rows (smoke tests)")
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--strict", action="store_true", help="generator_model in a checker role is an error")
    r.add_argument("-v", "--verbose", action="store_true")

    s = sub.add_parser("rescore", help="recompute every metric from out/raw without LLM calls")
    s.add_argument("--out", required=True)
    s.add_argument("--human-labels", help="reviewed m2_mutations_for_review.csv (human_valid column)")
    s.add_argument("--key-labels", help="CSV with item_id, human_key_correct")
    s.add_argument("--config", help="override the stored config (e.g. new thresholds)")

    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(a, "verbose", False) else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        if a.cmd == "validate":
            return runner.cmd_validate(a.input, a.config, a.out)
        if a.cmd == "estimate":
            return runner.cmd_estimate(a.input, a.config, runner.parse_metrics(a.metrics))
        if a.cmd == "run":
            return runner.cmd_run(a.input, a.config, a.out, runner.parse_metrics(a.metrics), a.limit, a.no_cache,
                                  a.strict, backend=backend, embed_backend=embed_backend)
        return runner.cmd_rescore(a.out, a.human_labels, a.key_labels, a.config)
    except (ValueError, FileNotFoundError) as e:   # bad config YAML, bad column_map, missing file
        print(f"error: {e}")
        return runner.EXIT_CONFIG


if __name__ == "__main__":
    sys.exit(main())
