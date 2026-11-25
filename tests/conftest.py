from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FIXTURE = ROOT / "tests" / "fixtures" / "items_small.csv"


def config_dict(tmp_path: Path, **over) -> dict:
    d = {
        "models": {
            "solvers": ["openrouter/va/s1", "openrouter/vb/s2", "openrouter/vc/s3"],
            "verifier": [],
            "judges": ["openrouter/va/j1", "openrouter/vb/j2"],
            "mutator": "openrouter/vc/mut",
            "mutation_validator": "openrouter/vd/val",
            "topic_classifiers": ["openrouter/va/t1", "openrouter/vb/t2"],
        },
        "llm": {"cache_dir": str(tmp_path / "cache"), "max_cost_usd": 100.0, "max_concurrency": 4},
        "judge_mutation": {"judge_runs": 2, "paraphrase_controls": 1},
    }
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    return d


def write_config(tmp_path: Path, **over) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(config_dict(tmp_path, **over)), encoding="utf-8")
    return p


@pytest.fixture
def fixture_csv() -> Path:
    return FIXTURE
