# item_metrics: item quality metrics calculator

A standalone tool that reads a CSV of assessment items and computes three quality metrics with several LLMs.
All calls go through LiteLLM, to OpenRouter (`openrouter/<vendor>/<model>`) or Gemini (`gemini/<model>`). The LLMs return answers and verdicts as JSON; every number is computed in Python.

| Quality | Metric | Headline |
|---|---|---|
| Accuracy | M1: blind multi-solver key consensus, planted-error audit, option-level ambiguity | Key Consensus Rate, Planted-Error Recall, Ambiguity Rate |
| Consistency | M2: judge accuracy on an answer-mutation set | Judge Balanced Accuracy |
| Faithfulness | M3: blind topic classification | Blind Topic Accuracy (raw and chance-adjusted) |

Each metric also gets a composite in [0, 1] (weighted geometric mean, 1 is best) with a bootstrap 95% interval.

This project imports nothing from the rest of the repository. Its only inputs are the CSV, an optional config
file and optional human-label CSVs.

## Setup

```bash
cd item_metrics
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Windows; use .venv/bin/python elsewhere
cp .env.example .env                                  # then set the key(s) your config uses
cp config.example.yaml config.yaml                    # OpenRouter or mixed; fill in real model ids
# or, with only a Gemini key:
cp config.gemini.example.yaml config.yaml
```

| Model string | Key in `.env` | Vendor (for the diversity rule) |
|---|---|---|
| `openrouter/<vendor>/<model>` | `OPENROUTER_API_KEY` | `<vendor>` |
| `gemini/<model>` (Google AI Studio) | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `google` |

Every role with several models must span at least two vendors. Mixing `gemini/` and `openrouter/` models
satisfies this. The mutation validator must differ from the mutator and every judge. `generator_model` (the
model that wrote the items) may not appear in a checker role: that is a warning, or an error with `--strict`.
`run` checks that the key of every provider in use is set.

**Several keys per provider.** Put up to nine keys in `.env` as `GEMINI_API_KEY`, `GEMINI_API_KEY_2` ...
`GEMINI_API_KEY_9`, or as one comma-separated `GEMINI_API_KEYS=k1,k2,...`. The same forms work with `OPENROUTER_`.
On each error the key is sorted into one of three cases:
- *Dead* (invalid, expired, revoked, no permission, daily quota or credit used up): skipped for the rest of the
  run.
- *Rate limited* (429 or per-minute quota): rested for `key_cooldown_s`, doubling per strike up to
  `retry_max_wait_s`, while the next key is used at once. When every key is resting, the run waits for the first
  one to recover.
- *Anything else* (timeouts, 5xx): retried on the same key.

If every key of a provider is dead, its calls fail and are recorded, but the run still finishes.
`key_rotation: failover` (the default) uses one key until it fails. `round_robin` spreads calls over all keys,
which adds their per-minute quotas together when the keys come from different Google projects. The manifest
(`llm.api_keys`) and the console report calls, rate limits and disabled keys by number (`gemini key #2`). Key
values are masked in every log and error message.

**Gemini only.** With just a Gemini key, every model is from one vendor, so the two-vendor rule cannot hold.
Set `allow_single_vendor: true` (as `config.gemini.example.yaml` does) to turn it into a warning. The manifest
records `single_vendor: true` and `report.md` adds a note: agreement-based numbers (key consensus, the ensemble
judge, the majority topic) are weaker when every checker comes from one vendor, because their mistakes tend to
coincide. Using different Gemini models per role (pro, flash, flash-lite) still keeps them from being one model.
Free-tier keys have low per-minute quotas, so the Gemini example uses `max_concurrency: 2` and
`retry_max_wait_s: 65`.

## Commands

```bash
python -m item_metrics validate --input items.csv [--config config.yaml] [--out out]
python -m item_metrics estimate --input items.csv --config config.yaml --metrics m1,m2,m3
python -m item_metrics run      --input items.csv --config config.yaml --out out/run1 \
                                --metrics m1,m2,m3 [--limit 20] [--no-cache] [--strict] [-v]
python -m item_metrics rescore  --out out/run1 [--human-labels reviewed.csv] [--key-labels keys.csv] [--config c.yaml]
```

- `validate` and `estimate` make no LLM calls and need no API key.
- `run` exits with 0 on success (even when quality is poor), 1 on an invalid config or a missing key, and 2 when
  the run stopped at `llm.max_cost_usd` (partial results are still written).
- `rescore` recomputes every metric from `out/run1/raw/*.jsonl` with no calls. Use it after filling `human_valid`
  in `m2_mutations_for_review.csv`, after labelling keys (`item_id, human_key_correct`), or with a new config
  (thresholds, weights).
- Responses are cached on disk (`llm.cache_dir`, keyed by model, prompts, temperature and prompt version), so
  re-running costs nothing until a prompt or model changes.

## Input CSV

One row per item. JSON cells hold JSON text; unknown columns are kept in `extra` and ignored.

| Column | Needed for | Notes |
|---|---|---|
| `item_id`, `format`, `stem` | all | format: `mcq_4`, `numeric`, `staged`, `short_text`, `rubric_task`, `code` |
| `options` | MCQ | `[{"id":"A","text":"..."}, ...]`, or columns `option_A` ... `option_D` |
| `answer_key` | keyed formats | MCQ option id; numeric `{"value":1000,"unit":"W/m^2","tolerance":0.02}` (a bare `1000 W/m^2` also works) |
| `stages` | staged | `[{"stage_id":"S1","prompt":"...","answer_key":...}, ...]`, at least 2 |
| `rubric` | short_text, rubric_task | `[{"criterion_id":"c1","text":"...","weight":2,"layer":"L3","anchor_pass":"...","anchor_fail":"..."}]` |
| `model_answer` | M2 | generated by the mutator when missing |
| `topic` | M3 | the syllabus topic the item was written for |
| `course_code`, `course_title`, `syllabus_topics` | M3 | `syllabus_topics` is pipe-separated; the union over a course's rows is used |
| `skill`, `bloom_level`, `difficulty_band`, `source` | report breakdowns | |
| `explanation`, `hints`, `textbook_ref` | never sent to solvers or classifiers | |
| `human_key_correct` | M1 ground truth | `true` / `false` |

`column_map` in the config renames columns of an existing export (`{csv_name: schema_name}`). A rename that
would create two columns with the same name is refused. The assessment engine's `questions.csv` already uses
`item_id`, `format`, `stem`, `option_A..D`, `answer_key` and `stages` (with `id` / `key`, which are accepted).
It has no `topic` or `rubric` column (rubrics are in `rubrics.csv`), so M2 and M3 need those columns added first.

Numeric tolerance is relative (0.02 = 2%) and is capped at `numeric.tolerance_cap`. Units are compared with
`pint` (`W/m^2` = `0.001 kW/m^2`).

## What each metric does

**M1 (accuracy).** Every solver solves every keyed item blind (no key, no explanation, no distractor tags), with
MCQ options shuffled per item and solver. A solver matches when its answer equals the key (MCQ) or is within
tolerance after unit conversion (numeric; staged needs the final value and every stage). Statuses:
`full_agree`, `majority_agree`, `split`, `all_disagree`, `solvers_agree_on_other` (the strongest sign of a
wrong key). Planted-error recall re-checks the stored answers against wrong keys planted into `full_agree` items
(each distractor; the key times 1 ± δ). It needs no extra calls, and it is an upper bound. The optional verifier
shows the key and measures sycophancy.

**M2 (consistency).** For each rubric item: a reference answer, one minimal mutation per criterion and
paraphrases, written by the mutator and checked by a separate validator model. Every judge grades every
variant `judge_runs` times. A `pass` counts only if its quote appears verbatim in the answer. Criteria that a
majority of judges fail on the reference are reported as rubric defects and excluded. The tool reports
sensitivity, specificity and balanced accuracy per criterion, plus collateral damage, paraphrase flips, score
error, Cohen's kappa against the expected labels, Krippendorff's alpha across judges and a majority-vote
ensemble. With two judges, a majority means both.

**M3 (faithfulness).** Every classifier sees the item content and the course's topics (shuffled, plus `NONE`),
but not the assigned topic. Majority-vote top-1 accuracy is reported raw and chance-adjusted (chance =
1 / (K + 1)), with and without items that name their topic in the stem. The tool also reports top-k, MRR,
off-syllabus rate, confusion pairs and inter-classifier kappa.

## Output (`out/run1/`)

`run_manifest.json` (models, prompt hashes, config, input hash, library versions, cost, `budget_stopped`),
`summary.json`, `report.md`, `invalid_rows.csv`, `m1_*.csv`, `m2_*.csv`, `m3_*.csv`, `raw/*.jsonl` (every parsed
LLM output, so `rescore` needs no calls), `llm_calls.jsonl` and `parse_failures.jsonl`.

## Cost tracking

The cost of each call is taken from LiteLLM, then from the cost OpenRouter reports in its usage data, then from
the vendor's price for the bare model name. Gemini models are priced directly from LiteLLM's bundled table. Calls with no known price are counted as `cost_unknown_calls` in the
manifest, and the run prints a warning. The `max_cost_usd` stop undercounts those calls.

## Tests

```bash
.venv/Scripts/python -m pytest -q
```

The tests are offline: `tests/fake_llm.py` replaces the network backend, which sits under the disk cache.
`tests/fixtures/items_small.csv` holds 8 items covering every scored format. One MCQ has a deliberately wrong key
and one item names its topic in the stem.

## Status

The offline suite passes (68 tests).

**First real run (2026-11-26).** The 56-item CSV in `data/items_for_metrics.csv` was run with
`config.items.yaml`, on four free-tier Gemini keys. Results are in `out/items_run/`; start with `report.md`.
- **M3** finished for all 56 items. Blind topic accuracy was 0.857 (0.810 without the items that name their topic
  in the stem). The M3 composite was 0.865, with a 95% interval of 0.766 to 0.948.
- **M1** finished for 1 of 52 keyed items. That item's key is wrong: all three solvers agree on 2503.5 W, and the
  key says 1712 W.
- **M2** did not run.

The free tier allows 20 requests per model per key per day, and the full run needs about 1,420 calls. To finish,
re-run the same command on later days (finished calls come from the cache), or use a paid key:

```bash
python -m item_metrics run --input data/items_for_metrics.csv --config config.items.yaml --out out/items_run
```
