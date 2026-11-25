# Item quality report

> **Not at temperature 0:** `gemini/gemini-3.6-flash` reject the temperature setting, so their outputs may vary between uncached runs.

> **Single vendor:** some roles use models from one vendor (`allow_single_vendor: true`). Their mistakes are more likely to coincide, so agreement-based numbers (key consensus, ensemble judge, majority topic) are less trustworthy than with mixed vendors.

Input: `data/items_for_metrics.csv` (sha256 `66e19715b9b4`), 56 valid rows, 0 invalid.

## Composite scores (0 to 1, higher is better)

| Score | Value | 95% CI | n | Components |
|---|---|---|---|---|
| M1 Accuracy | 0.046 `[#...................]` | [0.046, 0.046] | 1 | kcr 0.000, no_ambiguity 1.000 |
| M3 Faithfulness | 0.865 `[#################...]` | [0.766, 0.948] | 56 | chance_adj_accuracy 0.848, chance_adj_mrr 0.906 |
| M3 strict (topic not named in stem) | 0.820 `[################....]` | [0.701, 0.929] | 42 | chance_adj_accuracy 0.797, chance_adj_mrr 0.874 |

## Headline metrics

| Quality | Metric | Value | 95% CI | n | Threshold | Status |
|---|---|---|---|---|---|---|
| Accuracy | Key Consensus Rate | 0.000 | [0.000, 0.793] | 1 (small n) | 0.900 | below |
| Accuracy | Planted-Error Recall | n/a |  | 0 (small n) | 0.900 | n/a |
| Accuracy | Ambiguity Rate | 0.000 | [0.000, 0.793] | 1 (small n) | 0.050 | ok |
| Faithfulness | Blind Topic Accuracy | 0.857 | [0.743, 0.926] | 56 | 0.800 | ok |
| Faithfulness | Chance-adjusted accuracy | 0.848 | [0.734, 0.926] | 56 | | |

## M1 Accuracy: multi-solver key consensus

Each keyed item is solved blind by every solver (no key shown). The Key Consensus Rate is the share of items where every solver's answer matches the key. Planted-Error Recall plants wrong keys into trusted items and counts how many the solver check would catch.

Eligible items: 1 (incomplete: 51); gate rule `all`; acceptance rate 0.000.

Status counts: full_agree 0, majority_agree 0, split 0, all_disagree 0, solvers_agree_on_other 1

Planted-Error Recall by format: 

By perturbation: 

Single-solver recall: `gemini/gemini-3.5-flash` n/a, `gemini/gemini-3-flash-preview` n/a, `gemini/gemini-3.6-flash` n/a

Pairwise solver kappa: `gemini/gemini-3.5-flash|gemini/gemini-3-flash-preview` 1.000, `gemini/gemini-3.5-flash|gemini/gemini-3.6-flash` 1.000, `gemini/gemini-3-flash-preview|gemini/gemini-3.6-flash` 1.000

Worst items (full rows in [m1_key_consensus.csv](m1_key_consensus.csv)):

| item_id | item_score | status | ambiguity_rule |
|---|---|---|---|
| core-b3-numeric-7ed8187df3 | 0.000 | solvers_agree_on_other |  |

Keys to review are in [m1_keys_to_review.csv](m1_keys_to_review.csv).

**Limitations.** Solver agreement is not proof of correctness: several models can share a mistake. Planted-Error Recall is computed only on items the solvers already agree on (the easy ones), so it is an upper bound. Blind solving cannot measure sycophancy; the optional verifier does.

## M3 Faithfulness: blind topic classification

Classifiers see the item content and the course's topic list (shuffled, plus NONE) but not the assigned topic. Blind Topic Accuracy is the share of items whose majority-vote top-1 is the assigned topic.

Items: 56; excluding items that name their topic in the stem: 0.809 (n=42). Top-3 accuracy 0.964, MRR 0.912, off-syllabus rate 0.000, classifier kappa (cohen) 0.963.

Per classifier top-1: `gemini/gemini-3.5-flash` 0.857, `gemini/gemini-3-flash-preview` 0.875

By course: CHE200 1.000 (n=3), CHE201 1.000 (n=3), CHE212 1.000 (n=5), CHE213 1.000 (n=1), CHE221 0.500 (n=4), CHE251 1.000 (n=2), CHE261 0.500 (n=4), CHE311 0.750 (n=4), CHE331 0.600 (n=5), CHE352 1.000 (n=2), CHE362 1.000 (n=3), CHE381 1.000 (n=4), CHE391A 1.000 (n=3), CHE453 1.000 (n=3), CHE611 1.000 (n=3), CHE642A 1.000 (n=6), CHE888 0.000 (n=1)

Worst items (full rows in [m3_topic_classification.csv](m3_topic_classification.csv)):

| item_id | item_score | assigned | majority |
|---|---|---|---|
| core-b1-mcq_4-3628123566 | 0.000 | Overview of Chemical Industries | Inorganic bulk chemicals |
| core-b1-mcq_4-78c2a5deac | 0.000 | Laws of Thermodynamics | the second law |
| core-b3-numeric-209db6bb33 | 0.000 | basic concepts in chemical kinetics | isothermal design of batch |
| core-b3-numeric-20c7ea120b | 0.000 | Overview of Chemical Industries | n/a |
| core-b3-numeric-80fb61aabd | 0.000 | Laws of Thermodynamics | the first Law |
| core-b3-numeric-d0d2010a2b | 0.000 | nonelementary homogeneous reactions | isothermal design of batch |
| research-b3-short_text-d2d220fcde | 0.000 | Introduction to scientific Research | Framing of hypothesis and Karl Popper’s falsification principle |
| core-b3-staged-9dcb82a97c | 0.467 | Fluid flow | n/a |
| biz-b2-mcq_4-a7f1101a74 | 1.000 | Introduction to process design and process creation Heuristics for Process Synthesis Cost accounting and capital cost estimation | Introduction to process design and process creation Heuristics for Process Synthesis Cost accounting and capital cost estimation |
| biz-b3-short_text-a8b6a8d23b | 1.000 | Introduction to process design and process creation Heuristics for Process Synthesis Cost accounting and capital cost estimation | Introduction to process design and process creation Heuristics for Process Synthesis Cost accounting and capital cost estimation |

Confusion pairs (>= 2 occurrences) are in [m3_confusions.csv](m3_confusions.csv); they usually show overlapping syllabus topics rather than bad items.

**Limitations.** Accuracy depends on how many candidate topics a course has and how much they overlap. Items that name their topic in the stem are trivially classified; see the strict figures.

