# UHD eval report reference

Source of truth: `projects/hipdnn/tools/uhd_gen/evaluate.py` (`_build_report`,
`_holdout_integrity`) and `projects/hipdnn/tools/uhd_gen/README.md` §"`eval_report.json`"
in the checkout the model was trained with. Schema `uhd_gen.eval_report/1`. The same report
is written for the held-out score (beside the model) and every `uhd_gen evaluate` run.

## Metric definitions (RFC 0019.13 §11.1–§11.2)

For problem `p` with measured candidate set `V(p)`, `v*(p)` is the best measured
candidate (the oracle) and `v̂(p)` the model's first choice, both in the objective's direction.

| Metric | Definition |
|---|---|
| top-1 regret | `1 − t(v̂)/t(v*)` under `objective: max`, `t(v̂)/t(v*) − 1` under `min`; non-negative; mean, p50, p95, max |
| regret tail | fraction of problems with regret above the threshold (default 0.05) |
| top-k recall | fraction of problems whose oracle is in the model's top k, k = 1, 3, 5 |
| per-regime regret | mean top-1 regret grouped by the corpus regime column; §11.2's primary form |

A **problem** is `(benchmark, device)`, `device` being the device's kind (a hash of its
properties): the same graph on two kinds of card is two problems, on two GPUs of one kind one.
The split holds out problems, not rows, by a seeded SHA-256 of the problem key.

## Keys

| Key | Read it for |
|---|---|
| `metric`, `target`, `objective` | the UHD's `score.metric` (null for a metric-less ranker) and the column/direction regret is in |
| `corpus` | rows and problems |
| `grouping` | problem-identity columns; `degraded: true` means device identity was missing |
| `split` | method, seed, `eval_fraction`, train/eval problem counts, evaluated keys |
| `slice.exhaustive` | false: `v*(p)` is the best **measured**, so regret is a lower bound |
| `exclusions` | rows dropped and why (no usable measurement, single-candidate problems, non-positive oracle) |
| `metrics.problems_scored` | the denominator behind every figure |
| `metrics.deterministic_catalog` | true when nothing was scored because every problem had one candidate |
| `metrics.top1_regret` | mean / p50 / p95 / max |
| `metrics.regret_tail` | `threshold`, `problems`, `fraction` |
| `metrics.topk_recall` | `strict` (exact oracle row), `tie_aware` (within the tie band), `trivial` (problems with ≤ k candidates, a hit whatever the model does) |
| `metrics.two_stage` | grouped models only: `group_regret` + `in_group_regret` sum to top-1 regret |
| `metrics.per_regime`, `metrics.per_regime_status` | per-regime means, or why there are none |
| `metrics.calibration` | calibrated scores (every engine predictor, L1): rows, mean absolute error, mean absolute relative error, signed bias, RMSE, in the metric's unit |
| `metrics.immediate_selection` | L1: regret of the engine the predictions pick against the best measured engine, over problems where more than one engine was measured; `problems_compared: 0` when only one was |
| `per_problem`, `per_row` | with `--include-per-problem`: each problem's pick and each row's measured vs predicted value, keyed `[benchmark, device]` |
| `metrics.references` | `oracle`, `static_order`, `random`, each with the same `top1_regret`/`regret_tail`/`topk_recall`/`per_regime` block |
| `ties` | `rel_tolerance` (default 1%), `sigma`, `noise_band_applied` |
| `holdout_integrity` | `held_out`, `COMPROMISED`, or `unknown`, with `detail` |
| `not_implemented` | §11.2 weighted aggregates, §11.3 extrapolation splits, §11.4 MUST 4 (previous UHD) and item 5 (scoring time) |
| `warnings` | every loud condition, in print order |

## Warnings (prefix is stable; the rest is numbers)

| Prefix | Meaning | Action |
|---|---|---|
| `MODEL DOES NOT BEAT STATIC ORDER` | §11.4 MUST 2: mean regret not better than the engine's declared priority/id order | do not promote |
| `REGIME REGRESSION AGAINST STATIC ORDER` | §11.4 MUST 3: at least one regime is worse than static order | hold or retrain for that regime |
| `HELD-OUT SLICE COMPROMISED` | evaluated problems are among the recorded training problems | retrain on the train slice |
| `NO HELD-OUT SLICE` | `--eval-fraction 1.0` scored the training problems | re-evaluate with the recorded fraction |
| `Full supplied corpus evaluated; training overlap is reported separately.` | an engine predictor scored on every row it was given: the normal shape when those rows are the slice (`evaluation: eval_set`) | none when `holdout_integrity` is `held_out`; otherwise the rows were not held out |
| `DEGRADED PROBLEM GROUPING` | problems identified by `benchmark` alone | re-collect with device identity |
| `PARTIAL DEVICE IDENTITY` | some rows carry no device | treat oracle as possibly cross-device |
| `NO MEASUREMENT NOISE` | no `stddevMs` column; tie-aware recall uses the relative tolerance only | note it |
| `CALIBRATED SCORE IS BIASED` | calibrated score over- or under-predicts on picked candidates | fix before cross-engine use |

§11.4 MUST 4 (regression against a previous UHD for the same engine and arch) and
item 5 (scoring time against `native`) are listed under `not_implemented`, not
warned. The skill does the MUST 4 comparison itself, scoring the incumbent model on the same rows.

Reference points, explicitly not thresholds (§11.4): p95 top-1 regret under 10%;
calibrated scores: mean relative absolute error under 15%, |signed bias| under 5%.
