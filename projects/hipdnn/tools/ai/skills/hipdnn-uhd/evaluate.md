# Evaluate

A UHD is trained by pointwise regression, but the runtime decision is "pick the best kernel
(or engine) for this problem". Regression error can improve while that pick gets worse, so a
model is judged on problems it never saw (RFC 0019.13 §11,
`projects/hipdnn/docs/rfcs/0019.13_TrainingMethodology.md`). Field meanings and the warning
list are in [reference.md](reference.md).

| Score | Where | Question |
|---|---|---|
| held-out | `<model dir>/eval_report.json`, written by `uhd_gen generate` | how well does it pick on its evaluation slices: graphs reserved when its sets were made, never trained on |
| ad-hoc | `uhd_gen evaluate --output <file>` | how does it do on rows you name: a later collection, another device, the slice rows again |

Every set reserves a tenth of each regime as its evaluation slice ([data.md](data.md)). Every
model from the same sets is scored on the same graphs, so rounds compare.

The §11.4 metrics warn; they do not block training. Validation ([validate.md](validate.md))
supplies the installed-model evidence; this page makes the quality judgement explicit.

## Workflow

1. The model directory: `models[i].model_dir` in its run's `generation_manifest.json`. That
   manifest also says how it was scored (`evaluation`): `eval_set` (the slice ids passed as
   `--eval-benchmarks`, recorded in `eval_benchmarks`), `holdout` (`uhd_gen`'s seeded problem
   split, where no slice existed) or `recall` (a closed shape set). `models[i]` records
   `training_problem_keys` and `eval_problem_keys`; the two must be disjoint for anything
   but recall.
2. `eval_report.json`, in this order:
   1. `holdout_integrity.status`: `held_out`, `unknown` or `COMPROMISED`;
   2. `warnings[]`, verbatim;
   3. `metrics.problems_scored`;
   4. a **catalog ranker (L2)**: `metrics.top1_regret` (mean, p50, p95, max) against
      `metrics.references.static_order` and `.random`; `metrics.per_regime` against
      `references.static_order.per_regime`; `metrics.regret_tail`; `metrics.topk_recall`;
      `metrics.two_stage` for a grouped ranker; `metrics.deterministic_catalog`;
   5. an **engine predictor (L1)**: `metrics.calibration` (mean absolute error, mean
      absolute relative error, signed bias, RMSE in the metric's unit) and
      `metrics.immediate_selection` (regret against the best measured engine, when several
      engines were measured on the same problems);
   6. `split` (seed, eval fraction, problem counts), `metric`, `target`, `objective`. A model
      scored on its slice (`eval_set`) shows `split.method: full_corpus`, `eval_fraction: 1.0`
      and `train_problems: 0`, with the warning `Full supplied corpus evaluated; training
      overlap is reported separately.`: the scored rows are the slice alone, and
      `holdout_integrity: held_out` is the evidence that none was trained on.
3. **Score on other rows**, with the tools of the hipDNN `build` the model was trained with:
   ```bash
   python3 -m uhd_gen.dataset export --from <dataset dir> --out rows.json \
     (--contribution <ids> | --purpose eval) [--metric <metric>]
   python3 -m uhd_gen evaluate --input rows.json --model-dir <model dir> \
     --eval-fraction 1.0 --feature-evaluator <build>/bin/hipdnn_uhd_features \
     --include-per-problem --output <report>
   ```
   To score the slice rows, export the model's training contributions and keep the rows
   whose `benchmark` is in its slice. Rows the model trained on are recall, not evaluation:
   check them against `training_problem_keys` and say so if any are present.
   `--additional-model-dir` ranks another model on the same rows.
4. **The previous model** (§11.4 MUST 4, which `uhd_gen evaluate` does not compute): the
   incumbent -- the model the checkout already binds for the same engine, arch, role and
   metric, if any -- scored on the same rows (`--model-dir` on its directory in the
   checkout, or `--additional-model-dir`). Find it by its id: the UUID the UED's role map
   names for the arch key and metric (an opaque engine: the id its binding declares), then
   the `*.uhd.json` with that `id` in the checkout's descriptor sources; its directory is the
   model directory. An L1 incumbent installed without the training manifest `uhd_gen` writes
   beside a model (`train_manifest.json`) is refused by `uhd_gen evaluate` (`binding must be
   a JSON object`); compare it through Stage V's paired accuracy instead
   ([validate.md](validate.md)), which asks both models on the same graphs. A `native` L2
   incumbent has no model to score offline. Unless
   the incumbent's own manifest lists its training graphs, whether it trained on the slice
   is unknown: say the comparison may favour it.

## Decision rules

Decide one of **promotable**, **hold**, **retrain** or **grow**, and say which rule decided it.

- `holdout_integrity` `COMPROMISED` → **retrain**: the model scored problems it trained on.
  `unknown` → treat every figure as an optimistic bound and say so.
- `MODEL DOES NOT BEAT STATIC ORDER` (L2) → **not promotable**: a ranker that does not beat
  the priority/id ordering it replaces is "the one case where shipping is almost certainly
  wrong" (§11.4).
- `REGIME REGRESSION AGAINST STATIC ORDER` (L2) → **hold** unless the user confirms the
  losing regime is not a workload this engine targets; otherwise grow that regime
  ([data.md](data.md), sizing and `--regime-quotas`).
- A small slice → **retrain with a larger set** before trusting a small margin: §11.1 puts
  the problem count beside every figure. A few percent on a few hundred problems is noise.
- `deterministic_catalog: true` → the engine offers one candidate per problem; nothing for
  an L2 to rank. Train its L1 instead.
- Low `topk_recall.strict` with high `tie_aware` → misses within measurement noise that cost
  nothing. Both low with a high `regret_tail.fraction` → the misses are real. `trivial` close
  to `strict` → recall is a tautology of small candidate sets; rely on regret.
- `slice.exhaustive: false` (the normal case) → regret is against the best **measured**
  kernel, a lower bound; state it as optimistic.
- Calibrated scores (L1): bias, not spread, skews cross-engine selection. Read
  `metrics.calibration.signed_relative_bias` directly: `CALIBRATED SCORE IS BIASED` is raised
  only from the candidates a model picks, so it never fires for an engine measured alone. Reference points, not thresholds (§11.4): mean relative
  absolute error under 15%, |signed bias| under 5%; for rankers, p95 top-1 regret under 10%.

**Promotable** = holdout `held_out`, no `DOES NOT BEAT STATIC ORDER`, any regime regression
accepted by the user, and not worse than the incumbent on the same graphs. Installing it
then needs a passing Stage V ([validate.md](validate.md)). A model trained on one product serves every
product of its arch once installed; say which products it was measured on.

## Report

Model directory, engine, device, role, metric/objective, the dataset and contributions it
trained on; how it was scored and on how many problems; L2: mean/p95 regret for the model,
static order and random, regret tail, recall strict/tie-aware, regimes losing to static
order; L1: MAE, relative error, bias, RMSE; the incumbent on the same rows; every
warning verbatim; the verdict and the rule that decided it.

## Common failures

| Symptom | Fix |
|---|---|
| no `eval_report.json` | the generate run did not finish: [triage.md](triage.md) |
| `per_regime` null | the rows carry no regime column; the aggregate is the whole report, weaker than §11.2 asks |
| `NO HELD-OUT SLICE` | scored with eval fraction 1.0 (recall); expected only for a closed shape set |
| `DEGRADED PROBLEM GROUPING` | rows lack device identity; regret mixes devices, re-collect |
| regret negative | the objective contradicts `score.metric` (`tflops` is max, `time` is min) |
