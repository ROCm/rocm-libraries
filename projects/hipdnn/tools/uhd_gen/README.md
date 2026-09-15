# UHD Generation Tool

Train and export heuristic models for hipDNN's Universal Heuristic Descriptor (UHD) system.

## Overview

This tool takes benchmark timing data and produces:
1. A trained LightGBM model
2. A FlatBuffer model artifact (`model.bin`) for `TreeDataAdapter`
3. A UHD descriptor (`<stem>.uhd.json`) for the shared plugin-SDK runtime (RFC 0019 §4)

Promotion installs that pair into a descriptor tree. Every model -- L1 included -- is
reached through a reference in its owning UED's role map (RFC 0019 §3.1); a UHD never
names the engine it serves.

## Installation

```bash
cd projects/hipdnn/tools/uhd_gen
pip install -e .
```

## The pipeline

```
  sweep  ->  export-benchmarks  ->  results_import  ->  train  ->  evaluate  ->  promote
   |              |                       |              |           |             |
   |              |                       |              |           |             `- writes the UED's
   |              |                       |              |           |                role/architecture reference
   |              |                       |              |           `- eval_report.json (§11.2/§11.4 regret)
   |              |                       |              `- <stem>.uhd.json + model.bin
   |              |                       `- dataset.parquet: §8.3's checks applied, the
   |              |                          collector's is_valid/skip_reason rewritten as
   |              |                          `error`, tflops/gbs derived from the opmeta
   |              `- §8.3 collection CSV: appendable, resumable, one per shard
   `- ingestor benchmark log
```

```bash
# 1. sweep: run the graphs you care about with benchmark logging on
HIPDNN_LOG_LEVEL=info HIPDNN_LOG_FILE=sweep.log <run your graphs>

# 2. export-benchmarks: log -> the §8.3 collection CSV, one per shard, appendable
python -m uhd_gen export-benchmarks sweep.log -o bench.csv

# 3. results_import: the collected shards -> the published §8.3 dataset. This is where
#    §8.3's checks are applied, where a failed candidate's is_valid/skip_reason becomes
#    the dataset's `error`, and where tflops and gbs are derived from the operation's
#    declaration -- the collector measures times, it does not know an op's flop count.
python -m results_import.importer \
    --csv bench.csv \
    --opmeta ../corpus_gen/operations/matmul.opmeta.json \
    --out dataset.parquet

# 4. train: dataset -> descriptor + model artifact. A collected .csv works too, with a
#    target the CSV carries (a timing column) and none of §8.3's checks applied.
python -m uhd_gen train \
    --input dataset.parquet \
    --features q.M q.N q.K kernel.tile_m kernel.tile_n kernel.tile_k device.cu_count \
    --descriptor-tree ./descriptors --engine hipkernel:gemm \
    --training-arches gfx942 \
    --target tflops \
    --group-by benchmark device \
    --output-dir ./uhd_output \
    --descriptor-name gemm \
    --name "GEMM UHD"

# 5. evaluate: how much worse is the model's pick than the best kernel measured?
python -m uhd_gen evaluate \
    --input dataset.parquet \
    --model-dir ./uhd_output

# 6. promote: install the pair and update only its role/architecture reference
python -m uhd_gen promote \
    --model-dir ./uhd_output \
    --descriptor-tree ./descriptors \
    --engine hipkernel:gemm --arch gfx942
```

**Step 6 is not optional.** The UED's `sort_kernel_catalog` map must name the
model under the target architecture (or `default`). Promotion updates that entry;
other architectures and roles retain their existing models. An unavailable or
incompatible model disables only that model, leaving a valid engine usable with
deterministic priority/descriptor-ID ranking. A broken explicit architecture entry
does not silently select the default model.

### Feature columns use published symbol names

`--features` takes exact published column names without the leading `$`.
For example, `attention_dense.seqlen_q` becomes `$attention_dense.seqlen_q`.
The runtime does not add a synthetic `q.` namespace.

| Source | Example |
|--------|---------|
| Engine-published graph, node, or tensor binding | `attention_dense.seqlen_q`, `q.dims[2]` |
| Per-candidate UKD metadata | `kernel.tile_m`, `kernel.split_k` |
| Device properties | `device.cu_count` |

Use the names returned by candidate enumeration. Renaming columns without changing
the engine's published bindings produces a model the engine cannot evaluate.

### Constant feature columns are kept, and reported

A column with one value across the whole input cannot separate one candidate from
another. `train` detects those before fitting, names each one and its value, and
**keeps** them:

```
WARNING - 2 feature column(s) never vary in bench.csv: kernel.tile_m=128,
device.cu_count=304. They cannot separate one candidate from another, so no tree will
split on them.
WARNING - KEEPING them in features_signature. Constancy is measured against this
corpus, not the pack: a field the sweep failed to cover looks identical to one the
kernels pin ...
```

Keeping is the default because a CSV cannot tell two opposite situations apart. rocKE's
attention kernels bake their geometry in, so the matcher pins 8 of their 14 fields
before ranking begins and those 8 can never vary — dropping them is harmless. But a
column that *does* vary in the world, sampled at one value because the corpus is thin,
reads identically. Dropping there produces a model that cannot generalise on that axis,
and makes `features_signature` — and `features_hash`, the contract the runtime checks —
a function of which problems happened to be swept. The same engine trained on two
corpora would ship two different contracts.

`train_manifest.json` records `requested_features`, `constant_features` (with values)
and `dropped_constant_features`, so the provenance says what never varied whether or not
it was dropped.

- pass **`--drop-constant-features`** only to remove constant model inputs.
  Training and ordinary promotion preserve the UED's authored knobs. Explicit
  `promote --remove-knob NAME` requires a model trained against the intended
  major-revised UED and rejects removal of a field the model still consumes;
- when **two thirds or more** of the requested columns are constant, `train` warns that
  the proportion looks like a thin corpus and points at the input file. The threshold
  sits above the 8-of-14 rocKE shape (57%) on purpose: a warning that fires on every
  normal run is one people learn to ignore;
- when **every** requested column is constant, `train` fails and names each column with
  its value. `--drop-constant-features` does not override this — it changes the
  signature, not the fact that nothing varies. A model over zero varying features scores
  every candidate identically, and shipping one is worse than shipping none: the engine
  ranks by a model that cannot discriminate instead of falling back to its declared
  order.

### `train` arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Path to benchmark CSV/JSON |
| `--features` / `--feature-signature` | One | Exact published columns, or a JSON file containing inline feature expressions |
| `--descriptor-tree` / `--provenance` | One | Descriptor snapshot used for collection, or its recorded identity/revision provenance |
| `--engine` | If ambiguous | UED name or UUID in the descriptor tree |
| `--feature-evaluator` | For expressions | Shared `hipdnn_uhd_features` executable; alternatively set `HIPDNN_UHD_FEATURE_EVALUATOR` or put it on PATH |
| `--target` | No | Target column name (default: `tflops`) |
| `--objective` | No | `max` or `min` (default: `max`). Pass `min` for a cost target such as `latency_ms`, or the runtime will prefer the *worst* kernel. |
| `--score-units` | No | Units the score is expressed in (default: the `--target` column name) |
| `--calibrated` | No | Declare the score cross-engine comparable — RFC 0019 §4.1's `score.calibrated` header, which §11.3 reads when it compares predicted throughput across engines. Off by default; nothing here verifies the claim, but RFC 0019.13 §11.2 pins a calibrated score to `avgTimeMs`, so `--timing-statistic avgTimeMs` is required alongside it. |
| `--timing-statistic` | With `--calibrated` | Which measured timing the target was derived from (`avgTimeMs`, `minTimeMs`, `robustMeanMs`). Recorded in the manifest per §10.5: §11.2 refuses cross-engine comparison between models trained on different statistics, so it has to be readable off the artifact. |
| `--group-by` | No | Columns for GroupKFold CV |
| `--output-dir` | Yes | Output directory |
| `--name` | No | UHD display name |
| `--descriptor-name` | No | Stem for the emitted descriptor (default: `heuristic`), producing `<stem>.uhd.json` |
| `--uhd-id` | No | Reuse this UUID as the descriptor's id instead of minting a fresh one |
| `--num-boost-round` | No | Max boosting rounds (default: 500) |
| `--early-stopping` | No | Early stopping patience (default: 50) |
| `--keep-lgbm` | No | Keep intermediate .lgbm file |
| `--drop-constant-features` | No | Drop constant model inputs, never authored knobs (default: keep inputs) |
| `--training-arches` | No | Architectures the model was trained on, for §9.2 OOD detection |
| `--model-version` | No | Semantic version embedded in the model metadata |

`--uhd-id` makes retraining a no-edit operation: pass the id the engine's UED already
names and the pair is simply overwritten in place. A value that is not a UUID is
rejected before training starts — a typo'd id becomes the descriptor's *identity*, so
the UED would point at an id nothing defines and the engine would load with no
heuristic and no error.

### `promote` arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--model-dir` | Yes | The `train --output-dir` result: one `<stem>.uhd.json` plus its artifact |
| `--descriptor-tree` | Yes | Tree holding the engine's `<name>.ued.json`; searched recursively |
| `--engine` | If ambiguous | The UED's `name` (e.g. `hipkernel:pointwise_model`) |
| `--arch` | Unless unambiguous | Target architecture, or explicit `default`; may be inferred from one training architecture |
| `--role` | No | UED model role (default: `sort_kernel_catalog`) |
| `--remove-knob` | No | Explicit authored knob removal; requires compatible major-revised training provenance |
| `--dry-run` | No | Print the plan, write nothing |

`promote` copies the descriptor and artifact into the UED's directory and updates
`<role>.<arch>`. It preserves other model references and authored knobs.

It validates everything before writing anything, and refuses rather than half-succeed:

- the descriptor must satisfy the canonical schema and its artifact must exist;
  missing or incompatible models otherwise leave runtime selection in fallback;
- with more than one UED in the tree, `--engine` is **required**. Promoting into the
  wrong engine fails twice over: the engine you retrained keeps its old model, and one
  you never touched starts ranking with a model trained for a different kernel set.
  Both load cleanly and report nothing, so this is never guessed;
- replacing a descriptor or artifact still used by another engine, role, or
  architecture is refused. Use distinct artifact and descriptor paths;
- recorded UED, KMD, and applicable UMD identities/revisions must remain compatible.

## `evaluate`: regret against the best kernel that was measured

RMSE on `log1p(target)` is what `train` reports, and it can improve while the model's
*choice* gets worse. `evaluate` measures the choice, per RFC 0019.13 §11.2 and §11.4:

- **top-1 regret** — how much worse the model's pick is than the oracle `v*(p)`, the
  best measured candidate for that problem. `t(v̂)/t(v*) − 1` under `objective: min`,
  `1 − t(v̂)/t(v*)` under `max`; non-negative either way, reported as mean, p50, p95,
  max;
- **regret tail** — the fraction of problems whose regret exceeds 5%;
- **top-k recall** — how often the oracle is in the model's top k, for k = 1, 3, 5;
- **per-regime regret** — the same mean, grouped by the corpus's regime column. §11.2
  makes this the *primary* form: an aggregate hides a model that is excellent on the
  dense middle of the corpus and useless on decode-shaped or prime-dimension problems.
- **§11.4 references** — the same figures for the two things the model has to be read
  against: the **static order** the engine ships (its `priority`/`id` ordering, read
  off the corpus's enumeration order) and **random** choice from `V(p)`, plus the
  oracle's zero. The model's regret is not interpretable alone: §11.4 wants to know
  whether it beats the ordering it replaces. It also warns when it does not, in
  aggregate (MUST 2) or in any one regime (MUST 3).

It writes `eval_report.json` — the artifact §10.4 names — into `--model-dir`.

```bash
python -m uhd_gen evaluate --input bench.csv --model-dir ./uhd_output
```

```
Regret report (0019.13 §11.2, §11.4) -- ./uhd_output/eval_report.json
  target/objective:   minTimeMs (min)
  problems grouped by: benchmark, device
  split:              group_holdout_by_problem, seed 0, 12 eval / 48 train problem(s)
  problems scored:    12
  top-1 regret:       mean 0.7012  p50 0.0000  p95 2.1976  max 2.2109
  regret tail (>5%):  0.4167 (5 problem(s))
  top-1 recall:       strict 0.5833   tie-aware 0.5833
  vs §11.4 references: static order 1.2210   random 1.8043   oracle 0.0000 (mean top-1 regret)
  per-regime regret:  (from column 'regime')
    decode                   mean 1.4025  (6 problem(s))
    prefill                  mean 0.0000  (6 problem(s))
  holdout integrity:  held_out
```

### A problem is `(benchmark, device)`

The same graph on two GPUs is two problems with two different best kernels. Grouped on
`benchmark` alone, the oracle becomes the best kernel on whichever card is faster and
the regret figure is a different quantity — on the demo corpus above, conflating two
devices moved the mean from 0.70 to 0.26.

A corpus exported before the `device` column existed carries it empty on every row.
`evaluate` degrades to `benchmark` alone rather than refusing, and says so on stdout,
in `grouping.degraded`, and in `warnings[]`:

```
!! DEGRADED PROBLEM GROUPING: problems are identified by 'benchmark' ALONE because
no 'device' column in this corpus ... They are not comparable with figures from a
corpus that carries device identity. Re-export from a sweep that logs the `device`
column.
```

### The split holds out problems, not rows

Regret belongs to the evaluation slice (§5.6.4); on training data it is optimistic and
is not the number anyone wants. `evaluate` holds out `--eval-fraction` of the
**problems**, assigned by a seeded SHA-256 of the problem key — reproducible from
`(corpus, --seed)` alone, and independent of row order, so concatenating a log
differently does not move the slice.

Splitting *rows* would put some of a problem's candidates in training and the rest in
evaluation. The evaluation-side oracle would then be the best of a subset, and a
mediocre pick would look correct because the better candidate was not there to compare
against. On the fixture in `tests/test_evaluate.py` that turns a true regret of 3.00
into 2.25; the tests assert the group-aware figure.

The model must not have trained on the evaluation problems, and `evaluate` checks what
it can: if the manifest says the model was trained on the very corpus being scored, the
report says `holdout_integrity: COMPROMISED` and the reason is printed first. The fix
is one extra step:

```bash
# write the training side of the split, then fit on that
python -m uhd_gen evaluate --input bench.csv --model-dir ./uhd_output \
    --emit-train-slice train_slice.csv
python -m uhd_gen train --input train_slice.csv ... --output-dir ./uhd_honest
python -m uhd_gen evaluate --input bench.csv --model-dir ./uhd_honest   # same --seed
```

On the demo corpus that raises the reported mean regret from 0.45 to 0.70: the leak was
worth a third of the number.

### What is excluded, and what is not

§5.6.3 warns that dropping a configuration from the evaluation slice removes it from
the oracle. So only rows that carry no usable measurement are dropped, and every drop
is counted in `exclusions`:

| Excluded | Why |
|----------|-----|
| `is_valid=False` rows | A candidate that never ran has no time and cannot be the best. Its empty timing column would otherwise read as a zero and win every `min`. Only a collected CSV carries the flag. |
| Rows whose target is empty or non-numeric | Same reason, without the flag -- which is how a published dataset spells it, since §8.3 has no validity column and records the failure in `error` instead. |
| Problems with one measured candidate | With nothing to choose between, a correct pick is not evidence; scoring it as regret 0 would dilute the mean. |
| Problems whose oracle value is not positive | Both formulas divide by it, and under `max` the ratio's sense flips. |

Every measured candidate of an evaluated problem stays in `V(p)`.

### Ties within noise

Regret needs no tie rule — it is measured in the target metric, so two kernels a
fraction of a percent apart produce a regret a fraction of a percent from zero, which
is §11.2's stated reason for measuring it that way. **Top-k recall does need one**: it
is a rank test, and it scores the second of two statistically indistinguishable kernels
as an outright miss.

So recall is reported twice. `strict` demands the exact oracle row in the top k.
`tie_aware` accepts any candidate that is tied with the oracle, where tied means either

- within `--tie-rel-tolerance` (default 1%) of the oracle's measured value — unit-free,
  works for either objective, and it is the same quantity the regret column reports, so
  "tied" means exactly "costs less than 1%"; or
- within `--tie-sigma` standard errors of it, using `stddevMs` and `iters`. Applied
  **only** when the target is a millisecond timing (`minTimeMs`, `avgTimeMs`,
  `robustMeanMs`), because `stddevMs` is in milliseconds and widening a TFLOPS
  comparison by it would be a units error. For `avgTimeMs` that band is exact; for
  `minTimeMs` — §8.5's default target — and for `robustMeanMs`, the sample spread is a
  scale for the noise rather than that estimator's own error, so the band is
  approximate and deliberately so: the alternative is no noise notion at all for
  either.

The band needs the columns to be there. §8.3 makes `stddevMs` and `iters` part of the
result envelope, and both `export-benchmarks` and `generate` emit them; a corpus that
drops them turns the band off, and `ties.policy` then names the missing column rather
than blaming the target's units. `evaluate` also warns loudly, because nothing else in
the report changes when the band goes away.

`topk_recall.trivial` records the fraction of problems with no more than k measured
candidates, so a recall@5 of 1.0 on 4-candidate problems is legible as the tautology it
is.

### `evaluate` arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Benchmark CSV/JSON to evaluate on |
| `--model-dir` | Yes | A `train --output-dir` result |
| `--model` | No | Artifact to rank with (default: `model.lgbm` if kept, else the descriptor's `tree_data.artifact` — the file the engine itself loads) |
| `--output` | No | Report path (default: `<model-dir>/eval_report.json`) |
| `--eval-fraction` | No | Fraction of **problems** held out and scored (default: 0.2; `1.0` scores everything and says loudly that the figure is optimistic) |
| `--seed` | No | Seed for the problem-level split (default: 0), recorded in the report |
| `--target` | No | Measured column regret is computed in (default: the manifest's `target`) |
| `--objective` | No | Override the direction read from the descriptor/manifest |
| `--device-column` | No | Column holding device identity (default: `device`) |
| `--regime-column` | No | Regime column for the per-regime table (default: the first of `regime`, `corpus_regime`, `q.regime`, `problem.regime` that is present) |
| `--tie-rel-tolerance` | No | Tie tolerance for tie-aware recall (default: 0.01) |
| `--tie-sigma` | No | Noise-band width in standard errors (default: 2.0) |
| `--regret-tail-threshold` | No | Tail cutoff (default: 0.05, §11.2's 5%) |
| `--include-per-problem` | No | Write every problem's oracle, pick and regret into the report |
| `--emit-train-slice` | No | Write the training side of this split to a CSV |

The **objective is read, never assumed** — from the descriptor, falling back to the
manifest. Both directions are legal and the wrong one inverts every number, so a corpus
that offers neither is an error rather than a guess. Regret is asserted non-negative;
a negative one means the direction is backwards, and `evaluate` fails instead of
printing a plausible small number.

### `eval_report.json`

| Key | Contents |
|-----|----------|
| `schema` | `uhd_gen.eval_report/1` |
| `corpus` | path, row count, problem count |
| `target`, `objective` | what regret was measured in, and in which direction |
| `grouping` | the problem-identity columns, `degraded`, and why |
| `split` | method, unit, seed, fraction, train/eval problem counts, and the evaluated problem keys |
| `slice` | that `V(p)` is what the sweep measured rather than every applicable configuration, so `v*(p)` is a lower bound (§11.1) |
| `exclusions` | counts by reason, plus the policy that produced them |
| `metrics` | `problems_scored`, `top1_regret` (mean/p50/p95/max), `regret_tail`, `topk_recall` (`strict`/`tie_aware`/`trivial`), `per_regime`, `per_regime_status`, and `references` |
| `metrics.references` | §11.4's `oracle`, `static_order` and `random`, each carrying the same `top1_regret`/`regret_tail`/`topk_recall`/`per_regime` block plus a note on how it was derived |
| `ties` | tolerance, sigma, whether the noise band applied, and the policy |
| `model` | artifact, features, what it was trained on, how many rows |
| `holdout_integrity` | `held_out`, `COMPROMISED`, or `unknown`, with the reason |
| `not_implemented` | the parts of §11.2/§11.3/§11.4 this command does not compute |
| `warnings` | every loud condition, in the order printed |
| `per_problem` | with `--include-per-problem`: key, regime, candidate count, oracle, pick, regret, ranks |

`per_regime` is `null` when the corpus has no regime column, and `per_regime_status`
says so — an absent metric someone expected is worse than a stated gap.

`not_implemented` names what is missing rather than leaving a reader to infer it:
§11.2's regime-weighted aggregates (nothing declares weights yet), its calibration
metrics (required only when `score.calibrated` is true), §11.3's leave-one-regime-out
and leave-variants-out splits (both need retraining per fold), §5.6.3's round-0
core versus full slice and steering versus reserved portions (properties of a corpus
collected by the campaign loop, which does not exist yet), and two of §11.4's
obligations — MUST 4's regression check against a previously promoted UHD, which has no
loader here, and item 5's scoring-time comparison, which §11.4 itself notes is blocked
on the §11.6 (B5) harness.

§11.4's static-order reference is read off the corpus: a problem's rows are in the
order the engine enumerated its catalog, which is the `priority`/`id` order Stage 1
ships, so the static pick is the first row carrying a usable measurement. A corpus
re-sorted after collection no longer carries that order, and the reference then
describes a permutation rather than the shipped one; the note in the report says so.
The random reference is an exact expectation over `V(p)`, never a sampled draw, so the
sanity floor does not move between runs.

## Input Format

`--input` takes three forms and the suffix decides, for `train`, `evaluate`, `knobs` and
`merge` alike:

| Suffix | What it is | When |
|--------|------------|------|
| `.parquet` | The dataset `tools/results_import` publishes from collected shards (§8.3) | The route a model anyone ships should come by |
| `.csv` | A collected corpus, read directly | A quick local run |
| `.json` | The same rows as records | Hand-written corpora and fixtures |

**Collection stays CSV; publication is Parquet.** The format that has to survive a
two-day sweep and the format a trainer wants are not the same format. Parquet writes
its footer last, so a run killed mid-flight leaves a file that cannot be read at all,
against §8's requirement that the benchmark step be resumable from a partial result;
and a Parquet file cannot be appended to, so §8.8's "shard outputs merge by appending"
would become a full rewrite. CSV has both properties. So `hipdnn_bench` and
`export-benchmarks` keep writing CSV, and `results_import` reads the merged shards
once, derives `tflops` and `gbs` from the operation's declaration, and writes the
typed dataset:

```bash
python -m results_import.importer \
    --csv shards/*.csv \
    --opmeta ../corpus_gen/operations/matmul.opmeta.json \
    --out dataset.parquet

python -m uhd_gen train --input dataset.parquet ...
```

Prefer the dataset for the reason that argued for Parquet in the first place: it
carries its own column types, so a column that is empty in one shard and populated in
another cannot concatenate to `object` and quietly change what the trainer sees. The
CSV branch is the escape hatch, not the route -- nothing §8.3 specifies is checked on
it (no measurement-or-error rule, no completeness agreement, no candidate-set
comparison across a merge), which is what the importer exists for.

**A failure is spelled differently at the two ends, and the importer is where it is
translated.** The collector writes what the runtime record carries at the moment of
failure: `is_valid=False`, a `skip_reason`, and empty timing columns. §8.3's published
dataset carries no validity flag at all -- a failed candidate is a null measurement plus
a non-empty `error` -- so that no two columns can disagree about whether a row was
measured. `results_import` rewrites the one into the other and then drops `is_valid` and
`skip_reason` as collection bookkeeping, which is why the chain above composes: a sweep
containing a failure is a normal sweep, not a corpus the importer refuses.

`train` drops a row that carries no measurement in either spelling -- the flag, a
non-empty `error`, or a target that is not a finite number -- and logs how many went by
each. The failed rows stay in the dataset, because a candidate that could not run is
information about feature space and §8.3 keeps it; they are excluded at the fit, exactly
as `evaluate` excludes them from the oracle.

**Identity columns are read as text on every route.** `benchmark`, `device`, `graph_id`
and `device_id` name something; nothing computes with them. Left to inference a device
called `0007` becomes the integer 7 from one format and the string `"0007"` from the
other, and one problem becomes two.

Whichever form it arrives in, the corpus must carry:
- Feature columns (problem dimensions, kernel config, device properties)
- Target column (typically TFLOPS or time)

The §11.2 label rule is applied to all three alike: a `--calibrated` model must be
trained with `--timing-statistic avgTimeMs`, and the published dataset earns no
exemption from it.

Example CSV:
```csv
M,N,K,tile_m,tile_n,tile_k,cu_count,tflops
1024,1024,1024,128,128,32,120,50.5
2048,2048,2048,256,128,32,120,75.2
...
```

### Inline computed features and device coverage

Pass `--feature-signature features.json` instead of `--features`:

```json
[
  "$kernel.tile_m",
  "$device.cu_count",
  {"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]}
]
```

Build `hipdnn_uhd_features` and pass `--feature-evaluator` (or set
`HIPDNN_UHD_FEATURE_EVALUATOR`). Training and runtime use the same compiled
descriptor-expression evaluator. Expressions and categorical vocabularies are
part of the feature hash; there is no separate named `derived` block.

An explicit computed expression using a device field that never varied in the
training corpus is rejected. Automatic feature proposals omit such expressions
and retain the raw device field. Variation is recorded in `train_manifest.json`.

### Reproducible generation

`python -m uhd_gen generate --help` describes the combined workflow. It accepts
graph files or corpus directories, enumerates matched candidates through
`hipdnn_bench enumerate`, checks identities and knob tuples during timing, then
trains, evaluates a held-out problem/device split, and promotes.
`--no-promote` validates installation without changing the shipping tree.

`--graphs` takes both serialized forms: hand-written or exported `*.json`, and the
binary FlatBuffers `hipdnn_corpus_gen` writes as `problems/<operation>_<n>.fb`, so a
generated corpus composes with `generate` directly. Directories are searched
recursively for both. Form is decided by content rather than extension, the same way
`hipdnn_bench` decides it, so a renamed file still loads. An ID-less JSON graph is
given a reproducible UUID5 of its canonical content; a serialized graph already
carries its own id and the bench preserves it, so nothing is injected there.

Collection uses STANDARD autotune for each explicitly enrolled candidate; an
internal exhaustive sweep must not substitute a different kernel. Providers that
do not implement enumeration report unsupported, not an empty catalog.
Existing timing-based `is_valid` semantics are unchanged: this workflow does not
establish per-candidate numerical correctness.

Every collected row carries §8.3's envelope, `stddevMs` and `iters` included, so
`evaluate`'s noise band works on a generated corpus rather than being inert on it.

**A failure never destroys the measurements.** Collection is the expensive half of a
run, and §8.7 is explicit that measurements outlive the strategy that requested them,
so a failure anywhere after collection — training, evaluation, an empty holdout,
installation validation — leaves the staging directory in place and names it in the
error. It is only consumed by the rename into `--output-dir` that a successful run
performs, so nothing accumulates from runs that worked. Delete a reported stage once
you no longer need what it measured.

**What the catalog model is scored in.** When the engine publishes `graph.flops`,
`generate` derives `tflops = graph.flops / (avgTimeMs * 1e9)` per candidate and trains
`sort_kernel_catalog` on `tflops`/`max` with `score.calibrated: true` —
`avgTimeMs` because RFC 0019.13 §11.2 pins a calibrated score to the mean. That is what
gives the role the cross-engine standing RFC 0019 §11.1 describes, and with it §11.2's
`B only` ranking row. Without a published work count it falls back to
`robustMeanMs`/`min`/uncalibrated, which is legal (§2.5, §15.1) and ranks this engine's
own catalog just as well, and warns that the score is no longer comparable with another
engine's — Mode B then falls back to the engine's L1 prediction.

`generate` takes the work count from the engine's own published `graph.flops`, which is
the effective count the runtime computed for the graph it just ran. A corpus that was
collected as CSV and published by `results_import` instead carries `tflops` and `gbs`
derived from the operation's `.opmeta.json` declaration and the same `avgTimeMs`. The
two paths agree on the arithmetic and differ only in where the count comes from: a
measured graph knows its own, a CSV row needs its operation to declare one.

### Engine-level immediate predictions

`predict_engine_tflops` trains an engine's **normal untuned performance**, not
the best configuration found by a sweep. Collection builds only that engine's
plan with `global.benchmarking=0`, warms it up, and measures ordinary execution
with HIP events. It does not enumerate configurations or invoke autotune; normal
engine cache behavior is unchanged. Full-graph work and elapsed time determine the
TFLOPS label; unsupported work accounting is not replaced with a guessed label.

The label is `graph.flops / (avgTimeMs * 1e9)`. RFC 0019.13 §11.2 (:2003) requires a
UHD declaring `calibrated: true` to train on `avgTimeMs`, and §10.6.2 repeats it for
this role specifically; L1 always declares it, so the mean is the label and never the
minimum or the robust mean. `robustMeanMs` stays on every corpus row as §8.5's
informational statistic, alongside `stddevMs` and `iters`. Which statistic produced the
label is recorded as `timing_statistic` in `train_manifest.json`, because §11.2 refuses
cross-engine comparison between models trained on different ones.

```bash
hipdnn_bench --graph graph.json --engine-name vendor:gemm \
    --describe-engine-prediction --workspace-limit 67108864
hipdnn_bench --graph graph.json --engine-name vendor:gemm \
    --collect-immediate --workspace-limit 67108864

python -m uhd_gen generate \
    --graphs ./graphs --descriptor-tree ./descriptors \
    --engine vendor:gemm --engine-id <ENGINE-ID> \
    --role predict_engine_tflops --arch gfx942 \
    --workspace-limit 67108864 \
    --features graph.flops device.cu_count \
    --output-dir ./immediate-model

HIPDNN_DESCRIPTOR_PATH=./descriptors hipdnn_bench \
    --graph graph.json --engine-name vendor:gemm --predict-engine \
    --workspace-limit 67108864
```

Choose features from the description's published graph, device, and constraint
fields. L1 signatures cannot consume candidate metadata. A workspace bound must
be identical during collection and prediction when the model depends on it.
The example feature set demonstrates the workflow, not an accuracy recommendation.

Generation preserves supplied graph UUIDs and assigns reproducible IDs to ID-less
JSON inputs. It records the physical device ID, selector revision, commands,
constraints, warmup, timing statistics, and disjoint training/evaluation graph-device
identities. L1 evaluation reports calibration errors and cross-engine immediate
selection regret; a corpus with only one measured engine cannot establish
cross-engine selection quality.
UED role-map keys use the bare architecture (for example, `gfx942`); candidate
collection retains feature-suffixed architecture strings in `device_arch`.

The runtime lives in `hipdnn_plugin_sdk/heuristics/uhd/` and is available without
`HIPDNN_ENABLE_KERNEL_INGESTOR`, but an engine only reaches it through its
`predict_engine_tflops` UED role. An opaque engine has no UED, so no L1 model can be
authored for it: it contributes no score and falls back to static ordering, which is
the outcome RFC 0019 §11.2 and its Open Question 7 sanction.
Start a fresh consumer process after installing a model: a compiled model is cached
for the lifetime of the engine that owns it.

Install the intended catalog-ranking model before collecting L1 measurements.
Changing that model changes the descriptor engine's immediate selector and can
invalidate an existing L1 model. Collect and train L1 against the final selector;
do not reuse labels from the previous ranking policy.

### Prediction queries and engine-selection policies

Predictions are a generation-tool surface, not a consumer API: there is no `Graph`
method for them (RFC 0019 Open Question 12, RFC 0017 §2). `hipdnn_bench` publishes
them for a finalized graph, and a C++ tool can read the same descriptor attributes
through `hipdnn_frontend::detail::getEnginePrediction()`:

```bash
# L1 (engine kind), described without evaluating a model:
hipdnn_bench --graph graph.json --engine-name <engine> --describe-engine-prediction
# L1 evaluated:
hipdnn_bench --graph graph.json --engine-name <engine> --predict-engine
# L2 (configuration kind): add the knob constraints that name the configuration.
hipdnn_bench --graph graph.json --engine-name <engine> --predict-engine --knob tile=128
```

Under the hood the engine descriptor answers the engine-kind query
(`HIPDNN_ATTR_ENGINE_PREDICTION_EXT`) and an engine config descriptor answers the
configuration-kind query (`HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT`); the descriptor
queried states the kind, and the `*_PREDICTION_EVALUATE_EXT` input selects describe
(0) versus evaluate (1). Consumers still select engines the ordinary way:

```python
error = graph.create_execution_plans([hipdnn.HeuristicMode.B,
                                      hipdnn.HeuristicMode.FALLBACK])
```

- **Mode A** ranks applicable engines by L1 TFLOPS, without querying L2 or
  materializing losing engines' configuration catalogs. The chosen engine uses
  its normal selector with tuning disabled.
- **Mode B** uses an engine's calibrated L2 configuration prediction when
  available, otherwise its L1 prediction. The L2 result owns an `EngineVariant`
  containing the engine ID and explicit knob settings. Plan construction
  preserves those settings to execute the scored configuration.
- Engines without a usable prediction remain eligible after scored engines. If
  no engine has a usable score, the prediction policy declines rather than
  fabricating a ranking. An explicitly supplied fallback mode can then run.

`AVAILABLE` carries physical TFLOPS. `UNAVAILABLE` and `INVALID` do not remove
engine applicability. Description queries do not evaluate a model; evaluated
queries can omit binding/features metadata to keep the policy path lightweight.
Neither prediction kind times GPU work; L2 may prepare a candidate to ensure the
returned selection is executable.

A configuration-kind query scores the configuration its knob constraints name, so
replaying an `AVAILABLE` result is replaying those same knobs:

```python
knobs = [hipdnn.KnobSetting("tile", 128)]
error = graph.create_execution_plan_ext(engine_id, knobs)
```

Check the returned error, then call `build_plans()` before execution.
Preserving a configuration does not confer compiled-plan serialization support.
Graph serialization embeds a built plan only when the engine advertises that
capability; otherwise it stores the graph alone. For an engine without that
capability, restore the graph and explicitly reapply the owned configuration.

L2 cross-engine comparison requires a model trained against actual TFLOPS with
`--target tflops --objective max --score-units tflops --calibrated
--timing-statistic avgTimeMs`. `generate` produces exactly that whenever the engine
publishes `graph.flops`, and otherwise says in a warning that the model it produced
ranks a catalog without being a calibrated L2 throughput estimate. Do not relabel a
latency or arbitrary-score model as TFLOPS.

## Output

`train` generates:

```
output_dir/
├── <stem>.uhd.json     # the UHD: features, objective, score units, artifact
├── model.bin           # FlatBuffer GbdtModel for TreeDataAdapter
└── train_manifest.json # training provenance
```

`<stem>` comes from `--descriptor-name` (default `heuristic`).

`DescriptorLoader` globs `<stem>.uhd.json` and reads `tree_data.artifact` as a
path relative to that file, so the directory relocates as a unit.

For a descriptor-backed engine, the UED names the heuristic by id. `train` prints
the id and records it in `train_manifest.json`; `promote` updates the UED and
isolates model files by engine, role, and architecture:

```
descriptor_tree/
├── <engine>.ued.json   # <role>.<arch> names the new UHD
└── heuristics/
    └── <engine-uuid>/
        └── <role>/
            └── <arch>/
                ├── <stem>.uhd.json
                └── model.bin
```

L1 and L2 may both use the default source filenames without overwriting each
other: `<role>` keeps them in separate directories, and each is referenced from its
own entry in the UED role map.

## Generated FlatBuffers bindings

`model.bin` is written through flatc-generated Python bindings that
live in `_generated/`, committed alongside the tool the same way the C++
`*_generated.h` headers are committed alongside the SDK.

```
_generated/hipdnn_flatbuffers_sdk/data_objects/
└── GbdtModel.py, GbdtTree.py                                        # gbdt_model.fbs
```

`uhd_gen/__init__.py` prepends that directory to `sys.path`, so `import
hipdnn_flatbuffers_sdk.data_objects.GbdtModel` resolves to the bindings that
match the schema shipping beside this tool rather than to any other copy
installed on the system.

Regenerate after editing the schema — a build with `HIPDNN_GENERATE_SDK_HEADERS=ON`
does it automatically, and so does the `flatc-hipdnn` pre-commit hook:

```bash
python projects/hipdnn/scripts/run_flatc.py \
    projects/hipdnn/flatbuffers_sdk/schemas/gbdt_model.fbs
```

That command emits both the C++ header and these bindings from one invocation, so
the two cannot drift. Requires flatc 25.9.23 on PATH (see
`projects/hipdnn/CONTRIBUTING.md`).

**Do not hand-write FlatBuffers vtables here.** The writer this tool replaced did,
declaring `StartObject(11)` against a 13-field table: every field from
`features_signature` on landed one slot low, so every descriptor it produced failed
verification. Nothing caught it, because the only structural assertion in the test
suite was the four-byte file identifier — which sits before the root table and
survives any vtable error. The descriptor is JSON now; `model.bin` is the only
FlatBuffer this tool writes, and it goes through generated bindings.

### `<stem>.uhd.json`

```json
{
  "version": "1.0",
  "id": "...",
  "name": "GEMM UHD",
  "adapter": "tree_data",
  "features_signature": ["$q.M", "$q.N", "$q.K", "$kernel.tile_m", ...],
  "features_hash": "sha256:...",
  "objective": "max",
  "score": {"units": "tflops", "calibrated": false, "transform": "log1p"},
  "categorical_encoding": {"$kernel.dtype": {"bf16": 0, "fp16": 1}},
  "tree_data": {"artifact": "model.bin"}
}
```

`categorical_encoding` maps each string-valued feature to the codes the model was
fitted with. It is derived from the training corpus, keyed by the full `$reference`
from `features_signature` (never the trailing field name — `kernel.dtype` and
`q.attention_dense.dtype` are two vocabularies, not one), and holds the values exactly
as the corpus spells them; codes run from 0 in sorted order. It is written only when
the corpus has a string column, and it is folded into `features_hash`, so changing the
map is a contract change even though the signature text is unchanged. The same dict is
recorded in `train_manifest.json` (as `{}` when there is none) and is the one
`evaluate` scores through, so evaluation encodes exactly as the fit did.

## Training Details

- **Target transform**: `log1p(tflops)` for scale-invariant training
- **Cross-validation**: GroupKFold when `--group-by` specified
- **Early stopping**: Prevents overfitting
- **Model format**: LightGBM → FlatBuffer GbdtModel

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Integration

The output files are loaded by hipDNN's UHD system:

`DescriptorLoader` parses `<stem>.uhd.json` while walking a descriptor tree, and
`makeKernelHeuristic` builds the scorer from it: `TreeDataAdapter` loads
`tree_data.artifact` relative to the descriptor, and `FeatureExtractor`
recomputes `features_hash` from `features_signature` and refuses the pair if the
two disagree.
