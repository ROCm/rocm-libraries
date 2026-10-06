# tilewright training pipeline

This pipeline trains the models the tilewright engine (`shared/tilewright/`)
uses to rank the candidate kernels of one hipBLASLt library, and deploys them
into a rocm-libraries checkout.

One run targets one Tensile library (`hipblaslt.library_stem`): one data type
combination and layout on one architecture. GEMM problem space is split into
96 cells by M, N, K and batch tier, and cells are split further where the
model needs it; every leaf cell gets its own two-tower network. The pipeline
benchmarks every kernel of the library on synthetic shapes with
`hipblaslt-bench`, trains the cell models on the measured latencies, checks
each round's model on the next round's unseen shapes, and writes the result
as one model file that hipBLASLt loads next to the library.

The pipeline does not use origami. The analytical baseline it compares
against is the kernel hipBLASLt's own ranking puts first in a bench run
without tilewright (`is_origami_pick` in the enriched data), and all model
evaluation goes through the tilewright engine itself.

## Setup

- Python 3.10 to 3.13 with `pip install -r requirements.txt`
  (`requirements-report.txt` adds matplotlib for the report's cell maps).
- The tilewright Python module, built from this repository:

  ```bash
  pip install shared/tilewright/python
  ```

  It needs CMake and a C++17 compiler, and no ROCm. Training, evaluation,
  deployment and parity all rank through it.
- A built hipBLASLt for the target architecture (the *bench root*):
  `projects/hipblaslt/build/release/clients/hipblaslt-bench` and
  `projects/hipblaslt/build/release/Tensile/library/<arch>/` holding the
  configured library. A gfx1250 build also produces the `gfx1250v0` library
  subtree. `runtime.build_dir` overrides the build directory.
- For deployment, a rocm-libraries checkout (the *deploy root*, often the
  same checkout) and, for `deploy.build_after_apply`, its configured hipBLASLt
  build directory.

### Model files

The deployed models live in `shared/tilewright/weights/hipblaslt/<arch>/<arch>/`
as `*.tilewright.bin`, committed as plain binary files
(`shared/tilewright/.gitattributes`); new or retrained models are committed
the same way.

## Running

```bash
scripts/run_pipeline.sh configs/<config>.yaml --mode full
```

`run_pipeline.sh` exports default roots and runs `run_orchestrator.py` with
the remaining arguments. Configs refer to host paths through environment
variables, which the orchestrator expands in every string; it refuses to start
when a referenced variable is unset or empty.

| Variable | Meaning | Default in `run_pipeline.sh` |
|---|---|---|
| `TILEWRIGHT_BENCH_ROOT` | checkout whose hipBLASLt build is benchmarked | the checkout holding the script |
| `TILEWRIGHT_DEPLOY_ROOT` | checkout that receives the model | the checkout holding the script |
| `TILEWRIGHT_DATASETS` | directory of stage08 datasets, for configs that list any | unset |
| `PYTHON` | interpreter with the requirements and the tilewright module | `python3` |

Benchmarks never inherit `HIPBLASLT_TENSILE_LIBPATH` (`run_pipeline.sh` also
unsets it), so hipBLASLt loads the library subtree that matches the device's
ASIC revision; stage 7 sets it to the deploy build's library.

### Modes

```bash
python run_orchestrator.py --config C --mode initial    # round_0 only
python run_orchestrator.py --config C --mode active --run-dir R    # one more round
python run_orchestrator.py --config C --mode full       # round_0 .. full.total_rounds-1
python run_orchestrator.py --config C --mode validate --run-dir R  # stages 7 and 8 only
```

Without `--run-dir` a run goes to `runs/<config stem>_<timestamp>/`.
`--deploy` deploys after every round, round_0 included. `--no-report` skips
writing `report.html`.

Before any stage runs the orchestrator checks the config against its schema
(unknown keys, types, required keys and cross-key rules such as a split budget
that can never be met) and runs preflight checks: the bench binary, the
library directory, the configured library stem, the tilewright module and its
feature catalog. `--skip-preflight` skips the latter.

### Resuming

Every finished stage writes `.stage_complete.json` into its directory and
every finished round writes `.round_complete.json`. Re-running the same
command resumes: complete rounds are skipped and an interrupted round restarts
at its first stage that did not complete. `--from-round R` (full mode) re-runs
round R and all later rounds; `--from-stage N` (1, 2, 3, 4, 4.5, 5, 6; 7 or 8
in validate mode) re-runs the target round from stage N, provided the earlier
stages completed. A stage that runs again starts in an empty directory: the
previous attempt's directory and log move to `<round>/stale/<timestamp>/`.

Each invocation writes its expanded config to
`<run>/configs/<timestamp>_<mode>.yaml` (and the file as written to
`*.source.yaml`) and passes that snapshot to every stage, so editing the config
file never changes a run that is already going.

One orchestrator at a time uses a run directory: it holds `<run>/.lock`, and
a second invocation on the same directory exits with 2.

Every stage runs in its own process group, and stages 2 and 3 run each bench
in its own group too. SIGINT, SIGTERM or SIGHUP stops the running stage
(SIGINT to its group, SIGKILL after 30 s), the stage stops its benches, and
the orchestrator exits with 128 + the signal number; the same command resumes
the interrupted round.

A stage that fails stops the run with its exit code (a stage killed by signal
N gives 128+N). stage05 exits with 3 after writing its outputs when a cell
fails or a cell stage04b listed cannot be trained; the round then stops before
deployment (`round_<R>/stage05/metrics.json` lists `cells_failed` and
`cells_not_trained`), and the same command reruns stage 5. `--mode validate`
and a training run that ends with validation exit non-zero when stage 7 or 8
fails.

## Stages

| Stage | Script | Output |
|---|---|---|
| 1 | `stage01_generate_OOB_shapes.py` | `shapes.yaml`: new shapes per leaf cell; re-used history shapes in `reused_shapes.yaml` |
| 2 | `stage02_probe.py` | per-shape iteration counts; skipped with `bench.adaptive.enabled` |
| 3 | `stage03_load_balance_offline_tuning.py` | `logs/`: hipblaslt-bench logs, one work queue per GPU |
| 4 | `stage04_convert_to_enriched_dataset.py` | `chunk_*.csv`: one row per (GEMM, measured solution of the library); `kernels.json`: the solutions' kernel attributes |
| 4.5 | `stage04b_iterate_active_learning.py` | `decisions.json`, `retrain_cells.txt`, `splits.json` (active rounds) |
| 5 | `stage05_train.py` | `models.pt`, `cells.json`, `metrics.json` |
| 6 | `stage06_deploy_weights.py` | model file, index and `deploy_record.json` |
| 7 | `stage07_optional_validate_selection_time.py` | selection time and C++/Python pick parity |
| 8 | `stage08_optional_validate_selection_efficiency.py` | selection efficiency on external datasets |

Round 0 trains every populated cell. An active round R benchmarks new shapes,
evaluates round R-1's model on them (stage04b; shapes re-used from earlier
rounds are excluded), and retrains the cells that missed the bar on the union
of every round's data. Cells that keep missing the bar after retraining are
split on the eligible axis (a widest tier: Large M or N, LargeK K) whose
threshold best separates their well- and badly-served GEMMs (the largest
variance reduction of the per-GEMM log selection-efficiency ratios). When
stage04b lists no cell for stage05 (`retrain_cells.txt` is empty: every
evaluated cell passed and no leaf without a model of its own can be trained
yet), the previous round's `models.pt` and `cells.json` are carried forward
unchanged.

A leaf of the split tree without a model of its own, such as a split child
with too few GEMMs to train, is served by its nearest trained ancestor: stage05
keeps a split parent's model in the bundle until every leaf under it has its
own model (`fallback_parents` in `models.pt` and `cells.json`), and stage01
keeps sampling shapes for such leaves.

stage05's validation GEMMs choose the epoch and the whitelist fallback, so its
validation numbers are model-selection scores; the held-out measurement is
stage04b's. stages 4.5, 5 and 8 remove `TILEWRIGHT_FORCE_CELL`,
`TILEWRIGHT_PICK_LOG` and `TILEWRIGHT_DIAG` from their environment before they
load the engine, and fail when the engine serves any GEMM from another cell
than the pipeline routes it to.

Every stage can also run on its own; `python stages/<script> --help` lists its
arguments.

### Kernels and execution context

A kernel is one solution of the library: the data names it by its runtime
solution index, every ranking by its position in the library's kernel pool.
The pool is what hipBLASLt ranks with tilewright: the solutions of the
`.dat`'s Prediction table in table order (every solution in file order for a
file without one); the file can also hold solutions of other library rows,
which are not in the pool. Solutions whose `tilewright::Config` fields are
identical are not merged: each keeps its own rows and latency, the model
scores them alike, and the engine ranks them in pool order.

- stage04 writes one row per (GEMM, solution) and, in `kernels.json`, the
  named attributes of every solution its rows name: the `sizeMapping` values
  TensileLite hands the engine, as int64, with the C++ defaults for absent
  optional keys and the legacy `streamK` keys normalized the way TensileLite
  reads them (`lib/dat.py`). The engine `Config`s of a kernel pool read from
  the library, or from data with `kernels.json`, carry them; v2 models do not
  read them.
- stage05 trains on one pair per (GEMM, solution) and keeps each GEMM's
  candidates in pool order, so that its training-time picks break score ties
  like the engine. Whitelists stay per signature (macro tile, matrix
  instruction, cache hints); a signature's latency for a GEMM is the minimum
  over its solutions.
- Every evaluation (stages 4.5, 5 and 8) picks a pool position: the first
  kernel of the engine's ranking that the GEMM has a measurement for (of
  kernels with equal scores, the first in pool order), at that solution's own
  latency. Selection efficiency is relative to the fastest solution.
  stage08's `per_gemm.csv` gives the pick and the Origami pick as pool
  position and solution index.
- Stage 7 compares the pool position of the runtime's pick (`top1_index` of
  the engine's pick log) and its serving cell; the kernel signature is only
  reported (`pick_mismatch_same_sig`, `same_index_other_sig`).

The models serve only the exclusive execution context: every CU of the device
and the default tile scheduling. hipBLASLt calls with a CU budget or a
dynamic schedule are not scored by tilewright and keep the analytical
ranking. The pipeline measures and evaluates only the exclusive context:
benches do not inherit the scheduling override
`TENSILE_PERSISTENT_HYBRID_FORCE_MODE` (or its legacy name
`TENSILE_STREAMK5_FORCE_MODE`), and evaluations rank with the engine's
default context.

stage05 whitens each feature with its cell's statistics: (x - mean) / std,
dividing by 1 when the std is below 1e-6. An item feature whose stored std is
below 1e-3 was constant in training (such as the occupancy when every kernel
of the library has the same one) and whitens to 0, so an unseen value of it
does not move the score; the engine treats such features as constant at
inference too. The rule is `whitens_to_zero` in `stages/stage05_train.py`;
models store each std as measured.

### Active learning (stage04b)

For each leaf with a model of its own and at least `validate.min_cell_gemms`
new GEMMs:

- its widest splittable axis still spans at least
  `validate.structural_split_min_octaves` octaves (when set; down to
  `validate.structural_split_max_depth`): split, whatever its selection
  efficiency;
- held-out selection efficiency >= `validate.sel_eff_threshold`: pass;
- below the threshold, retrained `validate.split_after_attempts` times, and
  either below `validate.split_floor` (0 disables the floor) or improving by
  less than `validate.split_min_improvement` (<= 0: always) over its
  improvement baseline (the round R-2 model on the same shapes, else its
  previous measurement): split;
- otherwise: retrain.

A split becomes a retrain when the cell is at `validate.max_split_depth`, has
fewer than twice `validate.min_cell_gemms` evaluated GEMMs, a child would get
fewer than `validate.min_cell_gemms` of them or fewer than
`train.min_cell_gemms` GEMMs over all rounds (stage05 could not train it), or
this is the final round. A leaf without a model of its own is listed for
training once it has `train.min_cell_gemms` GEMMs over all rounds.

Both thresholds accept the string `origami`, meaning the cell's Origami
selection efficiency on the same shapes.

### Deployment (stage06)

stage06 writes the trained cells and the split tree implied by their labels
into one MLREC_v2 file (format in `shared/tilewright/README.md`). The file
carries the arch's model constants (matrix-instruction cycle table,
`parallel_mi_cu`, bandwidth coefficients; `lib/hardware.py`), so the engine
computes a model's features with the constants it was trained with. stage06
refuses a bundle whose feature names, dimensions or catalog hash differ from
the engine's, and loads the written file back with the tilewright module
before staging it; `deploy.apply_patch` requires the module.

It always stages the result under `<round>/stage06/staged_for_deploy/`
(the model, an index fragment per target and `INSTALL.md`). With
`deploy.apply_patch` it installs the model into
`shared/tilewright/weights/hipblaslt/<t>/<t>/` for every directory `t` of
`deploy.targets` and registers the library stem in that directory's
`tilewright_index`. One model can serve several targets (for example the
gfx1250 MXFP8 model ships for `gfx1250v0` and `gfx1250`); for a target that
names libraries with another architecture suffix (`gfx1250-strict`) the stem
is rewritten accordingly. The index merge keeps every other library's entry,
resolves a stem listed twice the way the engine does (the first line wins)
and removes the duplicates. A re-deploy keeps the file name the stem already
uses; a file that no index entry references any more is removed.

Installation is all-or-nothing: files are written under temporary names,
synced and renamed, indexes last, and any failure restores the previous files.
With `deploy.build_after_apply` stage06 then builds the hipBLASLt target
`hipblaslt-tilewright-models` (part of the default build), which copies each
index and its models next to the Tensile library of the same architecture,
and verifies the copies; a failed build or a missing or wrong copy also
restores the previous files, in the checkout and the index and model copies
in the build tree. A build directory with no library directory for any target
is recorded as unverified (`deploy_record.json`: `built: false`,
`colocation: "unverified"`).

hipBLASLt ranks with the deployed model when `TENSILE_USE_TILEWRIGHT=1` is
set. Problems routed to a leaf without a model of its own or of an ancestor,
calls outside the exclusive execution context, and problems for which the
model scores no kernel keep the analytical ranking.

Only the model file and its index entry go into the checkout. The manifest,
summary and deploy record stay in the run directory.

### Validation (stages 7 and 8)

Stage 7 needs a model deployed by this run and the deploy checkout's build.
It times hipBLASLt's `Solution selection time` with the model on and off
(request sizes `stage07.timing_request_sizes`, `stage07.timing_repetitions`
alternating repetitions over the identical problem list; pick logging off),
pairs the two runs problem by problem, and reports the per-problem deltas.
In a separate run with `TILEWRIGHT_DIAG` and `TILEWRIGHT_PICK_LOG` it checks
that the runtime loaded exactly the deployed file and compares every
problem's pick (its position in the library's kernel pool) and serving cell
with the engine ranking the same model, library and hardware in Python. Both
sides run the same engine, so a mismatch points at how hipBLASLt maps its
problem, kernels, pool order or hardware to tilewright; parity below
`stage07.parity_min_match_rate`, or a pick-log line without `top1_index`,
fails the stage.

With `stage07.parity_sample_trained` > 0 stage 7 uses that many random
problems that benchmarked successfully during training instead of
`stage07.bench_yamls`.

Stage 8 evaluates the trained models on external datasets (bench yamls, bench
log directories or enriched CSV directories) and reports the model and the
Origami pick on the same GEMMs.

## Run directory

```
runs/<config stem>_<timestamp>/
    .lock                        held by the running orchestrator
    configs/                     config snapshots, one per invocation
    orchestrator.log             the orchestrator's own output, all invocations
    timing/                      wall-clock per stage, one file per invocation
    round_<R>/
        stage01/ .. stage06/     stage outputs (stage04b only in active rounds)
        stage01.log ..           stage logs
        stale/                   outputs of re-run stages
        .round_complete.json
    validate/stage07/ stage08/   validation outputs
    held_out_sel_eff_breakdown.json
    report.html                  self-contained report (scripts/make_report.py)
```

Run directories are not source: their logs, metrics and reports carry measured
performance numbers and must not be committed (`runs/` is ignored).

## Config reference

Every key the orchestrator accepts; anything else is an error. Defaults apply
when a key is absent.

**Top level**: `config_id` (required, unique per file), `arch` (required: the
library subtree, e.g. `gfx950`, `gfx1250v0`).

**`hipblaslt`** (required): `a_type`, `b_type`, `c_type`, `d_type`,
`scale_type`, `compute_type`, `transA`, `transB` (the bench line fields),
optional `scaleA` / `scaleB` (hipblaslt-bench scaling-format modes, e.g. 1
scalar, 3 MX block-32), and `library_stem`: the Tensile library file name
without `.dat` / `.dat.zlib`, for example
`TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950`.
stage04 keeps only measurements of this library's kernels, training and
evaluation rank its kernel pool, and stage06 indexes this stem.

**`hardware`** (optional): `n_cu`, `lds_bytes`, `l2_bytes`, the device values
hipBLASLt passes to the engine. Any key set here overrides the value read from
the KFD sysfs topology of the first device in `runtime.devices`.

**`runtime`** (required): `devices` (GPU indices), `rocm_libraries_root` (the
bench root), optional `build_dir` (hipBLASLt build directory; default
`<root>/projects/hipblaslt/build/release`).

**`seed`**: `target_per_cell` (round_0 shapes per cell, 1600),
`shapes_per_cell` (new shapes per leaf in an active round, 150), `seed` (42;
each round derives its own seed from it), `n_strata` (log-strata per axis
within a cell, 1), `exclude` (shape regions left out, a list of rules with
`min_` / `max_` bounds on `m`, `n`, `k`, `batch` and `mn` = min(M, N); a shape
is left out when it meets every bound of a rule).

**`probe`** (stage02; required when adaptive timing is off):
`first_line_iters`, `first_line_cold_iters`, `other_lines_iters`,
`other_lines_cold_iters`, `duration_us`, `skip_ratio_mode` (`sigmoid` with
`skip_ratio_min`, `skip_ratio_max`, `skip_ratio_center_us`, `skip_ratio_k`;
or `bucket` with `fast_us_threshold`, `ratio_light`, `ratio_heavy`).

**`bench`**: `blocks_per_gpu` (20), `startup_grace_s` (900) and
`stall_timeout_s` (300) (stages 2 and 3 kill a silent bench), `warmup_pass`,
`skip_slow_solution_ratio`, `rotating_mb`, `rotating_target_blocks`,
`device_memory_gib` (default: the KFD sysfs topology), and `adaptive`
(`enabled`, plus hipblaslt-bench's adaptive timing fields `warmup_time`,
`sample_time`, `measure_time`, `max_measure_time` in ms, `min_iters`,
`max_iters`, `noise_threshold`, `stability_threshold`, `stability_window`,
`stability_interval`).

**`train`**: `workers` (1), `threads_per_worker` (0 = cores / workers),
`epochs` (200), `device` (`cpu`), `eval_sample_frac` (1.0), `smart_k` (10),
`smart_k_max` / `smart_k_min` (4) / `smart_k_saturation_eps` (0.005) for an
adaptive whitelist size, `min_cell_gemms` (50; cells with fewer GEMMs over all
rounds are not trained, a leaf without a model is listed for training once it
has this many, and each side of a split needs this many; every leaf that a
model of the previous round serves gets at least this many shapes per round,
`seed.shapes_per_cell` new ones topped up with re-used history shapes),
`validation_frac` (GEMMs per cell held out to choose the epoch; stage05's
default when absent).

**`validate`** (stage04b): `sel_eff_threshold` (0.95), `split_floor` (0.85;
0 disables), `min_cell_gemms` (20), `split_min_improvement` (0.02),
`split_after_attempts` (2), `max_split_depth` (6),
`structural_split_min_octaves` (0 = off), `structural_split_max_depth` (-1).

**`full`**: `total_rounds` (3).

**`deploy`**: `target_sel_eff` (0.95; null disables; an active round whose
held-out selection efficiency reaches it deploys), `on_final_round` (true),
`stop_when_target_reached` (false), `weight_dtype` (`bf16`; one of `fp32`,
`bf16`, `int8`, `int4`; also the precision every evaluation uses),
`rocm_libraries_to_deploy`
(the deploy root, or null to only stage), `targets` (weights directories;
default `[arch]`; an entry is a directory name or `{dir, library_stem}`),
`apply_patch` (false), `build_after_apply` (false), `build_dir` (default
`<deploy root>/projects/hipblaslt/build/release`), `build_jobs`.

**`stage07`**: `parity_sample_trained` (0), `bench_yamls` ([]),
`timing_request_sizes` ([1]), `timing_repetitions` (3),
`parity_min_match_rate` (1.0).

**`stage08`**: `datasets` (a list of `{name, path, rocm_libraries_used}`; the
last is the checkout whose library produced a log or CSV dataset, read from
`runtime.build_dir` when it is the bench root and from its
`projects/hipblaslt/build/release` otherwise), `tie_tolerance`, `bootstrap`.

## Tests

```bash
python -m pytest            # from this directory; CPU only
```

The tests use stub stages, a fake `hipblaslt-bench` and fake Tensile
libraries; tests that need the tilewright module skip without it.

## Layout

```
run_orchestrator.py      orchestration, config schema, resume
configs/                 run configs; several can target one library (for
                         example full, int4 and smoke variants)
stages/                  stage01 .. stage08
lib/                     shared modules (shapes, bench, data, features,
                         model file format, evaluation, ...)
scripts/
    run_pipeline.sh      launcher
    make_report.py       HTML report of a run
    collect_bench_failures.py
    convert_mlrec_v1_to_v2.py
    experiments/         CPU microbenchmark of the engine's ranking call
tests/                   pytest suite
```
