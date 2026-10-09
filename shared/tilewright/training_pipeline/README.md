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
ASIC revision; stage 7 sets it to the deploy build's library. Nor do they
inherit `TENSILE_USE_TILEWRIGHT` or any `TILEWRIGHT_*` or
`TENSILE_TILEWRIGHT_*` variable: training benches run without tilewright, and
stage 7 passes its ML-on runs the variables they need, so a value exported in
the shell, such as `TENSILE_TILEWRIGHT_BANK`, does not change what stage 7
measures.

### Modes

```bash
python run_orchestrator.py --config C --mode initial    # round_0 only
python run_orchestrator.py --config C --mode active --run-dir R    # one more round
python run_orchestrator.py --config C --mode full       # round_0 .. full.total_rounds-1
python run_orchestrator.py --config C --mode validate --run-dir R  # stages 7 and 8 only
python run_orchestrator.py --config C --mode enhance --run-dir R --cells 'L1,L2'
```

Without `--run-dir` a run goes to `runs/<config stem>_<timestamp>/`.
`--deploy` deploys after every round, round_0 included. `--no-report` skips
writing `report.html`. `--origami-reference` adds Origami's numbers, as an
external reference on the same GEMMs as the model's, to the printout of
stages 4.5, 6, 7 and 8 and of the run summary, and the report's reference
section; they never enter a decision.

### Enhancing chosen cells

`--mode enhance` adds active rounds to a run with at least one complete round,
restricted to chosen cells. Every enhance round runs the active-round stages
on those cells only: stage01 draws `--shapes-per-cell` new shapes (default
`seed.shapes_per_cell`) for the leaves under them, including leaves without a
model of their own; stages 2-4 bench and enrich them; stage04b measures the
current model on them; stage05 retrains them (`--only-cells`) and carries the
other cells forward. A chosen leaf is retrained when it has at least
`train.min_cell_gemms` training GEMMs. When stage04b splits a chosen leaf, the
split children are retrained in its place. Cells that stage04b lists outside
the selection are left for a later active round. The last of
`--enhance-rounds N` (default 1) rounds is a final round: it never splits, and
it deploys when `deploy.on_final_round` is set (through the deploy gate, on
the gate GEMMs of the chosen cells) or `--deploy` is given.

The cells come from any combination of:

```bash
--cells 'Mid|Large|MidK|Bnone,Large|Large|LargeK|Bnone'  # labels; a split parent stands for its leaves
--weakest 5                     # the 5 lowest-efficiency cells below the target
--weakest 5 --rank-by gain      # ... ranked by GEMM share x (target - efficiency)
--below 0.9                     # every cell below 0.9
```

The ranking uses training-side evidence from the complete rounds only
(`lib/cell_evidence.py`). For each leaf it takes the newest of three sources:

- the stage04b row of a later round (the previous model on new training
  shapes);
- stage05's model-selection validation score;
- the in-sample score.

Every value is relative to the fastest measured kernel. The target is
`validate.sel_eff_threshold`; a non-numeric setting falls back to 0.95.
Validation datasets (stage 8) never select cells.

`round_<R>/enhance.json` records the cells, the evidence behind the selection
and the retrain list. An interrupted enhance round resumes with the recorded
cells when you run `--mode enhance` again; the other modes refuse to resume
it. `--dry-run` prints the selection and every stage command and writes
nothing.

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
round R and all later rounds; `--from-stage N` (1, 1.5, 2, 3, 3.2, 3.5, 4, 4.5, 5, 6;
7 or 8 in validate mode) re-runs the target round from stage N, provided the earlier
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
| 1.5 | `stage01b_plan_benchmarks.py` | `shapes.yaml` with each shape's kernel list, `plan.json`, `plan_summary.json`; only with `planner.enabled` |
| 2 | `stage02_probe.py` | per-shape iteration counts; skipped with `bench.adaptive.enabled` |
| 3 | `stage03_load_balance_offline_tuning.py` | `logs/`: hipblaslt-bench logs, one work queue per GPU |
| 3.2 | `stage03b_rebench_near_ties.py` | `logs/`: stage03's logs with the near-tie candidates re-timed, `rebench_summary.json`; only with `bench.rebench.enabled`. GEMMs of other execution contexts (stage03c) keep their single measurement |
| 3.5 | `stage03c_bench_contexts.py` | per execution context: `<label>/` with its bench wrapper and stage02/03 outputs, `holdout/stage04/`, `contexts.json`; only with `contexts` |
| 4 | `stage04_convert_to_enriched_dataset.py` | `chunk_*.csv`: one row per (GEMM, measured solution of the library); `kernels.json`: the solutions' kernel attributes |
| 4.5 | `stage04b_iterate_active_learning.py` | `decisions.json`, `retrain_cells.txt`, `splits.json`, the gate slice `gate/` and `train/` (active rounds); `rebench_misses.json` and `rebench/` with `validate.rebench_misses` |
| 5 | `stage05_train.py` | `models.pt`, `cells.json`, `metrics.json` |
| 6 | `stage06_deploy_weights.py` | model file, index, `gate.json` and `deploy_record.json` (`rebench/` when the gate re-benches its misses) |
| 7 | `stage07_optional_validate_selection_time.py` | selection time and C++/Python pick parity |
| 8 | `stage08_optional_validate_selection_efficiency.py` | selection efficiency on external datasets |

Round 0 trains every populated cell. An active round R benchmarks new shapes,
evaluates round R-1's model on them (stage04b; shapes re-used from earlier
rounds are excluded), and retrains the cells that missed the bar on the union
of every round's data, without the round's gate GEMMs (`deploy.gate_frac` of
its held-out GEMMs per leaf), on which stage06 measures the bundle the round
trained before it deploys it. Cells that keep missing the bar after
retraining are split on the eligible axis (a widest tier: Large M or N,
LargeK K) whose threshold best separates their well- and badly-served GEMMs
(the largest variance reduction of the per-GEMM log selection-efficiency
ratios). When stage04b lists no cell for stage05 (`retrain_cells.txt` is
empty: every evaluated cell passed and no leaf without a model of its own can
be trained yet), the previous round's `models.pt` and `cells.json` are
carried forward unchanged.

stage05 records the bundle's provenance in `models.pt` and `metrics.json`
(`lib/provenance.py`): the config's `config_id`, `training_data_hash` (SHA-256
over the contents of the enriched chunks it trained on, in training order),
`source_revision` (the pipeline checkout's commit, `-dirty` when tracked files
are modified, `unknown` outside git), the UTC `date` and, for v3, the
`launch_model` id of its launch plans (`HipblasltLaunch.id`). Every stage that
writes the bundle as a model file records it in the MLREC_v3 file, and the
engine then refuses to rank that model with another launch model. No value
holds a host or user name or a path.

A leaf of the split tree without a model of its own, such as a split child
with too few GEMMs to train, is served by its nearest trained ancestor: stage05
keeps a split parent's model in the bundle until every leaf under it has its
own model (`fallback_parents` in `models.pt` and `cells.json`), and stage01
keeps sampling shapes for such leaves.

stage05's validation GEMMs choose the checkpoint and the whitelist fallback,
so its validation numbers are model-selection scores; the held-out
measurements are stage04b's (the previous bundle) and the deploy gate's (the
bundle that ships). stages 4.5, 5, 6 (its gate) and 8 remove
`TILEWRIGHT_FORCE_CELL`,
`TILEWRIGHT_PICK_LOG` and `TILEWRIGHT_DIAG` from their environment before they
load the engine, and fail when the engine serves any GEMM from another cell
than the pipeline routes it to.

Every stage can also run on its own; `python stages/<script> --help` lists its
arguments.

### Student training (stage05)

Each cell's student trains in mini-batches of `--batch-gemms` GEMMs, the
loss of a GEMM taken over all its candidates (soft cross-entropy against
latency targets plus distillation from the LightGBM teacher), for
`train.epochs` passes over the cell's training GEMMs, with a linear warmup
and a cosine (or step) decay of the learning rate. The targets treat every
candidate within the tie tolerance of the GEMM's winner as a winner: they
share the largest target share and the teacher's top relevance grade. The
tolerance is the one the run's evaluations use (`stage08.tie_tolerance`;
lib/evaluate.py's default without it), or `--tie-tolerance`. The
embedding-term temperature is learned as its logarithm
(`--student-temperature fixed` keeps it at 1).

`--eval-points` checkpoints over the schedule are measured as deployed: the
whitelist is chosen with the checkpoint in the loop, the engine ranks the
cell's validation GEMMs at the deploy weight dtype with the model's serving
options (whitelist tier, cascade, soft rules), and the checkpoint with the
best selection efficiency is kept (equal ones: the lower validation loss).
`--patience N` stops after N measured checkpoints without a better one. A
cell without validation GEMMs keeps the checkpoint with the best in-sample
efficiency. For int8 and int4 models, `--qat-frac F` continues from the
kept checkpoint for F of the schedule's steps with the weights rounded as
the model file stores them (straight-through gradients) and keeps a better
checkpoint. `training_log.csv` has one row per measured checkpoint.

`--boundary-margin D` (`train.boundary_margin`, 0 = off) softens the cell
boundaries: each cell also trains on the training GEMMs of the neighbouring
leaves that lie at most D octaves outside its box along M, N and K
(`lib/boundaries.py`), with loss weight `--boundary-weight`
(`train.boundary_weight`, 0.5). Only inner sides are crossed (tier and split
thresholds, not the grid's outer edges), and only within the cell's batch
tier. The margin GEMMs come from the GEMMs stage05 trains on, minus each
neighbour's own validation GEMMs, so a gate slice or a hold-out split never
enters through them. The teacher, whitelist, early stopping and every
reported number use the cell's own GEMMs; `cells.json` lists each cell's
`n_margin_gemms`.

The listwise loss sees only measured kernels, so nothing constrains the
score of a kernel a GEMM's plan left off its list, which the deployed model
may still pick. `--unmeasured-samples U` (`train.unmeasured_samples`, 0 =
off; needs `--plan`) adds to each planned training GEMM's loss up to U pool
kernels off its list that pass the LDS gate and the serving rules, as
pseudo-negatives: their soft-CE target is that of a kernel
`train.unmeasured_margin` slower than the winner, scaled by
`train.unmeasured_weight`, and the teacher gives them none. Feature
statistics, the teacher, early stopping and every reported number use the
measured kernels; `cells.json` counts each cell's `n_unmeasured_pairs`.

The margin is off by default. Measured on the hold-out splits of two
first-principles round-0 runs (TF32 NT and FP8 TN, five seeds each, the
configured int4 serving policy), hold-out GEMMs within a quarter octave of an
inner side were worse than those an octave or more inside on TF32 but not on
FP8. Margins of 0.25 and 0.5 octaves at weights 0.3 and 0.6 did not raise the
overall selection efficiency beyond seed noise on both runs, and 0.25 at
weight 0.6 lowered TF32's, mostly on the GEMMs far from any side.

### Near-tie re-bench (stage03b)

With `bench.rebench.enabled`, stage 3.2 runs between stages 3 and 3.5: one
bench pass cannot order kernels whose latencies differ by less than its
noise, and the training targets come from that order. For every complete
problem of stage03's logs, the candidates within `bench.rebench.tolerance`
of the winner (tested rows, and skip rows by their warm-up) are timed again
in `bench.rebench.runs` runs through stage03: one bench line per GEMM with
an explicit `solution_index_list` of those kernels (at most
`max_kernels`, the fastest), no skip screen, the line's iteration counts
times `iters_factor` and its rotating buffer, the list order rotating from
run to run. Each re-timed candidate is replaced in the logs by the row of
the run with its median latency (the lower median for an even count, always
a measured row); a skip row becomes a tested row, ranks are kept, and the
`Winner:` block names the new fastest row. The original timings take no
part in the median, since they put the candidates in the set. stage04 then
reads `stage03b/logs/`. Only stage03's GEMMs are re-timed: those of other
execution contexts (stage03c) and of the planner's seed bench keep their
single measurement. `rebench.json` lists every candidate's original and
re-bench latencies, `rebench_summary.json` the winner changes and the wall
time of each run. The bench must know `solution_index_list`.

### Planned benches (stage01b)

By default stage03 benches every shape with every kernel hipBLASLt's
heuristic lists, in the heuristic's order, so the order-dependent skip screen,
stage02's calibration (the heuristic's first kernel) and stage04's
`is_origami_pick` all follow the library heuristic. With `planner.enabled`
the pipeline does not use the heuristic: stage01b decides the kernels of
every shape, stage02 calibrates on kernels of each shape's list, and every
bench line sets `skip_screen_prepass`, so the bench warms up every listed
kernel once and times only those within the skip ratio of the fastest warm-up,
whatever their order. This needs a hipblaslt-bench that knows
`solution_index_list` (up to 4096 kernels) and `skip_screen_prepass`
(preflight checks the bench's `hipblaslt_common.yaml`).

`planner.mode`:

- `exhaustive` (default): every shape benches every pool and bank kernel in
  library order. Only when they exceed the bench's list limit (4096) does it
  fall back to `algo_method: 1`, which also benches the problem type's other
  libraries: their warm-ups enter the skip screen, so a shape whose pool
  kernels all trail such a kernel can lose all its rows.
- `planned`: every shape benches an explicit list. Round 0 bootstraps from
  seeds: `planner.seed_per_cell` shapes per cell are benched exhaustively
  (stages 2 to 4 in `stage01b/seed/`), and the other shapes are ranked from
  them by `planner.ranker`: `neighbors` (the measured speed of each kernel on
  the cell's `neighbors_k` nearest seed shapes) or `stage05` (a stage05 model
  trained on the seeds). Later rounds rank with the previous round's
  `models.pt`; `planner.model_file` fixes the model for every round.
- `auto`: `planned` when the pool and the bank hold more than
  `exhaustive_max_kernels` kernels, else `exhaustive`.

A planned shape's list holds, among the pool kernels that fit it (LDS and the
non-temporal rule), in this order: the ranker's `top_k` kernels; every kernel
(at most `family_variants` when > 0) of the families of its first
`families_top_f` ranked kernels, a family being the kernels that share macro
tile, matrix instruction and cache hints (a whitelist signature); `explore_random_k` other
kernels drawn with `seed`; then, per cell, kernels added on random shapes
until every kernel is benched on at least `min_coverage` shapes of the cell
where it fits (seed shapes count). Lists stop at `max_list` kernels. The
bank kernels join the ranking and families, and stage04 keeps their rows.

The pool and bank are the ones TensileLite ranks (`lib/bank_pool.py`): the
Prediction table in table order, then the bank, the solutions of the Equality
rows (nested `Problem` libraries included) of every library file of the same
problem type (the stem without `_ID<chip ids>` and `_CU<count>` before
`_<arch>`) whose hardware predicate accepts the bench device, in ascending
solution index, minus the table's indices. Pool positions therefore equal the
runtime's: `top1_index` of `TILEWRIGHT_PICK_LOG` indexes the same list
(`scripts/check_pick_log.py` checks a bench log against it). `planner.bank:
auto` takes the bank from every file of the problem type;
`bank_library_stems` names the files instead and overrides `bank`. The device
is the processor of `arch`, the CU count of `hardware.n_cu` (else KFD) and the
PCI chip id of `hardware.pci_chip_id` (else the KFD `device_id`; required on
processors with chip-id predicates). The evaluation stages (04b, 05, 06, 07,
08) rank the same merged pool.

stage04 adds `is_planned` (0 only for a kernel off its shape's list) and
`n_planned` (the length of the shape's list; empty for exhaustive shapes and
runs without a plan) to every row; `is_origami_pick` is 0 for every GEMM of a plan. Seed GEMMs are part of
the round's data.

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
  read them, v3 models read those of their attribute spec.
- stage05 trains on one pair per (GEMM, solution) and keeps each GEMM's
  candidates in pool order, so that its training-time picks break score ties
  like the engine. Whitelists stay per signature (macro tile, matrix
  instruction, cache hints); a signature's latency for a GEMM is the minimum
  over its solutions.
- Every evaluation (stages 4.5, 5, 6 and 8) picks a pool position: the first
  kernel of the engine's ranking the runtime would run (of kernels with equal
  scores, the first in pool order). A kernel without a row is passed over,
  as the runtime passes over a kernel whose predicates fail, unless the GEMM
  was benched from a plan and the kernel is off its list. stage05 gets the
  plans its training data was benched from (`--plan`: each training
  round's `stage01b/plan.json` and the seed bench's; `--context-plan`: the
  context re-plans with `contexts.replan`); a GEMM's list is the union of
  its shape's lists, so its early stopping and deployed numbers count an
  off-list pick as stages 4.5 and 6 do instead of scoring the next
  measured kernel of the ranking. Those stages re-bench it; stage05 cannot,
  and the slowest latency the GEMM measured can be far from what an off-list
  pick costs, so `train.off_list_sel_eff` (`--off-list-sel-eff`; unset: the
  slowest) charges it the winner's latency over that efficiency instead, e.g.
  the geomean a re-bench of a previous model's off-list picks measured.
  `validate.pick_measurement` (stages 4.5, 5, 6, 7 and 8; stage05's early
  stopping and deployed numbers) says how a pick without a timed row of its
  own costs. `warmup` (the default) charges a pick with only a skip-screen
  row its warm-up latency, a close and slightly pessimistic stand-in, and
  counts a pick off the GEMM's planned list as a miss at the slowest latency
  measured for the GEMM, since nothing measured it. `timed` counts both as
  misses, a strict lower bound; `any` charges the warm-up and passes over
  off-list kernels like kernels without a row, an upper bound. `rebench`
  charges a miss by a measurement: `lib/rebench.py`'s `rebench_picks` times
  each missed pick in one bench line with the GEMM's timed winner and up to
  two more timed kernels, a few runs in rotated order, and the pick costs the
  winner's latency over the ratio of their medians (at most the winner's
  efficiency); a miss without a re-bench counts as under `warmup`.
  With `validate.rebench_misses` (on by default) the stages that decide,
  stage 4.5 (the previous model on the new GEMMs, and `--prior-models`) and
  stage 6's deploy gate (the model and its baseline), and stage 8 measure
  their misses that way (`lib/miss_rebench.py`): each evaluates under
  `validate.pick_measurement`, re-benches every missed GEMM (a skip-screened
  pick, or a pick off the GEMM's planned list, which its line then lists
  explicitly) on the run's devices with the run's bench, pool-only library
  and watchdog, in `validate.rebench_runs` runs, and evaluates again under
  `rebench`. Decisions use the re-benched numbers; `decisions.json`,
  `gate.json` and stage08's summaries keep the configured rule's as
  `pre_rebench` and count `n_rebenched`, `n_rebench_failed` (the bench could
  not run the pick, e.g. its predicates fail, so it stays a miss) and
  `n_context_skipped`; `rebench_misses.json` lists every miss. A GEMM of
  another execution context is re-benched in that context through stage
  3.5's wrapper (`--sm_count_target`, the schedule's hybrid policy, the
  cotenant); one whose context the config does not list, or whose cotenant
  is unavailable, is skipped and counted. Selection
  efficiency is relative to the fastest timed solution. stage08's
  `per_gemm.csv` gives the pick and the Origami pick as pool position and
  solution index, and each miss.
- Stage 7 compares the pool position of the runtime's pick (`top1_index` of
  the engine's pick log) and its serving cell; the kernel signature is only
  reported (`pick_mismatch_same_sig`, `same_index_other_sig`).

v2 models serve only the exclusive execution context: every CU of the device
and the default tile scheduling. hipBLASLt calls with a CU budget or a
dynamic schedule are not scored by them and keep the analytical ranking.
Benches do not inherit the scheduling override
`TENSILE_PERSISTENT_HYBRID_FORCE_MODE` (or its legacy name
`TENSILE_STREAMK5_FORCE_MODE`). stage04 records the context a bench ran under
in the `cu_budget` and `schedule` columns (`--cu-budget`, `--schedule`;
0 / `default`, or CSVs without the columns, mean exclusive). A problem
measured under another context is another GEMM; evaluations rank each GEMM
under its context, and v2 training drops the non-exclusive ones.

`train.catalog: v3` trains MLREC_v3 models: the v3 features (context-aware and
framework-agnostic: a kernel is its signature, plus item features from kernel
attributes through `train.attribute_spec`, a list of `{name, default,
transform, values}` with transforms `identity`, `log2p1`, `nonzero` and
`onehot`, for parameters every GEMM kernel framework defines only; the default
`DEFAULT_ATTRIBUTE_SPEC` in `lib/features.py` is empty), the launch plans
hipBLASLt makes for each training pair (`lib/hipblaslt_launch.py`, from the
kernel's occupancy and its `tile_processing_strategy`, `work_assignment`,
`stream_k_atomic` and `work_group_x/y/z` attributes; evaluations hand the
engine the same plans and `launch_key`s as its launch model;
`scripts/check_launch_plans.py` compares TensileLite's plans and keys
for a library's pool with these), launch keys
(`train.launch_keys`, default off: the model's serving flag that makes the
engine score each launch class of a view key on its own instead of the view
key's first kernel only; the integration's launch model returns the keys
either way and the model decides, and every evaluation groups as the engine
does for that model), the declared contexts (`train.min_cu_fraction`,
`train.schedules`; default the range and schedules in the data), per-cell clip
bounds (the training feature ranges; `train.clip_features`), whitelists keyed
by `train.whitelist` (`none` by default: every feasible kernel through the
cascade, `train.cascade_k` default 32; `signature` or `view-key`), and the
soft feasibility rules off unless `train.small_batched_rule`,
`train.dot2_rule` or `train.non_temporal_rule` is set. A v3 model scores
problems of base cells without a trained cell through the nearest trained base
cell; stage05 trains base cells down to `train.min_base_cell_gemms` GEMMs and
reports the base cells still uncovered (`uncovered_base_cells` in
`cells.json` and `metrics.json`). A round cannot carry cells forward from a
prior bundle of another catalog or attribute spec.

stage05 whitens each feature with its cell's statistics: (x - mean) / std,
dividing by 1 when the std is below 1e-6. An item feature whose stored std is
below 1e-3 was constant in training (such as the occupancy when every kernel
of the library has the same one) and whitens to 0, so an unseen value of it
does not move the score; the engine treats such features as constant at
inference too. The rule is `whitens_to_zero` in `stages/stage05_train.py`;
models store each std as measured.

### Execution contexts (stage03c)

A `contexts` section benches the round's shapes under other execution
contexts too, so a v3 model learns per-context winners (under a CU budget the
fastest kernel often changes):

```yaml
contexts:
  list:                                  # trained contexts
    - {cu_budget: 0}                     # exclusive: the round's own stage02/03
    - {cu_budget: 192, cotenant: true}
    - {cu_budget: 128, cotenant: true}
    - {cu_budget: 64, cotenant: true, fraction: 0.25}
    - {cu_budget: 128, schedule: dynamic, cotenant: true}
  holdout:                               # benched, never trained on
    - {cu_budget: 224, cotenant: true}
    - {cu_budget: 96, cotenant: true, fraction: 0.25}
train:
  catalog: v3
```

An entry has `cu_budget` (0 = every CU), `schedule` (`default`, `dynamic`,
`auto`), `cotenant` (false), `fraction` (1.0) and `name` (default
`cu<budget>[_cot][_<schedule>]`). After stage03, stage03c benches each
non-exclusive context in `stage03c/<name>/`: the shapes it takes (every shape,
or `ceil(fraction * n)` per cell drawn with `contexts.seed`), with the same
bench line, kernel list and skip settings as the exclusive bench, through a
generated `hipblaslt-bench` wrapper that adds `--sm_count_target <cu_budget>`
and, for `dynamic` / `auto`, `--hybrid_assignment_policy DynamicWorkQueue` /
`Auto`. With `cotenant` it starts the bench inside `hipblaslt-cotenant --cus
<device CUs - cu_budget>`, which keeps the other CUs busy so the kernels
really run on `cu_budget` CUs; the device CU count is `hardware.n_cu`, else
the KFD topology. The wrapper appends `contexts.path_append` (default
`$ROCM_PATH/bin` or `/opt/rocm/bin`; hipblaslt-cotenant needs `rocminfo`) to
`PATH`, and does not touch `LD_LIBRARY_PATH`. stage02 (without adaptive
timing) calibrates each context under itself.

The round's stage04 enriches every trained context's logs with its
`cu_budget` / `schedule` columns (`--context-log-dir`), so each (shape,
context) is its own GEMM and ranking group in stages 4.5 and 5 and in the run
history. stage05's declared `min_cu_fraction` defaults to the smallest
trained budget fraction; a configured `train.min_cu_fraction` above it, or
`train.schedules` without a benched schedule, is a config error. The
`holdout` contexts are enriched into `stage03c/holdout/stage04`, which no
stage trains on: stage04b reports the previous round's model per held-out
context (`aggregate.context_holdout` in `decisions.json`), and per context of
the round's new GEMMs (`aggregate.by_context`) when there are several.
Held-out budgets between trained ones test interpolation; preflight warns
about one below the smallest trained fraction, which the model does not
declare.

`contexts.replan: true` (with the planner) re-plans each context's lists in
rounds that rank with a model: stage01b scores the kernels under the context
(`stage01b/contexts/<name>/`), and stage03c / stage04 use those lists and
plans. Otherwise every context benches the exclusive list of the shape, and
seed shapes keep their exhaustive lists.

### The pool-only library directory

With `TENSILE_PREDICTION_LIB=1` a `requested_solution_num: -1` bench gets the
Prediction library's ranked pool and then `getAllSolutions()`: every other
kernel of the loaded libraries for the problem type, such as a `_CU256_`
exact-match library of the same type. Those kernels take bench time, stage04
drops their rows (they are not in the pool), and their warm-ups take part in
the skip screen, so they can screen pool kernels out. The planner's explicit
lists avoid this (`exhaustive` mode included, up to 4096 kernels). Without the
planner, `bench.pool_only_library: true` makes stage03 and stage03c bench
against `libdir_pool/`: the library directory linked file by file, with the
logic files of the problem type's other libraries replaced by copies without
solutions (the planner's bank files are kept;
`lib/pool_libdir.py`).

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
training once it has `train.min_cell_gemms` GEMMs over all rounds. GEMM
counts over all rounds leave out the round's gate GEMMs, which stage05 does
not train on in that round.

Both thresholds are absolute selection efficiencies (relative to the fastest
measured kernel); config validation rejects `origami`. stage04b records
Origami's selection efficiency only on the GEMMs that have an Origami pick,
next to the model's on the same GEMMs (`paired_sel_eff_new`), and prints it
only with `--origami-reference`.

With `deploy.gate_frac` F > 0, stage04b keeps round(F * n) of each leaf's n
held-out GEMMs (by a hash of the run seed, the round and the GEMM) out of
the round's training: it splits the round's rows into `stage04b/gate/` and
`stage04b/train/`, and stage05 trains on `train/`. Later rounds train on the
gate GEMMs like on the rest of the round's data.

### Deployment (stage06)

stage06 writes the trained cells and the split tree implied by their labels
into one MLREC_v2 file, or MLREC_v3 for a v3 bundle (formats in
`shared/tilewright/README.md`). The file carries the arch's model constants
(matrix-instruction cycle table, `parallel_mi_cu` and, in v2, the bandwidth
coefficients; `lib/hardware.py`), so the engine computes a model's features
with the constants it was trained with. stage06
refuses a bundle whose feature names, dimensions or catalog hash differ from
the engine's, and loads the written file back with the tilewright module
before staging it; `deploy.apply_patch` requires the module.

The deploy gate measures the bundle that ships: stage06 evaluates the packed
model file, the bytes it would install, on the round's gate GEMMs, which no
bundle trained on, and writes `gate.json` (overall and per cell). An active
round deploys when its bundle reaches `deploy.target_sel_eff` there. The
final round with `deploy.on_final_round` deploys a bundle below the target
too, unless an earlier round deployed one and the new bundle is worse than
it on the same gate GEMMs. A bundle the gate keeps out is staged and recorded
with `deployed: false` and the reason; stage 7 and stage 8 validate the
latest bundle that was deployed, and the run summary and
`held_out_sel_eff_breakdown.json` report that bundle's own numbers (its gate,
and any stage04b that evaluated the same `models.pt`). `--deploy`
deploys every round, with the measurement recorded. With
`deploy.gate_frac: 0` nothing measures a bundle before it ships: only the
final round or `--deploy` deploys, unmeasured.

It always stages the result under `<round>/stage06/staged_for_deploy/`
(the model, an index fragment per target and `INSTALL.md`). With
`deploy.apply_patch` it installs the model into
`shared/tilewright/weights/hipblaslt/<t>/<t>/` for every directory `t` of
`deploy.targets` and registers the library stem in that directory's
`tilewright_index`. One model can serve several targets (for example the
gfx1250 MXFP8 model ships for `gfx1250v0`, `gfx1250` and `gfx1250-strict`,
the tree that holds the library `gfx1250v0` is built from); for a target that
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
set. With a v2 model, problems routed to a leaf without a model of its own or
of an ancestor, calls outside the exclusive execution context, and problems
for which the model scores no kernel keep the analytical ranking; a v3 model
scores every problem under the contexts it declares.

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
fails the stage. The runtime's picks are also joined to the bench timings of
their problems (every round's stage04 data) for the runtime's selection
efficiency, with a bootstrap interval; on problems sampled from the training
data (`stage07.parity_sample_trained`) that is mostly an in-sample number.

A second parity run requests `stage07.parity_top_k` solutions (8; 1 or less
turns it off): the runtime's top-1 at that request size and the order of the
solutions it returns must match the engine's ranking (the engine's kernels
first and in its order, the analytical ranking's after them; kernels the
runtime leaves out must be ones whose predicates may fail, which a bench
timing of the problem rules out). Either rate below the minimum fails the
stage.

The selection-time report gives the ML-on times and the ML-on minus ML-off
deltas per request size, overall and split into the first call into each
serving cell (which builds that cell's caches) and the warm calls, each
median with a bootstrap 95% interval (`stage07.bootstrap` resamples). The
ML-off times are the analytical ranking's and are printed only with
`--origami-reference`.

With `stage07.parity_sample_trained` > 0 stage 7 uses that many random
problems that benchmarked successfully during training instead of
`stage07.bench_yamls`.

Stage 8 evaluates the deployed bundle on external datasets (bench yamls,
bench log directories or enriched CSV directories). A bench yaml is benched
like the training shapes: each entry's M, N, K and batch count become a
`lib.bench_yaml` bench line with the config's data types, layout and bench
knobs (rotating buffer, flush, skip screen, adaptive timing). Its headline is
the model against the fastest measured kernel over every evaluated GEMM,
with the misses; Origami appears only on the same GEMMs (`paired`) and is
printed only with `--origami-reference`. Stage 8 is report-only: its
datasets never select cells or feed a decision.

## Run directory

```
runs/<config stem>_<timestamp>/
    .lock                        held by the running orchestrator
    configs/                     config snapshots, one per invocation
    orchestrator.log             the orchestrator's own output, all invocations
    timing/                      wall-clock per stage, one file per invocation
    round_<R>/
        stage01/ .. stage06/     stage outputs (stage04b only in active rounds,
                                 stage01b only with planner.enabled, stage03b
                                 only with bench.rebench.enabled, stage03c
                                 only with contexts)
        stage01.log ..           stage logs
        stale/                   outputs of re-run stages
        enhance.json             cells and retrain list of an enhance round
        .round_complete.json
    validate/stage07/ stage08/   validation outputs
    held_out_sel_eff_breakdown.json  the deployed bundle's own held-out numbers
    report.html                  self-contained report (scripts/make_report.py)
```

Run directories are not source: their logs, metrics and reports carry measured
performance numbers and must not be committed (`runs/` is ignored).

### Report

```bash
python scripts/make_report.py R [out.html] [--jobs N] [--no-engine] [--no-bench] \
    [--target T] [--selection-budget-us US] [--diagnostics] [--origami-reference] \
    [--shipped-subset FILE]
```

The report is one HTML file with no external resources, built from the
pipeline's outputs only. Its headline is the bundle the run shipped: the
latest `stage06/deploy_record.json` the deploy gate let through (a later
bundle staged with `deployed: false` is named, not reported), its
`gate.json` and every stage04b that measured the same `models.pt`, and
stage 7's runtime selection efficiency (with its bootstrap interval) and
selection time on the first call into a cell vs warm. Every comparison
between two models is paired (the same GEMMs).

The bench sections re-rank GEMMs with the shipped bundle
(`lib/run_analysis.py`; the tilewright module is needed for the ranking
steps), each in its execution context, and compare against the fastest
measured kernel of each GEMM. They charge picks with
`validate.pick_measurement` of the run's config (`lib/evaluate.py`, as
stages 4.5, 5, 6, 7 and 8), so their numbers match the stages'. The training
benches are those the bundle learned from; the held-out benches are its
deploy-gate slice and later rounds' new GEMMs; held-out contexts
(`stage03c/holdout`) are reported separately.

A kernel is its signature: macro tile M and N, depthU, matrix instruction
M, N and K, cache hints A and B. Library solutions sharing a signature are
its members, listed as a detail.

- **Where to improve**: every leaf ranked by expected gain = GEMM share x
  (target - efficiency), with the efficiency taken from the training-side
  evidence of enhance mode. It names one action per leaf:
  - more data: too few GEMMs, too few evaluated GEMMs, or in-sample far above
    the evidence;
  - retrain;
  - split: retrained `validate.split_after_attempts` times without reaching
    the target;
  - tier policy: the non-temporal rule, the whitelist or the cascade loses
    the most.

  It also lists the cells `--weakest` / `--below` would select.
- **Loss decomposition** (training and held-out benches): per leaf, the
  efficiency lost at each step of the engine's own selection. The steps
  are the library pool, the hard validity rules, the non-temporal rule,
  tier 1 (the whitelist), the cascade, and the ranking. Per GEMM, the best
  measured latency left after each step is clamped to be monotone, so the
  step factors multiply to the cell's selection efficiency. GEMMs without a
  pick are counted by the step that empties; misses (picks without a timing
  of their own under the pick measurement) are counted per reason.
- **Execution contexts**: selection efficiency per context, trained and
  held-out contexts separately (stage04b's own numbers and the report's
  re-ranking), and the weak cells of each context.
- **Kernel usage**, **Kernel parameters**: wins (oracle) and picks (model)
  per signature and per value of macro tile, depthU, matrix instruction and
  cache hints, overall and per cell. The removal loss is the oracle loss if
  the signature or value were removed. `--diagnostics` adds a collapsed
  section with framework-specific parameters (occupancy, vector widths,
  varying kernel attributes).
- **Pool-drop candidates**: signatures that never win and are never picked,
  with:
  - the GEMMs where they come within 1%, 2% and 5% of the best;
  - the number of GEMMs that tested or screened them.
- **Selection time per cell**: stage 7's ML-on selection time per serving
  cell, for every bench yaml and request size.

- **Shipped candidate set**: the table and recommendation of
  `shipped_subset.json` (below), when the run has one (`--shipped-subset`
  names another file).

Stage 8 results appear as validation only. They never feed the actions or
the selection. Origami's numbers (and stage 7's ML-off times) appear only
with `--origami-reference`, in a collapsed section labelled as an external
reference.

### Shipped candidate set

```bash
python scripts/choose_shipped_subset.py R [--out-dir D] [--sizes 0,32,64,128,256,512,1024,all] \
    [--model-file F] [--bank-stem S ...] [--pick-measurement warmup|timed|any] [--jobs N] \
    [--no-latency | --bench-binary B] [--bench-shapes 1000] [--bench-core C]
```

Chooses which planner-bank kernels (`planner.bank_library_stems`) to ship
next to the library pool. For the bundle the report describes, it computes
each GEMM's engine scores once over pool + bank and then simulates the
engine's pick under any shipped subset (`lib/shipped_subset.py`: view keys
split into hipBLASLt's launch classes, feasibility with the subset's
non-temporal availability, whitelist tier, cascade by view key, pool-order
ties and the pick measurement's charging). A greedy
forward selection on the training-side GEMMs of every round (never a gate
slice) orders the bank; every size B is measured on the GEMMs the bundle
never trained on, per execution context, with the perfect-scoring ceiling
and, with `scripts/experiments/bench_shipped_rank`, the p50 / p99 time of
one warm ranking call on one pinned core. That benchmark ranks with
hipBLASLt's launch model (`hipblaslt_launch_rule.hpp`, a C++ port of
`lib/hipblaslt_launch.py`) when the bundle records one, as the engine
requires, and the script checks its plans against the Python rule first;
`TILEWRIGHT_SHIPPED_RANK_BINARY` runs the rule's parity test. A win-count order is the
baseline. A run without held-out GEMMs yet (round 0) chooses on stage05's
training GEMMs and evaluates on its validation (early-stopping) GEMMs, the
split recovered from stage05's metrics.json and cells.json with stage05's
own split function (`lib/stage05_split.py`) and checked against every
trained cell's counts; when that fails it holds a hashed share of the
training-side shapes out of the selection (`--held-out-split`). The output
names the evaluation set. `<out-dir>/shipped_subset.json` holds the orders, the numbers and,
per selection-time budget (`--budgets`, default 10 and 20 us), the smallest
B within 0.002 of the best held-out selection efficiency that fits.

### Identity ceiling

`lib/identity_ceiling.py` measures what a kernel identity costs before any
model: given enriched GEMMs (stage04 rows with candidates and attributes,
exhaustive or planned) and a grouping of kernels, a perfect scorer of the
groups picks per GEMM the best group, which runs its first measured member in
pool order. `ceiling()` returns per-GEMM and geomean efficiency against the
best measured kernel plus group counts; `engine_and_signature()` compares the
engine's v3 grouping (signature, spec attributes and the launch model's
`launch_key`) with signature-only. Planned benches leave kernels unmeasured,
which makes the ceiling optimistic.

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
hipBLASLt passes to the engine, and `pci_chip_id`, the device's PCI chip id
for the bank's hardware predicates. Any key set here overrides the value read
from the KFD sysfs topology of the first device in `runtime.devices`.

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

**`planner`** (stage01b): `enabled` (false), `mode` (`exhaustive`;
`planned`, `auto`), `exhaustive_max_kernels` (512), `seed_per_cell` (8),
`ranker` (`neighbors`; `stage05`), `neighbors_k` (4), `model_file` (null),
`bank` (`none`; `auto`), `bank_library_stems` ([]), `top_k` (32), `families_top_f` (8),
`family_variants` (0 = all), `explore_random_k` (32), `min_coverage` (2),
`max_list` (512; at most 4096, the bench's limit), `probe_kernels` (3, kernels stage02 times
per shape), `seed` (42).

**`contexts`** (stage03c; see Execution contexts): `list` and `holdout`
(entries `{cu_budget, schedule, cotenant, fraction, name}`), `seed` (42),
`cotenant_binary` (default: `hipblaslt-cotenant` next to the bench),
`path_append` (directories appended to the wrapped bench's `PATH`; default
the ROCm `bin`), `replan` (false).

**`bench`**: `pool_only_library` (false; see The pool-only library
directory), `rebench` (stage 3.2: `enabled` (false), `tolerance` (0.03),
`runs` (3), `iters_factor` (2), `max_kernels` (16)), `blocks_per_gpu` (20),
`startup_grace_s` (900) and
`stall_timeout_s` (300) (stages 2 and 3 kill a silent bench), `warmup_pass`,
`skip_slow_solution_ratio`, `rotating_mb`, `rotating_target_blocks`,
`device_memory_gib` (default: the KFD sysfs topology), and `adaptive`
(`enabled`, plus hipblaslt-bench's adaptive timing fields `warmup_time`,
`sample_time`, `measure_time`, `max_measure_time` in ms, `min_iters`,
`max_iters`, `noise_threshold`, `stability_threshold`, `stability_window`,
`stability_interval`).

**`train`**: `workers` (1), `threads_per_worker` (0 = cores / workers),
`epochs` (200; passes over each cell's training GEMMs), `device` (`cpu`), `eval_sample_frac` (1.0), `smart_k` (10),
`smart_k_max` / `smart_k_min` (4) / `smart_k_saturation_eps` (0.005) for an
adaptive whitelist size, `min_cell_gemms` (50; cells with fewer GEMMs over all
rounds are not trained, a leaf without a model is listed for training once it
has this many, and each side of a split needs this many; every leaf that a
model of the previous round serves gets at least this many shapes per round,
`seed.shapes_per_cell` new ones topped up with re-used history shapes),
`validation_frac` (GEMMs per cell held out to choose the epoch; stage05's
default when absent). The student schedule keys below reach stage05 only when
set; absent, stage05's default applies (see *Student training (stage05)*):

| key | stage05 option | default | values |
| --- | --- | --- | --- |
| `batch_gemms` | `--batch-gemms` | 16 | GEMMs per optimizer step, >= 0 (0: all) |
| `lr` | `--lr` | 0.002 | peak learning rate, > 0 |
| `lr_schedule` | `--lr-schedule` | `cosine` | `cosine`, `step`, `constant` |
| `warmup_frac` | `--warmup-frac` | 0.05 | share of the steps warming up, [0, 1) |
| `lr_min_frac` | `--lr-min-frac` | 0.02 | cosine: final rate / `lr`, [0, 1] |
| `eval_points` | `--eval-points` | 40 | checkpoints measured as deployed, >= 1 |
| `patience` | `--patience` | 0 | checkpoints without a better one before stopping, >= 0 (0: off) |
| `student_temperature` | `--student-temperature` | `learned` | `learned`, `fixed` |
| `qat_frac` | `--qat-frac` | 0 | int8/int4: share of the schedule's steps trained with stored weights, >= 0 (0: off) |
| `qat_lr_frac` | `--qat-lr-frac` | 0.1 | peak rate of that phase / `lr`, > 0 |
| `boundary_margin` | `--boundary-margin` | 0 | octaves of neighbouring cells' GEMMs a cell trains on, >= 0 (0: off) |
| `boundary_weight` | `--boundary-weight` | 0.5 | loss weight of those GEMMs, (0, 1] |
| `off_list_sel_eff` | `--off-list-sel-eff` | unset | early stopping and deployed numbers charge a pick off its GEMM's planned list the winner's latency over this, (0, 1] (unset: the slowest latency measured) |
| `unmeasured_samples` | `--unmeasured-samples` | 0 | pseudo-negatives per planned training GEMM: feasible pool kernels off its list, >= 0 (0: off) |
| `unmeasured_margin` | `--unmeasured-margin` | 0.05 | their pseudo latency: winner x (1 + margin), >= 0 |
| `unmeasured_weight` | `--unmeasured-weight` | 0.3 | scale of their soft-CE target, (0, 1] |

**`validate`** (stage04b): `sel_eff_threshold` (0.95), `split_floor` (0.85;
0 disables), both numbers, `min_cell_gemms` (20), `split_min_improvement`
(0.02), `split_after_attempts` (2), `max_split_depth` (6),
`structural_split_min_octaves` (0 = off), `structural_split_max_depth` (-1),
`pick_measurement` (`warmup`; `timed`, `any`, see
*Kernels and execution context*; read by stages 4.5, 5, 6, 7 and 8),
`rebench_misses` (true: stages 4.5, 6 (gate) and 8 re-bench their misses on
the GPUs and decide on the measured numbers, same section),
`rebench_runs` (3, at least 1: runs of that re-bench).

**`full`**: `total_rounds` (3).

**`deploy`**: `target_sel_eff` (0.95; null disables; an active round whose
bundle reaches it on its gate GEMMs deploys), `on_final_round` (true: the
final round deploys below the target too, unless its bundle is worse than the
one deployed before on the same gate GEMMs), `stop_when_target_reached`
(false), `gate_frac` (0.2, at most 0.5: the share of each leaf's held-out
GEMMs of an active round kept out of its training for the deploy gate),
`weight_dtype` (`bf16`; one of `fp32`,
`bf16`, `int8`, `int4`; also the precision every evaluation uses),
`rocm_libraries_to_deploy`
(the deploy root, or null to only stage), `targets` (weights directories;
default `[arch]`; an entry is a directory name or `{dir, library_stem}`),
`apply_patch` (false), `build_after_apply` (false), `build_dir` (default
`<deploy root>/projects/hipblaslt/build/release`), `build_jobs`.

**`stage07`**: `parity_sample_trained` (0), `bench_yamls` ([]),
`timing_request_sizes` ([1]), `timing_repetitions` (3),
`parity_min_match_rate` (1.0), `parity_top_k` (8), `bootstrap` (1000).

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
    make_report.py       HTML report of a run (sections in report_sections.py)
    choose_shipped_subset.py  bank kernels to ship with the library pool
    collect_bench_failures.py
    convert_mlrec_v1_to_v2.py
    experiments/         CPU microbenchmark of the engine's ranking call
tests/                   pytest suite
```
