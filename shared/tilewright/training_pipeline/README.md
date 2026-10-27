# tilewright training pipeline

This pipeline trains the models the tilewright engine (`shared/tilewright/`)
uses to rank the candidate kernels of one hipBLASLt library, and deploys them
into a rocm-libraries checkout.

One run targets one Tensile library (`hipblaslt.library_stem`): one data type
combination and layout on one architecture. GEMM problem space is split into
96 cells by M, N, K and batch tier, and a run can split cells further where
the model needs it (the training recipes do not); every leaf cell gets its
own two-tower network. The pipeline benchmarks every kernel of the library
on synthetic shapes with `hipblaslt-bench`, trains the cell models on the
measured latencies, checks each round's model on the next round's unseen
shapes, and writes the result as one model file that hipBLASLt loads next
to the library.

The pipeline does not use origami. The analytical baseline it compares
against is the kernel hipBLASLt's own ranking puts first in a bench run
without tilewright (`is_origami_pick` in the enriched data), and all model
evaluation goes through the tilewright engine itself.

## Setup

- Python 3.10 to 3.13 with `pip install -r requirements.txt`.
- The tilewright Python module, built from this repository:

  ```bash
  pip install shared/tilewright/python
  ```

  It needs CMake and a C++17 compiler, and no ROCm. Training, evaluation,
  deployment and parity all rank through it.
- A built hipBLASLt for the target architecture (the *bench root*):
  `projects/hipblaslt/build/release/clients/hipblaslt-bench` and
  `projects/hipblaslt/build/release/Tensile/library/<arch>/` holding the
  configured library, where `<arch>` is the name the bench device reports
  (`gfx950`, `gfx1250`, ...). `runtime.build_dir` overrides the build
  directory.
- Bench devices that load that subtree. Which subtree a device loads can
  depend on its mode: a gfx1250 device of ASIC revision 0, which no config
  here trains for, reports `gfx1250-strict` only when the environment of
  the run has `HSA_DISABLE_GFX12_STRICT=0`; by default it reports `gfx1250`
  and hipBLASLt loads its `gfx1250v0` subtree, the gfx1250-strict logic
  built for gfx1250 (a deploy target, not an `arch`). A config for such a
  mode sets the variable in `runtime.env`: the orchestrator puts
  `runtime.env` in the environment of every stage, and stages 3 and 7 pass
  it to every bench and cotenant they start, so the command needs no
  exported variable. For an `arch` whose subtree depends on the device's
  mode like this, preflight checks every device of `runtime.devices`
  against it, from the device's KFD topology node and that environment.
- For deployment, a rocm-libraries checkout (the *deploy root*, often the
  same checkout) and, for `deploy.build_after_apply`, its configured hipBLASLt
  build directory.

### Model files

The deployed models live with hipBLASLt's library logic, in the `Tilewright`
directory of each library subtree:
`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/<subtree>/Tilewright/`
(`lib/weights_layout.py`), as `*.tilewright.bin`, committed as plain binary
files (`projects/hipblaslt/.gitattributes`); new or retrained models are
committed the same way.

## Running

```bash
scripts/run_pipeline.sh configs/<config>.yaml --mode full
```

`run_pipeline.sh` exports default roots and runs `run_orchestrator.py` with
the remaining arguments. Configs refer to host paths through environment
variables, which the orchestrator expands in every string; it refuses to start
when a referenced variable is unset or empty, and names it. Outside a
rocm-libraries checkout (a standalone copy of the pipeline) the launcher
leaves the roots unset and prints `unset` for them: set the ones the config
refers to, or give the config absolute paths.

| Variable | Meaning | Default in `run_pipeline.sh` |
|---|---|---|
| `TILEWRIGHT_BENCH_ROOT` | checkout whose hipBLASLt build is benchmarked | the rocm-libraries checkout holding the script; unset outside one |
| `TILEWRIGHT_DEPLOY_ROOT` | checkout that receives the model | the rocm-libraries checkout holding the script; unset outside one |
| `TILEWRIGHT_DATASETS` | directory of stage08 datasets, for configs that list any | unset |
| `TILEWRIGHT_AUDIT_ROOT` | directory of audit sets, for configs that name one | unset |
| `PYTHON` | interpreter with the requirements and the tilewright module | `python3` |

Benchmarks never inherit `HIPBLASLT_TENSILE_LIBPATH` (`run_pipeline.sh` also
unsets it), so hipBLASLt loads the library subtree that matches the device's
ASIC revision and mode, and `arch` must name that subtree (see *Setup*);
stage 7 sets it to the library directory it serves the model from. Nor do
they inherit `TENSILE_USE_TILEWRIGHT` or any `TILEWRIGHT_*` or
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
python run_orchestrator.py --config C --mode audit --run-dir A --source-run-dir R
```

Without `--run-dir` a run goes to `runs/<config stem>_<timestamp>/`.
`--deploy` deploys after every round, round_0 included. `--no-report` skips
writing `report.html`. `--origami-reference` adds Origami's numbers, as an
external reference on the same GEMMs as the model's, to the printout of
stages 4.5, 6, 7 and 8 and of the run summary, and the report's reference
section; they never enter a decision. `report.origami_reference: true` in
the config does the same, so that one command ends with the report's
reference summary (see *Report*).

With `audit.dir` set, one `--mode full` command also builds the audit set
when it is not complete (`audit.build`), measures every round's bundle on
it under protocol v2 and the sustained composite (`audit.v2`,
`audit.sustained`), and at the end can retrain every cell on every round's
data and ship that bundle when the audit set finds it not worse under the
official protocol (`audit.official`, `full.final_retrain`): see *Audit
set* and *Final retrain*.

### Enhancing chosen cells

`--mode enhance` adds active rounds to a run with at least one complete round,
restricted to chosen cells. Every enhance round runs the active-round stages
on those cells only: stage01 draws `--shapes-per-cell` new shapes (default
`seed.shapes_per_cell`) for the leaves under them, including leaves without a
model of their own, without the weak-cell budget or the very-large-K stratum
(see *Weak cells and very large K (stage01)*); stages 3-4 bench and enrich
them; stage04b measures the
current model on them; stage05 retrains them (`--only-cells`) and carries the
other cells forward, except those trained with other model-shaping options
than the round's, which it retrains too (see *Stages*). The current model is
the run's latest bundle: the last round's, or a shipped final retrain's (see
*Final retrain*), whose cells stay as they are. A chosen leaf is
retrained when it has at least
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
that can never be met; what only stage 7 needs is a warning in `--mode
audit`, which runs no stage 7) and runs preflight checks: the bench binary,
the library directory, the configured library stem, the library subtree each
bench device loads (for an `arch` whose subtree depends on the device's ASIC
revision and the runtime's mode, such as gfx1250-strict; any other arch is
not checked), the tilewright module and its feature catalog (with
`train.catalog: v3`, one that reads the CU floor), and that no
shape list the config names (`audit.held_out`, `stage07.bench_yamls`,
`stage08.datasets`) is a registered never-train set (see
*Audit set*). Every mode that
ends with validation also checks what the enabled validation stages need:
stage 7's build (its bench, and that its TensileLite holds `MLREC_v3`, or
`MLREC_v2`, the format `train.catalog` trains, with `MLREC_v3` also
`CU floor`, the name its engine's loader gives the CU floor every v3 model
declares, when the kernel pool has a
bank `TENSILE_TILEWRIGHT_BANK`, the setting that serves it, and, when the
correctness gate is on or `correctness.lists` names a kernel of the pool,
`.tilewright.blocklist`, the reader of the blocklist sidecar) and every
stage 8 dataset (it exists, as a bench yaml or a directory of `block_*.log`
or `chunk_*.csv` files, and one of its first 1,000 problems, read file
after file and only until one matches, has the config's data types and
layout). `--mode full` also checks that `audit.dir`, when set, holds an
audit set when `audit.build` is off (an error without one; with `before`
or `parallel` the run builds or extends it, and the training modes check
that it can, and in every mode that it was built with the run's settings
and contexts: see *The audit set in the one command*); that stage 7, when
it is on, will have a bundle to validate (an error when the run has
deployed none and neither a round nor the final retrain will, as in a run
of one round without `--deploy` or `full.final_retrain`, or with
`deploy.on_final_round` false and no `deploy.target_sel_eff` or gate GEMMs
to reach it with; a warning when only a round that reaches the target, or a
final retrain its gate lets through, deploys); and the free space of the
run directory's file system. That space is judged by `runtime.min_free_gb`
when it is set (0: no check), else by the largest complete active round:
an error when it cannot hold one more such round, a warning when it cannot
hold the rounds left. With only the initial round complete, whose seed
benches make it larger than an active round, its size gives a warning only.
With `seed.weak_cell_factor` above 1, a round with more weak leaves than
the rounds before it draws more shapes than they did (see *Weak cells and
very large K (stage01)*), which this estimate does not foresee: a run with
little space to spare sets `runtime.min_free_gb`.
Before every round the orchestrator checks the free space the same way and
stops with 2 on an error. It also resolves the kernel pool the config names
on the first device (an error when it cannot) and prints what the pool is
made of (see *Kernel pool*). `--skip-preflight` skips the checks before the
first stage.

### Resuming

Every finished stage writes `.stage_complete.json` into its directory and
every finished round writes `.round_complete.json`. Re-running the same
command resumes: complete rounds are skipped and an interrupted round restarts
at its first stage that did not complete. `--from-round R` (full mode) re-runs
round R and all later rounds; `--from-stage N` (1, 2, 3, 3.2, 3.5, 3.7, 4, 4.5, 5, 6;
7 or 8 in validate mode; or the stage's name, such as `stage02` or `stage03c`)
re-runs the target round from stage N, provided the earlier stages completed.
A stage that runs again starts in an empty directory: the previous attempt's
directory and log move to `<round>/stale/<timestamp>/`.

The benchmark planner is stage 2, in `stage02/`, the number of the removed
probe stage. It was stage 1.5, in `stage01b/`: `--from-stage 1.5` and
`--from-stage stage01b` still name it, with a deprecation note. The
orchestrator records the layout in `run_layout.json` before the first stage
it runs in a run or audit directory, unless the directory was written
before the rename: one without that file that holds a `stage01b/`
directory, or a `stage02/` directory without a `plan.json`, at its top or in
a round. Such a directory stays as it is: every reader (the stages, the
final retrain, the audit tools, the report) and every resume keep its
planner in `stage01b/`, the rounds a resume adds included, and nothing in
it is renamed or moved (`lib/run_layout.py`). Its rounds may also
hold the `stage02/` outputs of the removed probe, which calibrated the
benches; no stage, resume, report or script reads or moves them, and a
round interrupted after its probe resumes at stage 3 on its own bench
lines, which stage03 times adaptively (see *Bench timing*).

Each invocation writes its expanded config to
`<run>/configs/<timestamp>_<mode>.yaml` (and the file as written to
`*.source.yaml`) and passes that snapshot to every stage, so editing the config
file never changes a run that is already going. The snapshot states the bench
timing the invocation used: every `bench.adaptive` field,
`skip_slow_solution_ratio`, `warmup_pass`, `warmup_line`, the `canary` and
the `rebench` options are written out at the values the stages apply, also
where the
config left them to their defaults, and its first comment lines sum them up,
including whether the bench lines carry the screen prepass (see *Bench
timing*).

One orchestrator at a time uses a run directory: it holds `<run>/.lock`, and
a second invocation on the same directory exits with 2.

Every stage runs in its own process group, and stage 3 runs each bench in its
own group too. SIGINT, SIGTERM or SIGHUP stops the running stage
(SIGINT to its group, SIGKILL after 30 s), the stage stops its benches, and
the orchestrator exits with 128 + the signal number; the same command resumes
the interrupted round.

An orchestrator that dies without stopping its stage (SIGKILL, the OOM
killer) does not leave it running into the next attempt:

- every stage gets SIGTERM when the orchestrator dies (Linux
  `PR_SET_PDEATHSIG`), and so does a sub-stage when the stage that runs it
  in its own process group dies (stage03 under stage03c and stage03d,
  stage08, the seed bootstrap and the miss re-benches); stage03 then stops
  its benches;
- every stage starts with a fresh attempt token in its environment
  (`TW_PIPELINE_ATTEMPT`, `lib/attempts.py`), which every process it starts
  inherits, benches and cotenants included. `<run>/.running.json` names the
  running attempt (token, pid, process group, command, the run directory and
  the orchestrator's pid and start time). When a stage ends, the orchestrator
  stops every process still carrying its token; the next invocation on the
  run directory stops the processes of any attempt `.running.json` still
  names (SIGTERM, SIGKILL after 30 s) before it takes a GPU lease or moves an
  output aside, and exits with 2, naming them, when one does not stop. It
  stops only the attempts of an orchestrator of this run directory that is
  gone: an entry of another run directory (a copy of a run directory made
  while a stage ran carries the file) is dropped with a warning and nothing
  is signalled, and an entry whose orchestrator still runs (the directory
  was replaced under it) stops the invocation with 2;
- the stage inherits the descriptors of its GPU leases, so a lease lasts as
  long as the stage;
- stage03 writes its bench yamls into `tmp/<its pid>/` and removes only that
  directory, so a stale attempt never removes a new one's files.

A stage that fails stops the run with its exit code (a stage killed by
signal N gives 128+N), except a failure that is likely transient: a GPU
lease another process tree still holds after `runtime.gpu_lease_wait_s`
(3600 s), exit status 2 of a stage that benches (stages 2 with the seed
bootstrap, 3, 3.2, 3.5 and 7), which these stages give only for trouble with
their GPUs (crash blocks beyond tolerance, repeated quarantines, every GPU
retired, a refused lease or platform check, a seed or context bench that
failed so, or one of their sub-stages killed by SIGHUP, SIGINT, SIGKILL or
SIGTERM), or a death of the stage by SIGHUP, SIGINT, SIGKILL or SIGTERM that
the orchestrator did not send (the OOM killer's, for one). Such a stage is
run again up to `runtime.stage_retries` (1) times, the first time after
`runtime.stage_retry_backoff_s` (300 s) and then after twice the last wait,
each attempt in a fresh directory (the failed one moves to `stale/`); its
marker and the timing records name the attempt. Setup, config and identity
errors exit with 1 and are not retried, nor is a lease that cannot be taken
for another reason than a holder (devices the environment does not show), a
crash (a death by any other signal, such as SIGSEGV) or a stage that does
not bench (argparse also exits with 2, so a bad argument to a stage that
benches is retried once). Exit status 3 of a stage that benches (stages 3,
3.2, 3.5, 3.7 and 2 with the seed bootstrap pass their stage 3's through)
means a GPU or the host is wedged (*Wedged GPUs*, under *Shared hosts*): it
is never retried, and no stage whose GPUs a wedge holds starts (`not
started: the GPUs it benches on are wedged`, also exit 3), whatever
`runtime.stage_retries` says. The run stops until the host reboots, which
lifts the hold by itself, or the hold is lifted with `scripts/run_guard.py
accept`; the same command then goes on.

stage05 exits with 3 after writing its outputs when a cell
fails or a cell stage04b listed cannot be trained (a cell whose training
worker failed has been trained once more in stage05's own process). With
`train.on_cell_failure: stop` (default) the round then stops before
deployment (`round_<R>/stage05/metrics.json` lists `cells_failed` and
`cells_not_trained`), and the same command reruns stage 5. With `carry` the
round goes on with the bundle stage 5 wrote, in which such a cell keeps the
model an earlier round trained for it, or has none of its own and is served
by its nearest trained cell; the cells, why and what serves them are logged
and recorded in `round_<R>/stage05/cells_carried.json`, the round's marker
lists them (`cells_carried`), the end of the run names them again and the
report counts them per round and names them in its notes. `carry` stops the
round as `stop` does when stage05 trained none of the cells it was to train,
or left more than `train.carry_max_fraction` (0.25) of them, since a
failure that wide is not one cell's. A round stopped by `stop` goes on with
the bundle stage 5 already wrote when the same command runs again with
`carry` (stage05 records its arguments and the pin in
`stage05/.untrained_bundle.json`; other arguments, another pin or
`--from-stage` run stage 5 again). Before it reads
any GEMM, stage05 packs an untrained one-cell model of the format it trains
(catalog, serving options, weight dtype and, for MLREC_v3, declared contexts
with a CU floor) and loads it with the tilewright module; a module that
cannot load it or reads it back otherwise (for example an older build that
knows only MLREC_v2, or none of the CU floor) stops the stage at once with
1, naming the module and the format.

A validation stage that passes records in its stage marker what it vouches
for: stage 7 the deploy record, the model's sha256, its build with the
size and time of the libraries its bench loads, and the `stage07` settings;
stage 8 the bundle's training directory and `models.pt` sha256, the weight
type, the `stage08` settings and the evaluation settings it reads. Every
`--mode full` invocation, also one whose rounds are all complete, runs each
enabled validation stage that has no pass for what it would validate now
(the bundle the run deployed last, a round's or the final retrain's; for
stage 8 the latest round's when none was deployed), so the same command
retries a failed validation, validates a newly deployed bundle, and runs
nothing once both passed. A training run
exits non-zero while an enabled validation stage has not passed: with the
stage's exit status when it failed, and in `--mode full` with 2 when stage 7
has nothing to validate because the run deployed no bundle (a run of one
round deploys only with `--deploy`, which `--mode full` refuses at its
start); the other training modes warn instead. An incomplete end of a full
run (a failed audit set build or final retrain, or a final retrain the audit
set cannot decide yet) gives the run's exit status before validation does.
`--mode validate` runs the stages it names again.

`<run>/run_status.json` sums the run up: the planned and complete rounds and
the round an invocation stopped at, the bundle the run deployed last (a
round's, or the final retrain's, `final_retrain`), the final retrain (see
*Final retrain*), the audit set's state, each validation stage (`passed`,
`failed`, `not_run` with the reason, `stale` for a pass of another bundle,
`passed_unrecorded` for a pass whose marker predates this record, which the
next `--mode full` runs again, `pending`, `off`), the report, and the
invocation that wrote it with its exit status.
`complete` holds once every planned round is complete, an enabled final
retrain is decided (and deployed when it ships), and every enabled
validation stage passed for the current bundle. The orchestrator rewrites
it when an invocation starts, after each round and when it ends.

**Pin.** A run is produced with one pipeline code, one engine module and one
config (`lib/pinning.py`). `<run>/pin.json` records them when the run's
first stage completes (`first_stage` names it, so a run resumed from before
pins existed is pinned from there on): the digest of the pipeline's Python
files that produce its data and models (`run_orchestrator.py`, `lib/`,
`stages/` without stages 7 and 8, which only validate) with each file's
digest, the checkout's revision and whether its tilewright tree has
uncommitted changes, the tilewright module the stages import (path, digest
of its files, feature catalogs), and every pinned config key at the value
the stages apply. Every stage marker and config snapshot records the pin it
ran with. An invocation whose code digest, engine digest, catalogs or pinned
config differ exits with 2 before any stage runs and names each change and
the files behind a code change (`config train.epochs: 200 -> 100`;
`pipeline code: ...; files: lib/planner.py`; a file that cannot be read is
named as unreadable); `--allow-code-change` accepts them and records them in
`pin.json`'s history. Unpinned keys may change freely: where and how
patiently the run executes (`runtime.devices`, `gpu_lease_wait_s`,
`stage_retries`, `stage_retry_backoff_s`, `min_free_gb`,
`bench.stall_timeout_s`, `startup_grace_s`, `train.workers`,
`threads_per_worker`, `pooled_threads`, `pooled_teacher_threads`), how far
it goes (`full.total_rounds`), what it does with an untrained cell
(`train.on_cell_failure`, `carry_max_fraction`),
where it installs a bundle (`deploy`'s checkout and build keys), how it
builds and measures with the audit set (`audit.build`, `audit.devices`,
`audit.v2` and the width of its sets), the final retrain
(`full.final_retrain`, which records its own options and trains again when
they change), what it validates (`stage07`, `stage08`) and what it reports
(`report`, and `planner.recall_min_ceiling`, below which stage02 warns about
its seed recall). The near-winner re-reads' `tolerance_us`, `inline_copies`
and `align_to_original` are pinned only away from the values at which they
change nothing (0, 1, false), so that a run or audit directory produced
before they existed keeps its pin. `audit.held_out` is pinned only when it
names a list, together with the digest of each list's GEMMs, so that a list
that changes what the rounds may not draw is a change the next invocation
names; an audit directory does not pin it. `contexts.granule` is pinned
only at a granule other than 32, the one every budget had before the key
existed (`PREDATING_GRANULE` of `lib/contexts.py`), which a pin without the
key means whatever the default becomes. The weak-cell budget's keys
(`seed.weak_cell_factor`, `strong_cell_factor`, `strong_cell_margin` and
`large_k_factor`) are pinned only away from their defaults (1, 1, 0.03 and
1), so that a run produced before them keeps its pin; an audit directory
does not pin them. Stages
start from the files on disk, so the orchestrator compares the code and
engine digests again before every stage starts and before it marks one
complete: a change while it runs stops the run with 2, and the same command
with `--allow-code-change` goes on with the new code.

Two kinds of invocation keep less of the pin. An audit directory, which runs
of other code share and extend, keeps only the config keys its benches read
(not the training, deploy and validation sections, nor the planner keys an
exhaustive bench ignores, nor the `hazards` section, whose quarantine only
leaves kernels out of the pool the set's completeness is checked against);
another code or engine module is recorded in its `pin.json` with a warning.
`--mode validate` produces no data or model of the run, so it names every
difference and goes on, and leaves `pin.json` as it is.

### GPUs: leases, quarantine and platform records

The latencies the benches measure are the training labels, so a bench must
have its GPU to itself and the settings it ran under must be on record.

**GPU numbering.** A GPU's index is the one `ROCR_VISIBLE_DEVICES` takes,
which is the order of the GPU's node in the KFD topology
(`/sys/class/kfd/kfd/topology/nodes`). That need not be PCI bus order, and
`amd-smi list` numbers GPUs by bus address: on some hosts amd-smi's GPU 0 is
not ROCR's GPU 0. A stage's `--devices` are indices into the GPUs its
environment makes visible (`ROCR_VISIBLE_DEVICES`, then
`HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`); with
`ROCR_VISIBLE_DEVICES=0,3,4` device 1 is GPU 3. `lib/kfd.py` maps devices
to GPUs and reads each GPU's KFD gpu_id and PCI address; stages check the
mapping against `amd-smi list --json` (which names the KFD node and gpu_id
of each GPU it lists) when amd-smi is installed. Every bench of stages 3
(and so 3.2, 3.5, 3.7, the audit parts and the miss re-benches) and 7 sees its
GPU alone: it runs
with `ROCR_VISIBLE_DEVICES=<the device's GPU>`, without
`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`, and with `--device 0`
(hipblaslt-bench also allocates memory on device 0 of the GPUs it sees).
Logs keep the stage's device index in their names (`..._gpuD.log`).

**Leases.** A stage that benches holds a lease on each GPU it uses
(`lib/gpu_lease.py`): an exclusive `flock` on `gpu<N>.lock` in the lease
directory, whose content names the holder (pid, user, purpose, run
directory, start time). The orchestrator takes the leases of a stage before
starting it and releases them when it ends; the stage, its sub-stages and
their benches work under them. A GPU that another process tree holds is left
to it: the stage benches on the others, and the stage log names the holder
(`GPU 3 left to its holder: pid ... for 'round_2 stage03' in <run> since
...`). Only while every GPU of a stage's devices is held does the stage wait,
for `runtime.gpu_lease_wait_s` (3600), for one of them; then it stops before
it starts, with exit code 2 and a message naming the holder (`refused (one
process tree per GPU): GPU 3 is leased by pid ...`), and is retried (see
*Resuming*). The wait is logged when it starts, naming the holders, then
every five minutes (`scripts/gpu_run.py --wait`, which takes one GPU, logs
its wait too). The stage inherits the leases' descriptors, so they last until
it exits even when the orchestrator dies first. Stages run on their own
take the same leases.
A stage leases the GPUs of its devices only. The stages that lease are 3, 3.2,
3.5 (also as the parts of an audit), 2 with the seed bootstrap, 4.5, 6 and
8 when they re-bench misses, and 7 (its one device). A lease ends with its holder's process; a lock file left
behind is free.

The lease directory is `TILEWRIGHT_GPU_LEASE_DIR` when set, else
`tilewright-gpu-leases` in the system temp directory (not `XDG_RUNTIME_DIR`,
which is removed when the user's last login session ends). Only processes
that use the same directory see each other's leases: give every run and
experiment on a host the same environment.

For ad hoc work, `scripts/gpu_run.py` runs a command on one GPU under its
lease:

```bash
python scripts/gpu_run.py --gpu 1 --purpose "kernel sweep" -- ./sweep.sh
python scripts/gpu_run.py --status        # who holds which GPU
```

The command runs with `ROCR_VISIBLE_DEVICES=1` and without
`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`, so GPU 1 is the only
device it sees (its device 0, also for a `hipblaslt-cotenant --device 0`).
The lease lasts until the command and the processes that inherited it exit;
a held lease is refused at once (or after `--wait S` seconds) with the
holder's record, and the exit status is the command's (2 for a refusal).
Processes that already hold memory on the GPU are named on stderr.

**Quarantine.** While a bench of stage 3 runs (stage 3 also runs the
benches of stages 3.2, 3.5, 3.7, the audit parts and the miss re-benches), the
stage samples which
processes hold memory on its GPU (`/sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>`,
readable without root) with the stall watchdog's polls (at most every 5 s)
and once more when the bench exits. A process that is not the stage's own
(a bench it started, or a process the bench started, such as the cotenant)
is another tenant, named by the reading that sees it. So is unattributed
use: VRAM of the card in use
(`/sys/bus/pci/devices/<PCI address>/mem_info_vram_used`) beyond what the
processes KFD lists hold plus a fixed 0.5 GiB allowance, in every reading
(0.25 s apart) over its settle window. That is memory of a process whose
KFD entry we cannot see (another container's, for one) or of a graphics
client; the allowance covers the driver's own use of an idle card, a
fraction of it that does not grow with the processes' use. Shorter
excursions are normal: memory a process frees, or holds when it exits,
stays in the card's use for a while after the process's KFD entry stops
counting it, the longer the more memory (a bench that held most of the
card leaves it there for seconds after it exits). So the window is 2 s
plus 1 s per GiB of the larger of the unattributed memory and the most the
block's benches held on the GPU at one reading, at most 30 s
(`lib/kfd.py` `settle_window_s`), and the check after a bench exits, or
before a block starts, waits that long for the memory to go before it calls
it someone else's. The window costs sensitivity: unattributed use that
comes and goes within it is not seen (after a bench that held tens of GiB,
anything shorter than 30 s), and while unattributed use lasts every check
of the GPU waits out the window before it returns, the watchdog's polls of
a running bench (which meanwhile check neither for a stall nor for a fatal
error) and `scripts/gpu_run.py`'s warning before a launch among them. In
stage 3 other use ends the block: the bench is
killed, the block's logs move to `quarantine/<block log stem>_q<N>/` with
`quarantine.json` (the rule: `foreign-process`, `unattributed-memory` or
`gpu-numbering`; the GPU; each other process: pid, command, user, start
time, memory, when it was first and last seen; the unattributed bytes with
the card's and the processes' use at the largest reading and the settle
window they outlasted), the stage log
names it, and the block goes back on the work queue to be benched again, at
most `--max-quarantines` (2) times. A GPU on which the other use is still
present after a quarantine, or before a block starts, is retired: it is
re-checked every 10 s and takes blocks again after 3 minutes without other
use. While every GPU is retired the stage waits for one to return, logging
each minute, for at most 30 minutes, and then leaves the remaining blocks.
The stage fails (exit 2) when a block is quarantined more often, when blocks
are left because every GPU stopped taking them, or when a bench holds memory
on a GPU other than its own (the GPU numbering is wrong; that GPU is retired
for good); `stage03_summary.json` lists the quarantines, the GPUs still
retired and why (`retired_devices`) and the ones that returned
(`returned_devices`). The check needs the stage to run in the host's pid namespace (KFD names
processes by host pids); elsewhere, and for GPUs the KFD topology does not
list, the stage warns that it cannot check.

Canary.** Other load on the host slows the benches, a launch-bound GEMM the
most, and the latencies of a block are only comparable with the others when
they were taken under the same conditions. With `bench.canary.enabled: true`
(stage03 `--canary`; off by default), every bench process of stage 3 (and so
of 3.2, 3.5, 3.7, the audit parts and the miss re-benches) benches a canary
right before the block's shapes (after the warm-up shape) and again after
them: the stage's least-work bench line restamped to time one kernel alone,
without a screen, `bench.canary.timings` (3) times in a row. The kernel is
the first the line lists; on a line without a list, the first the library
holds for the problem (`algo_method: 1`, `requested_solution_num: 1`); on a
block-scaled (MX) line without a list, which hipblaslt-bench does not
enumerate that way, the lowest solution index of the first library file the
stage's `--identity-stem` names (the orchestrator names the pool's first),
and without `--identity-dir` the stage runs without a canary. No ranking
picks the kernel. The canary entries are named apart from each other, like
every entry of a bench process's yaml (see *Bench timing*). The median of
each group is a reading; the output goes to `logs/canary_<block
id>_gpuD.log`, out of stage04's way. A GPU's reference is the median of
every reading taken on it during the stage. An attempt with a reading more
than `bench.canary.tolerance` (0.1) above its GPU's reference, and, with
`bench.canary.tolerance_us` above 0 (stage03 `--canary-tolerance-us`; 0, the
default, sets no floor), more than that many microseconds above it, drifted:
its logs move to `drift/<block log stem>_d1/` with `drift.json` (the
readings, the reference, the tolerance and its floor, the GPU, the canary
kernel's signature) and the block goes back on the work queue, to be benched
no sooner than `bench.canary.backoff_s` (120) later; when its next attempt
drifts too, that attempt's data is kept and the block flagged. While the
stage runs, an attempt is judged against the readings taken so far, its own
among them, which judges a GPU's first blocks hardly at all; so at the
stage's end every kept attempt is judged again against its GPU's final
reference, and one over it is flagged as well (not benched again). A flagged
block's log gets a `*** CANARY FLAGGED` line, and stage 4 marks every row of
that log (`canary_flagged`). `stage03_summary.json` gives the canary line,
its tolerance and floor (`tolerance`, `tolerance_us`), per GPU its reference
and the canary kernel's signature (`kernels`; the name's macro tile and hash
without `--identity-dir`), per block its readings, the blocks benched again,
flagged and flagged at the end, where the canary was turned off and why. The
canary is blamed for a bench process that fails only once its own output has
started: failing in the canary before the block's shapes, the process leaves
the shapes to a new one without the canary, and none of them is charged for
it; failing in it after the shapes, or getting no timing, it is turned off
on that GPU. A process that dies before its canary printed anything is
charged to the shape like any silent death. A process with the canary whose
output does not line up with its entries may have left shapes in the canary
log: what it wrote moves to `misaligned/<block log stem>_s<N>/` with
`misaligned.json`, the canary is turned off on every GPU, and a new process
takes the shapes. The stage log names every GPU the canary went off on.
Audit protocol v2 benches its parts without it, since it reads a canary of
its own around each part.

Shared hosts.** A label is a latency, so a bench must not run while other
work on the host slows it or on a GPU that is no longer in the state the run
started in, and a line that hangs must not hold its GPU for good. Every
bench of stage 3 (and so of 3.2, 3.5, 3.7, the audit parts and the miss
re-benches) runs under four guards, the first three on by default
(`lib/host_load.py`, `lib/gpu_health.py`). `stage03_summary.json` records
what they saw (`guard`, with the wedged GPUs, the lines that wedged their
GPU and the device probes of *Wedged GPUs*; per block `host_load`, `aborts`,
`hung`, `wedge_lines` and `health_events`), `platform.json` too
(`host_load`, `health`), and `<run dir>/guard/guard.json` what the whole run
decided: its GPUs' baseline, its hold, its canary and its quarantined lines
(`scripts/run_guard.py status <run dir>` shows it). The orchestrator names
the guard directory, its invocation and itself to every stage
(`TILEWRIGHT_GUARD_DIR`, `TILEWRIGHT_GUARD_SESSION`,
`TILEWRIGHT_GUARD_ROOT`); a stage run on its own keeps one in
`<out-dir>/guard/`, and `scripts/audit_v2.py run` on its own gives its
workers one in `<audit>/v2/guard/`.

*Quiet cores* (`bench.quiet_host`). Stage 3 samples the CPUs it may run on
(its affinity, which its benches inherit: pin the run with `taskset`) in
windows of `every_s` (10) seconds from `/proc/stat` and subtracts what the
run's bench stages used in each: its own process tree and, under the
orchestrator (or `scripts/audit_v2.py run`), every other stage 3 of the run
with its benches, such as the audit's per-part stages on the other GPUs.
The rest is other load, the run's own training beside a parallel audit
build included. A block (a slice of the stage's lines on one GPU, the unit
its GPUs take work in) starts only when the latest window is quiet: other
load at most `quiet_busy` (0.15) of the cores and, while nothing of the
stage runs, no core busier than `quiet_core_busy` (1.0, which leaves this
rule out: on many cores one busy core does not reach a bench; lower it for
a stage pinned to a few). The first window is 2 s, so that a stage hardly
waits for it. A GPU that waited for a block (its stage's last blocks still
run elsewhere) checks again once it takes one and puts it back when the
cores are not quiet. A block that sees `abort_samples` (2) windows of other
load above `hot_busy` (0.25) is stopped: its logs move to `aborted/<block
log stem>_a<N>/` with `aborted.json` (why, and the load of the windows it
ran in), and it is benched again once the cores are quiet, at most 10
times. The stage stops waiting after `max_wait_s` (28800 s, 0: no limit) and
fails with exit 2, which the orchestrator retries. The windows go to
`host_load.jsonl` as they are taken. On a host where `/proc/stat` gives
none of the stage's CPUs the stage log says so once and every block starts.

GPU health* (`bench.health`). The first time a run checks a GPU it records,
by the GPU's PCI address, its power management state (`pp_features` lists an
enabled feature: the test amd-smi's `POWER_MANAGEMENT` makes), performance
level, power cap and uncorrectable ECC count (the sum of the `ue:` counts in
the files of its PCI device's `ras/`) as the run's baseline; a value the GPU
does not give is not checked. Power management off is a departure whatever
the baseline holds, also when the GPU was off at the run's first check: no
GPU benches without it. Before every stage on GPUs (on the GPUs it leased;
the correctness gate on its `correctness.devices`, the hazard canary on its
`hazards.devices`), and every `poll_s` (30) seconds while stage 3 benches,
the GPUs are compared with the baseline, each departure read again 2 s
later, and, unless `ttm` (true) is off, the host is checked for kernel
workers of the ttm memory manager in D state in each of 10 readings 0.5 s
apart, as a GPU reset leaves them. A departure of a GPU (its power
management, performance level, power cap or ECC count) holds that GPU alone:
stage 3 stops the blocks it is benching on it (into `aborted/`, as above;
they go back on the work queue, for its other GPUs) and gives it none, while
its other GPUs go on, and the orchestrator starts no stage whose GPUs are
all held, nor the correctness gate, the hazard canary or stage 7 on a held
GPU. Stage 3 checks its GPUs before its platform check (*Platform records*),
which leaves out each GPU the run holds for a departure of its own (`held`
in `platform.json`): that GPU benches once the run no longer holds it and
its settings agree with those of the stage's other GPUs. A GPU is compared
with them again whenever the run stops holding it or takes its state as the
run's (`accept`, below), so one whose new state was accepted while the
others kept theirs waits (the summary's `guard.health.platform` lists the
GPUs held at the start, those that joined and those found unlike the
others). An audit worker takes no part while the run holds its GPU (see
*Protocol v2*). The host's problems (the ttm workers) and departures that a
check finds on every GPU it reads, two or more (a host-wide condition, such
as a reset of every GPU), hold all GPU work of the run: the orchestrator
starts no stage on GPUs and stage 3 stops every block. Both name what
changed, then again every 10 minutes. The hold keeps its problems per GPU
and the host's apart: a check settles those of the GPUs it reads (its own,
and every GPU the hold names) and the host's, never another GPU's, so a
stage on one GPU neither lifts another GPU's problem nor waits for a check
it cannot make. Once the run no longer holds a GPU it reads its canary again
before it benches. The waits have no limit unless `max_wait_s` (0) sets one:
past it a stage 3 GPU whose own hold, settings unlike the others', or
failing canary lasts that long takes no more blocks in that stage
(`retired_devices`) and its other GPUs go on, while a hold of every GPU
fails the stage with exit 2, as it fails a stage that waits before it
starts. A change made on purpose (a new power cap) that the run should go on
under: `scripts/run_guard.py accept <run dir> [--gpu PCI_ADDRESS]` takes the
GPUs' state now as the run's and drops their problems; a GPU that should be
left out goes into `runtime.exclude_gpus`, which drops its problems when the
run starts. Neither ends power management off, nor stuck ttm workers: those
are the host's, and their problem ends only when they do, or with `ttm:
false`. For a wedge, `accept` drops the hold of the GPU, or of the host,
though the stuck processes stay until the host reboots: use it once no
process stays stuck in the GPU driver, or for a wedge that proves wrong;
`runtime.exclude_gpus` leaves a wedged GPU out instead.

Health canary* (`bench.health.canary`, read only while the health checks are
on). Before a GPU's first block of every invocation of the run, and after
every hold, the GPU reads its canary: `lines` (4) reference lines that the
run's first stage 3 chose among its own lines (distinct shapes from the
least work up to the median, a single line the shape halfway between, each
restamped to time one kernel alone, as `bench.canary` restamps its line), a
warm-up copy of each and then `timings` (5) rounds of every line in turn, in
a bench process of its own, outside any execution context's wrapper. A
line's reading is the median of its rounds. The GPU's first canary of the
run is its reference; a later one passes when the median over the lines of
ln(reading / reference) is within ln(1 + `tolerance`) (0.05) of 0, or, with
`tolerance_us` above 0 (0, the default: no floor), when the median over the
lines of reading - reference is within that many microseconds: the reading
of a short line moves by a few microseconds from one bench process to the
next, which a relative tolerance alone reads as a departure. A canary whose
bench is stuck in the GPU driver, reports a fatal GPU error (a memory fault,
a queue abort), or hangs or prints nothing until the watchdog kills it
fails, the first or a later one (`faulted` and why in its record), and the
GPU benches no block until a canary passes. The run's canary lines and
kernels are fixed, so a canary kernel that faults at its shape faults every
time: a GPU whose canary faults so in three canaries in a row (stage03
`CANARY_FAULT_LIMIT`; a canary that does not fault starts the count again)
leaves the stage whatever `max_wait_s` says (`retired_devices`), and a stage
that loses every GPU so exits 2. When the canary faults on every GPU its own
kernel is the likely cause, and only turning it off
(`bench.health.canary.enabled: false`) lets the stage bench past it:
`max_wait_s` only makes a GPU leave sooner, and `scripts/run_guard.py
accept` and a reboot keep the same canary. A canary that cannot be read
otherwise is not judged, the first or a later one: its bench crashes, or
exits non-zero before any problem, without a fatal GPU error (`bench_error`
in its record: a bench binary, yaml or library that fails does that on any
GPU, and holding the GPU for it would hold every GPU for good), or fails
once its output began; its output does not line up with its entries; or it
reads fewer than half its lines. The first becomes no reference, a later one
gives no verdict (the canary record says why), the GPU benches, and its next
canary tries again, so that a canary line a kernel can no longer time never
holds a GPU. What the canary's wedge check finds wedged on other GPUs, or on
the host alone, the run holds (*Wedged GPUs*), but it tells nothing for or
against the canary's own GPU. The lines, the kernel each names (by solution
index, with its kernel id and signature), the rule, the tolerance and the
floor (a setup recorded before the floor existed has none) are recorded
once, before any canary is judged, and every later canary of the run keeps
them: a stage whose library holds a line's kernel under another solution
index now times it there, and one that holds fewer than half the kernels
chooses the canary anew from its own lines (every GPU's next canary is then
its reference). A line that timed another kernel than at the GPU's reference
(by the name the bench prints) is not judged; a canary left with fewer than
half its lines for that reason becomes the GPU's reference anew, saying why.
A GPU whose canary fails takes no block, the others go on (the retry pass
too passes it over), and it reads its canary again after `retry_s` (900);
`max_wait_s` bounds the wait as above. The canaries' logs are in `<run
dir>/guard/canary/`. This canary is not `bench.canary`, the drift check
around every bench process of a block.

*Hung lines.* A bench process whose output has not grown for
`bench.stall_timeout_s` (600) seconds (after its startup grace) is killed,
and the line it hung in leaves its block and is quarantined in the run,
under the measurement protocol it hung in: the retry pass does not run it
(the audit's protocols run it once more first, below), `skipped_gemms.txt`
gives it with reason `hung` and the stage log names the kernel in flight by
signature (the listed kernel after the last one the log reported, by its
solution index). The run's record keeps it with its protocol (`scope`), and
every later stage 3 of the run and of that protocol leaves its shape out
(reason `quarantined_line`), while the run goes on. The run's own benches
(its rounds, the audit set's build and the miss re-benches) are one protocol
(`bench`); audit protocol v2 and the sustained composite each have their own
(`audit_v2`, `audit_sustained`), so a line that hangs in one never takes a
GEMM out of another. Their stages bench a line that hangs once more in the
retry pass, in a process of its own, and quarantine it only when that does
not bench it to the end; the GEMM is then dropped from its part, which says
so, and the audit records count it (`n_dropped`, and their status line) and
leave it out of their scores instead of waiting for it (see *Protocol v2*).
The v2 canary's line is never quarantined. A stage 3 whose every line is
quarantined benches nothing and exits 1, its `skipped_gemms.txt` and
summary naming them, so that the audit drops their GEMMs too.
`scripts/run_guard.py release <run dir> --shape M,N,K,B` lets a quarantined
line be benched again (the record keeps it under `released_lines`).

*Wedged GPUs.* A GPU that stops answering leaves the processes in its driver
calls in uninterruptible sleep (state `D`), and a SIGKILL sent to such a
process only takes effect when the call returns; the exit it then starts can
leave the process, and the GPU, stuck until the host reboots. So
lib/procs.py sends no SIGKILL to a process group with a process in `D` in
each of four readings half a second apart (stage 3's watchdog and guard
kills of its blocks, retry pass, health canary and device probe, a stage's
stop, the hazard canary and the correctness gate; stage 7's watchdog and the
device-value probe still kill outright): a kill waits up to
`procs.STUCK_WAIT_S` (60 s) for the group to leave the call, as a slow call
does, kills it once it has, and else leaves it running. After any kill, and
after a process that failed without finishing a shape, stage 3 waits for the
process to end (`REAP_S`, 60 s) and looks for the processes the GPU driver
lists stuck in `D` (`/sys/class/kfd/kfd/proc`, ten readings half a second
apart; those stuck since before the stage started, or found by an earlier
check, count only when they are the bench's); a process other than the bench
counts only when it is still stuck after another `REAP_S`, as a slow driver
call or the teardown of a process that is ending is not. Wedged are: the
bench's GPU, when the bench does not end; the stage's GPUs a stuck process
holds memory on; and the host, when processes the stage started are stuck
holding memory on no GPU. A process stuck only on other GPUs, or on none,
that the stage did not start is recorded (`guard.wedged.elsewhere`) and
holds nothing: another run's wedge is not this one's. Nothing is judged
while the stage is being stopped. The run holds each wedged GPU, and a
wedged host, under the check `wedged` until the host reboots (the hold
records the boot it was raised under and lifts by itself under another;
`scripts/run_guard.py status` shows it, and `accept`, below, lifts it),
stage 3 starts no process on it again, no stage whose GPUs a wedge holds
starts (*Resuming*), and an audit worker whose GPU, or the host, is wedged
stops at once.

A wedge counts against the bench it was found after only when it holds that
bench's own GPU. Then the block goes back on the work queue for another GPU,
without the line the process faulted in, or was stuck in when a guard
stopped its block (the warm-up shape a process benches before its lines,
with `bench.warmup_pass`, is the shape of the block's next line, so a
process that wedged its GPU there did so in that line): no GPU benches that
line again in the stage (`skipped_gemms.txt` reason `wedged`, in the retry
pass too), and the run quarantines it under the stage's protocol as it does
a hung line (`why: wedged` in the record, `run_guard.py status` says so,
`release` lets it back in), so that a line that wedges every GPU it runs on
wedges one, also across audit workers. A process that wedged its GPU before
it started a line of its block (in its start-up, such as the library load,
or in the warm-up line or the canary) blames no line: the block goes back
whole. A check that finds only other GPUs, or the host alone, wedged holds
them, but the bench that triggered it is a crash like any other: its line is
skipped and benched again in the retry pass, and its GPU benches on unless
the wedge stops the stage. A wedged host, two or more wedged GPUs of the
stage (found in one check or in several, or held so by the run), or a wedge
of every GPU the stage has stops the stage: it aborts its running blocks
(`aborted/`, reason `wedged`), starts no bench process and exits 3 (`device
wedged, reboot`); a stage whose other GPUs left it for other reasons (a
failed probe or canary, another process on the GPU) ends as any stage that
runs out of GPUs, with exit 2. `stage03_summary.json` records under
`guard.wedged` each wedge found (the GPUs and processes, whether the bench
ended, the devices it took out), the processes found elsewhere, those stuck
when the stage started and why the stage stopped (`stopped`), under
`guard.wedge_lines` the lines that wedged their GPU (line, shape, kernel in
flight, GPU, protocol, whether quarantined), and per block its
`wedge_lines`. Like every hold, a wedge holds work only while
`bench.health.enabled` is on: with it off no stage reads the hold, and stage
3 leaves out only the GPUs it finds wedged itself.

*Device probe.* A bench process that finished every shape but had to be
killed, one that reported a fatal GPU error, and a block abandoned after
three processes in a row that finished no shape are GPU health events (per
block `health_events`): before its next block the GPU benches its device
probe, the stage's least-work line restamped to time one kernel alone, in a
process of its own under the watchdog (`probe/probe_<n>_gpu<D>.log`). A
probe that is stuck in the GPU driver, reports a fatal GPU error, or hangs
or prints nothing until the watchdog kills it holds the GPU as wedged. One
that crashes, or exits non-zero before any problem, without a fatal GPU
error takes the device out of this stage only: a bench binary, yaml or
library that fails does that on any GPU, so the run does not hold it, and a
stage that loses every GPU so exits 2, which another attempt may cure. What
the probe's wedge check finds on other GPUs, or on the host alone, tells
nothing for or against the probed GPU. The probe waits out the watchdog's
`startup_grace_s` on a GPU that prints nothing. `guard.probes` in the
summary records each probe (device, GPU, why it ran, whether it passed,
`fault` or `bench_error`, its log).

*GPUs to leave out.* `runtime.exclude_gpus` lists GPUs the run never uses:
physical indices (the ones `ROCR_VISIBLE_DEVICES` takes; a quoted one such
as `"3"` is an index too), PCI addresses (`0000:4c:00.0` or `4c:00.0`) or
`GPU-<hex>` ids. When the run starts their devices leave `runtime.devices`,
`audit.devices`, `correctness.devices`, `hazards.devices` and an audit's
`--devices`; preflight then checks only the devices left. Leased GPUs are
left to their holders (see *Leases*).

None of these keys changes a GEMM's rows: they are not pinned, and an
invocation may change them. A run directory written before them resumes
unchanged; its guard starts recording with the first invocation that has
them. What changes for such a run when it resumes, with the defaults: the
quiet-host gate, the health checks and the health canary are on (each GPU
reads its canary before its first block of every invocation and after every
hold of it); `bench.stall_timeout_s` is 600 (300 before) in stage 3, 3.2 and
3.7, and a line that stalls is quarantined for its protocol (see *Hung
lines*) where it was skipped as a crash and benched again in the retry pass;
and a stage leases the GPUs of its devices that are free when it starts and
benches on them at once, where it waited `runtime.gpu_lease_wait_s` for
every one of them, so a GPU another process tree holds then sits that stage
out even when it is freed soon after. Such a run also sends no SIGKILL to a
bench stuck in the GPU driver, holds a wedged GPU until the host reboots
(*Wedged GPUs*), quarantines a line whose bench leaves its GPU wedged,
probes a GPU after a health event, and fails a GPU's canary when its bench
faults, taking the GPU out of the stage after three such canaries in a row;
the canary floors are 0 and `bench.heavy_operand_gib` is 0, so the canaries
judge and stage 3 cuts its blocks as before.

Platform records.** Stages 3 and 7 write `platform.json`: per GPU its device
index, GPU index, KFD node and gpu_id, PCI address, unique id, amd-smi's
index, performance level (`power_dpm_force_performance_level`), power cap
and its default (hwmon `power1_cap`, W) and VBIOS version, read from sysfs,
the CUs and XCCs of its KFD node (`simd_count / simd_per_cu`, `num_xcc`) and
its compute partition mode (`current_compute_partition`); for a GPU that is
one partition of its card, the card's other partitions
(`partition_siblings`: each one's KFD node and identity, its VRAM in use as
its PCI device reports it, the processes KFD lists holding memory on it, by
pid, command, user and bytes, and their total, as the stage started; outside
the host's pid namespace pid and bytes only, since KFD names processes by
host pids); the amdgpu driver, kernel and ROCm versions; and whether amd-smi
agrees with the KFD topology. A value that cannot be read is null. A stage
refuses to start (exit 2) when its GPUs disagree on the performance level,
the power cap, the CU count, the partition mode or the XCC count (null
values are not compared), or amd-smi disagrees with the topology; stage 3
leaves a GPU the run holds for a departure of its own out of that comparison
(*Shared hosts*, GPU health). Preflight checks the devices of
`runtime.devices` the same way before any stage starts: a run reads its
device values (the CU count its `cu_fraction` budgets resolve against, the
XCD count, the feature inputs) from its first device alone. Stages 3 and 7
also record, under `environment`, the variables of the benches' environment
that select what they run: those `runtime.env` names and
`HSA_DISABLE_GFX12_STRICT`, null when unset. Stage 3 records, under
`warmup_line`, the warm-up line every bench process starts with (see *Bench
timing*), or why there is none. Stage 3's summary also records each block's
GPU, performance level and power cap when the block started. Stage 7 adds
the CPU side of its selection-time measurements: the CPU affinity the
benches inherit, whether it pins them, the cpufreq governors, boost, SMT and
the load average, and, per bench yaml, how busy the SMT siblings of pinned
CPUs were during its timing runs (`timing_cpu`).

**Launch settings.** TensileLite decides at every launch how the kernel a
bench times runs (its workgroup mapping and chiplet remap, the stagger of
its K loop, the persistent grid, the Stream-K reduction and hybrid schedule,
split-K), partly with Origami's heuristics, so a label measures a kernel as
the bench build launches it. A bench never inherits TensileLite's launch
overrides, which fix those choices or print them into its output:
`TENSILE_FIXED_*`, `TENSILE_PERSISTENT_*` (the hybrid force mode among
them), `TENSILE_STREAMK_*`, `TENSILE_STREAMK5_FORCE_MODE`,
`TENSILE_DISABLE_STAGGERU`, `TENSILE_PRECISE_SM_TARGET` and
`TENSILE_AUTO_GSU_ALGO`; a run that means to bench under one sets it in
`runtime.env`, and the config check warns about it. The pipeline sets none
of them itself: a context's schedule reaches its benches as
`--hybrid_assignment_policy`, which the hybrid force modes override, so the
check warns about those apart. Stages 3 and 7 record
under `launch` in `platform.json` (`lib/launch_settings.py`): `overrides`,
the launch overrides the benches had, and `stripped`, those the stage's own
environment had and the benches did not get; `heuristic`, the launch
heuristic of the bench build; and `workspace_bytes`, the workspace the bench
lines pass (hipblaslt-bench's `user_allocated_workspace`, from its argument
schema `hipblaslt_common.yaml` when a line sets none), which caps Stream-K
partials and so decides the grid of some launches. The heuristic is a sha256
`digest` over these files of the build's source tree, each recorded with its
own sha256 (`files`, null for one the tree lacks): TensileLite's
`src/ContractionSolution.cpp` (automatic workgroup mapping, stagger and
split-K, the persistent grid, the Stream-K reduction and hybrid choice, the
kernel arguments), its `include/Tensile/ContractionSolution.hpp`,
`ContractionProblem.hpp` (the rounding of the CU budget), `AMDGPU.hpp` (the
launch overrides and their defaults), `src/AMDGPU.cpp` and, where the build
has it, `DefaultPersistentLaunch.hpp`; Origami's `src/origami/origami.cpp`
(workgroup mapping and stagger), `streamk.cpp` (grid, reduction, hybrid
choice), `heuristics.cpp`, `hardware.cpp` and their headers in
`include/origami/`. The tree is the one the build compiled, as its
`compile_commands.json` names it (else the source directory its
`CMakeCache.txt` names; `found_by`); the build is the one whose `clients/`
holds the bench (for an execution context's wrapper, `runtime.build_dir`,
else the bench root's build). `newer_than_build` names the files changed
since the build compiled its library: the digest describes them as they are
now, not as the build compiled them, so a record with such files compares
as unknown, like a build that names no source tree and records a null
heuristic. stage04 sums up the records of the log directories it reads in
`enrich_summary.json` (`launch`; stage03b's merged logs carry theirs in
`rebench_summary.json`), stage05 those of its enriched folders in
`metrics.json` and `models.pt` (`bench_launch`), and stage06 copies that
into the deploy record (see *Deployment (stage06)*); stages 3.2, 4, 5 and 6
warn when the launch-rule sources of the builds differ. A digest marks a
source change, not necessarily a change of launches: an edit of
`ContractionSolution.cpp` that is not about launches, or a refactor of the
launch code with checked parity, changes it as well.

**Device values.** The pipeline computes the features and the LDS-fit rule
with the device values of the run's first device (`lib/hardware.py`): its KFD
node's, or the ones the config's `hardware:` block pins. hipBLASLt serves a
model with the values TensileLite reads through HIP: the physical CU count
(`hipDeviceAttributePhysicalMultiProcessorCount`), `sharedMemPerBlock` and
`l2CacheSize`. Preflight reads both on that device (`lib/device_values.py`):
the KFD node's CU count (`simd_count / simd_per_cu`), LDS (`lds_size_in_kb`),
first level-2 cache and wave size (`wave_front_size`), and HIP's through
`tools/device_props.cpp`, a host program without device code that preflight
builds on first use (into `<run dir>/tools/`, with the correctness gate's
compiler) and runs on that GPU alone, under its lease; a probe that does not
end, also once killed, is a device that does not answer. It refuses a run
whose values (the ones its stages read, `lib/hardware.py` `device_hardware`)
differ from HIP's or that its stages cannot read, so that a model is never
served with inputs it was not trained with (once you know which value is
right, pin it in the `hardware:` block of the run config kept outside the
repository), and one whose KFD node and HIP disagree on the wave size or the
architecture (the two then name different devices); a KFD value that differs
from a pinned value HIP agrees with is a warning. Only a run whose first
stage has not completed is refused: one that already holds data was made with
the values it uses, which only a new run changes, and an audit directory
serves runs that check their own device, so there each problem but a device
that does not answer is a warning. Each invocation records both sets, the
values the run uses and what the check found in
`configs/<snapshot>.platform.json`, next to its config snapshot.
`TILEWRIGHT_DEVICE_PROBE` names a prebuilt probe instead; set but empty, it
skips the HIP read, which a fake KFD topology (`TILEWRIGHT_SYSFS_ROOT`) does
as well.

`TILEWRIGHT_SYSFS_ROOT` replaces `/sys` for these reads and for the
device values read from the KFD topology (`lib/hardware.py`), and
`TILEWRIGHT_AMD_SMI` names the amd-smi executable (empty: do not run it);
the tests use both.

## gfx950 configs

`configs/gfx950_<library>_origami_v3.yaml` hold the training recipes for
gfx950 models that select from the Origami library alone
(`library.kind: origami`): BF16 (`bbs`) and TF32 (`tf32`) in the four
layouts, and FP8 with one f32 scale per tensor (`f8bs`) in TN. They size a
run as the production recipe does, as the gfx1250 training configs do:
`seed.target_per_cell` shapes per cell in round 0, `seed.shapes_per_cell`
new shapes per leaf in each active round, `full.total_rounds` rounds before
the final retrain, and no cell split: a cell below the bar is retrained
(`validate.max_split_depth: 0`, see *Active learning (stage04b)*). Like the
gfx1250 configs they hold no host value: set
`TILEWRIGHT_BENCH_ROOT` and `TILEWRIGHT_AUDIT_ROOT`, and list the host's
devices and evaluation sets in a run config outside the repository. Their
execution contexts are shares of the bench device's CUs, as in the gfx1250
configs: three quarters, half, a quarter and an eighth, each on a tenth of
a cell's shapes (`enabled: false` turns the eighth off for its bench time,
see *Execution contexts (stage03c)*). Each resolves to a whole multiple of
the 32-CU budget granule, at least one, so the smallest trained budget is
one granule, one XCD's worth of a gfx950 device, and every
compute-partition size of the device is also a trained budget;
`train.min_cu_fraction` stays unset, so the models declare CU shares down
to the smallest one they trained and no lower. The
non-temporal rule is on for TN and off for the other layouts, where it
would keep most of the Prediction table from being served;
`gfx950_bbs_tn_origami_v3.yaml` also sets a distillation weight and
temperature and pooled epochs of its own. Each active round gives a leaf the
round before it retrained twice the new shapes per leaf, and every leaf half
as many again
as its draw holds with K above 32,768, without a lower factor for any leaf
(`seed.weak_cell_factor`, `seed.large_k_factor`; see *Weak cells and very
large K (stage01)*).

## Running the gfx1250 configs

The gfx1250 configs run every round and stage with one command on devices
that report gfx1250 (ASIC revision other than 0): the training configs
`configs/gfx1250_*_tn_v3.yaml`, and the sanity configs
`configs/gfx1250_bbs_tn_sanity.yaml` and
`configs/gfx1250_192cu_bbs_tn_sanity.yaml` (*Sanity runs*). No config here
trains for a device of revision 0, which loads hipBLASLt's `gfx1250v0`
subtree, or `gfx1250-strict` with `HSA_DISABLE_GFX12_STRICT=0`. The configs
hold no device value: a run reads the CU count, LDS, L2 and XCD count from
the bench device when it starts, and the run config that lists the host's
devices stays outside the repository. Their kernel pool is the Origami
library alone (`library.kind: origami`, see *Kernel pool (`library`)*):
preflight refuses a library file without a Prediction table. The training
configs plan the production recipe's budget and rounds, as the gfx950
configs do: *First run on a new device* lists what to measure on the
host before the run, and a sanity run gives its first measurements.

1. **Compiler.** hipBLASLt builds the gfx1250 target with a ROCm compiler
   that accepts `--offload-arch=gfx1250`. Check before a long build:

   ```bash
   echo 'int main() { return 0; }' | amdclang++ -x hip --offload-arch=gfx1250 -c - -o /dev/null
   ```

   `unsupported HIP gpu architecture` means that compiler cannot build the
   target. The gfx1250-strict target, for devices of revision 0 in strict
   mode only, which no config here trains for, also needs a compiler that
   accepts `--offload-arch=gfx1250-strict`, which not every release does.

2. **hipBLASLt** of this repository, with its clients and tilewright (on by
   default), for the host's devices:

   ```bash
   cd projects/hipblaslt
   ./install.sh -c -a gfx1250                     # devices that report gfx1250
   ```

   A gfx1250 build also builds the `gfx1250v0` subtree, the strict logic
   for revision-0 devices without the strict switch
   (`HIPBLASLT_BUILD_GFX1250V0`), which no config here trains for either.
   `projects/hipblaslt/build/release` must
   then hold `clients/hipblaslt-bench`, whose argument schema
   (`clients/hipblaslt_common.yaml`) has `solution_index_list` of 4096
   entries and `skip_screen_prepass`, `clients/hipblaslt-cotenant` with its
   kernel, and `Tensile/library/<arch>/` with the config's library.
   `--logic-yaml-filter` builds part of the logic, such as one layout, in
   less time. A rebuild renumbers the solution indices: never rebuild a
   build a run is using.

3. **The engine module** of the same checkout (CMake, a C++17 compiler, no
   ROCm), in the interpreter the pipeline runs with:

   ```bash
   pip install -r requirements.txt            # from this directory
   pip install ../python
   python -c "import tilewright; print(tilewright.feature_catalog_hashes())"
   ```

   The list holds the v3 catalog's hash (the engine's `AGENTS.md`).

4. **A run config** outside the repository: a copy of the config of the
   data type with the host's devices in `runtime.devices` (one partition per
   card unless a noise replicate shows that benches on partitions of one
   card do not disturb each other, *First run on a new device*) and, once a
   value a device reports is known to be wrong, the right one in a
   `hardware:` block (*Device values*).

5. **The one command**, with the audit sets in a directory of their own:

   ```bash
   TILEWRIGHT_AUDIT_ROOT=<audit sets> scripts/run_pipeline.sh <run config> --mode full
   ```

   On a new host or partition, run it with a sanity config first
   (*Sanity runs*), then with the training config of the data type.

   It runs the hazard canary, the correctness gate where the config turns
   it on (`correctness.enabled`; the exact harness runs bf16 and f32 data,
   f8 and bf8 data with one f32 scale per tensor, and f8, bf8 and f4 data
   with block-32 E8M0 scales), the audit set, every round, the final retrain
   and stage 7 in the bench build, and stages the model
   (`deploy.apply_patch: false`). Leave such models out of the
   repository: a model's feature statistics describe the device it was
   trained with. Stage 7's production route says whether production calls
   reach the model at all: where a GridBased row of a few kernels comes
   before the Prediction row, they do not unless the build's TensileLite
   searches a loaded model first (see *Validation*).

**What preflight checks** before any stage runs, each failure naming its
fix: the bench and its argument schema (the planner's kernel lists and the
order-independent skip screen); the config's library in
`Tensile/library/<arch>/`, a library of a problem type the build holds no
library of at all being named as missing; that every device of
`runtime.devices` loads that
subtree, from its KFD node's ASIC revision and `runtime.env`; that the
devices agree on CU count, partition mode, XCC count, power cap and
performance level; that the first device's KFD node and HIP report the
same CU count, LDS, L2 and wave size (*Device values*); the XCD count of a
pool with dynamic-queue kernels; the contexts and their co-tenant; the
bank's stems; the audit set; what the stage 7 build's TensileLite serves;
the stage08 datasets; the free disk space; and the engine module, its
feature catalogs and, for v3, its CU floor. The config check refuses
`f4_r`, `f6_r` and `bf6_r` operands without a block format in
`hipblaslt.scaleA` / `scaleB`.

## First run on a new device

The `gfx1250_*_tn_v3` configs train with the production recipe's sizing,
as the gfx950 configs do, with settings that the measurements below confirm
or change for a new device:

- the production budget and rounds: `seed.target_per_cell` shapes per cell
  in round 0, `seed.shapes_per_cell` new shapes per leaf in each active
  round, `full.total_rounds` rounds, then the final retrain, and no cell
  split: a cell below the bar is retrained (`validate.max_split_depth: 0`,
  see *Active learning (stage04b)*).
  `full.total_rounds` is not pinned: a run whose audit set still climbs
  goes on with more rounds when the same command runs with a larger one
  (see *Final retrain*);
- planned lists sized to each library's kernel pool (`planner.top_k`);
- the gfx950 configs' weak-cell budget: twice the new shapes for a leaf
  the round before retrained, and for every leaf half as many again as its
  draw holds with K above 32,768 (`seed.weak_cell_factor`,
  `seed.large_k_factor`; see *Weak cells and very large K (stage01)*);
- near-winner re-reads in a wide band, twice for a GEMM whose winner is
  below `fast_below_us` and once for the others (`bench.rebench`);
- the default stall timeout after the long start-up grace of the library
  load (`bench.stall_timeout_s`, `startup_grace_s`);
- execution contexts at three quarters, half, a quarter and an eighth of
  the CUs (WGPs here), as in the gfx950 configs, each a whole multiple of
  the 32-WGP budget granule, at least one: the smallest trained budget is
  one granule on any device or partition with a budget below its CUs, one
  XCD's worth on the 256-WGP part, so every compute-partition size of that
  part is also a trained budget (`enabled: false` turns the eighth off for
  its bench time); none held out: no budget below a small partition's CU
  count is left for a held-out context, and a share the partition cannot
  give is left out (*Execution contexts (stage03c)*);
  `train.min_cu_fraction` stays unset, so the models declare CU shares
  down to the smallest one they trained and no lower;
- protocol v2 as the official exclusive number (`audit.sustained: false`,
  `audit.official: v2`);
- the correctness gate for the problem types its harness runs (BF16, FP8
  with one f32 scale per tensor, MXFP8 and MXFP4).

Before the run's GPU-days, measure these on the host and set what they show
in the run config kept outside the repository:

1. **Which rows serve the problem type.** hipBLASLt asks the model, or
   Origami, only for a GEMM that reaches the library's Prediction rows; a
   row of the same type ahead of them that matches the GEMM answers on its
   own: an Equality row its exact size, a GridBased or FreeSize row a whole
   range of sizes. Only stage 7's production route (*Validation*) sees
   this, at the end of a run: the pipeline's other benches run with
   `TENSILE_PREDICTION_LIB=1`, which in this hipBLASLt passes
   every other kind of row over. On the stage 7 build, bench a few plain
   GEMMs of the config's type (sizes no Equality row names) with
   `requested_solution_num: 1` and `print_kernel_info: true`, once with
   `TENSILE_PREDICTION_LIB=1` and once without: a GEMM that gets another
   kernel without it is served by an earlier row. With a model staged as
   stage 7 serves it, `TENSILE_USE_TILEWRIGHT=1` and
   `TILEWRIGHT_PICK_LOG=1`, every GEMM the model ranks prints a
   `[TILEWRIGHT_PICK]` line. Such rows change neither the training nor the
   audit set's scores, but the model cannot serve their GEMMs until the
   library's rows change. Check the block-scaled types first.
2. **Device values.** Preflight refuses a run whose KFD node and HIP report
   different device values (*Device values*); hipBLASLt serves with HIP's,
   so pin HIP's value in the run config's `hardware:` block. On one device,
   `hipblaslt-cotenant --cus <CUs> -- <a short bench command>` must log
   `READY` before the contexts' benches can be trusted. Set an `HSA_XNACK`
   the benches need in `runtime.env` (the MXFP4 config says where).
3. **Bench cost per line.** The first stage 3 blocks give it: their
   `wall_s` in `stage03_summary.json` over the lines they benched. With the
   bench lines the budget implies, it gives each round's GPU time, and so
   the GPU-days the run needs and how many devices to give it.
4. **A noise replicate.** A sample of GEMMs whose winners range from
   launch-bound to long, each benched in several separate bench processes
   (one stage 3 run each), and on a card with several partitions once with
   the others idle and once with them busy. Compare one kernel's readings
   across processes: a spread of a fixed time per process, rather than a
   share of the latency, reorders the near-winners of short GEMMs most, and
   a busy neighbour that widens it calls for benches on a quiet card. The
   spread sets the
   re-read band (`bench.rebench`) and says whether the canaries' relative
   tolerances (`bench.health.canary`, `bench.canary`) hold for short
   GEMMs.
5. **The skip screen.** Round-0 planned lines with the config's
   `bench.skip_slow_solution_ratio` against the same lines without the
   screen (0): every kernel close to the winner must stay timed (*Bench
   timing*, choosing the ratio).
6. **A power soak.** Compute-bound GEMMs timed in long windows, with the
   device's power and clock readings. Where the clock settles below its
   peak under the power limit, the sustained composite is the regime that
   counts (`audit.sustained: true`, `audit.official: sustained`; *The
   sustained composite*); otherwise v2 stays the official number.
7. **A fault-rate smoke.** A sample of the data type's shapes through
   stage 3 before a full run, and its crashed, hung and quarantined lines in
   `stage03_summary.json`: every crash costs a new bench process and its
   library load, so a device that faults on a share of the shapes needs
   GPU time budgeted for it, or a fix first.

### Sanity runs

`configs/gfx1250_bbs_tn_sanity.yaml` runs BF16 TN through every stage of
the one command (the hazard canary when the pool holds a kernel of a hazard
class, the correctness gate, the audit set, the production recipe's
`full.total_rounds` rounds, the final retrain and stage 7) on the smallest
budgets that still let a cell train, be evaluated and split: a few shapes
per cell in round 0 and per leaf in each active round, one exhaustive seed
per cell, short planned lists, an audit set of a few GEMMs per cell, and
splits at most one level deep (`validate.max_split_depth: 1`): the training
configs do not split, but a sanity run keeps one level so that it still
exercises the split code. Besides the whole device it benches half
and an eighth of its CUs with a co-tenant, so that its models declare CU
budgets from an eighth of the device, rounded to the nearest multiple of
the budget granule (exactly an eighth on a device of a multiple of eight
granules, one granule on a smaller one), to all of it, as the training
configs' models do from their four shares; on a device of two
granules the eighth meets the half's budget and is left out, and a device
of a single granule has no budget below its CU count, so a run there
leaves both out and trains the exclusive context alone. A run on another
CU count or partition resolves other budgets, and a training run refuses
an audit set built with other contexts, so each device class and
partition mode needs an audit set of its own (`audit.dir`). Every other
key is the BF16 training
config's (`configs/gfx1250_bbs_tn_v3.yaml`, a config test holds them
equal), so a sanity run benches, trains, audits and stages its model as a
production run does and gives the measurements of the list above (which
rows serve the problem type, the device values and the co-tenant's
`READY`, the bench cost per line) before a production run spends its
GPU-days. Its model is staged, never shipped.
`configs/gfx1250_192cu_bbs_tn_sanity.yaml` is the same run for the device
class of hipBLASLt's `gfx1250_192cu` logic subtree: the device class comes
from the bench device, so the two configs differ only in their name and
their audit sets, one per class.

## Stages

| Stage | Script | Output |
|---|---|---|
| 0 | `stage00a_hazard_canary.py` | `hazards/canary/` (written as `canary.new/`, which replaces it when the canary succeeds): `quarantine.json`, `verdicts.json`, `canary_summary.json`, each run's bench yaml and output in `logs/`; once per invocation, before anything else, with `hazards.enabled` (the default) and a kernel of a hazard class in the pool |
| 0 | `stage00_correctness_gate.py` | `correctness/gate/` (written as `gate.new/`, which replaces it when the gate succeeds): `blocklist.json`, `verdicts.json`, `gate_summary.json`, the harness's records in `runs/`; once per invocation, before the rounds, only with `correctness.enabled` |
| 1 | `stage01_generate_OOB_shapes.py` | `shapes.yaml`: new shapes per leaf cell, none of them in a held-out list; re-used history shapes in `reused_shapes.yaml`; `categorization.json` with each held-out list it left out (`exclusion_sources`); `targeted_shapes.json` with an active round's targeted shapes (see *Targeted shapes (stage01)*); `per_cell_plan.json` with an active round's per-leaf targets and budget (see *Weak cells and very large K (stage01)*) |
| 2 | `stage02_plan_benchmarks.py` | `shapes.yaml` with each shape's kernel list, `plan.json`, `plan_summary.json`; only with `planner.enabled` (in `stage01b/` of a run written before the planner became stage 2, see *Resuming*) |
| 3 | `stage03_load_balance_offline_tuning.py` | `logs/`: hipblaslt-bench logs, one work queue per GPU, every kernel timed adaptively, every bench process led by the discarded warm-up line (`bench.warmup_line`), with `bench.canary` a canary around each block's shapes; `platform.json`, `quarantine/`, `drift/`, `misaligned/`, `aborted/`, `probe/` (device probes), `skipped_gemms.txt`, `stage03_summary.json` |
| 3.2 | `stage03b_rebench_near_ties.py` | `logs/`: stage03's logs with the near-winners read again and labelled by their median, `rebench.json`, `rebench_summary.json`; only with `bench.rebench.enabled`. GEMMs of other execution contexts (stage03c) keep their single measurement |
| 3.5 | `stage03c_bench_contexts.py` | per execution context: `<label>/` with its bench wrapper and stage03 outputs, `holdout/stage04/`, `contexts.json`; only with `contexts` |
| 3.7 | `stage03d_sustained_finalists.py` | `runs/run_<r>/`: the sustained re-measurement of the finalists of power-capped GEMMs, `sustained.json`, `sustained_summary.json`; only with `bench.sustained.enabled` |
| 4 | `stage04_convert_to_enriched_dataset.py` | `chunk_*.csv`: one row per (GEMM, measured solution of the library); `kernels.json`: the solutions' kernel attributes |
| 4.5 | `stage04b_iterate_active_learning.py` | `decisions.json`, `retrain_cells.txt`, `splits.json`, the gate slice `gate/` and `train/` (active rounds); `rebench_misses.json` and `rebench/` with `validate.rebench_misses` |
| 5 | `stage05_train.py` | `models.pt`, `cells.json`, `metrics.json`, `model_options.json`; `whitelist_tier.json` with `train.whitelist_tier_k` |
| 6 | `stage06_deploy_weights.py` | model file, index, `gate.json` and `deploy_record.json` (`rebench/` when the gate re-benches its misses) |
| 7 | `stage07_optional_validate_selection_time.py` | selection time, C++/Python pick parity and the production route, `platform.json` |
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
ratios), unless `validate.max_split_depth` is 0, as in the configs of the
training recipes: then a cell that misses the bar is retrained.

A cell the round does not retrain is carried forward with the model of its
latest training round, trained with that round's model-shaping options:
every stage05 option but its data and the library and device it was measured
on, which cells train, how the work runs, and the optimizer's budget and
schedule (`NOT_MODEL_SHAPING` in `stages/stage05_train.py`), and the kernel
identity rule its keys follow (`identity_rule`, see *Kernel identity across
hipBLASLt builds*). stage05 records them in `model_options.json`; for a
stage05 without that file they come from the command its
`.stage_complete.json` records. They are compared as given,
not as resolved from data: `train.off_list_sel_eff: auto` is `auto` in every
round, whatever charge it measures. The round retrains, besides stage04b's
list, every carried cell whose options differ from its own stage05's or are
unknown (every cell when that is every carried cell; a cell with fewer than
`train.min_cell_gemms` training GEMMs stays as it is), and logs the options
that differ. A change of a model-shaping option such as `train.cascade_k`,
`train.stage1_recall_weight` or `deploy.weight_dtype` thus reaches every cell
of the next bundle; a change of the optimizer's budget, such as
`train.epochs`, reaches the cells the round retrains anyway. When there is
nothing to retrain (`retrain_cells.txt` is
empty: every evaluated cell passed and no leaf without a model of its own can
be trained yet, and no carried cell differs), the previous round's
`models.pt` and `cells.json` are carried forward unchanged, with its
`model_options.json` and `member_order.json`, which describe that bundle.

stage05 records the bundle's provenance in `models.pt` and `metrics.json`
(`lib/provenance.py`): the config's `config_id`, `training_data_hash` (SHA-256
over the contents of the enriched chunks it trained on, in training order),
`source_revision` (the pipeline checkout's commit, `-dirty` when tracked files
are modified, `unknown` outside git), the UTC `date` and, for v3, the
`launch_model` id of its launch plans (`HipblasltLaunch.id`). Every stage that
writes the bundle as a model file records it in the MLREC_v3 file, and the
engine then refuses to rank that model with another launch model; the model
stage06 publishes records less (`deploy.provenance`, see *Deployment
(stage06)*).
No value holds a host or user name or a path. Next to it, `models.pt` and
`metrics.json` name the training recipe (`recipe`, see *Student training
(stage05)*), which the model file does not record.

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

A GEMM's label of a kernel is one latency. When the data hold several
rows of one kernel for one GEMM (its shape benched again in another round,
a retried problem), they merge by `train.repeat_aggregation`
(`--repeat-aggregation`, `lib/repeats.py`): its tested rows if it has any,
else its skip rows, at their `median` (the default; the mean of the two
middle readings of an even count), `median_low` (always one of the
readings) or `min`. The minimum of several noisy readings is biased low,
the more readings the lower, so it favours the kernels measured most often;
before the option existed stage05 took it, and a round whose recorded
options lack the key counts as trained with `min` (a changed rule retrains
the carried cells, see *Stages*). The merged kernel keeps the `cv` and
`status` of its median reading for quality checks, and `metrics.json`
counts `rows_merged` and `kernels_repeated` under `load_stats`. Stages
4.5, 6 (the gate) and 8 read the same key, so every evaluation of the run
labels its GEMMs alike. A kernel with sustained rows (stage03d) takes its
label from them before any other row, and the GEMM's other kernels are
scaled to match (see *Sustained finalists (stage03d)*).

Each cell's student trains in mini-batches of `--batch-gemms` GEMMs, the
loss of a GEMM taken over all its candidates (soft cross-entropy against
latency targets plus distillation from the LightGBM teacher, mixed by
`train.distill_lam` at temperature `train.distill_T`), for
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
`--patience N` stops after N measured checkpoints without a better one
(`train.patience`; 0, the default, trains the whole schedule). A patience
such as 8 saves training time; it does not shape the model (a carried cell
is not retrained for it). A
cell without validation GEMMs keeps the checkpoint with the best in-sample
efficiency. For int8 and int4 models, `--qat-frac F` continues from the
kept checkpoint for F of the schedule's steps with the weights rounded as
the model file stores them (straight-through gradients) and keeps a better
checkpoint. `training_log.csv` has one row per measured checkpoint.

The teacher's own selection efficiency on the validation GEMMs, its scores
ranking each GEMM's candidates as the student's do, is recorded beside the
student's: `teacher_val_sel_eff` per cell in `cells.json`, and
`per_cell_teacher_val_sel_eff` and `global_teacher_val_sel_eff` beside
`per_cell_val_sel_eff` and `global_val_sel_eff` in `metrics.json`. The
validation GEMMs chose the student's checkpoint and nothing of the
teacher's, so the student's number is the more optimistic of the two; a
teacher well ahead of its student says how much a better distillation could
still gain. The teacher only predicts for it: every weight stays what it
was. Without a teacher the per-cell numbers are null and
`global_teacher_val_sel_eff` is NaN, as `global_val_sel_eff` is without
validation GEMMs; under `pooled_ft`, `pooled.teacher.elapsed_s` includes the
teacher's validation predictions.

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
may still pick. `--pseudo-negatives` (`train.pseudo_negatives`; needs
`--plan`) adds to each planned training GEMM's loss up to
`--unmeasured-samples U` (`train.unmeasured_samples`) kernels that pass the
LDS gate and the serving rules, as pseudo-negatives: their soft-CE target is
that of a kernel `train.unmeasured_margin` slower than the winner, scaled by
`train.unmeasured_weight`, and the teacher gives them none. Feature
statistics, the teacher, early stopping and every reported number use the
measured kernels; `cells.json` counts each cell's `n_unmeasured_pairs`. Each
cell draws with its own seed, so a cell's draws do not depend on the others.
The draws:

- `random`: pool kernels off the GEMM's list. They hold a GEMM's fastest
  kernel whenever the plan missed it, so they also teach the next model the
  ranking errors of the model that planned the bench. U has no default;
  margin 0.05 and weight 0.3 unless set.
- `targeted` (v3): keys no training GEMM of the cell has a row of, each as
  its first pool kernel, the one the engine scores the key by; a key with a
  kernel on the GEMM's list is left out (listed without a row, it is one the
  runtime skips for that GEMM). The model scores such keys by extrapolation
  alone, and picks of keys the leaf never measured are the catastrophic
  ones: far slower than the winner, and on exhaustively benched seed shapes
  no candidate model trained on, such keys never held a winner. U 16,
  margin 1.0 (a pseudo latency of twice the winner's) and weight 0.1 unless
  set. A leaf whose GEMMs measured every key (exhaustive seed shapes) gets
  none.
- `none`.

Without `--pseudo-negatives`, U above 0 draws `random` (what
`--unmeasured-samples` did before the option existed) and U = 0 none. The
recorded model-shaping options name the draw only when U does not imply it,
so a round whose recorded options predate the option counts as trained with
the draw its U implies. `none` with U above 0, and `random` or `targeted`
with U = 0, are errors.

A v3 model ranks keys (a kernel's signature and spec attributes, and its
launch class with launch keys): the engine scores a key with the launch plan
of its first kernel in pool order, and the runtime runs the key's first
kernel whose predicates pass. Other kernels of the key have rows of their
own, which per-kernel targets reward although the runtime never runs them
for that key. `--served-candidates` (`train.served_candidates`, v3 only,
default off) keeps one candidate per key in each training GEMM, the kernel
the runtime runs: the key's first pool kernel with a row, passing over pool
kernels the GEMM was benched with but has no row of (their predicates
fail); a key whose first such kernel is off the GEMM's planned list is
dropped, since nothing measured what the runtime runs for it. Targets and
the teacher's grades are relative to the fastest kept candidate
(`target_winner_us`); early stopping and every reported number still compare
against the fastest measured kernel. `metrics.json` counts the candidates
kept and dropped under `load_stats.served_candidates`.

The margin is off by default and experimental: check held-out selection
efficiency before turning it on.

`--recipe` (`train.recipe`) chooses how the cells' students are trained:

- `per_leaf` (default): every cell's student starts from fresh weights,
  learns from a LightGBM teacher of its own and keeps the checkpoint the
  engine measures best, as described above.
- `pooled_ft`: one student learns every cell's training GEMMs together, then
  every cell fine-tunes it on its own GEMMs. One LightGBM teacher learns all
  cells' measured training pairs (`--pooled-n-estimators` 1000 trees of
  `--pooled-num-leaves` 255 leaves, at least `--pooled-min-data-in-leaf` 50
  pairs per leaf, `--lgb-lr`). The pooled student, at the widest per-leaf
  widths (embedding, hidden and interaction widths 80, 192 and 112;
  `ML_FORCE_CAPACITY` overrides them), trains on all cells' training pairs
  with their own feature statistics and bounds for `train.pooled_epochs`
  (12) passes of `train.pooled_batch_gemms` (64) GEMMs at a peak rate of
  `train.pooled_lr` (0.002), and keeps the best of `train.pooled_eval_points`
  (24) checkpoints, measured on every cell's validation GEMMs together with
  the weights as the deploy dtype stores them. Each cell to train then
  starts from it, keeps its features, and trains on its own pairs for
  `train.fine_tune_epochs` (12; 0 keeps the pooled student) passes of
  `--batch-gemms` GEMMs at `train.fine_tune_lr` (0.001), measured on its
  validation GEMMs after each pass; it keeps the pooled student unless a pass
  beats it. The loss, its teacher weight and temperature (`train.distill_lam`,
  `train.distill_T`), the tie tolerance, the stage-1 recall loss, weight
  decay, gradient clipping and the learning-rate schedule's shape are the
  per-leaf ones, and pseudo-negatives join both phases, each cell's drawn
  as for `per_leaf`.

  Every cell of `pooled_ft` reads the same feature statistics and widths,
  so the bundle's format and serving are unchanged; a cell that `per_leaf`
  would serve with narrower widths costs slightly more selection time. The
  v3 training configs use it. The pooled phase
  covers every cell stage05 loads, also those it does not train
  (`--only-cells`, `--skip-cells`, too few GEMMs): those options choose the
  cells to train, not the data. The pooled student runs on
  `--pooled-threads` threads (0: every CPU the stage may use), the pooled
  teacher on `--pooled-teacher-threads` (0: as many), the fine-tuning of
  the cells to train in `--train-workers` processes of
  `--train-threads-per-worker` threads. Thread counts change the results:
  LightGBM's trees (with `deterministic` on, the same data, seed and thread
  count give the same trees, another thread count can give others) and the
  students' MKL arithmetic depend on them. The same thread counts and
  `--pooled-seed` (0, which seeds the pooled teacher, the pooled student and
  the fine-tuning batches; `--seed` still splits the GEMMs and draws the
  pseudo-negatives) give the same models bit for bit, so the record names
  the counts each part ran on: `pooled.threads` (the student),
  `pooled.teacher.threads` and `pooled.fine_tune.workers` processes of
  `pooled.fine_tune.threads_per_worker` threads. The per-leaf
  schedule (`train.epochs`, `lr`, `eval_points`, `patience`,
  `--n-estimators`, `--num-leaves`) does not apply, and the config check
  warns when the train section sets it (`epochs` other than its default);
  `boundary_margin`, `qat_frac` and a `validation_frac` of 0 are errors.
  `models.pt` and `metrics.json` keep the pooled phase under `pooled`
  (models.pt also its student), with `pooled.teacher.val_sel_eff`, the
  teacher's efficiency on every cell's validation GEMMs together, beside
  `pooled.student.best_val_sel_eff`; `cells.json` keeps each cell's
  `pooled_val_sel_eff`, the pooled student's validation efficiency on the
  cell (its `teacher_val_sel_eff` is the pooled teacher's), and
  `training_log.csv` the pooled checkpoints under the cell `pooled`.

The recipe shapes the model, so changing it retrains every carried cell; its
schedule does not.

### Whitelist tier (stage05)

`train.whitelist_tier_k` (`--whitelist-tier-k`; v3 with `train.whitelist`
`none`; 0, the default, is off) gives every cell of the bundle stage05
writes, carried cells included, a whitelist tier of K view keys. The engine
scores a cell's tier keys first (tier 1, the model's whitelist-tier serving
flag) and no other key while one of them is feasible, so a problem the tier
suits skips the cascade's first stage over every feasible key. The weights
are those trained without it: the tier is chosen after training
(`lib/whitelist_tier.py`), on

- GEMMs**: the training side of the enriched data stage05 trained from, read
  again as stage05 read it (the library's kernel identity,
  `train.repeat_aggregation`, the blocklist, the rows of the pool's kernels
  alone) and split as it split it, so that no validation GEMM counts; a GEMM
  counts for the cell the bundle routes it to, also when its own leaf has no
  model. - **scores**: the deployed model's: the bundle packed at the deploy
  weight dtype with cascade 0 and without a tier, so that the engine gives
  every key it serves its full score (embedding term and interaction MLP),
  under the bundle's serving rules (the soft rules, the family-wise
  non-temporal rule, the trained-key gate) and hipBLASLt's launch plans,
  over the pool the runtime ranks: the library's table, then the bank,
  without the kernels the blocklist names. - **cost**: a key's latency in a
  GEMM is that of its first pool kernel with a row, the one the runtime runs
  for it (a skip row counts with its warm-up latency). A key without a row,
  or that the engine does not score (the serving rules reject it), does not
  cover the GEMM: a kernel no training GEMM measured, a blocked kernel or a
  rejected key never enters a tier. - **keys**: per cell, the greedy of the
  smart_K whitelist (`select_smart_k_signatures`): the key that covers the
  most GEMMs not yet covered, then the key that most improves the mean log
  selection efficiency of the model's picks, up to K keys (every measured
  key when it has fewer). A cell no training GEMM reaches gets no tier. -
  **contexts**: `train.whitelist_tier_contexts`
  (`--whitelist-tier-contexts`) says how the execution contexts the cell's
  GEMMs ran in (exclusive use, the CU budgets and schedules of
  `contexts.list`) count: - `pooled` (the default): every GEMM alike, so the
  contexts count by their share of the training GEMMs; - `balanced`: every
  context of the cell weighs the same in all, in both steps of the greedy: a
  key's coverage is the share of each context's GEMMs it newly covers,
  summed over the contexts, and the regret step improves the sum over the
  contexts of their mean log efficiency; - `reserve`: the K keys are taken
  on the cell's exclusive GEMMs (on all of them when it has none), then up
  to 8 more on its other GEMMs, by the same greedy, each covering one of
  them or making their picks faster in all.

  The tier serves every context of the cell: a key taken for one context
  is scored in every other. A cell whose GEMMs share one context gets the
  pooled tier in every mode, and `whitelist_tier.json` gives a cell whose
  GEMMs ran in several its GEMMs, covered GEMMs and in-sample efficiency
  per context.

The greedy serves each cell's mean, so a key only a few of its GEMMs
need, such as the winners of a context the cell's GEMMs rarely run in,
can drop out of the tier while the model's pick from the remaining keys
takes more than twice the winner's time on those GEMMs. The tail guard
(`train.whitelist_tier_guard_keys`, `--whitelist-tier-guard-keys`, 8 by
default, 0 off; `lib/whitelist_tier.py` `guard`) runs after the greedy:

- a GEMM of the cell's own training GEMMs whose pick from the tier is
  below half its winner's speed (`GUARD_FLOOR`, the per-cell gate's tail
  floor) is a collapse; its rescue is its fastest measured key outside
  the tier that is within `GUARD_NEAR` (0.9) of the winner and that the
  model scores above the tier's pick, so that keeping it moves the pick
  there (the GEMM's winner when the model ranks the winner above the
  pick, else the near-winner it does). A collapse whose every near-winner
  the model ranks below the pick is a ranking error the tier cannot fix,
  and is only counted;
- before each try, the collapses still open and their rescues are found
  again under the tier with the keys kept so far, so that no key joins
  for a collapse an earlier one lifted; the rescue lifting the most open
  collapses is tried next, each key at most once. One joins the tier only
  when, on the cell's training GEMMs and on the GEMMs stage05 held out of
  the cell's training (its validation GEMMs), the tier with it and the
  keys kept before it, compared with the tier the greedy chose, adds no
  new pick below the floor (each GEMM and context on its own), lets none
  of the picks the keys before it lifted fall back below the floor, and
  keeps at least `GUARD_MEAN_FLOOR` (0.99, the per-cell gate's tolerance)
  of the held-out picks' geomean (the training side's when the cell holds
  none out), so that the keys of a cell keep these bounds together; at
  most the configured number of keys per cell join.

Guard keys follow a cell's greedy keys in its tier. The tier record names
the guard (`guard`: its keys, floors, the held-out GEMMs it read, the
cells with collapses, the keys added and the collapses they lift), and
`whitelist_tier.json` gives each cell with a collapse what the guard
found (`guard`: collapses, rescuable ones, added keys, the check of the
last kept key, which covers every kept key against the greedy's tier,
and the first refused candidates with why). The guard is
model-shaping: `model_options.json` records its keys
(`whitelist_tier_guard_keys`), None without a tier, which it does not
shape, and a record from before the guard reads as None (stage05's
`PREDATING_OPTIONS`), so a tier trained before it is trained again.

A problem none of whose tier keys is in the pool or feasible is ranked as
by the model without the tier: through the cascade (`train.cascade_k`) over
every feasible key. Keys outside the tier rank behind it, through the
cascade, only for a caller that asks for more kernels than the tier holds.

`models.pt` holds the tier as the cells' `view_key_hashes` with
`serving.whitelist_tier` and the record `whitelist_tier` (K, how it counted
the contexts, `contexts`, with `reserve_k` for `reserve`; a record without
`contexts` is pooled; the cells and keys, the training GEMMs it was chosen
on, the weight dtype, and the pool's kernels, keys and key digest); stage05
writes `whitelist_tier.json` (per
cell its keys in the order taken, and the in-sample selection efficiency of
the tier next to that of every measured key), `metrics.json` keeps the
record, and `cells.json` each trained cell's `n_whitelist_tier_keys`. Every
stage that packs the bundle (stages 4.5, 6, 7 and 8, the audit) serves it
with its tier. stage06's `manifest.json` and `deploy_record.json` record it
(`whitelist_tier`); the deploy record also says how the tier meets the pool
of the library the model is deployed for (`served_pool`: the tier keys that
pool lacks, the cells left with none), with a warning when keys are
missing. A round's stage05 carries cells from the bundle before it without
that bundle's tier; with `train.whitelist_tier_k` it gives its own bundle a
new one.

The option works with both recipes, the soft rules, the bank, the
trained-key gate (a tier key has training rows by construction) and the
blocklist; it is refused with launch keys (the tier lists view keys, launch
keys score each launch class of a view key on its own), with the smart_K
whitelist, without the library (`--library-dir`, `--library-stem`) and
without the engine. It is model-shaping: a change retrains the carried
cells, and the final retrain records it among its options
(`full.final_retrain.recipe` can set it for the final retrain alone).
Building a tier reads the training data again and scores every training
GEMM with the engine, in `train.workers` processes.

When the kernel pool changes (another hipBLASLt build, another bank or
blocklist), a model keeps serving: the engine skips the tier keys the pool
lacks, a cell whose tier keys are all missing or infeasible is served
through the cascade over the new pool, and a kernel the pool adds ranks
only where the cascade serves. To build the tier again without training:

    python scripts/build_whitelist_tier.py --train-dir <run>/<round>/stage05 \
        --out-dir <dir> [--library-dir <dir> --library-stem <stem>] \
        [--config-yaml <config>] [--k K] [--contexts MODE] [--guard-keys G] \
        [--jobs N]

It chooses the tier on the GEMMs that stage05 recorded (`metrics.json`: the
enriched folders, the split tree, the validation split, the repeat
aggregation), read with the kernel identity of the pool it names (else the
one stage05 ranked): rows of kernels the pool lacks are left out, also
from each GEMM's winner, and a folder benched with another build counts
its kernels under their indices in this one. It writes the stage05 files
with the new tier into `<dir>` (the source stays as it is) for stage06
(`--train-dir <dir>`); `--k 0` removes the tier, and without `--k` the
tier keeps its size; `--contexts pooled|balanced|reserve` chooses it as
`train.whitelist_tier_contexts` does, and without it the tier keeps the
way its record names. A kernel the pool adds enters a tier once training
data measures it. `--guard-keys` bounds the keys the tail guard adds per
cell (default 8; 0 builds the tier without it), checked on the GEMMs
stage05 trained each cell on and on those it held out;
`model_options.json` records it as `whitelist_tier_guard_keys`.

The tier differs from the smart_K whitelist (`train.whitelist` `signature`
or `view-key`, sized by `train.smart_k` or, with `smart_k_max`, at the knee
of each cell's regret curve) in when and on what it is chosen. The smart_K
whitelist is part of training: every measured checkpoint gets its own,
chosen with the student's scores as stage05 computes them, and stage05 keeps
the checkpoint whose whitelist serves the validation GEMMs best, so it
shapes the weights; it costs a key by its fastest feasible kernel, and
`--smartk-fallback-delta` drops it where it hurts. The tier is chosen once, after training, for the
bundle as it ships: with the engine's scores at the deploy dtype, a key
costed as the runtime serves it, over the shipped pool with the bank, at a
fixed K, without touching the weights or the checkpoints. It therefore
covers carried cells too and can be built again for another pool or K
without training. Both fill the cells' whitelists, so a bundle has one or
the other.

### Bench timing (stage03)

Adaptive timing is the only timing mode: the `bench.adaptive.enabled` switch
was removed together with the probe stage that calibrated fixed iteration
counts, and a config that still sets it is an error that says so. Every bench
times each kernel with hipblaslt-bench's adaptive timing: a bench line has
`adaptive: true` and every `bench.adaptive` field, and no iteration counts.
The repository's configs spell out every field.

The bench warms each kernel up for `warmup_time` ms, then times it in
samples of about `sample_time` ms until, past `min_iters` iterations and
`measure_time` ms, the relative standard error of the mean is below
`noise_threshold` or the sample spread has settled (`stability_threshold`
over `stability_window` readings taken every `stability_interval` samples),
and at most until `max_measure_time` ms or `max_iters` iterations (0: no
limit); the row's `us` is the median per call. The config check applies the
bench's own rules to these fields (a ceiling is set, the floor is below it).

stage03 benches the lines as written, except that a line with fixed
iteration counts (from a run directory written before adaptive timing was
the only mode) gets `adaptive: true` and the config's `bench.adaptive`
fields in their place; it warns how many lines it converted. Every bench
process starts with the discarded warm-up line (`bench.warmup_line`, see
*Warm-up line* below). Its discarded warm-up shape (`bench.warmup_pass`)
only loads the kernel modules, with one cold and one timed call per kernel.
The canary each bench process times before and after the block's shapes
(`bench.canary`, see *GPUs: leases, quarantine and platform records*) runs
on the GPU in the same process and its timings stay out of the block logs.

Every entry of a bench process's yaml also gets a `name` of its own
(`warmup_line`, `warmup`, `canary_start<i>`, `line<N>` for line N of the
stage's input, `canary_end<i>`), an argument the bench does not read:
hipblaslt-bench converts its yaml with `hipblaslt_gentest.py`, which runs
only the first of entries equal in every argument, and stage03 maps a
process's output to its entries by position. A process whose output does
not line up with its entries (a problem of another shape than its entry,
more problems than entries, or fewer from a process that exited cleanly) is
noted in its block log and in `stage03_summary.json` (`misaligned`), and
fails the stage (exit 2); with the warm-up line or the canary in the
process, the process's output is set aside and its shapes benched again
without them instead (see *Canary*).

The skip screen is `bench.skip_slow_solution_ratio` alone. At 0, its
default, a bench line carries no ratio, the bench's default 0 applies, and
every listed kernel is timed. With a ratio r the bench first times one call
of each kernel, its warm-up, and times a kernel adaptively only when r times
its warm-up does not exceed the fastest warm-up: of every listed kernel on a
line with `skip_screen_prepass` (planned lines; each kernel is launched once
untimed first, which loads it), else the fastest so far. Every other kernel
gets a skip row with that single-call warm-up latency: under
`validate.pick_measurement` `warmup` and `any` a pick of such a kernel costs
its warm-up, and the miss re-bench re-times it. Near-winner re-read and
miss re-bench lines have no screen (ratio 0) and keep their GEMM line's
adaptive fields (a slow-band re-read raises its `min_iters`).

Sacrificial kernel: the screen's call of the first listed kernel comes right
after the bench initialised the problem's data, on a device that has been
idle, and reads several times too slow, so the screen drops that kernel
whatever its speed and its warm-up becomes its label. With
`bench.sacrificial_kernel` (the default) stage02 lists the first kernel of
every line with a skip ratio twice. The copy at position 0 takes that call;
every reader of the logs (`lib/bench_log.py`) sets aside the row of rank 0
when the row of rank 1 names the same solution index, which no other line
does, so stage04 writes no row for it (`sacrificial_rows` in
`enrich_summary.json` counts them), stage03b and the miss re-bench never
take it for a candidate, and the other rows keep their bench rank. A line
at the bench's list limit has no room for the copy and goes without it, as
do lines without a list.

Warm-up line: a bench process loads a library file's code object when it
first runs one of the file's kernels, and the screen reads the cold call of
that kernel several times too slow, after the prepass's untimed call too,
and drops it. The sacrificial copy takes that call only for the file of the
line's first kernel, so in the first problem of every process the first
listed kernel of each other file (the bank's, when the line's first kernel
is the pool's) was screened out whatever its speed. With
`bench.warmup_line.enabled` (the default) every bench process of stage03,
and so of stages 3.2, 3.5, 3.7, the miss re-bench and the audit's benches,
starts with a throwaway warm-up line: a fixed problem of `size` cubed
(batch 1) of the stage's problem type, timed and screened as the stage's
lines are, over `kernels` of the kernels the lines list, one of each
library file of `--identity-stem` in turn and at least one of each (without
the library files, spread over the listed solution indices; without a list,
the first `kernels` the bench takes for the problem). It is one line for the
whole stage, the entry `warmup_line`, first in every process, the retry
pass's too; its output goes to the warm-up logs (`logs/warmup_*.log`, marked
`*** WARM-UP (discarded)`), which no stage reads, and the stage's
`platform.json` and `stage03_summary.json` record it (`warmup_line`:
its settings, the line, its shape and its kernels with their library file
and signature, or why there is none). A process that fails before it got past the warm-up
line, where the block's first shape never started, or whose output does not
line up with its entries, turns the line off for the rest of the stage, and
a new process benches the shapes without charging any of them; the summary
says why (`warmup_line.off`). The keys are not pinned: a run or audit
directory benched before the warm-up line existed resumes with it, and each
stage's `platform.json` says whether its processes had it; an audit v2
measurement keeps what it was planned with (its `warmup_line` setting). The
warm-up line must reach every library file, not just run first.

Kernel order: a kernel timed early in a long line runs before the device
settles at its power limit and reads faster than the kernels timed after
it, so a list in a ranker's order flatters the kernels the ranker put first
and the next model learns its predecessor's picks. With
`bench.kernel_order: shuffled` (the default) every explicit kernel list of
a round is listed in a shuffle seeded with `bench.order_seed` and what the
line benches: stage02's planned, exhaustive, seed and context lines
(seeded with the shape; plan.json keeps each list in the planner's order
and records `kernel_order` and `order_seed`, so `lib.planner.shuffled`
gives the bench order again), and the near-winner re-read, sustained
and miss re-bench lines (seeded with the shape, then rotated by one place
per run, so that every kernel takes another place in each run). `listed` benches
every list in the order it was made. A line without a list
(`algo_method: 1`, a heuristic line) is benched in the order the bench
chooses; the audit set's increments and protocol v2 keep their own
order.

Block sizes: every work block of a bench is a bench process of its own, and
its start-up (HIP, the library, the problem's buffers) costs seconds
whatever the block holds. stage 3 cuts its lines into `bench.blocks_per_gpu`
blocks per GPU; the near-winner re-reads (stage03b), the sustained finalists
(stage03d, `gemms_per_block` per block) and the miss re-bench of stages 4.5,
6 and 8 cut theirs by their lines per GPU instead (`lib/rebench.py`
`blocks_per_gpu`: a block per thousand-odd lines, at most 20), so a call of
a few lines runs one block per GPU, each GPU still in a bench of its own.
With `bench.heavy_operand_gib` above 0 (0, the default, keeps them among the
others; a config sets a size to turn it on), stage 3 puts the lines with an
operand (A, B, C or D over the batch) of that many GiB or more into blocks
of their own at the end of its work queue, at least one per GPU where there
are enough of them, so that a GPU such a line wedges has benched the other
lines first; a stage of one block per GPU keeps them with the others. The
key orders and groups blocks only, never what a line benches, and is not
pinned. `stage03_summary.json` gives the threshold and the count
(`heavy_operand_gib`, `heavy_lines`) and marks each such block (`heavy`).

Long lines (`bench.adaptive_long`, off by default): a shorter sample
averages fewer calls of a short kernel and reads it noisier, but a kernel of
hundreds of microseconds takes few calls per sample either way. So the line
of a GEMM that cannot run faster than `min_roofline_us` (100) warms up,
samples and stops sooner: it takes the block's `warmup_time` (0.5),
`sample_time` (0.25) and `max_measure_time` (4.0) over `bench.adaptive`, and
every other line keeps `bench.adaptive`. The bound is the GEMM's roofline:
the larger of 2 M N K batch over the data type's peak matrix rate
(`peak_tflops`, TFLOP/s) and of reading A and B and writing D once over the
memory bandwidth (`bandwidth_gbs`, GB/s). It needs only the shape and the
config, so every stage gives a GEMM the same knobs (its planned and seed
lines, the near-winner re-reads, the context benches, the miss re-benches,
the audit, stage 8's datasets), and no kernel faster than the bound can land
on a long line. The orchestrator reads the rates the config leaves out from
the first bench device when a run starts (`lib/hardware.py`
`roofline_rates`: the family's best matrix instruction for the type in the
MI cycle table times `parallel_mi_cu`, the CU count and the KFD node's
engine clock; the bus width times the clock of the node's frame-buffer
memory times the transfers per clock of the family's memory) and the
snapshot states them. A rate it cannot read leaves its term out, and so
does a data type `lib/shapes.py` `BYTES_PER_ELEMENT` does not know the
memory term, which keeps the bound a lower bound; a run whose bound keeps no
term (neither rate, or no peak and such a data type) is refused when it
starts. The defaults are set for gfx950 BF16 lines: measure default
against long knobs on the same GEMMs (bench time, the repeat noise of short
kernels, winners against a re-bench median) before turning the block on for
another data type or architecture, and set the rates of unreleased hardware
in a run config outside the repository.

Choosing the ratio: a single cold call is slower than the warmed median by
a margin that varies from kernel to kernel, so the near-winners' warm-ups
spread out above the fastest warm-up. Choose a ratio that keeps the
near-winners' cold calls in and still skips the clearly slow kernels; the
configs here use 0.45 on planned lines. An unplanned line (no
`skip_screen_prepass`) has warm-ups that also pay the kernel's load and are
compared with the fastest so far. Measure that cluster (screen off against
screen on, same GEMMs) before changing a ratio.

### Measurement protocol

The keys that decide how the labels are measured, as a next run sets them
(the comments give each key's default):

```yaml
bench:
  skip_slow_solution_ratio: 0.45  # 0: no skip screen
  kernel_order: shuffled          # shuffled; listed keeps the planner's order
  order_seed: 42                  # 42
  sacrificial_kernel: true        # true
  rebench:
    enabled: true                 # false
    tolerance: 0.05               # 0.05
    tolerance_us: 0.0             # 0.0: no absolute floor
    fast_runs: 3                  # 3: winners below fast_below_us (10 us)
    runs: 1                       # 1: winners in between
    slow_runs: 3                  # 3: winners from slow_from_us (1000 us) on
    slow_min_iters: 0             # 0: the line's own windows
    inline_copies: 1              # 1: every candidate once per run
    with_original: true           # true
    align_to_original: false      # false
  adaptive_long:
    enabled: false                # false; see Bench timing
  sustained:
    enabled: false                # false; see Sustained finalists (stage03d)
train:
  repeat_aggregation: median      # median; median_low, min
  patience: 8                     # 0: no early stopping
```

- `kernel_order` and `order_seed`: every explicit kernel list in a seeded
  per-GEMM shuffle, so no kernel is timed first because a model ranked it
  first (*Bench timing*).
- `sacrificial_kernel`: a screened line's first kernel is listed twice, so
  the screen's first cold call, which reads too slow, cannot drop it
  (*Bench timing*).
- `rebench`: each GEMM's near-winners read again in bench processes of
  their own, as often as the band of its winner latency asks, and labelled
  by the median of their readings (*Near-winner re-reads (stage03b)*).
  `tolerance_us`, `inline_copies` and `align_to_original` are for a device
  whose readings carry an offset per bench process; at their defaults (no
  floor, one copy, no shift) they change nothing.
- `adaptive_long`: shorter timing windows for the lines of GEMMs whose
  roofline is long (*Bench timing*). Off by default: it saves bench time on
  those lines and changes no other.
- `sustained`: the finalists of power-capped GEMMs re-timed in long
  self-soaked windows, a label source of their own (*Sustained finalists
  (stage03d)*). Off by default: it costs GPU time per finalist and run.
- `repeat_aggregation`: several readings of one kernel of one GEMM merge by
  their median, not the fastest, which is biased low (*Student training
  (stage05)*).
- `patience`: early stopping of each cell's student (*Student training
  (stage05)*); it saves training time and does not change what the bench
  measures.

### Near-winner re-reads (stage03b)

With `bench.rebench.enabled`, stage 3.2 runs between stages 3 and 3.5. One
bench pass cannot order kernels whose latencies differ by less than its
noise, and the training targets come from that order: single readings of
launch-bound kernels flip a GEMM's winner far more often than the median of
a few readings in separate processes, and a kernel of a millisecond or more
gets only a few iterations into a default window. So for every complete
problem of stage03's logs, the candidates within `bench.rebench.tolerance`
of the winner, or within `tolerance_us` of it when that is wider (tested
rows, and skip rows by their warm-up; at most `max_kernels`, the fastest),
are read again through stage03, each re-read in a bench process of its own
(`lib/rebench.py`). The absolute floor is for readings whose error does not
scale with the latency, such as an offset per bench process, which a
relative tolerance misses on a short winner:

- How often depends on the band of the GEMM's winner latency:
  `fast_runs` times below `fast_below_us`, `slow_runs` times from
  `slow_from_us` on, `runs` times in between; a band of 0 runs is not
  read again. With `slow_min_iters` the slow band's re-reads take at least
  that many iterations per window (a floor that outranks
  `max_measure_time`); by default they take the line's own windows, three
  times (see *The slow band* below). With `bench.sustained.enabled`
  stage03d times the finalists of power-capped GEMMs again in long
  self-soaked windows on top of this.
- Run r benches the GEMMs that ask for more than r reads: one line per GEMM
  with an explicit `solution_index_list` of its candidates, no skip screen,
  the line's adaptive timing and rotating buffer, the list in the run's
  kernel order (see *Bench timing*: shuffled, then rotated by one place per
  run) and, with `sacrificial_kernel`, led by a copy of its first kernel
  whose row the log readers set aside (the first timing of a line reads
  slow); the GEMMs in a seeded shuffle per run (`gemm_order: shuffled`,
  with `bench.order_seed`; `listed` keeps the input order).
- With `inline_copies` k above 1, a run's line lists the GEMM's candidates
  k times, each copy in the run's order rotated one place further than the
  copy before (`lib/rebench.py` `reread_list`), so the run reads every
  candidate k times in one bench process, at k places of the line. A line's
  data set-up is paid once instead of once per read, and every candidate's
  k readings share one process's offset, so their medians compare the
  candidates under the same offset (medians over separate runs can each
  come from another run's).
- Each re-read candidate is replaced in the logs by one row. With
  `with_original` its latency is the median of its original timing and its
  re-reads (with one re-read the median is the mean of the two), in a copy
  of the row of the lower middle reading. A skip row's warm-up is no timing
  and takes no part, so a GEMM whose near-winners include a skip row labels
  all of them from their re-reads alone: an offset between stage03's
  process and the re-reads' then shifts no candidate against another.
  Without `with_original`, the row of the re-read with the lower median
  latency. Ranks are kept, a skip row becomes a tested row, and the
  `Winner:` block names the new fastest row. stage04 then reads
  `stage03b/logs/`.
- With `align_to_original` (it needs `with_original`), each run's re-reads
  of a GEMM are first shifted by their run's offset: the median, over the
  GEMM's candidates, of the difference between a re-read and the
  candidate's original timing. Where a bench process reads every kernel
  late or early by about the same time, the plain median moves a re-read
  candidate by part of that offset against the GEMM's kernels that were
  not read again; the shifted re-reads keep the original reading's level.
  The candidates were chosen by their original timings, which therefore
  lean fast, and the shift carries that lean into the re-reads, so the
  labels can come out slightly fast against the kernels that were not read
  again: keep it off where bench processes read alike. A run is shifted
  only when its offset rests on at least three differences
  (`lib/rebench.py` `MIN_ALIGN_DIFFS`) and leaves every reading of the run
  above 0; any other run merges as it was read. The median of two
  differences is their mean, which one slow re-read raises by half its
  excess, and the shift would take as much off the other candidate, down
  to a label at or below 0 (a row stage05 drops) or a winner too fast.
  So a GEMM with two near-winners is shifted only with `inline_copies` 2
  or more; the config check and stage03b warn when `align_to_original`
  comes with one copy. A set with a skip row is labelled from its unshifted
  re-reads alone, as above.
- Kernels the run's blocklist names (*Correctness gate*) are no
  near-winners and do not set the winner the tolerance counts from, so the
  window is the one of the kernels a GEMM keeps.

Only stage03's GEMMs are re-read: those of other execution contexts
(stage03c) and of the planner's seed bench keep their single measurement.
`rebench.json` lists every GEMM's band, runs and copies, whether its labels
merge the original timings (`with_original`), the offset each run's
re-reads were shifted by (`offsets_us`, null unless `align_to_original`
shifted them), why each other run was left as read (`unaligned`:
`skip_row`, `few_differences` or `nonpositive`; null unless one was), and
every candidate's original latency, re-read latencies in run order
(`runs_us`; shifted, `aligned_us`) and label, with the library digest of
their solution indices, each candidate and the winner before and after
named by signature and kernel id when stage03 recorded its library;
`rebench_summary.json` the options, the GEMMs and candidates per band, the
winner changes, the GEMMs and runs whose re-reads were shifted with the
median and largest offset, the GEMMs none of whose runs was shifted
(`n_gemms_unaligned`; for a skip row in their set,
`n_gemms_unaligned_skip_row`) and the runs left as read by reason
(`n_runs_unaligned`), the wall time and median offset of each run and how
the benches launched the kernels (`launch`, see *Launch settings*). The
bench must know `solution_index_list`.

Cost: a run is a stage03 bench of the lines it re-reads, a process start per
block, then one adaptive window per listed kernel. A GEMM costs
`inline_copies` windows per candidate, and one for the sacrificial copy, in
each run of its band, and one data set-up of its line per run.
With planned lists of a hundred-odd kernels per GEMM, of which a few are
near-winners, that is a small share of stage03's bench time: most GEMMs are
read once, the launch-bound and the slow ones three times in the line's own
windows. A wider `tolerance` or `tolerance_us`, a larger `max_kernels`, more
runs or more copies grow it in proportion; where a line's set-up costs more
than its windows, copies cost less than runs. The stage's settings before
the bands, three runs of every GEMM, cost about three such passes. They
stay reachable as `tolerance: 0.03`, `tolerance_us: 0.0`, `runs: 3`,
`fast_runs: 3`, `slow_runs: 3`, `slow_min_iters: 0`, `inline_copies: 1`,
`with_original: false`, `align_to_original: false`,
`sacrificial_kernel: false` and `gemm_order: listed` (`lib/rebench.py`
`PREDATING_REREADS`, which pins every key, so later default changes leave
it as it was).

The slow band reads three times in the line's own windows; `slow_runs: 1`
with `slow_min_iters: 10` (`lib/rebench.py` `PREDATING_SLOW_BAND`) is a
cheaper band. The first run after a long bench pass can read slow against
later runs on the same GPU, a shift common to the whole run. It does not
reorder the kernels within a run, but it biases whatever compares runs or
merges a re-read with the original reading (`with_original`). Keep the
kernel orders rotating across runs, as the stage does, and do not compare
a GPU's first minute after a long pass with later runs unnormalised.

### Sustained finalists (stage03d)

A long or compute-bound GEMM runs at the device's power limit in sustained
use (back-to-back GEMMs, training loops), and the limit slows some kernels
more than others. Adaptive timing's windows of a few milliseconds end before
the clock settles, so for such a GEMM they rank the kernels at a clock it
never runs at for long, and its labels pick another winner than sustained
use would. With `bench.sustained.enabled` (off by default), stage 3.7
times the finalists of these GEMMs again in long self-soaked windows and
stores the readings as a label source of their own (`lib/sustained.py`).

- Which GEMMs (`power_capped`): of the complete problems of the round's
  exclusive bench (stage03b's merged logs with the near-winner re-reads, else
  stage03's), those whose winner takes at least `min_us`, or that reach
  `compute_bound_tflops` (2 M N K batch over the winner's latency) while
  taking at least `compute_bound_min_us`. Context benches (stage03c) and
  the planner's seed bench are not re-measured.
- Which kernels (`finalists`): the `top_k` fastest candidates and every
  candidate within `tolerance` of the winner (tested rows by their latency,
  skip rows by their warm-up), at most `max_kernels`, the fastest.
- How: `runs` runs (one by default: a long window's readings vary far less
  between repeats than the short windows' do, and every run costs as much
  as the first) through stage03 on every device of `runtime.devices`,
  one GPU per bench process, under the stage's leases; one line per GEMM
  with an explicit list of its finalists, no skip screen, and the adaptive
  fields `warmup_time` (long enough for the device to reach its power
  limit), `measure_time` and `max_measure_time`, in ms. The lists follow
  the run's kernel order (shuffled, then rotated by one place per run), the
  GEMMs a seeded shuffle per run, and a block holds about
  `gemms_per_block` GEMMs, which run back to back on one GPU.
- Labels: stage04 reads `runs/run_<r>/stage03/logs` with
  `--sustained-log-dir`, marks their rows `is_sustained`, and counts their
  problems apart from the round's GEMMs. A finalist takes its label from its
  sustained rows, merged by `train.repeat_aggregation`; since the long
  windows read the finalists slower than the short windows read the rest,
  every other kernel of the GEMM is labelled with its short-window latency
  times the median ratio of sustained to short-window latency over the
  finalists with a timed short-window row (`lib/repeats.py`; the GEMM's
  `sustained_ratio`, the kernel's `us_short`). Stages 4.5, 5, 6 and 8 read
  the labels the same way.
- `sustained.json` lists every re-measured GEMM with why it is power-capped,
  its short-window winner and each finalist's short-window and sustained
  latencies; `sustained_summary.json` the counts and every run's wall time.

The cost is one window per finalist and run, a few seconds per GEMM per
run with the default windows, and with the default thresholds a sizeable
share of a round's GEMMs is power-capped. `tolerance: 0` keeps the
finalists to the `top_k`, and a large `min_us` keeps the pass to the
compute-bound GEMMs (set `compute_bound_tflops` for the data type), whose
kernels the short windows read furthest from their sustained speed.

### Planned benches (stage02)

By default stage03 benches every shape with every kernel hipBLASLt's
heuristic lists, in the heuristic's order, so the order-dependent skip screen
and stage04's `is_origami_pick` follow the library heuristic. With
`planner.enabled` the pipeline does not use the heuristic: stage02 decides
the kernels of every shape, and every bench line sets `skip_screen_prepass`,
so with a skip ratio the bench warms up every listed kernel once and times
only those within the ratio of the fastest warm-up, whatever their order
(see *Bench timing*). This needs a hipblaslt-bench that knows
`solution_index_list` (up to 4096 kernels) and `skip_screen_prepass`
(preflight checks the bench's `hipblaslt_common.yaml`).

`planner.mode`:

- `exhaustive` (default): every shape benches every pool and bank kernel
  (an alias group, see *Kernel identity across hipBLASLt builds*, as its
  first member in pool order, like every mode does).
  Only when they exceed the bench's list limit (4096) does it
  fall back to `algo_method: 1`, which also benches the problem type's other
  libraries: their warm-ups enter the skip screen, so a shape whose pool
  kernels all trail such a kernel can lose all its rows.
- `planned`: every shape benches an explicit list. Round 0 bootstraps from
  seeds: `planner.seed_per_cell` shapes per cell are benched exhaustively
  (stages 3 and 4 in `stage02/seed/`), and the other shapes are ranked from
  them by `planner.ranker`: `neighbors` (the measured speed of each kernel on
  the cell's `neighbors_k` nearest seed shapes) or `stage05` (a stage05 model
  trained on the seeds), with lists checked on the seeds (*Seed recall*).
  Later rounds rank with the previous round's
  `models.pt`; `planner.model_file` fixes the model for every round.
  `planner.top_k_round0`, when set, is the `top_k` of the lists the seeds
  rank, so that round 0 can list more kernels than the rounds a model ranks;
  with `planner.model_file` no round ranks from seeds.
- `auto`: `planned` when the pool and the bank hold more than
  `exhaustive_max_kernels` kernels, else `exhaustive`.

A planned shape's list holds, among the pool kernels that fit it (LDS and the
soft rules of `rules`: `config`, those the `train` section serves the next
models with, or `model`, the ranking model's own; a v3 model serving with
other rules ranks under the configured ones), in this order: the ranker's `top_k` kernels; every kernel
(at most `family_variants` when > 0) of the families of its first
`families_top_f` ranked kernels, a family being the kernels that share macro
tile, matrix instruction and cache hints (a whitelist signature; the cache
hint of an operand the tensor data mover loads is 0); `explore_random_k` other
kernels drawn with `seed`; then, per cell, kernels added on random shapes
until every kernel is benched on at least `min_coverage` shapes of the cell
where it fits (seed shapes count). Lists stop at `max_list` kernels. The
bench line lists them in the order `bench.kernel_order` gives (by default a
seeded shuffle; see *Bench timing*), plan.json in this one. The
bank kernels join the ranking and families, and stage04 keeps their rows.

When the next models serve the family-wise non-temporal rule
(`train.non_temporal_family_rule`), a planned shape inside the rule's region
also lists every kernel only that rule admits there: the member it keeps of
each family the non-temporal rule rejects whole (a shape's `family_rule`
source in plan.json). These come after the other kernels and beyond
`max_list` (up to the bench's limit), in the round's plan and in every
context re-plan, with the same shuffle and sacrificial copy. A model serves
those members only when it trained on measurements of them, and a ranker
that never saw them measured rarely lists them on its own.
`plan_summary.json` counts them (`family_rule_added`,
`family_rule_shapes`); seed and exhaustive shapes already bench every
kernel.

**Seed recall.** A round planned from seeds ranks each shape from at most
`seed_per_cell` measured shapes of its cell, so stage02 checks the lists on
the seeds before it writes them (`lib.planner.seed_recall`). It ranks every
seed of a cell the round plans from the other seeds of its cell, as
`neighbors` ranks the cell's other shapes but from one seed fewer, builds the
list the planner would give it (its `top_k`, family and random kernels, not
the coverage additions), and scores the list with the seed's own exhaustive
bench, among the kernels the plan may list for the seed. `plan_summary.json`
records under `seed_recall` the share of the seeds whose fastest kernel is
listed (`winner_recall`), the share with a listed kernel within 5% of it
(`near_recall`) and the geometric mean of the fastest kernel's latency over
the fastest listed one's (`ceiling`, the selection efficiency a perfect pick
from the lists reaches); the same per cell (`cells`) and at other list
lengths, up to the longest a seed can get (`by_top_k`). stage02 warns when
`ceiling` is below `planner.recall_min_ceiling` (0.97), with the weakest
cells and the shortest `top_k` from which every scored length reaches it, a
value for `planner.top_k_round0`. The check changes no list. A seed without
another measured seed in its cell is not scored (`n_alone`), and with
`ranker: stage05` the seeds score the neighbors ranker the model falls back
to, as the model trained on every seed.

The pool and the bank are those of the config's kernel pool (*Kernel
pool*): the kernels of its first kind, by default the Prediction table in
table order, then those of the kinds after it. For the pools TensileLite
ranks, pool positions equal the runtime's: `top1_index` of
`TILEWRIGHT_PICK_LOG` indexes the same list (`scripts/check_pick_log.py`
checks a bench log against it). The evaluation stages (04b, 05, 06, 07, 08)
rank the same pool.

stage04 adds `is_planned` (0 only for a kernel off its shape's list) and
`n_planned` (the length of the shape's list; empty for exhaustive shapes and
runs without a plan) to every row; `is_origami_pick` is 0 for every GEMM of
a plan. Seed GEMMs are part of the round's data. stage04 also counts the
kernels a plan lists that the bench reported no row of, timed, screened or
sacrificial (`listed_kernels`, `listed_kernels_without_row` and
`problems_with_listed_kernels_without_row` in `enrich_summary.json`, with a
warning): the library did not support them for the problem, or the bench did
not enumerate them, and those GEMMs' labels lack them. The sustained
finalists' logs are not counted: their lines list the finalists alone.

### Kernel pool (`library`)

The kernel pool is the set of library kernels a run benches, trains on and
ranks, in the order the model ranks them; `lib/library_pool.py` resolves
it. The `library` section names it:

- `kind`: `origami`, the Prediction table of the library file
  (`hipblaslt.library_stem`) in table order, the kernels Origami ranks (a
  file without a Prediction table is an error); `bok`, a bank of kernels:
  the solutions of the rows of the `bok_rows` kinds of every library file of
  the library's problem type, the library's own included, whose hardware
  predicate accepts the bench device, in ascending solution index; or a
  list of both, taken in its order. The default is `origami`.
- `bok_rows`: the row kinds a bank of kernels takes, `freesize` and
  `equality` (the default) and `gridbased`, a bridge for a bank built as
  GridBased rows, whose kernels TensileLite does not rank with the model (a
  warning). Within one problem type TensileLite orders the rows Equality,
  Range, Prediction, GridBased, FreeSize, so a GridBased file of the
  Prediction table's problem type sits behind that table's row, which
  answers a call first: its kernels are reached only when every solution is
  enumerated or the Prediction row returns fewer than requested, and
  `TENSILE_TILEWRIGHT_BANK` takes Equality rows only. In another placement
  (a placeholder of another problem type, an earlier hardware row) which
  row answers first depends on their order.

Two files share a problem type when their stems agree once `_ID<chip ids>`
and then `_CU<count>` are removed before `_<arch>`. A solution index is one
entry of the pool, that of the first kind (and file) holding it, as
TensileLite tells the kernels it ranks apart by solution index; solutions
that repeat the name of another under another index stay, and the
printout counts them. Of a row only the solution indices are read, never
the problem sizes its table is keyed by. The device is the processor of
`arch`, the CU count of `hardware.n_cu` (else KFD) and the PCI chip id of
`hardware.pci_chip_id` (else the KFD `device_id`; required on processors
with chip-id predicates); it is read for a pool with `bok`, and by the
preflight check below when another file of the problem type holds a
Prediction table. The hardware predicates are evaluated the way TensileLite
does: `Processor`, `CUCount` (physical CUs), `PciChipId` (with TensileLite's
chip-id fallbacks) and `TargetProcessor` under `And`, `Or` and `Not`; a
predicate the pipeline does not know is an error.

hipBLASLt can give a class of devices of one architecture a library file of
its own: a stem with a `_CU<n>_` token, from a CU-count logic subtree.
TensileLite walks the hardware rows of a problem type from the most specific
(the device's own PCI chip id, then a chip id it falls back to, then the
higher CU count) and ranks a GEMM with the first Prediction table it
reaches, so a model serves a device class only through the stem that class
loads. Preflight stops a run, but `--mode audit`, whose first bench device
would rank the problem type with another file's Prediction table than
`hipblaslt.library_stem`'s (`lib/library_pool.py`
`prediction_stems_ahead`): set the stem that device loads, or bench on
devices that load the configured one.

The kinds after the first are the pool's bank: an MLREC_v3 model trained on
a pool with a bank says so, and stage 7 serves it
(`TENSILE_TILEWRIGHT_BANK=1`). TensileLite ranks the Prediction table and
then the Equality rows of the files of its problem type in ascending
solution index, which is the pool of `[origami, bok]` with bok rows
`equality`; it serves tilewright through a Prediction library only. So
preflight warns when `bok` comes first or stands alone (such a model is
trained and evaluated but cannot be served yet) and when bank kernels come
from rows TensileLite's bank does not take. A pool with `bok` needs
`train.catalog: v3` and `train.trained_key_gate` (kernels a bank gains
after training are then served only where no trained key fits).

Without a `library` section the planner's legacy keys name the pool, as
before, and only when `planner.enabled`: `planner.bank: none` (the
default) is `origami`; `planner.bank: auto` is `[origami, bok]` with bok
rows `equality`; `planner.bank_library_stems` is the same with the bank
read from the files it names (it overrides `bank`; preflight warns when one
is of another problem type than the library). A host file without a
Prediction table then gives every solution of it, in file order. A config
that sets `library` and one of these keys is an error. With
`TILEWRIGHT_GFX950_LIBRARY_DIR` naming a gfx950 library directory, the
tests check that the bf16 TN pool of `planner.bank_library_stems` (its
`_CU256_` file) and of `library: {kind: [origami, bok]}` are the same
kernels in the same order.

Given the library (`--library-dir`, `--library-stem`, as the orchestrator
gives it), stage05 trains on the rows of the pool's kernels alone. The rows
of any other kernel (a bank's, in data benched with a bank under a pool
without it, or any other solution of the files the bench loaded) are left
out before a GEMM's rows merge, so none of them is a candidate, a GEMM's
winner or a label, and stage05 warns with their number (`metrics.json`
`load_stats`: `rows_outside_pool` of `kernels_outside_pool` kernels, and
`gemms_outside_pool`, the GEMMs left without a row). A model retrained with
`library.kind origami` on data benched with the bank is thus labelled as a
run that benched the table alone labels it. Its whitelist tier and
`deploy_loss.py` read the data the same way. The stage06 gate and stage04b
still read every row of their folders, and stage08 leaves out other kernels'
rows only after a GEMM's rows merge: give a model trained this way gate and
held-out folders benched with its pool alone.

Preflight prints the pool on the first device: per kind its kernels, keys
(signatures) and the largest key, the kernels per row kind, the files they
come from, the copies of solution indices held earlier and the files of
the problem type that add none, with why (no rows of the bank's kinds, the
device's hardware predicate, kernels the pool holds already); the files of
the library's data types and operation that hold another problem type and
are not read; and the whole pool. The report's Kernel pool section shows
the same for the pool the bundle ranks. `lib/bank_pool.py` keeps the
pipeline's older names for the pool (`config_pool`, `bank_source`,
`bank_library_stems`, `config_identity`) as wrappers.

### Kernels and execution context

A kernel is one solution of the library: the data names it by its runtime
solution index, every ranking by its position in the library's kernel pool.
The pool is what hipBLASLt ranks with tilewright (*Kernel pool*): by default
the solutions of the `.dat`'s Prediction table in table order; the file can
also hold solutions of other library rows, which are in the pool only with a
bank. Solutions whose `tilewright::Config` fields are
identical are not merged: each keeps its own rows and latency, the model
scores them alike, and the engine ranks them in pool order. An alias group
(next section) is one kernel: stage02 benches its first member in pool
order, the one the runtime serves, and every member keeps its position.

- stage04 writes one row per (GEMM, solution) and, in `kernels.json`, the
  named attributes of every solution its rows name: the `sizeMapping` values
  TensileLite hands the engine, as int64, with the C++ defaults for absent
  optional keys and the legacy `streamK` keys normalized the way TensileLite
  reads them (`lib/dat.py`); `hand_written` 1 on a solution whose
  `customKernel` names a kernel that was not generated, which only
  hipBLASLt's launch rule reads (TensileLite hands it to no engine); and
  their kernel ids and signatures (next section). The engine `Config`s of a
  kernel pool read from the library, or
  from data with `kernels.json`, carry them; v2 models do not read them, v3
  models read those of their attribute spec. With the run's config
  (`--config-yaml`, which the orchestrator passes) a tested row whose data
  types, compute type or scale modes, as the bench prints them, are not
  those of the config's `hipblaslt` section stops stage04 with 1: those logs
  hold another problem type than the run trains.
- stage05 trains on one pair per (GEMM, solution) of the pool, or with
  `train.served_candidates` on one per key (the solution the runtime runs
  for it), and keeps each GEMM's candidates in pool order, so that its
  training-time picks break score ties like the engine. Whitelists stay per signature (macro tile, matrix
  instruction, cache hints); a signature's latency for a GEMM is the minimum
  over its solutions.
- Every evaluation (stages 4.5, 5, 6 and 8) picks a pool position: the first
  kernel of the engine's ranking the runtime would run (of kernels with equal
  scores, the first in pool order). A kernel without a row is passed over,
  as the runtime passes over a kernel whose predicates fail, unless the GEMM
  was benched from a plan and the kernel is off its list. stage05 gets the
  plans its training data was benched from (`--plan`: each training
  round's `stage02/plan.json` and the seed bench's; `--context-plan`: the
  context re-plans with `contexts.replan`); a GEMM's list is the union of
  its shape's lists, so its early stopping and deployed numbers count an
  off-list pick as stages 4.5 and 6 do instead of scoring the next
  measured kernel of the ranking. Those stages re-bench it; stage05 cannot,
  and the slowest latency the GEMM measured can be far from what an off-list
  pick costs, so `train.off_list_sel_eff` (`--off-list-sel-eff`; unset: the
  slowest) charges it the winner's latency over that efficiency instead, e.g.
  the geomean a re-bench of a previous model's off-list picks measured.
  With `auto` that geomean is measured each round: stage04b re-benches the
  previous model's misses on the round's new GEMMs (`rebench_misses.json`),
  and stage05 (`--off-list-rebench`, `lib/miss_rebench.py`
  `off_list_charge`) charges the geomean of min(1, ratio) over its measured
  off-list misses, a context with at least `train.off_list_min_misses` (30)
  of its own gets its own geomean. With fewer overall, as in round 0, it charges
  `train.off_list_fallback` (unset: the slowest latency). `metrics.json`
  keeps the charge under `off_list_charge` (value, source, n per context).
  `validate.pick_measurement` (stages 4.5, 5, 6, 7 and 8; stage05's early
  stopping and deployed numbers) says how a pick without a timed row of its
  own costs. `warmup` (the default) charges a pick with only a skip-screen
  row its warm-up latency, a close and slightly pessimistic stand-in, and
  counts a pick off the GEMM's planned list as a miss at the slowest latency
  measured for the GEMM, since nothing measured it (stage 7 reads no planned
  list: it leaves such a problem out, see *Validation*). `timed` counts both as
  misses, a strict lower bound; `any` charges the warm-up and passes over
  off-list kernels like kernels without a row, an upper bound. `rebench`
  charges a miss by a measurement: `lib/rebench.py`'s `rebench_picks` times
  each missed pick in one bench line with the GEMM's timed winner and up to
  two more timed kernels, a few runs in the run's kernel order (shuffled,
  then rotated per run; see *Bench timing*), and the pick costs the
  winner's latency over the ratio of their medians (at most the winner's
  efficiency); a miss without a re-bench counts as under `warmup`.
  With `validate.rebench_misses` (on by default) the stages that decide,
  stage 4.5 (the previous model on the new GEMMs, and `--prior-models`) and
  stage 6's deploy gate (the model and its baseline), and stage 8 measure
  their misses that way (`lib/miss_rebench.py`): each evaluates under
  `validate.pick_measurement`, re-benches every missed GEMM (a skip-screened
  pick, or a pick off the GEMM's planned list, which its line then lists
  explicitly) on the run's devices with the run's bench, pool-only library
  and watchdog (one block per GPU for a call of up to about a thousand
  lines per GPU), in `validate.rebench_runs` runs, and evaluates again under
  `rebench`. Decisions use the re-benched numbers; `decisions.json`,
  `gate.json` and stage08's summaries keep the configured rule's as
  `pre_rebench` and count `n_rebenched`, `n_rebench_failed` (the bench could
  not run the pick, e.g. its predicates fail, so it stays a miss) and
  `n_context_skipped`; `rebench_misses.json` lists every miss, with the
  library digest of its solution indices and each pick and re-benched
  winner named by signature and kernel id. A GEMM of
  another execution context is re-benched in that context through stage
  3.5's wrapper (`--sm_count_target`, the schedule's hybrid policy, the
  cotenant); one whose context the config does not list, or whose cotenant
  is unavailable, is skipped and counted. Selection
  efficiency is relative to the fastest timed solution. stage08's
  `per_gemm.csv` gives the pick and the Origami pick as pool position,
  solution index, kernel id and signature, and each miss.
- Stage 7 compares the pool position of the runtime's pick (`top1_index` of
  the engine's pick log) and its serving cell; the kernel signature is only
  reported (`pick_mismatch_same_sig`, `same_index_other_sig`).

v2 models serve only the exclusive execution context: every CU of the device
and the default tile scheduling. hipBLASLt calls with a CU budget or a
dynamic schedule are not scored by them and keep the analytical ranking.
Benches do not inherit the scheduling override
`TENSILE_PERSISTENT_HYBRID_FORCE_MODE` (or its legacy name
`TENSILE_STREAMK5_FORCE_MODE`); a config that sets one in `runtime.env`
forces that schedule on every bench, over the one each context asks for, and
the config check warns about it. stage04 records the context a bench ran
under in the `cu_budget` and `schedule` columns (`--cu-budget`,
`--schedule`; 0 / `default`, or CSVs without the columns, mean exclusive). A
problem measured under another context is another GEMM; evaluations rank
each GEMM under its context, and v2 training drops the non-exclusive ones.

`train.catalog: v3` trains MLREC_v3 models: the v3 features (context-aware and
framework-agnostic: a kernel is its signature, plus item features from kernel
attributes through `train.attribute_spec`, a list of `{name, default,
transform, values}` with transforms `identity`, `log2p1`, `nonzero` and
`onehot`, for parameters every GEMM kernel framework defines only; the default
`DEFAULT_ATTRIBUTE_SPEC` in `lib/features.py` is empty), the launch plans
hipBLASLt makes for each training pair (`lib/hipblaslt_launch.py`, from the
kernel's occupancy and its `tile_processing_strategy`, `work_assignment`,
`stream_k_atomic`, `work_group_x/y/z` and `hand_written` attributes:
TensileLite reduces the partial tiles of a hand-written Stream-K kernel in a
tree only, so such a kernel that can run on a static grid is a launch class
of its own; evaluations hand the engine the same plans and `launch_key`s as
its launch model;
`scripts/check_launch_plans.py` compares TensileLite's plans and keys for a
library's pool with these, and counts apart, without failing, the plans of a
static Stream-K launch of batch 1 with one K iteration per tile and more
tiles than CUs, where TensileLite takes the grid Origami's
`select_grid_size` chooses, which can differ from rule version 1's; a dynamic-queue
Stream-K launch also needs the queue count hipBLASLt compiles into those
kernels, the table's for the architectures that have one, else the device's
XCD count: `hardware.num_xcd`, which a run that leaves it unset reads when it
starts from the KFD `num_xcc` of its first bench device when the device's PCI
`current_compute_partition` is `SPX` (one partition) and writes into its
snapshot; preflight stops a v3
run whose pool holds a kernel that can launch with a dynamic queue under its
schedules and budgets while the count is unknown; a persistent kernel,
DataParallel or Stream-K, launched in clusters, whose ClusterDim spans more
than one workgroup, is outside the mirror: TensileLite plans its launch with
cluster rules the mirror does not model, so preflight stops a v3 run whose
pool holds one, blocked or not, except `--mode audit`), launch keys
(`train.launch_keys`, default off: the model's serving flag that makes the
engine score each launch class of a view key on its own instead of the view
key's first kernel only; the integration's launch model returns the keys
either way and the model decides, and every evaluation groups as the engine
does for that model), the declared contexts (the CU fractions from the
smallest in the data, or `train.min_cu_fraction` when that is larger, to 1;
the CU floor, the fewest usable CUs in the data; `train.schedules`, default
the schedules in the data), per-cell clip
bounds (the training feature ranges; `train.clip_features`), whitelists keyed
by `train.whitelist` (`none` by default: every feasible kernel through the
cascade, `train.cascade_k` default 32, whose first stage ranks by the
embedding term alone, which `train.stage1_recall_weight` trains to rank as the
full score does; `signature` or `view-key`) or a whitelist tier chosen after
training (`train.whitelist_tier_k`, see *Whitelist tier (stage05)*), and the
soft feasibility rules off unless `train.small_batched_rule`,
`train.dot2_rule` or `train.non_temporal_rule` is set.
`train.non_temporal_family_rule` (default off; needs
`train.non_temporal_rule`) refines the non-temporal rule for batched
problems with K <= 512 that are thin along exactly one of M and N: a kernel
family (equal `mt` and `mi`) the rule rejects whole keeps one member
(`lib/feasibility.py`). It is model-shaping, and a model served with it
needs training data that measures the members it keeps. Neither rule, nor
an attribute holding the temporal hint of an operand the tensor data mover
loads, can serve a pool with such operands (*Kernel identity across
hipBLASLt builds*). A v3 bundle records
the view keys with training rows (`trained_view_keys` in `models.pt`: every
loaded round's, plus the prior bundle's), and with `train.trained_key_gate`
(default on) the model scores only those keys unless none of them is
feasible for a problem, so kernels added to the pool after training wait
for a model trained on them; stage05's deployed evaluation serves the same
way. A v3 bundle also records whether the pool stage05 ranked held the bank
(`serving.bank`), which the model file carries for the integration. A v3 model scores
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

### Kernel identity across hipBLASLt builds

A solution index names a kernel of one hipBLASLt build only:
TensileCreateLibrary numbers the solutions of every architecture and library
file in sorted order, so adding or removing any library file or architecture
renumbers the others. Model files hold no solution index (signatures and
view-key hashes), but the pipeline's CSVs, plans and records do, so each
records the kernels its indices named (`lib/kernel_identity.py`):

- a kernel id is `<signature>:<name hash>`: the 8-field signature written
  `MT<m>x<n>x<depthU>_MI<m>x<n>x<k>_NTA<a>_NTB<b>`, and a hash of the
  canonical solution name (the `.dat` `name`, which the bench prints as
  `--Solution name`, as the identity rule below reads it) that tells the
  kernels of one signature apart;
- a file digest hashes one library file: its stem, every (solution index,
  kernel id) pair and its Prediction table order, the order the runtime ranks
  the pool in; its index digest hashes the same without the table, what the
  file's indices name; the library digest hashes the file digests of the
  files a stage reads (the library's file and the files of the pool's bank,
  `lib/bank_pool.py` `config_identity`).

The identity rule, `tdm_th_v1` (`IDENTITY_RULE` in
`lib/kernel_identity.py`), reads TDMInst, the bit mask of the operands the
tensor data mover loads (bit 0 A, bit 1 B): the solution's `sizeMapping`
value where its library has one, else the `_TDMI<n>_` token of its name, else
0 (a name that spells two, or a value the name contradicts, is an error,
`lib/dat.py`). A loaded operand's temporal hint (TemporalHintA/B) does not
reach its code, so it decides nothing:

- its cache hint is 0, in the signature and everything built on it (view
  keys, v3 features, non-temporal availability, planner families,
  whitelists), and TensileLite's adapter hands the engine the same: it
  reads TDMInst from the solution name, since TensileLite's own cache hint
  ignores it (`tests/data/tdm_cache_hint_cases.tsv` holds cases both read
  alike);
- the canonical name sets its `_THA<v>_` / `_THB<v>_` token to 0 (the D
  store's `_THD<v>_`, `_NTA<v>_` / `_NTB<v>_` and the MX-scale hints stay
  as written; `tests/data/tdm_identity_cases.json` holds cases the
  integration's adapter must read alike), and the kernel id hashes the
  canonical name;
- solutions with a loaded operand that share a kernel id are one kernel, an
  alias group: stage02 benches its first member in pool order, the one the
  runtime serves (the members share their predicates), in every mode; the
  pool keeps every member at its position, so the runtime's positions stay;
  a GEMM's planned list, read with the library's identity (stages 4.5, 5, 6
  and 8, the report), holds the other members of the groups it lists, so
  an evaluation passes over a member without a row as the runtime does and
  none is drawn as an unmeasured kernel; a blocklist name blocks the group
  whole; the audit set's kernel digests read the canonical name and the
  hint as 0, and leave a group's other members out of the pool's digests;
- the non-temporal rules read cache hints, and an attribute holding a
  loaded operand's temporal hint would split its group into keys: preflight
  and stage05 refuse `train.non_temporal_rule` (v2 serves it unless it is
  false), `train.non_temporal_family_rule` and a `train.attribute_spec`
  naming `temporal_hint_a` / `temporal_hint_b` of a loaded operand for such
  a pool, preflight in the train section and in the final retrain's
  recipe.

Kernels without a loaded operand keep the kernel ids, keys and digests they
had before the rule. stage05 records the rule in `model_options.json`
(`identity_rule`: a record without one was trained with `tdm_th_v1`, whose
keys every pool trained before it gives), `models.pt` and `metrics.json`,
stage06 in `deploy_record.json`.

A record, under `kernel_identity`, holds the library digest, every file's
digest and index digest and, per file, the kernel id of every index the
artifact names, with the identity rule (`rule`) and the aliases of the alias
groups it names a member of (`aliases`, alias index -> first member, when
there are any; a plan's are the members it left unbenched). A recorded file
names other kernels when its index digest differs (its file digest, in a
record written before index digests were), so a rebuild that only reorders
the Prediction table changes none. Records:
stage04's `kernels.json` (the digest also in `enrich_summary.json`), every
stage02 `plan.json` (the round's, the seed bootstrap's, the context
re-plans), the audit v2 canary, generation plans and model registries, an
audit increment's `increment.json` and `shipped_subset.json`. A bench records
the files its lines name kernels of, without kernels: stage03 in
`stage03_summary.json`, stage03b in `rebench_summary.json` when stage03
recorded them (logs nothing checked against a library keep no record). The
audit manifest gives the library digest of each generation; stage07's summary,
stage08's and the audit's records, `rebench.json` and `rebench_misses.json`
the digest of the library their indices name. Persisted outputs name a
kernel by its signature (`MT256x256x64 MI16x16x32 NTA0 NTB4`) next to its
kernel id: stage08's `per_gemm.csv` (`pick_kernel_id`, `pick_signature`, and
the Origami pick's), `kernels.json` (`signatures`), `shipped_subset.json`
(`kernels`), `rebench.json` (each candidate and the winner before and after,
when stage03 recorded its library), and `rebench_misses.json` and the
re-bench record in `gate.json` (each pick and re-benched winner). stage04's
name checks name the kernel the bench ran by its kernel id and solution
index in `--dat-dir` when the library holds its name.

Before a stage joins such an artifact with the library it reads now, it
compares the record with that library file by file. When no recorded file
the stage reads names other kernels, the indices are kept, those of
recorded files it does not read included (a stage without the bank reading
rows of bank kernels), except an index that a file the stage reads holds
another kernel under, which is unresolved. Otherwise each recorded index of
a file the stage reads maps to the index its kernel has now, an alias
group's first member's for a group; a kernel the library no longer holds
under one index (retuned under its own index, removed, or held twice
outside an alias group) is unresolved, and so is every index of a recorded
file the stage does not read. Then:

- joins that can follow the kernels read the indices as their kernels'
  current ones and leave the unresolved ones out: stages 4.5, 5, 6 and 8
  read enriched rows and plans so, stage 7 its timing CSVs, the report's run
  analysis every bench folder and stage 8 dataset, the audit v2 score its
  medians; the rows left out are counted (`rows_kernel_not_in_library`).
  stage04 reads a recorded plan's kernel lists so too and fails only when a
  kernel on a list is unresolved, so an audit increment enriches with the
  exhaustive generation 0 plan after any rebuild;
- joins that cannot follow them refuse, naming the kernels by kernel id: the
  orchestrator benches no round whose plans name other kernels now (plan it
  again from stage 2); stage04 refuses logs whose bench recorded library
  files that name other kernels in `--dat-dir` now, which covers the skip
  rows the bench prints no name for, and fails on the first tested row of
  such logs whose `--Solution name` is not the `.dat` name of its index (in
  logs without a record a problem with such a row is dropped, and more than
  ten such rows fail the stage); stage03b does not merge re-bench runs into
  logs stage03 benched with other library files; `scripts/audit_v2.py`
  neither plans on top of nor benches a generation recorded with another
  build (or, written before records, whose indices name other kernels by
  the audit set's kernel digests as of its plan, against every kernel of
  the pool, those a blocklist leaves out too); `--mode audit` stops when
  the build changed since its generation 0 or an increment not benched yet
  was planned; `choose_shipped_subset.py` refuses bench data whose indices
  name other kernels.

Enriched rows written before identities were recorded carry each kernel's
engine config, and their `kernels.json` its attributes: joined with a
library, a solution index whose config or attributes differ from the
library's kernel there names another kernel, and its rows are left out with
a count that names the kernels (the joins that refuse refuse instead); when
all agree they are read as they are. Plans written before identities were
recorded carry neither: the stages that read one (stage04, and every stage
that attaches plans to rows) and the orchestrator, before it benches a
round's lines, take its kernel lists as they are and warn once per plan.
Without a library's identity (a stage given no library) indices are taken
as they are; the report's run analysis then merges bench folders by index
only when their records agree and each folder without a record names the
kernels the records name under the same indices (by the signature of its
rows and the attributes of its `kernels.json`), and otherwise skips the
analysis with a note saying why. An index no record names is not checked,
so folders without a record are not compared with each other.

The audit set follows its generations by kernel digest (`lib/audit.py`); see
"Audit set" below.

A rebuild that only renumbers keeps the pool order (library files in name
order, then each file's own order); a retuned pool file can also reorder the
members of a key, which changes the member the runtime serves (the first in
pool order whose predicates pass). stage05 records the member order of every
key with several members in the pool it trains with (`member_order.json`, a
digest of the members' kernel ids per key, the keys grouped as the model
groups kernels); stage06 records it for the library the model is served from
in `deploy_record.json` (`member_order`) and warns about each key whose order
differs from the training build's. Measure such a model on the audit set
again (an increment) before it ships with that build.

### Execution contexts (stage03c)

A `contexts` section benches the round's shapes under other execution
contexts too, so a v3 model learns per-context winners (under a CU budget the
fastest kernel often changes):

```yaml
contexts:
  list:                                  # trained contexts
    - {cu_budget: 0}                     # exclusive: the round's own stage03
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

An entry has `cu_budget` (0 = every CU) or `cu_fraction` (below),
`schedule` (`default`, `dynamic`, `auto`), `cotenant` (false), `fraction`
(1.0, the share of the shapes), `name` (default
`cu<budget>[_cot][_<schedule>]`) and `enabled` (true; below). After
stage03, stage03c benches each non-exclusive context in
`stage03c/<name>/`: the shapes it takes (every shape,
or `ceil(fraction * n)` per cell drawn with `contexts.seed`), with the same
bench line, kernel list and skip settings as the exclusive bench, through a
generated `hipblaslt-bench` wrapper that adds `--sm_count_target <cu_budget>`
and, for `dynamic` / `auto`, `--hybrid_assignment_policy DynamicWorkQueue` /
`Auto`. With `cotenant` it starts the bench inside `hipblaslt-cotenant --cus
<device CUs - cu_budget>`, which keeps the other CUs busy so the kernels
really run on `cu_budget` CUs; the device CU count is `hardware.n_cu`, else
the KFD topology. The wrapper appends `contexts.path_append` (default
`$ROCM_PATH/bin` or `/opt/rocm/bin`; hipblaslt-cotenant needs `rocminfo`) to
`PATH`, and does not touch `LD_LIBRARY_PATH`. Adaptive timing sizes each
kernel's measurement under the context it runs in.

`cu_fraction` (between 0 and 1) gives the budget as a share of the device's
CUs, so one config fits devices and partition modes of any CU count and
states none. A run resolves it when it starts, from the CU count of the
first bench device (`hardware.n_cu`, else the KFD topology). Inside a
compute partition (DPX, QPX, CPX) that is the partition's: each partition
is a device of its own, with its own KFD node and HIP device, so the shares
are of the partition's CUs, and the run's first line about its contexts
says which CUs, and which partition, they are shares of. The budget is
the multiple of the budget granule nearest `cu_fraction` times the CU
count, at least one granule and below the CU count, and for a trained
entry raised to `train.min_cu_fraction` of the CUs where a multiple below
the CU count allows. The granule is `contexts.granule`, by default 32
(`CU_BUDGET_GRANULE` of `lib/contexts.py`), in CUs, which are WGPs on
gfx1250. It is the budgets' own, apart from the launch rule's rounding of
the SM target (`SM_TARGET_GRANULE` of `lib/hipblaslt_launch.py`), and must
be a multiple of that (a config error otherwise), so that hipBLASLt takes
every budget as its SM target as it is and the kernels size their grids
for exactly the CUs the cotenant leaves them. One granule is one XCD's
worth of CUs on gfx950 and on gfx1250's 256-WGP part: there the smallest
budget a share resolves to is one XCD's worth, however small the share,
and every compute-partition size of the device is a whole number of
granules. An entry whose resolved budget and schedule are another's (one
with a `cu_budget`, or an earlier one) is left out with a warning, as on
a device too small for all of them, and so is one the device has no
budget for: a device of one granule or fewer CUs has none, so it trains
the exclusive context alone, and the run warns when every context of
fewer CUs is left out. Trained entries come before held-out ones, so a
small device leaves out held-out entries first; when it leaves out every
one, the run warns that it measures the models on no held-out context, and
the report says so in its notes. Preflight also warns for each held-out
context whose kernels hipBLASLt sizes as a trained context's (its SM target,
a multiple of 32 CUs, and its schedule are a trained context's, the
exclusive one included), and again when every held-out context does: the run
then measures the models on no held-out launch. The report's Execution
contexts section names every entry left out, from the run's source config
and its latest snapshot. The snapshot keeps every resolved entry's
`cu_fraction` beside its `cu_budget` and its first comment lines say what
each became; a config that gives both must give the budget the fraction
resolves to. The budgets are then checked like configured ones (below the
device's CUs, covered by `train.min_cu_fraction`), and a trained
`cu_fraction` below `train.min_cu_fraction` is a config error.

`enabled: false` turns an entry off: no stage benches or trains it, the run
names it when it starts and leaves it out of its snapshot, and the report's
Execution contexts section names it. The production configs train the
eighth of the CUs beside three quarters, half and a quarter, each on a
tenth of a cell's shapes, so that their models train and serve CU budgets
from one granule up to the whole device; `enabled: false` turns the
eighth off for its bench time: the fewer the CUs, the longer each shape
benches. The eighth adds the multiple of the granule nearest an eighth of
the device's CUs, at least one, where no other share takes that budget,
and the models then declare it as their smallest share; elsewhere it is
left out. The exclusive entry cannot be turned off. Turning the eighth on
or off, or running on another CU count or partition, changes the contexts
the audit set is bound to (*Audit set*): a run refuses an audit set built
with other contexts, complete or not, so it needs a new `audit.dir`.

```yaml
contexts:
  list:
    - {cu_budget: 0}
    - {cu_fraction: 0.75, cotenant: true, fraction: 0.1}
    - {cu_fraction: 0.5, cotenant: true, fraction: 0.1}
    - {cu_fraction: 0.25, cotenant: true, fraction: 0.1}
    - {cu_fraction: 0.125, cotenant: true, fraction: 0.1, enabled: true}
  holdout:
    - {cu_fraction: 0.375, cotenant: true, fraction: 0.03}
```

The round's stage04 enriches every trained context's logs with its
`cu_budget` / `schedule` columns (`--context-log-dir`), so each (shape,
context) is its own GEMM and ranking group in stages 4.5 and 5 and in the run
history. A model declares the CU fractions from the smallest trained
budget's to 1 (stage05's `min_cu_fraction`), so that the engine ranks a call
of fewer CUs as the smallest trained budget (the kernels still launch with
the call's own budget); `train.min_cu_fraction` can only raise that
floor. It also declares the fewest usable CUs it trained, its CU floor
(stage05's `min_effective_cus`, a budget or the bench device's CU count):
a device of fewer CUs than the bench device, a compute partition for one,
is a whole device to the engine, so a call there is in the declared range
of shares whatever its CUs, and the floor makes the engine rank a call of
fewer usable CUs than the model trained as the floor there too (as the
whole device on a device with fewer CUs than the floor).
`train.min_cu_fraction` raises the shares only. An engine that predates the
CU floor refuses every such model (`unknown serving flags`), so port the
engine to the deploy checkout and the stage-7 build before such a model
ships: stage05 stops at once with a module that predates it, and preflight
refuses one and a stage-7 TensileLite without it. `lib/mlrec.py`'s
`with_cu_floor(data, None)` removes the floor from a model file without
retraining; the file then ranks as a model without a floor. One above the
smallest trained budget fraction, or `train.schedules`
without a benched schedule, is a config error, and one below 1 without a
trained context of fewer CUs than the device has is a warning: the models
then declare every CU only. The
`holdout` contexts are enriched into `stage03c/holdout/stage04`, which no
stage trains on: stage04b reports the previous round's model per held-out
context (`aggregate.context_holdout` in `decisions.json`), and per context of
the round's new GEMMs (`aggregate.by_context`) when there are several.
Held-out budgets between trained ones test interpolation; preflight warns
about one below the smallest trained fraction, which the model does not
declare.

`contexts.replan: true` (with the planner) re-plans each context's lists in
rounds that rank with a model: stage02 scores the kernels under the context
(`stage02/contexts/<name>/`), and stage03c / stage04 use those lists and
plans. It re-plans only the shapes stage03c benches in the context: the
same `fraction` per cell drawn with `contexts.seed`
(`lib.contexts.select_shapes`), so a context's coverage quotas
(`planner.min_coverage`) count the shapes it benches. Otherwise every
context benches the exclusive list of the shape, and seed shapes keep their
exhaustive lists.

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
solutions (the files of the pool's bank are kept;
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

An improvement baseline counts only when another network than the one
stage04b evaluates measured it. A cell carried since its baseline, such as
one that passed the round before, has not been retrained, so it has no
baseline, like every cell in round 1 (below). `decisions.json` names the
network behind each cell's reading (`model_digest`: a digest of the cell's
weights, feature statistics and dimensions, which a carried cell keeps; its
whitelist tier, rebuilt every round, is not part of it) and the baseline the
cell was judged against (`improvement_baseline_source`:
`same_data_prior_model`, `cross_round_prior_decisions`, `model_unchanged` or
`no_baseline`).

A split cuts the cell at the log median of the eligible axis whose cut
reduces the variance of the cell's per-GEMM log selection-efficiency ratios
most. Only equal reductions (equal up to rounding) fall to an axis not yet
in the cell's split path, then M before N before K.

`validate.max_split_depth: 0` turns splitting off, as the configs of the
training recipes do (the gfx1250 sanity configs keep one level, see *Sanity
runs*): a cell below the threshold retrains (reason `splits_off`), and the
split keys and the structural width trigger have no effect (config
validation warns about a structural trigger it overrides).

A base cell whose splits a repair revoked (see *Repair on refusal*) never
splits again: stage04b reads the revocations of its prior rounds and of
the bundle it evaluates, leaves the revoked splits out of the tree, and
retrains the cell's leaves instead of splitting them (reason
`__split_barred`).

A split becomes a retrain when the cell is at `validate.max_split_depth`, has
fewer than twice `validate.min_cell_gemms` evaluated GEMMs, a child would get
fewer than `validate.min_cell_gemms` of them or fewer than
`train.min_cell_gemms` GEMMs over all rounds (stage05 could not train it), or
this is the final round. A leaf without a model of its own is listed for
training once it has `train.min_cell_gemms` GEMMs over all rounds. GEMM
counts over all rounds leave out the round's gate GEMMs, which stage05 does
not train on in that round.

Round 1 has no improvement baseline (round 0 runs no stage04b and no
bundle precedes round 0's), so there only the split floor and the
structural trigger can split a cell. Config validation warns when the
rounds leave no other way: three rounds with
`validate.split_after_attempts: 0`, where round 1 is the only round that
can split; it also warns when `validate.split_after_attempts` or fewer than
three rounds leave no round that can split at all.

Both thresholds are absolute selection efficiencies (relative to the fastest
measured kernel); config validation rejects `origami`. stage04b records
Origami's selection efficiency only on the GEMMs that have an Origami pick,
next to the model's on the same GEMMs (`paired_sel_eff_new`), and prints it
only with `--origami-reference`.

With `deploy.gate_frac` F > 0, stage04b keeps round(F * n) of each leaf's n
held-out GEMMs (by a hash of the run seed, the round and the GEMM) out of
the round's training: it splits the round's rows into `stage04b/gate/` and
`stage04b/train/`, and stage05 trains on `train/`. Each slice keeps what
the round's `stage04/enrich_summary.json` records of how its rows were
produced (the library, the blocklist that filtered them, how the benches
launched the kernels; not its counts), so that stage05's `metrics.json` and
the deploy record know the labels of the round's own GEMMs too. Later rounds
train on the gate GEMMs like on the rest of the round's data.

A cell's mean hides the few GEMMs it serves badly, so stage04b also reads
every held-out GEMM on its own:

- a held-out GEMM of an evaluated cell whose pick is below
  `validate.tail_floor` (0.7) of its winner is a tail GEMM. decisions.json
  lists it per cell, whatever the cell's decision (`tail_gemms`: shape,
  execution context and efficiency, worst first, at most 64;
  `n_tail_gemms`), and the next round's stage01 samples new shapes near
  each (see *Targeted shapes (stage01)*);
- picks below half the winner's speed are counted (`n_low_seleff_lt_0p50`
  beside `n_low_seleff_lt_0p70`, and their totals in `aggregate`), and with
  `validate.collapse_retrain` (true) a cell that passes on its mean with a
  held-out pick below the per-cell gate's tail floor (`gate.tail_floor`) is
  retrained in the same round (`collapse_retrain`, listed in
  `retrain_cells.txt` and `cells_collapse_retrain`; the cell's decision
  stays `pass`, so it counts as no retrain attempt toward a split). The
  retrain trains on the GEMMs it collapsed on, and its new tier's tail
  guard sees them among the cell's own GEMMs.

### Targeted shapes (stage01)

An active round's stage01 draws, besides `seed.shapes_per_cell` new
shapes per leaf, `seed.targeted_per_tail_gemm` (4; 0 off) new shapes near
each tail GEMM of the previous round's stage04b (`tail_gemms`, worst
first), at most `seed.targeted_max_per_leaf` (32) per leaf
(`--targeted-per-gemm`, `--targeted-max-per-leaf`, `--targeted-from`).
They are first-principles draws (`lib.shapes.synthesize_near`): each of
M, N and K log-uniform within an octave of the tail GEMM's own, inside the
box of the leaf of this round's split tree that holds the GEMM, in its
alignment class (a value on a macro-tile multiple stays on that multiple,
an unaligned one stays off the tile grid), its batch class kept (batch 1,
or a batch within an octave of its own), K on the block of a
block-scaled type, through the same filter, history dedup and exclusions
as every other shape: never a shape of an earlier round, of the audit set
or of an `audit.held_out` list, which stage01 checks again before it
writes the round. A leaf the round samples nothing in gets none. They are
new held-out shapes like the others, so the next stage04b measures the
model there. `targeted_shapes.json` lists, per leaf, each tail GEMM and
the shapes drawn near it, and categorization.json and per_cell_plan.json
count them.

A tail GEMM measured under a CU budget or schedule (`contexts.list`) has
its targeted shapes benched in that context as well: stage03c and the
context re-plan of stage02 (`--targeted-shapes`) take them among the
shapes the context draws for their cells (`lib.contexts.select_shapes`,
`targeted_for`), ahead of the cell's random draw.

### Weak cells and very large K (stage01)

An active round's stage01 sets each leaf's number of new shapes from the
previous round's stage04b (`decisions.json` of the last prior round,
`--decisions-from`):

- a leaf that stage04b retrained, for its selection efficiency (decision
  `retrain`: below `validate.sel_eff_threshold`) or for a held-out collapse
  (a pass with `collapse_retrain`), is weak: it draws
  `seed.weak_cell_factor` times `seed.shapes_per_cell` (rounded half up);
- a leaf that passed with a fresh held-out reading at least
  `seed.strong_cell_margin` (0.03) above the threshold stage04b applied is
  strong: it draws `seed.strong_cell_factor` times `seed.shapes_per_cell`,
  and at least `train.min_cell_gemms`, but the strong leaves together give
  up no more shapes than the weak leaves gain;
- every other leaf draws `seed.shapes_per_cell`, and so do the leaves of a
  split the previous round made (the split gave their region a leaf each).
  A leaf whose split a repair revoked reads the rows of the leaves it held:
  weak when one was retrained, strong when all were strong.

The budget rule: a round's target is `seed.shapes_per_cell` per leaf,
plus the weak leaves' extra, minus the strong leaves' cut, which never
exceeds that extra. A round's target therefore never falls below
`seed.shapes_per_cell` per leaf and never exceeds a weak leaf's target per
leaf (`seed.weak_cell_factor` times `seed.shapes_per_cell`, rounded half
up); the very-large-K and targeted shapes come on top. A round with more
weak leaves than the rounds before it is larger than they were (see
*Running* for the free space it needs). Execution contexts bench a
fraction of each cell's shapes (see *Execution contexts (stage03c)*), so a
weak cell's CU budgets grow with it.

The very-large-K stratum: with `seed.large_k_factor` F above 1, a leaf
whose new shapes hold n with K above 32,768 draws (F - 1) * n more there
(rounded half up), inside its box and under the same filter, history dedup
and exclusions as every other shape, from a random stream of its own, so
its other draws stay as they were; fewer when its box holds no more.
Batched cells hold none: the filter keeps every dimension of a batched
shape at or below 8,192. They are new held-out shapes like the others, on
top of the leaf's target, so the round's stage04b reads such a leaf on F
times as many GEMMs with K above 32,768 as its draw held. Those GEMMs
read below the rest of their LargeK cells, so a LargeK leaf near the bar
can read below it and be retrained.

Round 1, which has no stage04b before it, keeps `seed.shapes_per_cell`
per leaf, with the stratum; an enhance round (`--only-cells`, see
*Enhancing chosen cells*) draws `--shapes-per-cell` for each chosen leaf,
with neither. The round after an enhance round reads that round's
decisions, in which only the chosen leaves have new GEMMs: only they can
be weak or strong. With every factor at 1, the default, each leaf draws
`seed.shapes_per_cell` and no stratum shape, stage01 reads no decisions,
and the orchestrator passes stage01 only the keys away from their defaults
(a factor of 1, a margin of 0.03), so a config without them runs the same
stage01 command as one that sets them to their defaults.
`per_cell_plan.json` records the round's `budget` (the decisions it read,
the factors, the threshold, the counts of weak and strong leaves, the
round's count at `seed.shapes_per_cell` per leaf, the weak leaves' extra,
the strong leaves' cut and the round's target) and each leaf's `need`
(`weak`, `strong` or null) and `n_large_k`; `categorization.json` counts
the stratum's shapes (`large_k`).

A round's readings over all of its new GEMMs weight each leaf by its new
GEMMs: stage04b's `aggregate` (`global_sel_eff_new`, `all_held_out`), the
round's `held_out_sel_eff` in its round marker and in `run_status.json`,
and the deploy gate's overall reading against `deploy.target_sel_eff`,
whose gate GEMMs are a share of each leaf's. With the budget they lean
toward the weak leaves and, in LargeK leaves, toward K above 32,768, as
they leaned toward split regions when cells split, so they do not compare
as they are with the rounds of a run without the budget: compare runs per
cell, or with every cell weighted alike.

The training recipes (see *gfx950 configs*) set `seed.weak_cell_factor`
to 2 and `seed.large_k_factor` to 1.5 and leave `seed.strong_cell_factor`
at 1: a weak cell draws twice the recipe's new shapes per round, every
leaf half as many again as its draw holds with K above 32,768, and no
leaf fewer than the recipe's shapes per leaf.

### Hazard canary (stage00a)

A kernel can stop its device in a way that no signal to the bench ends: the
device takes no more work until it is reset, and other GPUs of the host can
stop with it. A bench process runs many kernels, so one such kernel would
take its whole block with it. With `hazards.enabled` (the default) the
orchestrator therefore runs stage00a before anything else of an invocation
(training, enhance, validate and audit modes), before the correctness gate
and the audit set, which run many kernels per process. Every kernel of the
pool (the library's table and the bank) of a class of `HAZARD_CLASSES` in
`lib/hazards.py`, parameter values checked as a precaution, runs alone
first:

- `cluster_dim_y4`: ClusterDim with a second dimension of 4;
- `prefetch_gl2_2`: PrefetchGL2 2;
- `tdm_iterate_mode_3`: TDMIterateMode 3.

The class comes from the kernel's logic parameters where the library
records them (`clusterDim`) and from its solution name otherwise: TensileLite
abbreviates a parameter by the capitals of its name, so `_CD2_4_` is
ClusterDim [2, 4], `_PGL2_` PrefetchGL2 2 and `_TDMIM3_` TDMIterateMode 3.
Each kernel runs in a hipblaslt-bench process of its own on two problems:
two by two of its macro tiles per cluster with K four DepthU iterations (a
multiple of the MX block), then partial tiles in M and N with a batch of 2.
One such process runs at a time on each GPU of `hazards.devices` (default:
the first of `runtime.devices`), each GPU leased while the canary runs,
`hazards.runs` (1) processes per kernel, and before each one the canary
waits, as the correctness gate does, for other processes to leave the GPU.
The benches' stall watchdog kills a
process that prints nothing for `hazards.timeout_s` (300; until it has
loaded its library, which it says by finding a problem's solutions, not
before `bench.startup_grace_s`). Before the first kernel, and after every
kernel that hangs or faults, a kernel of no hazard class (the first of the
pool's first three that runs a canary problem) runs the same way as a
control. When none of them runs a canary problem before the first kernel,
the canary's lines do not fit the library: it stops with exit 1 and blames
no kernel.

A kernel is quarantined when its process hangs (`hang`), ends by a crash, a
fatal GPU error or a non-zero exit (`fault`), or runs none of the canary's
problems (`untested`: a kernel that never ran alone does not run among others
either); a process killed from outside or by the stage blames no kernel, but
a kernel whose run was cut short from outside stays out of the invocation all
the same, and the next one runs it alone again. The canary stops (exit 3),
and with it the invocation, when a bench it killed does not end, when the
control kernel fails, or when a process the GPU driver lists stays in
uninterruptible sleep before the first kernel runs (whatever stopped it would
be blamed on that kernel); after a hang or a fault also when no control
kernel runs a canary problem, when such a process shows up, or when the pool
has no kernel outside the hazard classes to check the device with. The
verdicts so far are recorded, and the same command goes on once the device is
reset. With several devices, a kernel that takes the driver down can stop the
kernels running on the other GPUs too, and they are quarantined with it; one
device blames one kernel.

Records go to `canary.jsonl` in `hazards.cache_dir` (default `<run
dir>/hazards/cache`; give a run, its audit set and the other runs of a host
one directory, as a run's snapshot does for the audit set it builds): a
`start` record, on disk before a kernel's process starts, then an `end`
record with its verdict, each naming the process that wrote it. A run whose
start has no end (the bench, the stage or the host went down while it ran)
quarantines its kernel as well, unless its process still runs: another
invocation sharing the directory runs that kernel alone now, and this one
keeps it out unchecked unless that run has passed by its end. A quarantine
holds for the kernel's name on the run's `arch` whatever build, device or
protocol later invocations have, since a kernel that stopped a device once
may do so again (to check a kernel again, remove its lines from
`canary.jsonl`). A pass holds for the kernel id, the device (processor, PCI
chip id and CU count), the digest of the logic file and code objects that
hold the kernel and the canary's protocol (its problems, `hazards.runs` and
the hipBLASLt host libraries). A kernel with a verdict does not run again;
when every hazard-class kernel of the pool has one, the canary runs nothing
and takes no GPU.

stage00a writes `quarantine.json` (the pool's quarantined kernels, and those
kept out unchecked, as a blocklist: name, kernel id, signature, reason),
`verdicts.json` and `canary_summary.json` into `hazards/canary.new/`, which
replaces `hazards/canary/` when the canary succeeds. The orchestrator adds
the quarantine to the invocation's blocklist (*Correctness gate*): the gate
blocks its kernels as listed and never runs them, and without the gate the
invocation's blocklist is the quarantine itself. A quarantined kernel thus
leaves every plan and pool, stage04's rows and the deployed sidecar, as a
kernel that computes wrong results does; the audit set does not bench it,
and its completeness check does not wait for it. A kernel kept out unchecked
leaves them the same way for its invocation alone; an audit set built for
the run takes it from neither the run's blocklist nor its sidecars unless a
config snapshot of the run names it for a finding (the gate lists a kernel
the canary keeps out with the canary's entry alone, so a keep-out can hide
what the gate found in another invocation), and the set's own canary runs it
alone first (*Audit set*). Stage 7 skips the runs in which the analytical
ranking picks kernels (the ML-off timing runs, and with them the timing, the
Origami reference, the order parity and the production route) while the pool
holds one, and every run while the validated bundle's blocklist leaves one
in (*Validation*); its production route also keeps out the kernels of the
hazard classes that rows ahead of the model hold outside the pool, which the
canary never runs. A pool that keeps quarantined kernels needs
`planner.enabled`, since a bench line without a kernel list runs every
kernel of the library: the orchestrator refuses to go on without it.

### Correctness gate (stage00)

Some kernels of a library compute wrong results for some problems. With
`correctness.enabled` the orchestrator runs stage00 before anything else of
an invocation (training, enhance and validate modes; an audit set runs no
gate of its own: its config, `lib.audit.bench_config`, lists the blocklist
of the runs it is built for, *Audit set*). It checks the kernels of the
pool, the library's Prediction table and the bank, with an exact harness,
`tools/errmap.cpp`: A, B and C hold small integers, so every product and
partial sum of the f32 accumulation is exact and a correct kernel writes the
reference bits into every element of D; any other element is wrong, and an
element a kernel does not write keeps a NaN pattern (the harness prints NaN
and infinite values as printf does, and the gate reads such a line as the
failing repetition it is). The harness picks kernels by solution index and
links the run's hipBLASLt build (`runtime.build_dir`), whose library it
checks against the names of the logic files the pipeline reads: every line
it prints is checked as it arrives, and a kernel it names otherwise, or a
process that finds none of its kernels, stops the gate at once (a first
launch, which holds one kernel, leaves a kernel the harness finds nowhere
untested, and stops the gate only when another process found it). It is
built the first time a kernel needs a check, into the cache directory, with
`correctness.compiler` (default: `amdclang++` of `ROCM_PATH`, else `hipcc`;
it has no device code); `correctness.harness` names a prebuilt one instead.
It runs bf16 and f32 A, B, C and D with f32 or xf32 compute in every layout
(a padded leading dimension exceeds the rows its matrix is stored with in
the config's layout), A and B of `f8_r` (E4M3) or `bf8_r` (E5M2) with one
f32 scale per tensor (`hipblaslt.scaleA` and `scaleB` 1, the libraries'
`UseScaleAB: Scalar`, gfx950's F8BS problem type among them), and A and B of
`f8_r`, `bf8_r` or `f4_r` (E2M1) with block-32 E8M0 scales (MX:
`hipblaslt.scaleA` and `scaleB` 3), with bf16 or f32 C and D; a config of
another problem type (vector or other block scales, A and B scaled
differently, unscaled 8-bit inputs, 6-bit inputs) cannot enable the gate and
gives `correctness.lists` instead.

With one f32 scale per tensor, A and B hold the same small integers, the
scale of A is 2 and of B 4, and the operands are stored with K as it is.
hipBLASLt passes the scales to the kernel as device pointers, unchanged, and
the kernel multiplies them into alpha, so the reference is alpha times both
scales times the dot product, plus beta times C and the bias: every value
stays exact, and a kernel that leaves out a scale or applies one operand's
to both is wrong wherever a dot product is not 0 (one that swaps the two
computes their product all the same). To see that the harness sees such a
kernel on a new device or build, run it by hand on one kernel with
`--drop-scale A`, `B` or `AB` (the gate never passes it): the kernel is
given scales of 1 for those operands while the reference keeps 2 and 4, so
a correct kernel is reported wrong by those factors.

Block-scaled A and B hold the same small integers, which those types hold
exactly (f4 two to a byte), and every scale of A is 2 and of B 4: every
product stays exact, and a kernel that leaves out a scale or applies one
operand's to both is wrong wherever a dot product is not 0. With every scale
of an operand alike, the harness need not know the scale layout, which
differs between devices and hipBLASLt versions: it writes as many scales as
the largest padding hipBLASLt gives the scale tensor (K/32 blocks rounded up
to 8 times M or N rounded up to 32, per batch). hipBLASLt's own data holds
zero scales in that padding, and some kernels rely on them to cancel what
they read past K, so the harness stores a block-scaled A and B with K
rounded up to a multiple of 256 and zeros past K (in the leading dimension
where K is the stored rows, which a padded shape pads beyond that; in the
batch stride where K counts the columns): whatever such a kernel reads there
multiplies to 0. The other side of all this: the harness does not check how
a kernel indexes its scales, which only differing scales would show, nor a
leading dimension of exactly K when K is not a multiple of 256. To see that
the harness fails a kernel on a new device or build, run it by hand on one
kernel with `--corrupt-ref I,J,BATCH` (the gate never passes it): that
element of the reference is one unit in the last place off, and a correct
kernel is reported wrong there and only there. `--drop-scale` gives
block-scaled operands scales of 1 as well.

The harness runs in the benches' environment without TensileLite's launch
overrides (`lib/bench_env.py` `bench_env_with`; the run's other `runtime.env`
variables reach it): it checks every kernel as the runtime launches it in
production, where no override is set, since the sidecar guards those
launches, whatever overrides a run's benches time under. It checks the
exclusive whole-device context only: the reduced-CU budgets, co-tenants and
schedules of the trained execution contexts (stage03c) launch the same
kernels with other grids and Stream-K decompositions, and are not
checked.

Every kernel runs the shapes of `SHAPES` in `lib/correctness_gate.py`: one
partial M tile under persistent launches that loop over many tiles, long K
over many tiles (data-parallel and split along K), wide N, operands over 4
GiB, alpha != 1 with beta != 0 and a partial last M tile (with padded
leading dimensions, and with a separate D), and K of one DepthU iteration; M
and N are rarely tile multiples. Block-scaled kernels run those of
`mx_shapes` instead: the same problems with K a multiple of 32 (an odd one,
which no DepthU divides, unless the shape is about K), M not a multiple of 8
and mostly prime (a store vector width that does not divide M is where
out-of-bounds stores would hide), an operand of 4 GiB or more as one matrix
and as many small batches, at the least K (an odd multiple of 32) that takes
it there at its element size, A and B leading dimensions padded by whole
scale blocks, K of one iteration of DepthU 128 and of 256, and the bias
epilogue (a vector of M small integers added to every column of D, of the
bias type hipBLASLt defaults to) on a batched persistent shape, on alpha !=
1 with beta != 0 and a separate D, and on the batched operand over 4 GiB,
all three also run without it. 8-bit kernels with one f32 scale per tensor
run `SCALAR_SHAPES`: their libraries assert no multiple of K, M or N and no
alignment of a leading dimension, as the bf16 and f32 ones do, so these are
the problems of `SHAPES` (K not a multiple of a DepthU unless the shape is
about K, odd pads), with the operands over 4 GiB at the least odd K that
takes them there at one byte per element, also as many small batches, K of
one iteration of DepthU 64, 128, 256, 512 and 1024, K of one DepthU-512 or
DepthU-1024 iteration plus an odd tail of 7 on the same M and N (a kernel
wrong only in its tail loop fails there alone), and the bias epilogue of
their libraries (a vector over the rows of D, of bf16, hipBLASLt's default
for 8-bit A and B, and once of f32, both of which their kernels take) on a
batched persistent shape, on alpha != 1 with beta != 0 and a separate D, and
on the batched operand over 4 GiB, each also run without it. The shapes
whose failures come and go run `correctness.trigger_repetitions` times, the
others `correctness.repetitions` times (at least 3 each), every repetition a
separate launch. A hand-written kernel (`correctness.first_launch: custom`;
`all` for every kernel, `none` for none) also runs as the only GEMM of
`correctness.repetitions` fresh processes at K = its DepthU, since some
kernels fail only on the first launch of a process. A shape a kernel does
not support (a hand-written kernel needs K to be a multiple of its DepthU)
runs again with K rounded up to that multiple, and a failure is followed up
on the other kernels of the same macro tile and matrix instruction,
`trigger_repetitions` times.

Every kernel also runs the tiny shapes (`tiny_shapes_of`), each
`correctness.repetitions` times: M of 1, 2 and 3 with N of 257, then N of 1,
2 and 3 with M of 257, K 383 (416, an odd multiple of 32, for block-scaled A
and B). A kernel that reads A along M (A not transposed) or B along N (B
transposed) several elements at a time, under predicates that admit a
smaller free size, reads outside the operand there. When hipMalloc places
the operands, a small one sits next to the problem's other buffers in one
block: such a read may land in another buffer, return bytes the kernel then
leaves out, and go unseen, the result exact. So the tiny shapes run with
every operand at the start of an allocation of its own (the harness's
`--fresh 1`): for each of A, B, C, a separate D, the scales and the bias the
harness reserves a range of virtual addresses (`hipMemAddressReserve`), maps
a physical allocation of the buffer's own (`hipMemCreate`, `hipMemMap`,
`hipMemSetAccess`) after its first 2 MiB and leaves those unmapped, so that
a read before the buffer faults: a crash, which runs again alone and blocks
the kernel. A read past the end of a buffer is not certain to fault: it may
land in the rest of the buffer's last granule or in memory mapped after its
range. Placing the buffers so takes the device's virtual memory management
(`hipDeviceAttributeVirtualMemoryManagementSupported`); without it, or when
one of its calls fails, the buffers come from hipMalloc and the harness's
ready line says so (`"fresh": false`). The gate asks the harness for the
option first (`errmap --caps`, which calls no HIP function); a prebuilt
harness built before it (`correctness.harness`), or one whose `--caps`
fails, runs the tiny shapes on its usual allocations. The gate warns in both
cases and records per tiny-shape verdict whether every run of it had fresh
operands (`evidence.fresh`); its summary says whether the harness takes the
option (`tiny.harness_fresh`) and how many tiny-shape processes had fresh
operands (`tiny.fresh_tasks` of `tiny.tasks`). A tiny-shape pass or untested
verdict made without fresh operands is cached but never taken: the next gate
runs those kernels on the tiny shapes again (`tiny.cached_not_fresh`), so on
a device without virtual memory management every gate runs them; a crash or
a failure stands wherever it happened. The tiny shapes are a stage of their
own (`TINY`), with a protocol that names the stage and cached verdicts of
their own: a kernel with cached verdicts of the other shapes runs the tiny
shapes alone, and a kernel's verdict is the worse of the two stages' (its
row in `verdicts.json` holds the tiny stage's in `tiny`).

Before the harness runs anything, the gate checks every hand-written kernel
of the pool (a solution whose `customKernel` was not generated) statically,
unless `correctness.static_check` is false. TensileLite builds such a kernel
with global-read vector widths of 1 and derives its LeadingFree size
predicates from those widths, while the kernel reads A and B as wide as the
`custom.config` of its source declares (`GlobalReadVectorWidthA` and
`GlobalReadVectorWidthB`). Along a leading free dimension, M of a
non-transposed A and N of a transposed B, a free size below that width reads
outside the operand. A kernel whose predicates admit such a size
(`LeadingFree0SizesGreaterOrEqual` and `Free0SizeMultiple` bound M, their
`1` twins N, and `SizeEqual`, `SizeMultiple` and `SizeGreaterThan` the size
of their index) is `unguarded`: blocked with the reason and what the check
compared (the row's `static`: per operand the width declared and built, the
least free size and the predicates that set it, the source and its sha256),
and never run. The source is the kernel's `<name>.s`, found as TensileLite
finds it (at the top of the directory, else the first of a sorted walk), in
TensileLite's `CustomKernels` of the hipBLASLt source the build was
configured from (`CMakeCache.txt`), or below `correctness.custom_kernels`.
A kernel whose source cannot be found or read is `unchecked` by the static
check, and one whose `custom.config` declares no width for an operand it
reads along a leading free dimension `undeclared`; the gate warns, and both
run on every shape. The check reads the logic files and sources at every
gate and is not cached, so a rebuilt library or an edited source is checked
as it is now.

A harness process that prints nothing for `correctness.timeout_s` (before
it is ready, between two records or after its last one) hangs and is
killed. When a process crashes or hangs, the first kernel it found that
lacks a repetition runs again alone in a fresh process, and the kernels
after it in another; a process killed by a signal from outside (an
operator, the OOM killer) blames no kernel and stops the gate. A kernel
fails (`fail`) when a repetition returns a wrong element, crashes (`crash`)
when it crashes or hangs also alone, passes when it ran and did neither, and
is untested when no shape supported it or the harness did not find it.
Failing, crashing and unguarded kernels are blocked.

The gate runs one harness process at a time on each GPU of
`correctness.devices` (default: `runtime.devices`), with the GPUs leased
while it runs (*GPUs: leases, quarantine and platform records*) and the
stage's CPUs split between them; before each process it waits for other
processes to leave the GPU. Verdicts are cached in `correctness.cache_dir`
(default `<run dir>/correctness/cache`; give runs one directory to check a
pool once) by kernel id, device (processor, PCI chip id and CU count) and the
digest of the logic file and code objects that hold the kernel, so a rebuilt
library is checked again. A failure holds under any protocol, a later
harness's too: a kernel that returned a wrong element may do so rarely, and a
check that missed it would serve it (to check a kernel again, for one after a
fault of the harness, remove its lines from `verdicts.jsonl`). A crash or a
pass holds only under the stage, shapes, repetitions, first-launch policy,
harness and hipBLASLt host libraries it was made with, and a crash whose
every lone run hung only while `correctness.timeout_s` is no longer than the
one it hung under (the cache keeps it as `timeout_s`); a tiny-shape pass or
untested verdict made without fresh operands holds under none. A verdict is
cached as soon as no later pass of the gate can change it, and a gate that
stops early (an error or a signal) still caches the failures it saw. A kernel
a list names is not checked. When every kernel of the pool has a verdict of
both stages the gate runs nothing, builds nothing and takes no GPU.

`correctness.lists` names files of kernels known to compute wrong results: a
list with one kernel per line, `<class><TAB><name>` or the name alone (`#`
starts a comment), or the `blocklist.json` of another run. An entry without
a path separator or suffix names a shipped list: `gfx950_bbs_tn`
(`blocklists/gfx950_bbs_tn.tsv`) lists the kernels of the gfx950 BF16 TN
table and CU256 bank found that way. A kernel is named as its library's
logic file names it (the solution name hipblaslt-bench prints and
TensileLite's `ContractionSolution::name()` returns); names are
integration-side data, and models know a kernel only by its signature. A
name blocks every kernel of its canonical name, an alias group whole (*Kernel
identity across hipBLASLt builds*), and the sidecar lists every member. It
also blocks every kernel whose name equals it once both drop their IA
(InterleaveAlpha) and PKA (PreloadKernArgs) tokens, which builds of
TensileLite spell differently for one kernel (`lib.correctness.blocklist_key`,
the rule TensileLite reads the sidecar with), so a list or sidecar written
from one build blocks the same kernels in another; a kernel that differs from
another in those tokens alone is blocked with it. A kernel id is matched by
its name hash only, which those tokens change: a loader given the library
matches its solutions by the names of their kernels there, and the kernel
ids of a record read without the library by that hash.

The gate writes `blocklist.json`, the listed kernels and those that failed,
crashed or are unguarded, by name, kernel id and signature, into
`correctness/gate.new/`, which replaces `correctness/gate/` only when the
gate succeeds, so a gate that fails leaves the last one's blocklist in
place. The orchestrator copies `gate/blocklist.json` to the invocation's own
file in `correctness/blocklists/`, which the invocation's config snapshot
names (`correctness.blocklist`) and every stage marker of the invocation
records (file, sha256 and number of kernels), so an older round read with
its own snapshot applies the blocklist it ran with. The run pins the
contents of the `correctness.lists` files (`correctness.lists_sha256`):
changing one needs `--allow-code-change`, like any pinned value. From there
on every stage leaves the blocked kernels out (`lib/correctness.py`):

- every pool built from the config (`lib.bank_pool.config_pool`): the
  planner's kernel lists, stage05's pool and served candidates (a key whose
  served member is blocked is served by its next member, a key whose members
  are all blocked is no candidate), the deploy gate and the audit set's
  evaluations (under protocol v2 the winner is the fastest median of a
  kernel of the pool), which also leave out every kernel a blocklist the
  audit set was built with names (*Audit set*);
- stage04 drops their rows, so a blocked kernel is never a GEMM's winner or
  a label, and stage05, stage04b, stage06, stage07, stage08 and the report
  drop the rows of folders enriched before it was blocked (stage05's
  loader, which stages 4.5, 6 and 8 use too, knows a row's kernel by its
  name in every library file the folder's record names, so it drops a
  blocked bank kernel's rows also when the pool has no bank);
- stage03b's near-winner re-reads and stage03d's finalists leave them out,
  so a blocked kernel neither takes a re-read nor sets the winner a window
  counts from;
- stage06 names them in the integration-side sidecar of the model
  (*Deployment*), which stage 7 serves with the model (*Validation*).

Stages 7 and 8 validate a bundle with its own blocklist, which may be an
earlier invocation's (`--mode validate`, a resumed run): the sidecar its
deploy record names (stage 8: else the blocklist its stage05 applied, read
through the config snapshot its `metrics.json` names), since the runtime
serving it leaves out exactly those kernels. Stage 7 fails when the library
directory it runs serves another sidecar than the bundle's, and both warn
when the invocation's blocklist names other kernels of the pool; stage 7's
`summary.json` records the three (`blocklist`). Deploying the bundle again
ships the invocation's blocklist.

The cache, `verdicts.json` and the harness's records hold what each check
saw (its failing repetitions per shape); the lists and the sidecar hold names
and classes only. A script that reads a run's data (`audit_eval.py`,
`audit_v2.py`, `choose_shipped_subset.py`, also with `--bank-stem`) applies
the gate's blocklist when it is given the run's config snapshot (`<run
dir>/configs/`); given the training config it applies its lists alone and
warns. `deploy_loss.py` applies the blocklist stage05 applied (through the
snapshot its `metrics.json` names), and `check_pick_log.py` the file
`--blocklist` names, else the sidecar the runtime read (`<library
stem>.tilewright.blocklist` in `--library-dir`, next to the model), else
the config's. The report notes a blocklist it cannot read and analyzes the
whole pool.

### Deployment (stage06)

stage06 writes the trained cells and the split tree implied by their labels
into one MLREC_v2 file, or MLREC_v3 for a v3 bundle (formats in
`shared/tilewright/README.md`). The file carries the arch's model constants
(matrix-instruction cycle table, `parallel_mi_cu` and, in v2, the bandwidth
coefficients; `lib/hardware.py`), so the engine computes a model's features
with the constants it was trained with. stage06
refuses a bundle whose feature names, dimensions or catalog hash differ from
the engine's, and loads the written file back with the tilewright module
before staging it; `deploy.apply_patch` requires the module. An MLREC_v3
file says whether its training pool held the bank (as stage05 recorded it,
or for an older bundle whether the config's pool has a bank) and, with
`train.trained_key_gate`, carries the trained-key gate over the view keys
the bundle records; a bundle trained before that record ships without the
gate, with a warning.

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

The per-cell gate keeps a bundle from shipping when it regresses where the
bundle that ships is good, which a mean over cells can hide. With
`audit.dir` and once the run ships a bundle (a round's, or its final
retrain's), the orchestrator gives stage06 that bundle's directory
(`--cell-gate-baseline`), and stage06 scores its staged model on the audit
set beside the candidate's audit record, leaving out the shapes of the
rounds it trained on, and pairs them GEMM by GEMM per group, under the
official protocol (`audit.official`: the sustained composite, which only
the exclusive group has) where both records have it for the group, else
v2, else v1 (`lib.audit.cell_gate`).
Either of two rules keeps a candidate out, whatever its other gates say:

- **Cell rules**, on the groups `gate.groups` names (the exclusive group by
  default; a context's cells hold a few GEMMs each, so its cells are tested
  the same way and reported). All their cells are tested at once, as one
  family: random GEMM-wise sign flips of the paired log ratios
  (`audit.bootstrap` of them, one sign per shape) give each cell a
  Westfall-Young step-down p-value for a decline, from two statistics: the
  cell's mean log ratio (a large decline) and its studentized sum (a
  consistent one), each GEMM's change capped at a factor of two, the
  smaller p doubled. When the two bundles are equally good (each GEMM's
  two efficiencies as likely the other way round) the chance that any cell
  of the family is flagged is at most `gate.alpha`. A cell is flagged when
  its p is below alpha and
  - it is *significantly worse beyond `gate.cell_tolerance`*: its paired
    ratio is below 1 - tolerance, or
  - it *confidently fell below `gate.cell_floor`*: its geomean on the
    paired GEMMs was at or above the floor, and the 95% interval of the
    candidate's geomean there lies below it.

  The tests need at least 2 / alpha sign flips (40 at 0.05; config
  validation and stage06 refuse fewer). A gate judged with fewer says its
  cell rules were off and leaves its verdict undecided, unless the tail
  rule keeps the candidate out: stage06 then leaves the decision to its
  other gates, and the final retrain does not ship.
- **Tail rule**, over every group, contexts included, each GEMM of each
  group on its own: a new collapse is a pick of the candidate below
  `gate.tail_floor` where the shipped bundle's pick was at or above it (any
  step across the floor; `gate.tail_collapse` below 1 counts only drops to
  at most that fraction of the shipped pick's efficiency). A collapse the
  candidate heals (the shipped pick below the floor, the candidate's at
  or above it) is reported beside them and never offsets one, and so is a
  collapse both bundles share. Each new or healed pick counts only when a
  repeated measurement confirms it: under v2 or the sustained composite
  (medians of repeated benches) always, under v1 only when the
  candidate's v2 record of the group says the same, so with `audit.v2:
  off` the rule only reports. The candidate is kept out when its confirmed
  new collapses over all groups number more than `gate.tail_budget` (0).
  This rule is a policy on measured GEMMs, not a test: no test over a
  cell's few GEMMs can tell one collapse from chance, yet one collapse can
  sink a context's cell. `gate.tail_budget` loosens it,
  `gate.tail_collapse` below 1 counts only the deeper drops, and
  `gate.tail_floor: 0` turns it off. With the default
  `gate.tail_collapse: 1` a round whose pick steps across the floor on
  one audit GEMM, however slightly, is kept out. A bundle either rule
  keeps out for findings in a few base cells is repaired: a round's with
  `gate.repair` (see *Repair on refusal* below), the final retrain's with
  `full.final_retrain.repair` (see *Final retrain*).

`cell_gate.json` names every finding of every group (the cells with their
ratios, intervals and p-values; the new, healed and shared collapses with
both efficiencies and whether a repeated measurement confirms them), and
the deploy record and `gate.json` carry the verdict (`cell_gate`, with
`n_tail_new`, `n_tail_healed` and `n_tail_persisting`) with the
reason naming the first few cells and GEMMs of each kind. When the audit
set cannot measure both bundles (incomplete, or a model file missing) the
other gates decide and the record says why. `--deploy` measures and
records it without keeping a bundle out (`--cell-gate-record-only`);
`gate.per_cell: false` turns it off. A round's per-cell gate is decided
once, with its stage06; the final retrain's gate applies the same check
(see *Final retrain*).

#### Repair on refusal

With `gate.repair` (false; the training recipes set it), a round's bundle
that stage06 keeps out on the per-cell gate alone (its deploy gate passed:
`deploy_record.json` `deploy_gate`) is repaired instead of refused whole,
when every finding that decides names a base cell (a cell of `gate.groups`
significantly worse or confidently fallen below the floor, or a confirmed
new collapse in any group) and at most `gate.repair_max_cells` (3) base
cells are flagged; with more it stays out whole. The new bundle stays the
default: only the cells the gate itself flags change, never cells that
merely read lower on the audit's point estimates, since a per-cell best-of
chooses noise and loses on GEMMs it was not chosen on. Per flagged base
cell, in this order (`lib/cell_merge.py`, `scripts/repair_bundle.py`, in
`<round>/repair/`):

1. a collapse gets a tier key of its leaf's own network when the tail
   guard keeps it, as the final retrain's collapse repair does: the new
   networks stay;
2. every finding still open (a collapse no kept key lifts, a cell the cell
   rules flagged) carries the whole base-cell subtree, the cell and every
   leaf with its network, tier keys and feature statistics, from the
   incumbent: the bundle the run ships, which the gate compared with. The
   incumbent is found by content, as the bundle whose stage06 deployed the
   model the gate scored (by its sha256) and whose `stage05/models.pt` is
   the one that deploy packed (`deploy_record.json` `models_sha256`), never
   by its directory (a final retrain a later one replaced moves to
   `stale/`), and the repair refuses a `models.pt` of other content. The
   audit set must score that same model for the incumbent when it judges
   the repair (its record's `model_sha256`), or the round stays out.

A carried subtree can lack splits the round's tree holds. The repaired
bundle revokes them (`stage05/revoked_splits.json`): every split tree read
through that round (stage01, stage04b, stage05 and the final retrain, the
repair, the reports; `lib/subcells.py`) leaves them and everything under
them out, and stage04b never splits that base cell again in the run (its
leaves retrain: reason `__split_barred`, `cells_split_barred` and
`split_barred_base_cells` in `decisions.json`). Without the revocation the
next round and the final retrain would train the reverted leaves again. A
round that builds on a final retrain samples on that bundle's leaves: its
stage01 takes the bundle's directory (`--base-bundle-dir`), whose
`stage05/cells.json` gives the plan's labels and whose revocations leave
the tree before the round samples (`per_cell_plan.json` names both). The
round copies those revocations into its own `stage05/` before it deploys
(a round resumed after its stage05 too), and judges the cells it carries
forward on a tree read through that bundle.

The repaired bundle is registered with protocol v2 (with `audit.v2:
rounds`), scored on the audit set beside the incumbent
(`repair/audit.json`) and judged again: by the per-cell gate, each group
under the protocol the refusal used (while a group lacks it the round
stops, and the same command tries again), and by an all-group check: no
audit group, the exclusive context or a CU budget, significantly worse than
the incumbent beyond `gate.cell_tolerance`, family-wise at `gate.alpha`
over the groups with the gate's sign flips. A bundle that passes replaces
the round's `stage05/cells.json`, `whitelist_tier.json`, `metrics.json`,
revoked splits and, last, `models.pt` (the originals and the refused
stage06 are kept in `repair/unrepaired/`; `repair/installed.json` names the
installed, the unrepaired and the incumbent's `models.pt` by sha256 before
the copy, and an interrupted install is finished before stage06 runs). Its
carried cells take their origins from the incumbent (`cell_origins.json`),
and stage06 runs again: it measures the repaired bundle on the round's
deploy gate (re-benching its misses on the run's GPUs) and per-cell gate,
and deploys it with what the repair changed in the deploy record
(`repair`, with `selected_by_audit` and its `selection_note`; the run
summary repeats the note when it ships that round). When that stage06
keeps it out, the refusal is recorded first, then the install is withdrawn
(the unrepaired bundle and its stage06 back in place, the repaired stage06
kept in `repair/stage06.repaired`) and the round stays out whole, as it
does when the repair does not pass. An interruption anywhere in between,
or before the round recorded what a stage06 of the repaired bundle
decided, is finished by the next invocation before its stage06 runs. The
refusal that stage06 writes into its deploy record (`deployed: false` with
`not_deployed`, no `error`, `models_sha256` of the repaired bundle) is the
decision whether or not the orchestrator marked that stage06 complete (a
code change found as it ended, or a kill after it wrote the record, leaves
it unmarked), and the stage06 that measures an installed repair runs in
one attempt, without `runtime.stage_retries`: a repaired bundle a stage06
kept out is never measured again.
`repair/decision.json` records every flagged cell with its findings, what
was done there (`tier add` or `subtree carry`) and the sha256 of the
bundle whose networks now serve it, the judgement, the revoked splits, the
install, the outcome, and that the same audit GEMMs flagged and judged the
cells (`selected_by_audit`): fresh held-out audits confirm a repaired
bundle. A decision for a bundle under the same settings (`refused`,
`not repairable`, or `not repaired`: no incumbent found, no repair
possible, or the incumbent scored as another model) is not made again, so
that measuring once more cannot turn a refusal into a pass; a repair that
failed on an error (`repair failed`) is tried again by the same command,
and other settings decide again. A repair the audit set could not judge
yet (it did not measure the repaired bundle, or measured a group only under
a weaker protocol than the refusal's) stops the round with 1 (`wait` in
`decision.json`, with its `attempts`) for the same command to try again,
at most `gate.repair_max_attempts` (3) times for one bundle under the same
settings: the attempt that waits for the last time records `not repaired`
with the attempts in its reason, and the round stays out whole and the run
goes on, so that a bench that never completes cannot hold a run in a loop.
That decision stands like the others (`gate.repair_max_attempts` is not
one of the settings that decide again).

An active round builds on the latest bundle with a `models.pt`, deployed
or not, while its per-cell gate compares with the bundle the run ships. A
repaired round is deployed, so the next round's stage04b evaluates the
repaired bundle (its collapse retrains included) and its stage05 carries
the repaired networks, the round after takes it as the improvement
baseline (`--prior-models`), and the next per-cell gate and the final
retrain's gate compare with it. A round kept out whole is still the bundle
the next round builds on, while the gates compare with the bundle the run
ships.

It always stages the result under `<round>/stage06/staged_for_deploy/`
(the model, an index fragment per target and `INSTALL.md`). An MLREC_v3
model it stages records the neutral provenance of a published model
(`deploy.provenance: neutral`, the default; lib/provenance.py
`published_provenance`): the library stem it is deployed for as its
`config_id`, the bundle's `training_data_hash` and its `launch_model`, which
the engine checks. The training config's id, the source revision and the
date describe the run rather than the model and stay in `models.pt` and
`metrics.json`; `deploy.provenance: full` records them in the file too.
The audit set's evaluations write a `models.pt` with the same provenance
(lib/provenance.py `deployed_provenance`), so that they score the bytes,
and the sha256, that stage06 deploys. `deploy_record.json` names the mode
and what the file records (`provenance`). With
`deploy.apply_patch` it installs the model into the `Tilewright` directory of
every subtree `t` of `deploy.targets`, `<arch>/<t>/Tilewright/` of hipBLASLt's
library logic (`projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full`;
`<arch>` is `t`, or for the build alias `gfx1250v0` the `gfx1250-strict`
logic it is built from), and registers the library stem in that directory's
`tilewright_index`. A target whose `<arch>` logic the deploy root does not
hold is refused, so a deploy never adds an architecture to the logic tree,
and nothing a deploy writes there ends in `.yaml`: TensileCreateLibrary reads
every `.yaml` file of the logic tree. One model can serve several targets: a
model of the gfx1250-strict logic, for one, can also be installed for hipBLASLt's
`gfx1250v0` subtree, the same logic built for gfx1250, which a device of
ASIC revision 0 loads without the strict switch (devices of other revisions
load `gfx1250` and the model of a gfx1250 config). For a target whose
libraries carry another architecture suffix (those of `gfx1250v0` end in
`_gfx1250`) the stem is rewritten accordingly. The index
merge keeps every other library's entry and the rows for the devices of one
PCI chip ID (`chip=0x<id>`; see the engine README's *Weights*), whose files a
deploy never writes over or removes; it writes the stem's row without a chip
ID, first among the stem's rows, and resolves a stem listed twice the
way the engine does (the first line wins) and removes the duplicates and
lines of one field, with a warning. It refuses an index with a malformed
row (a third field that is not a chip ID, or a fourth field) and writes
nothing: the engine loads no model for that row's stem on any device, and
a rewrite without the row would serve the stem's other rows on the devices
it may have been meant for; correct or remove the row first. A stem with
rows for chip IDs only gets a warning: an engine that predates the chip ID
serves the model of its first row on every device. A re-deploy keeps the
file name the stem already uses; a file that no index entry references any
more is removed. A deploy never writes over a file a row for a chip ID
names, so a stem whose row for a chip ID names `<stem>.tilewright.bin` is
not deployed (`... is already used by another row of the index`) until
that row names a file of its own.

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

A bundle with a whitelist tier ships it in the model file;
`deploy_record.json` records it and how it meets the pool of the library it
deploys for (`whitelist_tier`, see *Whitelist tier (stage05)*).

`deploy_record.json` also says how the bundle's training data was benched
(`launch.benched`: stage05's `bench_launch`, the launch heuristics, launch
overrides and workspaces of its benches; see *Launch settings*) and, with a
build directory, that build's launch heuristic (`build_heuristic`, null
when unknown) and whether the training benches used it (`build_matches`,
null when either side is unknown), with a warning when the launch-rule
sources differ: that build may launch the picks otherwise than they were
measured.

Blocked kernels (*Correctness gate*): stage06 stages and installs, next to
the model of every target, the sidecar `<library stem>.tilewright.blocklist`:
after `#` comment lines, one kernel name per line, the kernels of the served
pool (the table and the bank) that the config's blocklist names. TensileLite
reads the sidecar of a library stem from the directory of its
`tilewright_index` and leaves those kernels out of every pool it ranks with
the model, matching names as a blocklist does (without their IA and PKA
tokens), so its pool positions and the members it serves are the
pipeline's; it warns when names of the sidecar match no kernel of a pool it
ranks. A deploy whose pool holds no blocked kernel removes the sidecar
a former deploy of the stem left, in the checkout and in the build. With
`deploy.build_after_apply` the build's copy is verified like the model's, so
the deploy fails on a hipBLASLt whose build does not copy sidecars. Every
bundle's `INSTALL.md` says which sidecar files to copy, or to remove when it
ships none. The deploy record keeps the provenance of the blocklist: under
`blocklist`, the digest of the one the deploy applied (`digest`, sha256 of
the blocked names; the same for the same kernels whatever files named them)
and the blocklists that filtered the bundle's labels (`trained_with`:
stage05's `blocklist`, the digest it applied and those its enriched folders'
stage04 applied, from their `enrich_summary.json`), with a warning when the
deploy applies another than training did; under each target, the sha256 of
its sidecar (`blocklist_sha256`).

Only the model file, its index entry and its sidecar go into the checkout.
The manifest, summary and deploy record stay in the run directory.

### Audit set

The audit set is the measure of the goal: a fixed set of GEMMs per grid
cell, benched once over every kernel the model can rank, that no round
trains on. Every round measures its models on it against the fastest
kernel of each GEMM, so the numbers do not depend on what a round's plan
chose to bench.

`--mode audit --config C --run-dir A` builds it in its own directory with
C's library, bank, bench settings and contexts (`lib.audit.bench_config`):

- stage01 draws `audit.shapes_per_cell` shapes per cell with `audit.seed`
  (the first-principles generator, initial mode), leaving out every shape of
  the `--source-run-dir` runs' rounds and every GEMM of the `audit.held_out`
  lists outside the audit directory (an entry inside it is the set's own and
  is skipped);
- stage02 plans them exhaustively: every pool and bank kernel per shape
  that the blocklist of the runs the set is built for leaves in (below),
  always in an explicit list (`--listed-only`: when the pool and bank hold
  more kernels than one bench line lists while that blocklist leaves one of
  them out, the stage fails rather than bench every kernel of the problem
  type);
- the bench runs in resumable parts of `audit.part_shapes` shapes:
  `bench/exclusive/` holds every shape, `bench/<label>/` every configured
  context, trained and held out, with `audit.shapes_per_context` shapes per
  cell through the stage03c wrapper (cotenant, CU budget, schedule,
  pool-only library). Each part runs stage03 and the stage03b near-winner
  re-reads, and is complete when it has `.stage_complete.json`; an
  interrupted invocation resumes at the first incomplete stage of the first
  incomplete part. `progress.json` has the parts and GEMMs done per context,
  the rate and the expected finish;
- as soon as a group's parts (`exclusive` or one context) are all benched,
  stage04 enriches them into `bench/<label>/stage04/` (the exclusive logs as
  `--log-dir`, a context's as `--context-log-dir`); generation 0 in
  `audit_set.json` lists these directories with the kernels benched, and
  `.audit_complete.json` marks the set complete. (Audits built before
  per-group enrichment have a single `stage04/`.)

A later run of the same command after a kernel pool change benches only the
kernels the audit has not benched, by kernel digest (name, engine config and
attributes, as the identity rule reads them; an alias group is its first
member) under any solution index, on every audit GEMM, on generation
0's bench lines (each part's `shapes.yaml`, timed adaptively), into
`increments/inc_NNN/`: a rebuild that only renumbers the pool benches
nothing, one that retunes a kernel under its own index benches that kernel.
Every generation's rows are read at the indices the current pool gives their
kernels; kernels that left the pool cost nothing, their rows are ignored.
`--devices` overrides `runtime.devices` for the audit only.

The set never benches a kernel the runs it is built for leave out of their
served pools. Each invocation takes the union of the blocklist of C (its
`correctness.lists`, and the gate blocklist a run's config snapshot names)
and, for every `--source-run-dir` run, the blocklist its newest config
snapshot whose files can be read names (its lists and the invocation's copy
of its gate's blocklist) and every sidecar its deploys staged (each bundle's
`stage06/deploy_record.json`; one that cannot be read is noted in the file);
a training run that builds its set itself gives the child its own snapshot
and run, so its set leaves out what its gate blocks. The union takes no
kernel a run's hazard canary only kept out unchecked, its run alone cut
short or never run (`lib.hazards.kept_out_unchecked`), and that no config
snapshot of the run names for a finding (`lib.audit.run_unchecked`): not
from the run's blocklist, not from a sidecar staged meanwhile, and not from
C's files when a `--source-run-dir` run shows the kernel so. Nothing was
found against such a kernel, so the set's own canary runs it alone first,
and the set benches it once it passes. The lists, the gate's findings
(failed, crashed or unguarded), the canary's quarantine and every other
keep-out are taken: the gate lists a kernel the canary keeps out with the
canary's entry alone, so a keep-out can hide what the gate found in another
invocation. The invocation adds every kernel a blocklist the set applied
before names (`lib.audit.build_blocklist`), writes the union to
`blocklists/source_<sha256 prefix>.json` and lists it as the one
`correctness.lists` entry of the set's config, so every stage applies it
with the hazard canary's quarantine: the plan lists none of its kernels, a
part benched in a later invocation leaves out of its lines what the
blocklist names then (a part whose `shapes.yaml` lists such a kernel and
whose stage03 is not complete is moved aside and benched again), an
increment holds none, and a generation records as benched only the kernels
every line of every part listed. `audit_set.json` names each blocklist an
invocation applied by its file and sha256 (`blocklists`, the latest in
`blocklist`, the one a generation was finished with in its `blocklist`) and
the kernels of the pool it left out by kernel digest (`left_out`): no
completeness check, of the set's builder or of a run that evaluates on it,
waits for those, whatever its own blocklist says, and every evaluation
leaves out of the pool it ranks and scores every kernel those blocklists
name (`lib.audit.evaluation_pool`), so that none is a winner, a ceiling or a
pick under any protocol; an evaluation record names what it applied
(`blocklist`: the run's by the digest of its names, the set's by sha256). A
kernel those blocklists name is thus never measured again for any run of the
set, also once no run's blocklist names it: a kernel unblocked later needs a
new audit set.

An audit directory is bound to its seed, counts, arch, library stem and
contexts; other settings need another directory, and a training run refuses
one built with other settings, complete or not. The contexts follow the
device: a config of CU shares resolves other budgets on another CU count or
compute partition, and turning an entry on or off adds or removes a context,
so such a run needs an audit set of its own.

A training config points at it with `audit.dir`. Then:

- stage01 of every round excludes the audit shapes (`--exclude-shapes-yaml`;
  they are neither drawn nor re-used by the top-up);
- stage05 refuses any input inside an audit directory;
- stage04b writes `audit.json` for the model entering the round, stage06 for
  the bundle it stages (its headline, and the official exclusive number
  with v2's beside it, also go into `deploy_record.json`),
  and a round that runs no stage06 evaluates its stage05 bundle into
  `stage05/audit.json` (`scripts/audit_eval.py`). The report shows them per
  round; an entering model that is the previous round's bundle (the same
  model file) has no row of its own, and that bundle's row shows the more
  complete of the two evaluations.

`audit.held_out` registers the library's other held-out lists, those a
training run must never draw although they are not its `audit.dir`: fresh
audit sets of the same generator with another seed, out-of-distribution
lists, an older audit of the library. An entry is an audit directory (any
directory holding `stage01/shapes.yaml`), a bench yaml, a file of
`- [M, N, K, B]` rows, or a CSV with `m`, `n`, `k` and `batch` (or
`batch_count`) columns (`lib/held_out.py`). A GEMM is held out by its exact
(M, N, K, batch). Then:

- stage01 of every round (initial, active and enhance) excludes the
  GEMMs of every list as it excludes the audit shapes: none is drawn or
  re-used by the top-up, and a cell takes its next draws instead, so it
  keeps its count; an active round's targeted shapes (see *Targeted shapes
  (stage01)*) are checked against the lists too;
- a list is read whole or refused: an entry line that names no GEMM, a CSV
  row without the columns or with a value that is not a positive integer,
  and a list without any GEMM fail the config check and stage01 (a list read
  short would hold out less than it names). Two entries that name one file,
  and an entry naming `audit.dir`'s own list (which every round leaves out
  already), are errors;
- every round's `stage01/categorization.json` records each list in
  `exclusion_sources`: the entry, the file read and its sha256, the number
  of GEMMs and the digest of their sorted (M, N, K, batch) set
  (`gemms_sha256`, the same for one set of GEMMs in any format), how many
  of its GEMMs the draws of the leaves' targets hit and left out
  (`n_dropped`; the targeted and very-large-K draws leave them out too,
  uncounted), and how many of its GEMMs the round's history holds
  (`n_in_history`, with a warning): GEMMs earlier rounds drew before the
  list was registered, which are training data of those rounds;
- every invocation that trains (`full`, `initial`, `active`, `enhance`)
  warns once per list that shares GEMMs with shapes the run's rounds drew,
  naming the rounds: they stay in the training data, also of the final
  retrain, and a measurement on the list must leave them out;
- `--mode audit` leaves the lists out of the audit set it builds too, so
  that a fresh audit set is disjoint from the earlier ones.

The lists are pinned when set, with the digest of their GEMMs
(`audit.held_out_sha256`): adding, removing or changing a list of a run
needs `--allow-code-change`, which records the change in `pin.json`. A run
without the key keeps the pin it had before it existed. An audit directory
does not pin them; its stage01 records what it left out.

No shape list of the registered never-train sets, or of the GEMM corpus
they are drawn from, trains, tunes or validates a model
(`lib/external_sets.py`): they may overlap the sets the models are
validated on. A list is recognised whole by its
fingerprint, the sha256 of its distinct (M, N, K, batch) sorted and written
one `m,n,k,batch` line each, joined by newlines, whatever its format, order
or repeats; a CSV also by those sets' columns (`legacy_category`,
`refined_category`, or a `source` column holding `log_sample` or
`model`). The loaders refuse such a list: held-out lists, bench yamls read
as shapes (rounds, histories, audit sets), stage 7's bench yamls, and stage
8's datasets (a bench yaml before it is benched, every dataset once its
GEMMs are read); preflight names the config's lists that are one
(`audit.held_out`, `stage07.bench_yamls`, `stage08.datasets`). A part of
such a set, or a list that mixes it with other GEMMs, has another
fingerprint and is not recognised.

A fresh audit set of a library (built after the training rounds it must not
contain, for a measurement the existing audit set did not select on): run
`--mode audit` into a new directory with a copy of the training config
whose `audit.seed` is a seed no round or audit used, whose `audit.dir` is
the new directory, and whose `audit.held_out` names the earlier audit sets
and lists of the library, with `--source-run-dir` naming every run whose
rounds it must not contain. Then add the new set to `audit.held_out` of
every later training config of the library, so that no later round draws
it.

An evaluation runs the engine with the deployed bytes, the pool the runtime
ranks (library and bank) and `validate.pick_measurement`, and drops audit
shapes found in rounds the model trained on (counted in
`n_excluded_trained`). Per context (`exclusive` and every context label) it
reports, per base cell (split cells merged): the geometric-mean selection
efficiency with a bootstrap interval, the ceilings (the best kernel the
model's serving rules allow over library and bank; the library alone with
and without the rules; the served ceiling, the best key as the runtime
serves it, that is grouped as the model's engine groups the pool and costed
by its first kernel in pool order with a row; every kernel costed as a pick
is) and the class: *passing* (the selection efficiency reaches
`audit.target`), else *pool-limited* (the rules' ceiling is below it too),
*identity-limited* (the served ceiling is: no model of this kernel identity
reaches the target) or *model-limited*, with whether the intervals settle
it. Per-GEMM efficiencies and ceilings
are capped at 1: a pick costed by its warm-up call can come in under the
timed winner. Over the
cells: the equal-weight mean of the cell geomeans (interval resampling
within every cell), the GEMM-weighted geomean, the minimum cell, the cells
below the target and the class counts, and every GEMM's efficiency
(`per_gemm`) for paired comparisons.

A complete audit set covering the current pool gives `status` `ok`. While
generation 0 is benched an evaluation is `partial`: it scores the groups
whose parts are all benched and enriched and gives every other group null,
with its state in `group_status` (`benching` with the parts done, or
`not_enriched`); `status_text` says which (`partial: exclusive complete, 5
contexts benching`), in `audit.json`, `deploy_record.json`, the CLI's output
and the report. A group with a missing part is never scored. Without any
such group, while a kernel increment is benched, or when the plan misses a
kernel of the current pool, the record is null (`status` `missing`,
`incomplete` or `error`, every number null); there is never a fallback.

`lib.audit.compare` pairs two evaluations GEMM by GEMM on one group, only
when both scored it: per cell the geomean ratio after / before with a
bootstrap interval (a cell *improved* or *regressed* when it excludes 1),
and the equal-weight and GEMM-weighted ratios with intervals. The report's
headline compares the described bundle with the previous candidate this
way on the exclusive group, and its audit section compares each candidate
with the one before on every group both scored. `lib.audit.cell_gate`
judges the cells of such a pairing against the bundle that ships (the
per-cell gate, see *Deployment (stage06)*).

```bash
python scripts/audit_eval.py --audit-dir A --config R/configs/<snapshot>.yaml \
    --round-dir R/round_0 --round-dir R/round_1 --out scores.json \
    [--enrich-dir E]
```

scores the candidates of finished rounds (the stage06 staged bundle, else
the stage05 models; `model_source` says which; a `final_retrain`
directory leaves out the rounds its `final_retrain.json` names), or
`--model-file` files with `--trained-round-dir`, and with several models
adds `changes`, each against the one before. `--enrich-dir` enriches, under E and never in the
audit directory, the benched groups the audit builder has not enriched yet
(an audit built before per-group enrichment), and scores them too; E is
reused while the group's logs are unchanged.

#### Protocol v2

Everything above is protocol *v1*: each kernel's single reading in the
exhaustive pass, and every v1 number says so (`protocol`, `protocol_text`
"v1 single reading"). One reading lets a host glitch on a launch-bound
kernel decide a winner, and scores a pick with one reading against a
re-benched winner. Protocol *v2* (`lib/audit_v2.py`, data in `A/v2/`,
never touching the audit's own files) measures each audit GEMM's set again:

- the set of a GEMM in a group is its kernels within `set_tolerance` of the
  exhaustive winner, or within `set_tolerance_us` of it when that is wider
  (at most `set_max`, fastest first), plus the top-1 pick of every
  registered model (`v2/models/<name>.json`, keyed by sha256). A set that
  misses the true best kernel scores every pick against a slower winner, so
  on a device whose single readings carry an error that does not scale with
  the latency, which a relative tolerance misses on a short winner, widen it
  with the absolute floor, or with a wider `set_tolerance` and a larger
  `set_max`;
- the sets are benched jointly in parts (the audit's units): every part is
  benched `repetitions` times, each a separate stage03 process on one GPU
  over the part's GEMMs in a shuffled order, each GEMM's whole set in a
  seeded shuffle of its own per repetition (`kernel_order: shuffled`) led
  by a sacrificial copy of its first kernel (`sacrificial_kernel`),
  adaptive timing, no skip screen, contexts through stage03c's co-tenant
  wrapper, and every process led by stage03's warm-up line over a kernel of
  each library file of the bench setup (`warmup_line`, see *Bench timing*).
  A long line runs its first kernels before the device settles at its
  power limit and its last ones after, so a set in a fixed order, its picks
  appended, would time the picks of a power-capped GEMM at a lower clock
  than its near-winners. Each part's `result.json` records the kernel order
  of every repetition of every GEMM (`orders`) and the GEMM order of every
  repetition (`gemm_order`). A measurement planned before these settings
  keeps benching as it did, each set as planned rotated by the repetition
  without a copy or a warm-up line, so that its generations stay
  comparable; to measure an old audit set the new way, move its `v2/` aside
  (the next registration plans generation 0 again);
- a canary (the least-work exclusive GEMM and its winner, `v2.json`) is
  timed at the start and end of every part bench against the GPU's
  reference median (`v2/canary/reference_<gpu>.json`); a reading off by
  more than `canary_tolerance`, and, with `canary_tolerance_us` above 0, by
  more than that many microseconds (`audit.canary_tolerance_us`, which the
  orchestrator gives a first v2 and sustained plan), benches the part again
  after `canary_backoff_s`, up to `canary_attempts`, then flags it. The
  readings are kept in each part's `result.json`;
- scores use per-kernel medians: the winner is the smallest median of a
  kernel of the evaluated run's pool, as in protocol v1 (`lib.audit`
  `pool_medians`): a kernel the run's blocklist or a blocklist the audit
  set was built with leaves out, or one of a library file the run does not
  read (a bank an audit set measured for a run without one), takes no part,
  which can change the winner of a GEMM against what the set measured; a
  pick's efficiency is winner / pick median. A GEMM whose spread (max - min
  over the median) exceeds `spread_limit` for the winner or a pick is
  benched again once, then flagged;
- no set holds and no bench times a blocked kernel: `plan` writes the union
  of the run's blocklist (its config's) and the blocklists the audit set was
  built with into `v2/blocklists/` (`v2/sustained/blocklists/`), names it by
  its sha256 in the manifest (`blocklist`) and in the plan of the
  generation it plans, keeps its kernels out of every set and pick (a set
  an earlier generation planned drops them), and refuses a measurement
  whose canary it blocks; a worker leaves out of a part's lines whatever the
  manifest's latest blocklist blocks, which the part's `result.json` names
  (`left_out`, `blocklist`), and `bench` refuses a blocklist file that no
  longer holds what the sha256 the manifest records says;
- a model whose picks are missing from some sets is an increment: a new
  generation (`gen_NNN/`) benches those GEMMs' whole sets (candidates and
  every pick) the same way, and a GEMM is scored from its latest
  generation;
- a GEMM whose latest generation benched the pick but kept fewer than
  `min_readings` readings of it (its line died in too many repetitions, as
  hipblaslt-bench does on some block-scaled problems) is unmeasured: no
  increment plans a kernel its set holds, so it is not waited for either,
  but counted in the group's `n_unmeasured` and the record's status line,
  and left out of the scores;
- a line that hangs (see *Shared hosts*) is benched once more in its
  repetition's retry pass, then quarantined in the run under `audit_v2`
  (`audit_sustained` for the sustained measurement): its GEMM is dropped
  from the part (`dropped` in `result.json`: why, the attempt and the
  repetition; the part's `n_dropped`; `status` counts them), and later
  parts of the run leave it out. A line whose bench leaves its GPU wedged
  (*Wedged GPUs*) is quarantined so at once, and its GEMM dropped as
  `wedged`. A dropped GEMM keeps the measurement of an earlier generation,
  which still scores the picks it holds; a pick it lacks is not waited
  for: the GEMM is counted as dropped in the group's `n_dropped` and the
  record's status line, and left out of the scores;
- a stage03 run that fails without a bench (no block log, and no line hung,
  left out or wedged: its GPU could not bench, such as one the run holds
  alone, which these runs leave out at once rather than wait for it,
  `--leave-held-gpus`), or that stopped for a wedge (exit 3: a GPU it
  benched on, or the host, is now held as wedged), measures nothing: its
  part is not written and its claim goes back to the other workers. A
  worker takes no part while the run holds its GPU (*Shared hosts*),
  waiting while parts are left, at most `bench.health.max_wait_s`; one
  whose GPU could not bench for another reason stops with exit 1. `run`
  exits 1 when GEMM benches are left once its workers end, and says how
  many;
- plans, model registries and the canary record their kernel identities.
  Those written before identities were recorded are followed by the audit
  set's kernel digests. The code that wrote them planned only while each
  solution index of the pool named the kernel the latest generation
  benching that index had recorded under it, so under an index such a
  record meant the digest recorded there by the latest generation finished
  when it was planned; its medians are taken to come from that build,
  which nothing checked then. Scoring reads them at their kernels' current
  indices and leaves out those of kernels that left the pool. A GEMM with a
  median of a kernel the digests cannot place (they do not tell which
  kernel its index named, or the pool holds that kernel under several
  indices) is not scored. `plan` and the bench refuse such records once
  their indices name other kernels; the bench takes the library a manifest
  without `bench.library` benches with from its config, as `plan` does,
  and refuses to bench without one.

```bash
python scripts/audit_v2.py run --audit-dir A --config C --gpus 1,2 \
    [--round-dir R/round_N ...] [--model NAME=FILE ...]
python scripts/audit_v2.py status --audit-dir A
```

`run` plans (generation 0 on the first call, then an increment when a
model needs one) and starts one worker per GPU under `scripts/gpu_run.py`
leases (`v2/worker_gpu<N>.log`); workers claim parts, so a stopped run
resumes and several runs share the work. The settings flags apply to the
first plan only; the set width (`--set-tolerance`, `--set-tolerance-us`,
`--set-max`) is the config's `audit.v2_set_tolerance`,
`audit.v2_set_tolerance_us` and `audit.v2_set_max` unless a flag gives it,
and the absolute floor of the sustained composite's sets
(`--sustained-set-tolerance-us`) the config's `audit.v2_set_tolerance_us`
too. A setting asked for that differs from the one the measurement was
planned with is not applied, and `plan` and `run` say so; to measure with
it, move `v2/` (or `v2/sustained/`) aside. `v2/progress.json` tracks the
parts done.

`run --devices D1,D2` names the GPUs by device index in its own environment,
as `runtime.devices` does, instead of `--gpus`: it leases their GPUs itself
(or works under the lease an ancestor such as the orchestrator holds) and
gives each worker its device's GPU alone. `--name-with-sha` ends every
registered name in the first 12 hex digits of the model file's sha256, so
that a model trained again never clashes with the one registered before.

`audit_eval.py` scores both protocols by default (`--protocol v1` for v1
only, `--v2-dir` for another v2 directory) and `--v2-bench GPUS` benches
the increment its models need before scoring. A record's `v2` holds the v2
evaluation in v1's form (per cell means against the v2 winner, intervals,
classes, `per_gemm`) with its own `status`: `ok`, or `partial` with
`group_status` per group `benching` (parts not done), `needs_increment`
(a set lacks the model's pick) or `needs_new_measurement` (a measurement
written before kernel identities were recorded holds a median the audit
set's digests cannot place), with counts of spread- and canary-flagged
GEMMs. The CLI adds `changes_v2`, the stages' summaries add `audit v2`
lines, and the report shows v2 wherever it is complete for the model, else
v1, compares under v2 when both records have it for the group
(`lib.audit.paired_change`) and names the protocol of every number; with
the sustained composite (below) complete for the model, the official
protocol comes first (`lib.audit.view`).

#### The sustained composite: the official exclusive number

A long or compute-bound GEMM run back to back holds the device at its
power limit, and the limit slows some kernels more than others. Protocol
v2's windows of a few milliseconds end before the clock settles, so for
such a GEMM they compare kernels at clocks it does not run at for long. The
*sustained composite* (a record's `sustained`) scores every power-capped
exclusive GEMM on a measurement in which each kernel first runs alone long
enough to reach the limit, and every other GEMM as v2 does; its context
groups are v2's. `audit.official` (`sustained` by default, or `v2`) names
the protocol of the official exclusive number: the report's headline, the
run summary, `decision.json`, the stages' summaries and the deploy record
give it with v2's number alongside, and the final retrain's ship decision
(`lib.audit.not_worse`) and the per-cell gate (`lib.audit.cell_gate`) take
it for the exclusive group whenever both records have it, else v2, else v1
(`lib.audit.preference`), naming the protocol they used.

The measurement (`lib/audit_sustained.py`, in `A/v2/sustained/`, in v2's
format with `regime: sustained`):

- power-capped GEMMs, by their v2 winner (the smallest v2 median of a pool
  kernel): *compute-bound* when it takes at least `compute_bound_min_us`
  (100) at a rate of at least `compute_bound_tflops`; else *very long* from
  `very_long_us` (1000) on; else *long* from `min_us` (100) on. Unless
  given, `compute_bound_tflops` is `compute_bound_fraction` (0.3) of the
  bench device's peak dense matrix rate for the config's data types, read
  from the device when the measurement is first planned
  (`lib/hardware.py` `roofline_rates`; without a peak no GEMM is
  compute-bound) and recorded with where it came from (`compute_bound`).
  The latency bands were set on one device: check them on a new one, where
  the slowdown of long windows against short ones shows which GEMMs the
  limit binds;
- the set of a power-capped GEMM: its kernels whose v2 median is within
  `set_tolerance` (0.08) of the winner's, at most `set_max` (10), fastest
  first (a compute-bound GEMM's within `compute_bound_set_tolerance` (0.15),
  at most `compute_bound_set_max` (12)), or within `set_tolerance_us` (0,
  or the config's `audit.v2_set_tolerance_us` when it sets one) of it when
  that is wider, then every registered model's pick that the exhaustive
  pass benched on it;
- the bench: parts of at most `part_gemms` (24) GEMMs, benched one after
  another so that the device stays heat-soaked, each `repetitions` (3)
  times in separate stage03 processes, the GEMMs and each set shuffled per
  repetition (no sacrificial copy: every kernel soaks first), each line its
  v2 line with `warmup_time` (250), `measure_time` (300) and
  `max_measure_time` (400) ms; the v2 canary before and after, medians of at
  least `min_readings` (2) readings and the spread re-bench, as in v2;
- increments: each plan adds the GEMMs that the current v2 measurement
  makes power-capped and no generation planned, and benches again, whole,
  every set that lacks a registered pick; a GEMM once planned stays
  power-capped.

Scores: a power-capped GEMM's winner is the fastest sustained median of a
kernel of the pool and its pick's efficiency the winner over the pick's
median. For its ceilings a kernel outside its set is costed at its v2
median, else its exhaustive reading, times the GEMM's median ratio of
sustained to v2 medians; a GEMM where such a cost beats the measured winner
is counted (`n_imputed_faster`: its set may miss the sustained winner). A
power-capped GEMM without a sustained median of the pick (being benched, or
not planned yet) makes the exclusive group null with the counts
(`group_status`: `n_power_capped`, `strata`, `n_unplanned`, ...), as a
missing v2 measurement does; a measurement that found no power-capped GEMM
scores every GEMM as v2.

`scripts/audit_v2.py run` benches the composite's increment after v2's
(`--no-sustained` leaves it alone; the `--sustained-<setting>` flags apply
to its first plan; `bench --sustained` benches its parts; `status` prints
both). `audit_eval.py` writes it in each record's `sustained`, with
`changes_sustained`, and the protocol of the official number in `official`
(the config's `audit.official`, unless `--official` gives another).

Which regime counts is a deployment choice. The composite fits back-to-back
heavy work (training steps, saturated batched serving), where the device
sits at its power limit; short windows fit an isolated heavy call after
light work, and only when every kernel starts from the same state, which is
why v2 shuffles each set per repetition. Where a GEMM never reaches the
limit the two agree. The long windows assume the device settles within the
warm-up: check that before taking them to a device whose clock settles
later. Each bench's stage03 records the device's settings
(`platform.json`); a partitioned device shares its power limit with its
other partitions, which moves these numbers.

#### The audit set in the one command

A training run whose config sets `audit.dir` builds the audit set itself
when the set is not complete for the current kernel pool (missing, still
benching, or lacking new kernels), as `--mode audit` of its config snapshot
in a child orchestrator: it holds the audit directory's lock, leaves out
every shape of the run's rounds (the run is its `--source-run-dir`) and
every kernel the run's blocklist and sidecars leave out of its pool,
benches on `audit.devices` (default `runtime.devices`) in its resumable
parts, and writes its own log and snapshot into the audit directory (the
run keeps a copy of its output in `audit_build.log`). `audit.build` says
when:

- `before` (default): before the invocation's first round. A failure stops
  the run with its exit code; the same command resumes the build, unless
  the build refused to go on with a plan of another hipBLASLt build, which
  the message then says, or with a value of a key the set's benches read
  other than the one its `pin.json` holds: the message names the change, and
  the same command with `--allow-code-change`, which the build gets too,
  extends the set with the run's value and records it there.
- `parallel`: beside the rounds, on `audit.devices`, which must not be
  `runtime.devices`. The run waits only for the audit shapes (its stage01),
  which every round leaves out, and joins the build at the end of the run
  (an invocation that is not a full run joins it before it exits; an
  interrupted or failed one stops it, and the same command resumes it).
  Until the set is complete the rounds' audit records are partial and their
  protocol v2 waits.
- `off`: nothing is built; evaluations stay null or partial until
  `--mode audit` builds the set, and `--mode full` refuses an `audit.dir`
  that holds no audit set at all. A final retrain gated on the audit set
  waits for it at the end of the run (preflight warns).

Like a stage, the build's child orchestrator gets SIGTERM when the run's
orchestrator dies, also by SIGKILL, and carries an attempt token that
`<run>/.running.json` names while it runs (`lib/attempts.py`), so that the
next invocation stops whatever is left of it before preflight checks the
audit directory's lock.

Before anything runs, preflight refuses an audit set built with other
bound settings or contexts, complete or not and whatever `audit.build`
says, and the run checks it again when it starts, also with
`--skip-preflight`: its models would be scored on GEMMs benched in other
contexts. Preflight also refuses a set it could not build: an audit
directory inside the run directory, holding rounds or inside another
training run, one whose first generation or next increment was planned
with another hipBLASLt build, one another process is building, or a bench,
library, device, context, planner or bank problem of the audit's own config
(`lib.audit.bench_config`).

`audit.v2` measures the run's bundles under protocol v2 with
`scripts/audit_v2.py run --devices` on the run's devices, under their
leases, each registered as `<run>.<round>.<sha256 prefix>`, and with
`audit.sustained` (true by default) the sustained composite's increment
right after; `audit.official` (`sustained`, or `v2`; `sustained` needs
`audit.sustained`) names the protocol of the official exclusive number,
under which the per-cell gate and the final retrain decide:

- `rounds` (default): each round's bundle (its stage05 `models.pt`) right
  before stage06, or the stage05 audit evaluation of a round that does not
  deploy, scores it, so that its `audit.json` carries v2. The first
  registration benches generation 0, every audit GEMM; later ones only the
  GEMMs whose sets lack the new picks. A failure is reported and the run
  goes on.
- `final`: only the final round's bundle and the shipped one, at the end
  of the run.
- `off`: none (v1 only).

`audit.v2_set_tolerance`, `audit.v2_set_tolerance_us` and
`audit.v2_set_max` give the width of the v2 sets (`set_tolerance`,
`set_tolerance_us`, `set_max` above) to the measurement's first plan, and
`audit.v2_set_tolerance_us` the absolute floor of the sustained composite's
sets to its first plan (an error of single readings that does not scale
with the latency is the device's; the sustained sets keep their own
relative widths). A measurement keeps the width it was planned with, which
every later generation shares, so set them before the first registration
on a new audit set; a run whose config asks for another width than its
measurement's says so in `audit_v2.log`. Changing them needs no
`--allow-code-change`.

At the end of a full run, once every planned round is complete and the
audit set is, the run benches protocol v2 for the bundles that still lack it
(a parallel build, a failure, or `final`; a shipped final retrain whose v2
bench failed when it was decided, unless `audit.v2` is off) and scores again
every bundle whose own audit record is not complete for its protocols (v2
and, with `audit.sustained`, the sustained composite), into
`audit_results/<round>.json` (`audit_results/final_retrain.json`, leaving
out the shapes of the rounds it trained on); the report reads those, and the
final retrain's records, from the run directory as it reads
`--audit-results`, and shows the more complete record of each bundle, as do
the run summary and `held_out_sel_eff_breakdown.json`. A run directory
written before the one command measured under v2 gets this on its next
`--mode full` with `audit.dir` set: with the defaults it builds or extends
the audit set when the pool changed, benches v2 for every round (their
records predate it) and scores them again, which is GPU work; set
`audit.v2: off` (and `audit.build: off`) to keep it from doing so. One
whose records predate the sustained composite gets that the same way, its
bundles' increments benched at the end of its next full run;
`audit.sustained: false` with `audit.official: v2` keeps it from doing so.
A run that stops early (`deploy.stop_when_target_reached`) leaves this, and
the final retrain, to the invocation that completes the planned rounds, and
says so: set `full.total_rounds` to the complete rounds to end it there.

### Final retrain

A round trains the cells stage04b lists and carries the others forward, so
the final round's bundle holds cells trained on the data of different
rounds. With `full.final_retrain.enabled`, a full run whose planned rounds
are all complete retrains every cell once more, in `final_retrain/`:

- stage05 trains on every round's enriched data (`round_*/stage04`: the
  seed benches and the trained contexts, the final round's gate slice too)
  with the final options: the config's train section, with the overrides
  of `full.final_retrain.recipe`, whose other keys are stage05 options
  (`--key-with-dashes`; `true` gives a flag, `false` or null leaves it out,
  a list repeats it). The recipe is the hook for training recipes stage05
  gains (for example a pooled pretrain followed by per-leaf fine-tuning, or
  a targeted guard on pseudo-negatives); the config is refused when a key
  is neither a train key nor an option of this stage05, or is one the
  orchestrator sets (the run's data, rounds, library and device, the cells,
  the catalog and attribute spec), or sets the trained-key gate
  (`trained_key_gate`, `no_trained_key_gate`): stage06 deploys every bundle
  with the run's, and the gate must score the bytes stage06 deploys. The
  train section with the overrides goes through the train section's own
  checks (pseudo-negatives against their sample count and the catalog,
  `pooled_ft` against `boundary_margin`, `qat_frac` and a `validation_frac`
  of 0, ...), so a recipe stage05 would refuse is an error, and one whose
  options it would ignore a warning, named `full.final_retrain.recipe`,
  when the run starts. Every round's split tree and plans apply, and a cell
  with too few GEMMs keeps the final round's model.
- protocol v2 registers it (unless `audit.v2` is off), and
  `scripts/audit_eval.py` scores it and its baseline on the audit set
  (`final_retrain/audit.json`, leaving out the shapes of every round): the
  bundle the run ships when it is decided, as stage06 deployed it (a
  round's, or an earlier final retrain's, leaving out the shapes of the
  rounds that one trained on), or with nothing deployed the final round's
  bundle, staged or not.
- with `gate: not-worse` (default; needs `audit.dir`) it ships only when
  its equal-weight mean over cells of the paired per-cell ratios to the
  baseline, on the exclusive group's GEMMs both scored, is at least 1
  (`lib.audit.not_worse`, under `audit.official` when both records have
  it for the exclusive group, else v2, else v1; the reason says when the
  official protocol has not measured both) and, when the run ships that
  baseline, the per-cell gate finds
  no cell or GEMM that regresses against it (`lib.audit.cell_gate`, the
  `gate` section; see *Deployment (stage06)*): a final retrain better on
  the mean that adds a confirmed collapse in any group does not ship as
  it is (the repair below may fix it). `decision.json` keeps the gate's findings
  (`cell_gate`) and settings (`cell_gate_settings`) when it ran, and the
  official protocol it decided with (`official`, the one used in
  `protocol`, and v2's numbers beside the composite's in `final.v2` and
  `baseline.v2`); a decision whose per-cell gate ran with other settings
  than the config's, or that compared the two bundles under another
  `audit.official`, is made again. One the per-cell gate took no part in
  (nothing shipped, `gate: none`, `gate.per_cell: false`, or made before
  it existed) stands whatever the gate's settings, and one that compared
  nothing, or was made before the official protocol was recorded,
  whatever `audit.official` says. With `gate: none` it ships without a
  comparison.
  `final_retrain/decision.json` records the decision, both records' cell
  means and the paired change; while the audit set cannot measure both
  models (it is not complete) nothing is decided, the run exits 1, and the
  same command decides once it can. The audit set then chose which of the
  two bundles ships, so its numbers for the one that ships are selected on
  the same GEMMs and carry a small optimistic bias: `decision.json`
  (`selected_by_audit`, `selection_note`), the run summary and the report's
  audit headline say so.
- with `full.final_retrain.repair` (true), a final retrain not worse on the
  mean that the per-cell gate keeps out for findings in at most
  `gate.repair_max_cells` base cells (new collapses, cells significantly
  worse or confidently fallen below the floor) is repaired instead of
  discarded (`scripts/repair_bundle.py`, `lib/collapse_repair.py`,
  `lib/cell_merge.py`, in `final_retrain/repair/`), as *Repair on refusal*
  describes for a round. Each confirmed new collapse is localized to the
  leaf whose network served it. When that network, ranking every key it
  serves without the tier, picks another kernel outside the leaf's tier
  that reads at or above the floor there, its view key joins the leaf's
  tier, kept only when the tail guard keeps it on the leaf's own data (no
  new pick below the floor on the leaf's training or held-out GEMMs, its
  held-out picks' geomean kept; see *Whitelist tier (stage05)*) and it
  lifts the collapse. A collapse no kept key lifts (the network itself
  ranks the collapsing kernel first, or the guard refuses the key), and a
  cell the cell rules flagged, sends its base cell, the cell and every leaf
  under it, back to the bundle that ships, found by the sha256 of the model
  its gate scored: their networks, tier keys and feature statistics replace
  the final retrain's, so its picks there are that bundle's, and the splits
  that subtree lacks are revoked (`stage05/revoked_splits.json`). The
  repaired bundle is registered with protocol v2 (unless `audit.v2` is
  off), scored on the audit set (`repair/audit.json`) and judged again by
  the same two rules and the all-group check against the same baseline,
  each group under the protocol the refusal used: while a group of it lacks
  that protocol (its picks wait for their v2 bench) nothing is decided and
  the same command tries again. When all pass, the repaired bundle replaces
  the final retrain's `stage05/cells.json`, `whitelist_tier.json`,
  `metrics.json`, revoked splits and, last, `models.pt`, and its audit
  record (the originals, with
  `cell_origins.json`, are kept in `repair/unrepaired/`), the cells it
  carried take their origins from the bundle that ships
  (`cell_origins.json`), and it ships. The repaired `cells.json` keeps the
  entries of the networks the final retrain trained that the bundle still
  serves, each counting the tier keys it is served with
  (`n_whitelist_tier_keys`; `n_repair_tier_keys` of them added by the
  repair, after stage05 measured the cell), and none for a carried cell,
  even one the bundle that ships holds under the same label: its origin
  names the bundle that trained it, so reports, model-option checks and
  the next round's plan read that bundle's records. `metrics.json` keeps
  what stage05 trained and measured, with the repaired bundle's tier
  record, and both files say what the repair changed (`repair`).
  `decision.json` records every collapse, every flagged cell, what was
  found and done, the picks after, the sha256 of the bundles the networks
  come from and the revoked splits (`repair`), and the refused candidate's
  change and per-cell gate (`unrepaired`); its `selection_note` says that
  the same audit GEMMs flagged its cells and verified the repair, so a
  fresh audit set must confirm the repaired bundle's numbers. The repair
  carries from, and the audit set scores, the bundle found by that sha256
  wherever it is (a model scored for it of other content leaves the final
  retrain out). A repair that does not pass, or
  finds nothing to repair, leaves the final retrain out as before, with
  why. `repair/installed.json` names the installed bundle, the unrepaired
  one and the bundle the cells came from by their sha256 and is written
  before the copy, which replaces `models.pt` last: until
  `stage05/models.pt` is the repaired one the install is unfinished, and
  the next decision finishes it before it scores; a decision made again
  after an interruption still records the repair. The audit only detects
  and verifies: every added key is the leaf's own network's choice, kept
  on its own data, and every carried network is one the run shipped; the
  repaired bundle's audit numbers come from the GEMMs that found its
  collapses and carry the same small optimistic bias as the decision
  itself.
- a final retrain that ships is deployed by stage06 into
  `final_retrain/stage06` (installed with `deploy.apply_patch`, its audit
  record leaving out every round's shapes) and is from then on the run's
  bundle (`lib/run_bundles.py`): stages 7 and 8 validate it, `--mode
  validate` included; the run summary and
  `held_out_sel_eff_breakdown.json` name it as `shipped`, measured on the
  audit set (its stage06 `audit.json`, under the official protocol when
  complete, else v2, with v2's numbers beside the composite's), with
  `final_retrain` saying whether and why it ships; the report describes it
  (headline, leaf cells, *Where to improve*, bench analysis, with its own
  split tree and every GEMM of its rounds, gate slices included, as its
  training benches, so that only later rounds are held out); and a later
  final round's deploy gate takes it as the bundle to be not worse than. A
  later round that deploys (an enhance round, say) replaces it, and so does
  a round it trained on that runs again.
- a round that runs while it is the run's bundle (an enhance round, or an
  active round after `full.total_rounds` was raised) builds on it, not on
  the final round's bundle: stage04b measures its models on the round's new
  GEMMs (with the final round's bundle as the one before it,
  `--prior-models`), stage05 carries every cell it does not retrain from it
  (the final retrain is its last `--prior-round-dir`), the planner ranks
  with it, and a final round's deploy gate compares the round's bundle with
  it. A cell of the final retrain stays as it is while the final retrain's
  own options hold, also when the recipe trained it with other
  model-shaping options than the round's, so the cells a gate on chosen
  cells does not measure are the final retrain's. The round after that one
  builds on that round, as usual.
- a final retrain whose stage05 fails stops before its gate: the run still
  joins a parallel audit set build and completes the rounds' audit records,
  exits with the stage's status, and the same command trains it again.

`final_retrain/final_retrain.json` records the rounds it trained on and its
stage05 options but those that only place its inputs, outputs and work:
when the rounds change (a later round, a round run again), when those
options change (the optimizer's budget and schedule too, which carried
cells of a round never retrain for), or when its model-shaping options do
(`model_options.json`), the next full run trains it again; a changed gate
decides again. A round counts as changed when it completed at another time
than its marker recorded then, as the readers of the run's bundles judge
it, so a copy of the run directory, mtimes or not, keeps its final
retrain. An option the record does not name predates it: it is compared at
its value from before it was an option (stage05's `PREDATING_OPTIONS`),
else at its default, and an option stage05 no longer has is not compared.
The record names the pseudo-negatives' options as stage05 resolves them and
the draw only where `unmeasured_samples` does not imply it, as
`model_options.json` does, so a record from before `--pseudo-negatives`
compares equal while the draw is the same.

A final retrain that is no bundle of the run (rejected, undecided, not
deployed, or a round it trained on ran again) moves to `stale/` and the new
one trains in its place. One that is a bundle of the run stays in place
while the new one trains beside it, in `final_retrain_next/`, gated against
the bundle the run ships (the final retrain in `final_retrain/`, unless a
later round deployed after it), so that a recipe change, another training
budget, or rounds the final retrain did not train on (`full.total_rounds`
raised, an enhance round its gate kept out) never ship a final retrain
the audit set finds worse than the bundle that ships. The run ships what
it shipped before, as the deploy checkout holds it, while the new one is
undecided, rejected, fails to train or fails in stage06; once the gate
accepts the new one and its stage06 deployed it (installed with
`deploy.apply_patch`), the new one takes `final_retrain/`'s place, the
paths its records hold rewritten to name it there, and the old one moves
to `stale/` (an invocation interrupted in between finishes this when the
next one starts). The same
command after a rejection does nothing more: the run summary,
`held_out_sel_eff_breakdown.json` (`final_retrain.candidate`) and the
report's deploy gates show the new one's decision. When a changed gate
keeps out a final retrain whose stage06 installed it into the deploy
checkout, or a round it trained on ran again after it was installed, the
run warns that the checkout still holds its weights and names the bundle
that ships now. `run_status.json` reports the final retrain on the run's
rounds (`final_retrain`: `pending`, `undecided`, `accepted`, `rejected` or
`shipped`, the reason and its rounds; while a new one is beside the final
retrain it would replace, the new one's, with `replaces` naming that one),
and a run with it enabled is `complete` only once it is decided and, when
it ships, deployed.

A round's stage05, and a final retrain's, carries the cells it does not
train from the bundle before it. `stage05/cell_origins.json` records, for
each of them, the bundle that trained it (its name and the completion of
its stage05), the cell's cells.json entry there and the model-shaping
options it trained with. The option rule (a carried cell trained with
other model-shaping options is retrained; one a final retrain trained,
while a final retrain's options hold) and the report's leaf cells follow
it: to that bundle's own records while it is in the run with the same
training, else to what the record kept, so that a bundle still tells them
once the bundle that trained a cell has left the run or trained again (a
final retrain replaced, a round run again).

### Validation (stages 7 and 8)

Stages 7 and 8 validate the latest bundle the run deployed: a round's that
its gate let through, or the final retrain (see *Final retrain*).

Stage 7 needs a model deployed by this run and a hipBLASLt build that
serves it:

- `stage07.build_dir` names a build whose TensileLite serves the trained
  format (for MLREC_v3 models, one with the v3 adapter). Stage 7 then serves
  the staged model of the run's latest deploy record from
  `validate/stage07_lib/<arch>/`: a link to every file of the bench library
  (`runtime.build_dir`'s), the staged model under the deploy record's
  weights file name, the blocklist sidecar staged with it (*Correctness
  gate*) in place of any the library holds for the stem, and a
  `tilewright_index` whose one line for the library stem names the model on
  every device (the other stems' lines kept, those for the devices of one
  PCI chip ID too, and a malformed one as it stands, with a warning). That
  build's bench runs with
  `HIPBLASLT_TENSILE_LIBPATH` pointing there, so the kernels and solution
  indices are the run's own, and no checkout or build is written.
- Without it, stage 7 runs the deploy checkout's build
  (`deploy.rocm_libraries_to_deploy`), which holds the model when the run
  deploys with `deploy.apply_patch` and `deploy.build_after_apply`, or after
  a manual deploy and model build.

A config that enables stage 7 with neither is an error. Every ML-on run
states `TENSILE_TILEWRIGHT_BANK`: 1 when the config's pool has a bank (kinds
after its first; see *Kernel pool*), so that the runtime ranks
the pool the model was trained on, the Prediction table and the bank, and 0
otherwise. Left unset, TensileLite's adapter serves the bank only to a model
that reads kernel attributes, so a deployed bank model ranks the pool it was
trained on only when hipBLASLt runs with `TENSILE_TILEWRIGHT_BANK=1` or the
model reads attributes. The summary records the setting (`bank_served`), and
so does the report. A runtime that ranks a pool of another size than the
model's fails the stage, naming both sizes (a bank the runtime does not
serve, or one the config does not have). `stage07.cpus` pins stage 7 and its
benches to a CPU list (for timing, an idle core near the GPU and its SMT
sibling) before it records the platform.

Every stage 7 bench line runs one iteration on zero operands
(`initialization: zero`): the stage reads selection times and picks, which
no operand value changes, and hipblaslt-bench generates block-scaled (MX)
operands on the host, on the CPUs `stage07.cpus` pins the stage to, where
values cost far more time to generate than zeros. The selection
efficiencies stage 7 reports cost its picks against the bench timings of
the run's own benches (`--timings-csv-dir`), never against these lines.

hipblaslt-bench allocates the scale buffers of block-scaled (MX) problems
smaller than the library reads them, and whether the read faults depends on
what the process has mapped after the buffer: such a problem can take one
bench process down with an illegal memory access and run in the next. So
stage 7's screen, pick and order parity, Origami reference and timing charge
a bench process that dies of that fault on a block-scaled problem to the
problem after the last one it finished, leave that problem out of the run
and run the problems after it in a new process; the run's summary entry
names the problems it left out (`left_out`), its checks cover the problems
it ran, and the timing pairs a problem only where the ML-on and ML-off runs
both have it, keeping the first call of every bench process out (as does the
report's selection time per serving cell). Any other bench failure fails the
stage: a death on a problem without block scaling or of another fault (a
stall included), output that does not line up with the yaml, and processes
that die in a row before finishing any problem, which no single problem
explains. Before its other runs, a yaml that holds a block-scaled problem
runs every problem once (model on, one solution, no timing), and the
problems this screen leaves out (`screen`) are left out of all its runs, so
that they run one list; a yaml left without problems fails the stage. The
report counts, per yaml, the problems the screen and the single runs left
out.

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
efficiency, with a bootstrap interval; a problem whose top-1 kernel has no
row (one the benches never ran, or one that cannot run) is counted apart
(`n_top1_without_row`) and left out of it, since the stage reads no planned
list; on problems sampled from the training
data (`stage07.parity_sample_trained`) that is mostly an in-sample number.

A second parity run requests `stage07.parity_top_k` solutions (8; 1 or less
turns it off): the runtime's top-1 at that request size and the order of the
solutions it returns must match the engine's ranking (the engine's kernels
first and in its order, the analytical ranking's after them; kernels the
runtime leaves out must be ones whose predicates may fail, which a bench
timing of the problem rules out). Either rate below the minimum fails the
stage.

The selection-time report gives the ML-on times (minimum, mean, median, p99
and maximum) and the ML-on minus ML-off deltas per request size, overall and
split into the first call into each serving cell (which builds that cell's
caches) and the warm calls, each median with a bootstrap 95% interval
(`stage07.bootstrap` resamples). The ML-off times are the analytical
ranking's and are printed only with `--origami-reference`.

Every bench of the pipeline sets `TENSILE_PREDICTION_LIB=1`, which makes
TensileLite answer from Prediction rows alone. Production sets no such
switch: TensileLite then searches every row of the problem's library files
in their order, the files ordered by the length of their names, so an
Equality row of the problem's size, or a GridBased row of a file whose name
sorts before the Prediction table's, answers before the model is asked; a
GridBased row answers every size. Stage 7 therefore benches every yaml once
more as production does (the production route; `stage07.production_route:
false` or `--no-route` leaves it out): tilewright on with the pick log and
`print_kernel_info` at `requested_solution_num: 1`, without
`TENSILE_PREDICTION_LIB`. The yaml's summary entry `production_route` counts
the problems that printed a `TILEWRIGHT_PICK` line, which the model served
(`n_tilewright`, `tilewright_share`), and the problems the model's
Prediction row served without one (`n_unscored`): they reached the model,
which scored nothing for them (no cell or no survivors), and the analytical
ranking filled the pick. For the others it names the row that served them
(`served_by`): its kind (Equality, Range, GridBased, FreeSize, or the
Prediction row of another file), its library file and its kernel's
signature, from the solution index and kernel name the bench prints and the
rows of the library's files of the same data types and operation;
`n_ahead` counts the problems rows ahead of the model's answered.
`route.json` lists every problem, and the summary's `production_route` adds
the yamls up. When a problem did not reach the model (`n_reached` below
`n_problems`), or the runtime never loaded the model (`model_loaded`: no
`[TILEWRIGHT_DIAG FILE]` line names the deployed file), stage 7 warns
(`warned`; the summary's `warned_yamls`) without failing, and the report
marks the yaml's route and names the rows; a bench of the route that fails
is a warning too. A TensileLite that, with tilewright on, searches the
Prediction row of a loaded model before the GridBased rows ahead of it
still searches first the rows it reaches before the first GridBased row
(Equality rows answering the sizes they were tuned for, Range and FreeSize
rows, and Prediction rows without a model) and the rows other than
GridBased ones between that row and the model's; it looks ahead within the
rows of one hardware row's library, so a GridBased row in the subtree of a
CU- or chip-specific hardware row still answers before a model of the base
one.

The rows ahead of the model can hold kernels outside the pool, which the
hazard canary never runs alone. With the canary on (`hazards.enabled`),
stage 7 reads the kernels of every row of those files before the route: a
kernel of a hazard class that a row other than the model's Prediction row
holds, and that is none of the pool's kernels (table and bank) the served
blocklist leaves in, by solution index or name, is kept from launching.
The route leaves out the problems whose size an Equality row answers with
such a kernel (each yaml's `skipped` `production_route_problems` names them
with the kernel's signature, row and file), and skips in full when a row
that answers every size (any other kind) holds one, or when the files' rows
cannot be read; the summary's `hazards.route` lists the kernels.

A kernel the hazard canary keeps out never runs among others, and the
analytical ranking knows nothing of the quarantine: while the library's
pool holds such a kernel, stage 7 skips the runs whose kernels the
analytical ranking picks (the ML-off timing runs, and with them the timing,
the Origami reference, the order parity, whose list it fills, and the
production route), and when the validated bundle's blocklist leaves one in
(a bundle deployed before the quarantine), every run. The summary's
`hazards` and each yaml's `skipped` say which and why.

With `stage07.parity_sample_trained` > 0 stage 7 uses that many random
problems that benchmarked successfully during training instead of
`stage07.bench_yamls`. `stage07.parity_sample_audit` > 0 adds that many of
the audit set's GEMMs (`audit.dir`), which no round trained on, with the
audit's exclusive rows as their bench timings: the runtime's selection
efficiency and selection time on held-out problems.

With `--origami-reference` (or `report.origami_reference`) stage 7 also
runs the audit set's sample once more with tilewright off at
`requested_solution_num: 1`, which returns the kernel Origami picks, and
pairs it problem by problem with the runtime's ML-on pick on the audit's
timings (`origami_reference` in the yaml's summary entry). Like every other
bench it sets `TENSILE_PREDICTION_LIB=1`, so Origami's pick comes from the
Prediction row, which `origami_rows` and the report name; the production
route shows which rows production would answer from instead. This is the
report's pairing of the model with Origami on held-out GEMMs of the
training distribution: planned benches time kernel lists, so no round's
own GEMMs carry Origami's pick, and stage 7 pairs no trained problem. A
pass of stage 7 without the reference does not stand for a run that asks
for it, so the next `--mode full` with it runs stage 7 again; a pass with
it stands for one without.

Stage 8 evaluates the deployed bundle on external datasets (bench yamls,
bench log directories or enriched CSV directories). A bench yaml is benched
like the training shapes: each entry's M, N, K and batch count become a
`lib.bench_yaml` bench line with the config's data types, layout and bench
knobs (rotating buffer, flush, skip screen, adaptive timing). The model ranks
the library's pool and its bank, so bench logs are enriched with
the rows of both (stage04 `--extra-library-stem` for each bank file, as the
rounds enrich theirs), and an enriched directory whose `enrich_summary.json`
shows the bank's rows dropped fails the dataset: its bank picks would be
passed over and its winners taken from the library alone. Its headline is
the model against the fastest measured kernel over every evaluated GEMM,
with the misses; Origami appears only on the same GEMMs (`paired`) and is
printed only with `--origami-reference`. Stage 8 is report-only: its
datasets never select cells or feed a decision.

## Run directory

```
runs/<config stem>_<timestamp>/
    .lock                        held by the running orchestrator
    .running.json                the stage attempt running (lib/attempts.py)
    pin.json                     the code, engine and config the run is
                                 produced with, and the accepted changes
    run_status.json              rounds, deployed bundle, audit, validation,
                                 report and the latest invocation's status
    run_layout.json              the directory names of the run's stages
                                 (see *Resuming*)
    configs/                     config snapshots, one per invocation
    orchestrator.log             the orchestrator's own output, all invocations
    timing/                      wall-clock per stage, one file per invocation
    correctness/                 with correctness.enabled: gate/ (the last
                                 gate that succeeded), gate.new/ (a gate
                                 that runs or failed), blocklists/ (each
                                 invocation's blocklist), stage00.log,
                                 cache/ (the default verdict cache), stale/
    round_<R>/
        stage01/ .. stage06/     stage outputs (stage04b only in active rounds,
                                 stage02 only with planner.enabled, stage03b
                                 only with bench.rebench.enabled, stage03c
                                 only with contexts, stage03d only with
                                 bench.sustained.enabled; a run written
                                 before the planner became stage02 keeps it
                                 in stage01b, see *Resuming*)
        stage01.log ..           stage logs
        stale/                   outputs of re-run stages
        enhance.json             cells and retrain list of an enhance round
        .round_complete.json
    validate/stage07/ stage08/   validation outputs; their stage markers say
                                 what each pass vouches for
    validate/stage07_lib/<arch>/ the library stage 7 serves with
                                 stage07.build_dir
    audit_build.log              output of the audit set build (audit.build)
    audit_v2.log                 protocol v2 and the sustained composite
                                 at the end of the run
    audit_results/<round>.json   rounds scored again on the complete audit set
    final_retrain/               full.final_retrain: final_retrain.json,
                                 stage05/, audit.json, decision.json and,
                                 when it ships, stage06/
    final_retrain_next/          a final retrain trained again beside the
                                 shipped one, until its gate accepted it and
                                 stage06 deployed it
    held_out_sel_eff_breakdown.json  the deployed bundle's own held-out numbers
    report.html                  self-contained report (scripts/make_report.py)
    weak_spots/                  the report's Weak spots data: pool_gaps.json
                                 and pool_gaps.csv (for kernel generation),
                                 weak_spots.json (the cells and their classes)
```

Run directories are not source: their logs, metrics and reports carry measured
performance numbers and must not be committed (`runs/` is ignored).

### Report

```bash
python scripts/make_report.py R [out.html] [--jobs N] [--no-engine] [--no-bench] \
    [--target T] [--selection-budget-us US] [--diagnostics] [--origami-reference] \
    [--shipped-subset FILE] [--audit-results JSON ...] \
    [--audit-label NAME=LABEL ...] [--weak-spots-dir DIR] \
    [--pool-gap-fraction F] [--peak-tflops T] [--peak-bandwidth-gbs B]
```

The report is one HTML file with no external resources, built from the
pipeline's outputs only; every table sorts by a click on a column header
(numbers, percentages, intervals and durations by value, missing values
last, summary rows in place). Its headline is the run's position on GEMMs
the model never trained on, deployed or not. It describes the bundle the
run deployed (the latest `stage06/deploy_record.json` the deploy gate let
through, or the final retrain's while that is the run's bundle: see *Final
retrain*), else the latest staged one (`deployed: false`, labelled staged,
not deployed): its `gate.json` and every stage04b that measured the same
`models.pt` (the next round's), for the final retrain its audit record,
its leaf cells against the target, and
stage 7's runtime selection efficiency (with its bootstrap interval),
selection time on the first call into a cell vs warm, and production route
(the share of problems tilewright served without `TENSILE_PREDICTION_LIB`,
those it scored nothing for, the rows that served the others by kernel
signature, marked when a problem did not reach the model or the runtime
never loaded it, and the problems left out for kernels of a hazard
class). Once the audit set (below) has a complete evaluation of that bundle,
its equal-weight mean
over cells against the true best, its minimum cell and its cells below the
target come first, under the official protocol (`audit.official`) when
that is complete for the bundle, else v2, with v2's cell mean in a tile of
its own beside the composite's (the audit table gives it per row), and
every audit number names its protocol. `--audit-results` adds
`scripts/audit_eval.py` results (protocol v1, or v1 with v2), as do the
records the run wrote outside its rounds (`audit_results/`, the final
retrain's), which the report reads itself. A record of
one of the run's bundle files stands for the run's own evaluation only when
its `audit_dir` is the run's `audit.dir` and it charges picks by the run's
`validate.pick_measurement`: it then replaces the run's record when it is
more complete (v2 complete first) and fills in rounds the run did not
evaluate. Every other record (a model that is no round of the run, such as
an offline retrain, or a round's bundle scored on another audit set or pick
rule) appears as an extra model, in the headline and as an extra row of the
audit table, labelled with its audit set, never as a round; of several
records of one model file on one audit set only the most complete is shown.
An extra model is compared with the bundle on the same GEMMs when both are
on one audit set and pick rule. `--audit-label NAME=LABEL` shows the extra
model NAME (its label in the report, or at least six leading hex digits of
its model file's sha256) as LABEL. A record written before the
identity-limited class existed shows that class as not recorded. The
headline's audit numbers are also embedded in the page's data (`AUDIT`).
In-sample numbers are kept apart and labelled as in-sample. A result the
run does not have says why, from the run's config and records (no bundle
passed the gate, stages 7 and 8 not reached or not configured, the audit
still benching with its progress). A shipped final retrain trained on
every GEMM of its rounds and no round follows it, so it has no deploy gate
and no later stage04b: one line names its held-out evidence instead, the
audit set's evaluation and stage 7's runtime selection efficiency on the
problems held out of training. Every comparison between two models is
paired (the same GEMMs). Wall-clock rows that belong to no round (stages 7
and 8, the audit set build, the final retrain) are labelled `run`.

The bench sections re-rank GEMMs with that bundle (`lib/run_analysis.py`;
the tilewright module is needed for the ranking steps), each routed to the
leaves of the bundle's own split tree and ranked in its execution context,
and compare against the fastest measured kernel of each GEMM. They apply
the rules the bundle serves with: MLREC_v2's small-batched, Dot2 and
leading-free rules, or the soft rules an MLREC_v3 file's serving flags
switch on, and the leading-free rule as TensileLite's predicate, which a
kernel must pass to run whether or not the engine applies it. They charge
picks with `validate.pick_measurement` of the run's config
(`lib/evaluate.py`, as stages 4.5, 5, 6, 7 and 8), and take the pick the
runtime serves: the first scored kernel that runs. A kernel that fails the
leading-free rule cannot run (no bench has a row of it) and is passed over
for the next-ranked one. A pick off the GEMM's planned list that passes
it can run but was never benched: stages 4.5, 6 and 8 charge it the
slowest latency the GEMM measured unless a re-bench measured it
(`validate.rebench_misses`), stage05 the winner's latency over
`train.off_list_sel_eff` (*Student training (stage05)*; the slowest latency
when that gives no efficiency), and stage 7, which costs the runtime's top-1
without the planned list, leaves its problem out of its selection efficiency
(a top-1 kernel without a row); the report counts it at the next-ranked
kernel with a reading the pick measurement accepts, and shows the range from the
slowest latency to the winner's next to it, with the number of such
picks, never the slowest latency alone. A GEMM benched under an execution
context has the list its context's re-plan gave its shape (stage02's
`contexts/<label>/plan.json`, as stage03c benched it), else the round's.
The training benches are those the bundle learned from; the
held-out benches are its deploy-gate slice and later rounds' new GEMMs;
held-out contexts (`stage03c/holdout`) are reported separately, under the
names of the run's latest config snapshot, whose budgets are resolved.

The Kernel pool section shows what the pool the bundle ranks is made of
(see *Kernel pool*): the library, config and bench device the bundle's
stage05 ranked with, its blocklist applied.

A v3 kernel's identity is its 8-field signature (macro tile M and N,
depthU, matrix instruction M, N and K, cache hints A and B; framework-
agnostic by design) plus the reads of an attribute spec, empty by default
(`DEFAULT_ATTRIBUTE_SPEC`). Library solutions that share the key are one
candidate, scored once; TensileLite ranks with `TieBreak::PoolOrder`, so a
problem gets the key's first member in pool order (library table, then
bank) that it admits. With `train.launch_keys` each launch class of a key is
a key of its own. The report's Kernel identity section counts the members
per key over the run's pool and bank (largest and mean, keys with more than
one member) and gives the identity ceiling (`lib/identity_ceiling.py`) on
the run's exhaustively benched GEMMs, the audit set once complete, else
round 0's exhaustive seed bench, with and without launch keys: what the
identity costs before any model.

- **Weak spots** (below): every audit cell of the bundle with the class of
  what limits it and the lever that addresses it, and the pool gaps.
- **Deploy gate per round**: each stage06's deploy gate, its per-cell gate
  against the bundle the run shipped, and the final retrain's decision;
  then every per-cell finding (bundle, group, whether it decides, cell or
  GEMM, both efficiencies, ratio with its interval, the cell's
  family-wise p, whether a repeated measurement confirms a tail pick),
  lifted tail picks included.
- **Repairs** (see *Repair on refusal*): every round's repair decision
  (`<round>/repair/decision.json`) and every final retrain's repair
  (`decision.json` `repair`): the decision (with its attempts while it
  waits), whether the repaired bundle was deployed, what the repair did,
  the incumbent, the splits it revoked and the reason; then per flagged
  base cell its findings, the action (`tier add`, `subtree carry`), the
  tier keys added and the sha256 of the bundle whose networks now serve it;
  and the note of every repaired bundle that the same audit GEMMs flagged
  and judged its cells (`selected_by_audit`).
- **Where to improve**: every leaf of the bundle ranked by expected gain =
  GEMM share x (target - efficiency), with the efficiency and the action
  taken from held-out evidence only: the audit set's result of the leaf's
  base cell when it is complete, else the bundle's own measurements on GEMMs
  it never trained on (deploy gate and later stage04b, pooled per cell),
  else the report's re-ranking of those GEMMs; a column says which. It
  names one action per leaf:
  - kernel pool: the audit's pool-limited class, or the library pool step
    losing the most (tier policy: serving rules when the library alone
    would reach the target);
  - kernel identity: the audit's identity-limited class (launch keys, spec
    attributes, or a pool whose keys serve better kernels);
  - more data: too few GEMMs of its own, or too few held-out GEMMs;
  - retrain;
  - split: retrained `validate.split_after_attempts` times without reaching
    the target, for a cell that can split (a Large M or N or a LargeK axis,
    below `validate.max_split_depth`); one that cannot is told to retrain;
  - tier policy: the non-temporal rule, the whitelist or the cascade loses
    the most on the held-out decomposition.

  In-sample efficiency and the training-bench decomposition are context
  columns and never choose the action. It also lists the cells `--weakest` /
  `--below` would select (enhance mode's own evidence).
- **Loss decomposition** (training and held-out benches): per leaf, the
  efficiency lost at each step of the engine's own selection. The steps
  are the library pool, the hard rules (the LDS gate, the leading-free rule,
  and the small-batched and Dot2 rules where the bundle serves with them),
  the non-temporal rule, tier 1 (the whitelist, or the whitelist tier), the
  cascade, and the ranking. Per GEMM, the best measured latency left after
  each step is clamped to be monotone and never above the pick's own cost,
  so the step factors multiply to the cell's selection efficiency and the
  pick is charged what it costs. GEMMs without a pick are counted by the step that
  empties; misses (picks without a timing of their own under the pick
  measurement) are counted per reason, the GEMMs with at least one kernel
  ranked above the served pick that cannot run are counted (column
  "unrunnable kernel ranked above the pick"), and the cell's efficiency
  with every pick
  without a reading at its GEMM's slowest latency and at the winner's
  stands next to it.
- **Execution contexts**: selection efficiency per context, trained and
  held-out contexts separately: stage04b's own numbers, the report's
  re-ranking of the GEMMs the bundle never trained on, and, apart, its
  in-sample re-ranking of the training benches (context only) with the
  audit set's held-out result of each context beside it; and the weak
  cells of each context.
- **Kernel usage**, **Kernel parameters**: wins (oracle) and picks (model)
  per signature and per value of macro tile, depthU, matrix instruction and
  cache hints, overall and per cell. A kernel is named by its signature,
  e.g. `MT256x256x64 MI16x16x32 NTA0 NTB4` (macro tile M x N x depthU,
  matrix instruction M x N x K, cache hints A and B), with the number of
  library solutions sharing it; the report never names a kernel by its
  solution index, which is a position in one library build. The kernel
  tables also give each field of the signature a column of its own (MT_M,
  MT_N, DepthU, MI_M, MI_N, MI_K, NTA, NTB), so that every column sorts.
  The removal loss is the oracle loss if the signature or value were
  removed. `--diagnostics` adds a collapsed section with framework-specific
  parameters (occupancy, vector widths, varying kernel attributes).
- **Pool-drop candidates**: signatures that never win and are never picked,
  with:
  - the GEMMs where they come within 1%, 2% and 5% of the best;
  - the number of GEMMs that tested or screened them.
- **Selection time per cell**: stage 7's ML-on selection time per serving
  cell, for every bench yaml and request size.
- **Cells in M / N / K**: every leaf of the bundle in M and N, one panel per
  K tier and batch tier, filled by its held-out selection efficiency (inline
  SVG; hover for the label and value).

- **Shipped candidate set**: the table and recommendation of
  `shipped_subset.json` (below), when the run has one (`--shipped-subset`
  names another file).

Stage 8 results appear as validation only. They never feed the actions or
the selection. Origami's numbers (and stage 7's ML-off times) appear only
with `--origami-reference` (or `report.origami_reference`), in a collapsed
section labelled as an external reference. It opens with a reference
summary: the model and Origami on the same held-out GEMMs of the training
distribution and on each stage 8 dataset, and stage 7's ML-on selection time
(minimum, mean, p99, maximum) with Origami's median beside it. The held-out
GEMMs of the training distribution are the rounds' new GEMMs where their
benches timed Origami's pick (list-free benches; planned ones do not) and
the audit set's sample stage 7 pairs (see *Validation*); when neither has
numbers the summary says why, and so does every other table of the section
left empty or dashed for the same reason (per round, the GEMMs benched from
planned lists). Below the summary, the shipped bundle's line says that none
of its own held-out measurements (deploy gate, stage04b, audit record) pairs
its pick with Origami's, and whether stage 7 does. The wall-clock table has a
column for every
stage the timing records name, in pipeline order: the planner's is
`stage02`, or `stage01b` in a run written before the planner became stage02,
whose `stage02` column is the removed probe's.

The orchestrator writes the report at the end of every invocation, stopping
`make_report.py` after `report.timeout_s`; the report never changes the exit
status, and `run_status.json` records whether it was written. It writes the
Weak spots data into `<run>/weak_spots/` with it (`--weak-spots-dir`).

#### Weak spots

The section answers, per grid cell of the bundle the report describes,
what keeps it from the target and what would fix it (`lib/weak_spots.py`).
Each cell of each audit group gets the classes of what limits it, each
naming its lever:

| class | what limits the cell | lever |
|---|---|---|
| `rule` | the serving rules (non-temporal, small-batched, Dot2) leave out the winners: the best kernel they admit is below the target (the audit's pool-limited class) | refine the rule, e.g. a narrower non-temporal rule with training data for the keys it adds |
| `identity` | the kernels one signature groups are served by a slower member (the audit's identity-limited class) | member order, launch keys or an attribute in the identity |
| `model` | a better pick exists among what the rules and the identity serve | retrain the cell: `--mode enhance --cells`, another recipe |
| `pool` | the true best of the cell's GEMMs reaches on average less than `report.pool_gap_fraction` of its roofline | generate kernels for the cell's pool gaps |

A cell can have several (`model+pool`). The first three come from the
bundle's audit record (under the official protocol where the group has
it, else v2, else v1); `pool` from the
exhaustively benched exclusive GEMMs of the run: the audit set when it is
complete for the bundle's pool (held out, so it names cells, never their
shapes), else round 0's exhaustive seed bench. The weak cells are those
with a class, a regression against the previous candidate (the per-cell
gate's family-wise cell test, under the `gate` section; the row shows the
paired ratio, its interval resampled within the cell, and the cell's p) or
a pick below `gate.tail_floor`; each row also counts the picks below
`gate.tail_floor` and `gate.cell_floor` and gives the true best's roofline
fraction (geomean over the cell's GEMMs) and its number of pool gaps. A
count the audit record cannot give is n/a, not 0: picks below the floors
of a group whose record holds no per-GEMM results (records written before
them), and regressions where the previous candidate cannot be paired, the
section saying why. With `--origami-reference` the external reference
section compares the weak exclusive cells with Origami on the audit set's
problems stage 7 timed; the exported data never holds Origami's numbers.

Roofline: a GEMM's attainable time is the larger of the benches' roofline
(`lib.bench_yaml.roofline_us`: its matrix work, 2·M·N·K·batch FLOPs, at
the peak matrix rate and its compulsory traffic, A and B read and D written
once, at the memory bandwidth) and the latency floor, the best time of the
fastest exclusive GEMM of the set (no launch finishes sooner, so a
launch-bound shape is not a gap). Its fraction is attainable / best
measured time. The rates are `--peak-tflops` / `--peak-bandwidth-gbs`,
else the run's `bench.adaptive_long.peak_tflops` / `bandwidth_gbs`, else
the device's (`lib.hardware.roofline_rates`, see *Bench timing*); each is
raised to the best any GEMM of the set reached when that is higher or the
rate is unknown, and the report names the source of both.

The pool gaps are the exclusive GEMMs of the training side below the
fraction, farthest first, with the signature of their best measured
kernel. They come from round 0's exhaustive seed bench only, without any
shape of the audit set: kernels generated for the held-out shapes would fit
the shipped pool to the GEMMs the audit and the per-cell gate judge it on.
Both sources share one set of peaks and one latency floor.
`pool_gaps.json` (their source, how many seed shapes it left out for being
audit shapes, the digests of the kernel library and of the pool they were
measured on, the layout and types, the peaks, the threshold and every gap)
and `pool_gaps.csv` (one row per gap: shape, cell, best and attainable
time, fraction, the bounding term, the signature and its eight fields) are
the input for kernel generation; `weak_spots.json` holds every cell row,
the counts per group and where the cells' roofline came from. Each file is
written atomically.

### Shipped candidate set

```bash
python scripts/choose_shipped_subset.py R [--out-dir D] [--sizes 0,32,64,128,256,512,1024,all] \
    [--model-file F] [--bank-stem S ...] [--pick-measurement warmup|timed|any] [--jobs N] \
    [--no-latency | --bench-binary B] [--bench-shapes 1000] [--bench-core C]
```

Chooses which kernels of the pool's bank (see *Kernel pool*) to ship
next to the library pool. For the bundle the report describes, it computes
each GEMM's engine scores once over pool + bank and then simulates the
engine's pick under any shipped subset (`lib/shipped_subset.py`: view keys
split into hipBLASLt's launch classes, feasibility with the subset's
non-temporal availability, whitelist tier, cascade by view key, pool-order
ties and the pick measurement's charging; a model with the family-wise
non-temporal rule or the trained-key gate is refused, since the simulation
models neither). A greedy
forward selection on the training-side GEMMs of every round (never a gate
slice) orders the bank; every size B is measured on the GEMMs the bundle
never trained on, per execution context, with the perfect-scoring ceiling
and, with `scripts/experiments/bench_shipped_rank`, the p50 / p99 time of
one warm ranking call on one pinned core (the script builds the benchmark
without the engine's re-rank memo, which can answer every other one of its
repeated rankings of a shape; build a `--bench-binary` so too). That
benchmark ranks with
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
architecture as hipBLASLt names its library subtree, the name the bench
device reports, e.g. `gfx950`, `gfx1250-strict`; a build alias such as
`gfx1250v0` is an error: list it in `deploy.targets` instead).

**`hipblaslt`** (required): `a_type`, `b_type`, `c_type`, `d_type`,
`scale_type`, `compute_type`, `transA`, `transB` (the bench line fields),
optional `scaleA` / `scaleB` (hipblaslt-bench scaling-format modes, e.g. 1
scalar, 3 MX block-32; required, and a block format, for `f4_r`, `f6_r` and
`bf6_r` operands, whose libraries are all block-scaled), and `library_stem`: the Tensile library file name
without `.dat` / `.dat.zlib`, for example
`TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_ID75a0_gfx950`.
stage04 keeps only measurements of this library's kernels, training and
evaluation rank its kernel pool, and stage06 indexes this stem.

**`library`** (see *Kernel pool*): `kind` (`origami`; `bok`, or a list of
both in pool order), `bok_rows` (`[freesize, equality]`; `gridbased`). A
config without the section names its pool with the planner's legacy
`bank` / `bank_library_stems`; setting both is an error. The section is
pinned only when set.

**`hardware`** (optional): `n_cu`, `lds_bytes`, `l2_bytes`, the device values
hipBLASLt passes to the engine, `pci_chip_id`, the device's PCI chip id for
the bank's hardware predicates, and `num_xcd`, the device's XCD count, the
queue count of dynamic-queue launches on an architecture without a fixed one
(on one with a fixed count it must equal it; see *Kernels and execution
context*). Any key set here overrides the value read from the KFD sysfs
topology of the first device in `runtime.devices`. Keep device values of
unreleased hardware in a run config outside the repository.

**`runtime`** (required): `devices` (GPU indices, into the GPUs the
environment makes visible), `rocm_libraries_root` (the bench root), optional
`build_dir` (hipBLASLt build directory; default
`<root>/projects/hipblaslt/build/release`), `gpu_lease_wait_s` (3600:
seconds a stage waits while every GPU of its devices is leased by another
process tree before the attempt fails; see *GPUs: leases, quarantine and
platform records*), `exclude_gpus` ([]: GPUs the run never uses, by
physical index, PCI address or `GPU-<hex>` id; see *Shared hosts*),
`min_free_gb` (null: the free space a round needs is judged by the rounds
before it, which a round with more weak leaves can exceed; a number: the GB
free the run directory's file system must have
before every round; 0: no check; see *Running*), `stage_retries` (1:
further attempts of a stage after a transient failure) and
`stage_retry_backoff_s` (300: the wait before the first of them, doubled for
each next one; see *Resuming*).

`runtime.env` (optional) is a mapping of environment variables to non-empty
strings or integers that every stage, bench and cotenant of the run gets over
the caller's environment, such as `HSA_DISABLE_GFX12_STRICT: "0"`. Variables
the pipeline sets or removes itself are errors: the device lists,
`TENSILE_USE_TILEWRIGHT` and any `TILEWRIGHT_*` or `TENSILE_TILEWRIGHT_*`
variable, `HIPBLASLT_TENSILE_LIBPATH`, `TENSILE_PREDICTION_LIB` and
`HSA_HOTSWAP_DISABLE`. TensileLite's launch overrides reach a bench from
here alone, with a warning (see *Launch settings*). The snapshot's first
comment lines state it, and preflight checks the bench devices with it.

**`seed`**: `target_per_cell` (round_0 shapes per cell, 1600),
`shapes_per_cell` (new shapes per leaf in an active round, 150), `seed` (42;
each round derives its own seed from it), `n_strata` (log-strata per axis
within a cell, 1), `exclude` (shape regions left out, a list of rules with
`min_` / `max_` bounds on `m`, `n`, `k`, `batch` and `mn` = min(M, N); a shape
is left out when it meets every bound of a rule), `targeted_per_tail_gemm`
(4: new shapes near each tail GEMM of the previous round's stage04b, 0 off;
see *Targeted shapes (stage01)*), `targeted_max_per_leaf` (32: at most that
many per leaf), `weak_cell_factor` (1: the new shapes of a leaf the
previous round's stage04b retrained, as a multiple of `shapes_per_cell`;
>= 1), `strong_cell_factor` (1: the same for a leaf that passed at least
`strong_cell_margin` above the threshold, in (0, 1]; the strong leaves
give up at most what the weak leaves gain, and config validation refuses a
factor that leaves a strong leaf fewer new GEMMs than
`validate.min_cell_gemms`), `strong_cell_margin` (0.03), `large_k_factor`
(1: a leaf's new shapes with K above 32,768, as a multiple of the number
its draw holds; >= 1); config validation warns about a `weak_cell_factor`
or `large_k_factor` above 4; see *Weak cells and very large K (stage01)*.

**`planner`** (stage02): `enabled` (false), `mode` (`exhaustive`;
`planned`, `auto`), `exhaustive_max_kernels` (512), `seed_per_cell` (8),
`ranker` (`neighbors`; `stage05`), `neighbors_k` (4), `model_file` (null),
`bank` (`none`; `auto`) and `bank_library_stems` ([]), the legacy form of
`library` (see *Kernel pool*), `rules` (`config`; `model`), `top_k` (32), `families_top_f` (8),
`family_variants` (0 = all), `explore_random_k` (32), `min_coverage` (2),
`max_list` (512; at most 4096, the bench's limit), `seed` (42),
`top_k_round0` (null: `top_k`; the `top_k` of the lists the seeds rank,
pinned only when set), `recall_min_ceiling` (0.97; the seed recall's
warning threshold, unpinned; see *Seed recall*).

**`contexts`** (stage03c; see Execution contexts): `list` and `holdout`
(entries `{cu_budget or cu_fraction, schedule, cotenant, fraction, name,
enabled}`; a run resolves a `cu_fraction` when it starts, as a share of the
bench device's CUs, a partition's inside one, and leaves out an entry with
`enabled: false`), `granule` (32: the CUs every budget a `cu_fraction`
resolves to is a whole multiple of, at least one; a multiple of the
launch rule's SM-target granule; pinned only at another value),
`seed` (42),
`cotenant_binary` (default: `hipblaslt-cotenant` next to the bench),
`path_append` (directories appended to the wrapped bench's `PATH`; default
the ROCm `bin`), `replan` (false).

`bench`**: `pool_only_library` (false; see The pool-only library directory),
`rebench` (stage 3.2, see *Near-winner re-reads (stage03b)*: `enabled`
(false), `tolerance` (0.05), `tolerance_us` (0.0: no absolute floor),
`max_kernels` (16), `runs` (1), `fast_below_us` (10.0), `fast_runs` (3),
`slow_from_us` (1000.0), `slow_runs` (3), `slow_min_iters` (0: the line's
own), `inline_copies` (1, at least 1: times a run's line lists every
candidate; times `max_kernels` at most 4096), `with_original` (true),
`align_to_original` (false; needs `with_original`; with `inline_copies` 1 it
leaves GEMMs with two near-winners as read, and the config check warns),
`sacrificial_kernel` (true), `gemm_order` (`shuffled`; `listed`)),
`adaptive_long` (see *Bench timing*: `enabled` (false), `min_roofline_us`
(100.0), `warmup_time` (0.5), `sample_time` (0.25) and `max_measure_time`
(4.0) in ms, `peak_tflops` and `bandwidth_gbs` (null: read from the first
bench device when a run starts)), `sustained` (stage 3.7, see *Sustained
finalists (stage03d)*: `enabled` (false), `min_us` (100.0),
`compute_bound_tflops` (700.0), `compute_bound_min_us` (100.0), `top_k` (4),
`tolerance` (0.08), `max_kernels` (12), `runs` (1), `warmup_time` (250.0),
`measure_time` (300.0) and `max_measure_time` (400.0) in ms,
`gemms_per_block` (24)), `kernel_order` (`shuffled`: every explicit kernel
list in a seeded per-GEMM shuffle; `listed`: in the order it was made; see
*Bench timing*), `order_seed` (42), `sacrificial_kernel` (true: a line with
a skip ratio leads with a copy of its first kernel whose row is set aside;
see *Bench timing*), `blocks_per_gpu` (20: the work blocks per GPU of stage
3's own benches, and of stage 3.5's; the near-winner, sustained and miss
re-benches size theirs to their lines, see *Bench timing*),
`heavy_operand_gib` (0: off; the lines with an operand this many GiB or more
in blocks of their own at the end of stage 3's queue, see *Bench timing*;
not pinned), `startup_grace_s` (900) and `stall_timeout_s` (600) (stage 3
kills a silent bench and quarantines the line it hung in; see *Shared
hosts*), `quiet_host` (`enabled` (true), `every_s` (10), `quiet_busy`
(0.15), `quiet_core_busy` (1.0), `hot_busy` (0.25), `abort_samples` (2),
`max_wait_s` (28800; 0: no limit)), `health` (`enabled` (true), `poll_s`
(30), `max_wait_s` (0: no limit), `ttm` (true: check the host's ttm
workers), `canary` (`enabled` (true, while `enabled` is), `lines` (4),
`timings` (5), `tolerance` (0.05), `tolerance_us` (0.0: no absolute floor),
`retry_s` (900))), `warmup_pass` (true), `warmup_line` (the throwaway line
every bench process starts with, see *Bench timing*; not pinned: `enabled`
(true), `kernels` (4: listed kernels, one of each library file in turn and
at least one of each; at most 64) and `size` (512: the problem `size` x
`size` x `size`, batch 1)), `canary` (stage 3's host-load guard, see *GPUs:
leases, quarantine and platform records*: `enabled` (false), `tolerance`
(0.1), `tolerance_us` (0.0: no absolute floor; pinned only when above 0),
`timings` (3), `backoff_s` (120)), `skip_slow_solution_ratio` (0: no skip
screen), `rotating_mb`, `rotating_target_blocks`, `device_memory_gib`
(default: the KFD sysfs topology), and `adaptive`, the adaptive timing of
every bench (see *Bench timing*): `warmup_time` (1.0), `sample_time` (0.5),
`measure_time` (1.5) and `max_measure_time` (8.0) in ms, `min_iters` (3),
`max_iters` (0 = no limit), `noise_threshold` (0.05), `stability_threshold`
(0.05), `stability_window` (32) and `stability_interval` (16).

Keys of the removed probe stage and of fixed iteration counts (`probe`,
`bench.adaptive.enabled`, `bench.rebench.iters_factor`,
`planner.probe_kernels`) are errors that say what replaced them.

**`train`**: `workers` (1), `threads_per_worker` (0 = cores / workers),
`epochs` (200; passes over each cell's training GEMMs), `device` (`cpu`), `eval_sample_frac` (1.0), `smart_k` (10),
`smart_k_max` / `smart_k_min` (4) / `smart_k_saturation_eps` (0.005) for an
adaptive whitelist size, `min_cell_gemms` (50; cells with fewer GEMMs over all
rounds are not trained, a leaf without a model is listed for training once it
has this many, and each side of a split needs this many; every leaf that a
model of the previous round serves gets at least this many shapes per round,
`seed.shapes_per_cell` new ones topped up with re-used history shapes),
`validation_frac` (GEMMs per cell held out to choose the epoch; stage05's
default when absent), `served_candidates` (false; v3: train on one candidate
per key, see *Student training (stage05)*), `whitelist_tier_k` (0; v3: a
whitelist tier of this many view keys per cell, chosen after training,
model-shaping; see *Whitelist tier (stage05)*), `whitelist_tier_contexts`
(`pooled`; with `whitelist_tier_k`, how the tier counts the execution
contexts of a cell's GEMMs: `pooled`, `balanced` or `reserve`,
model-shaping; validation warns when there is no tier or no trained
context besides exclusive use), `whitelist_tier_guard_keys` (8; with
`whitelist_tier_k`, the keys the tail guard may add to a cell's tier, 0
off, model-shaping; validation warns when there is no tier; see *Whitelist
tier (stage05)*), `non_temporal_family_rule`
(false; v3 with `non_temporal_rule`: the family-wise non-temporal rule,
model-shaping; stage02 then lists the members it keeps, see *Planned
benches (stage02)*; preflight refuses it, `non_temporal_rule` and a spec
attribute holding the temporal hint of an operand the tensor data mover
loads for a pool with such operands, see *Kernel identity across hipBLASLt
builds*), `trained_key_gate` (true; v3: the model scores only
the view keys it has training rows for while one of them is feasible, see
*Kernels and execution context*), `repeat_aggregation` (`median`;
`median_low`, `min`: how several rows of one kernel of one GEMM merge into
its label, read by stages 4.5, 5, 6 and 8; model-shaping), `on_cell_failure`
(`stop`; `carry`: a round whose stage05 trained some of its cells goes on
with the bundle, see *Resuming*), `carry_max_fraction` (0.25: the largest
share of the cells stage05 was to train that `carry` goes on without). The
student schedule keys below reach stage05 only when
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
| `off_list_sel_eff` | `--off-list-sel-eff` | unset | early stopping and deployed numbers charge a pick off its GEMM's planned list the winner's latency over this, (0, 1] or `auto` (stage04b's re-bench; unset: the slowest latency measured) |
| `off_list_fallback` | `--off-list-fallback` | unset | `auto` without enough measured off-list misses, (0, 1] (unset: the slowest latency) |
| `off_list_min_misses` | `--off-list-min-misses` | 30 | `auto`: measured off-list misses needed overall and for a context's own charge, >= 1 |
| `pseudo_negatives` | `--pseudo-negatives` | `random` when `unmeasured_samples` > 0, else `none` | `none`, `random` (feasible pool kernels off a planned GEMM's list), `targeted` (v3: keys no training GEMM of the cell measured) |
| `unmeasured_samples` | `--unmeasured-samples` | 16 for `targeted`, else 0 | pseudo-negatives per planned training GEMM, >= 0 (0 only with `none`) |
| `unmeasured_margin` | `--unmeasured-margin` | 1.0 for `targeted`, else 0.05 | their pseudo latency: winner x (1 + margin), >= 0 |
| `unmeasured_weight` | `--unmeasured-weight` | 0.1 for `targeted`, else 0.3 | scale of their soft-CE target, (0, 1] |
| `stage1_recall_weight` | `--stage1-recall-weight` | 0 | weight of a loss training the embedding term alone to rank as the full score does, for the cascade's first stage, >= 0 (0: off) |
| `distill_lam` | `--lam` | 0.5 | weight of the distillation from the LightGBM teacher: loss = lam x KL(teacher_T, student_T) x T^2 + (1 - lam) x soft-CE, [0, 1] (both recipes, both phases of `pooled_ft`; model-shaping) |
| `distill_T` | `--T` | 2.0 | temperature of that KL term's per-GEMM softmaxes, > 0 (model-shaping) |
| `recipe` | `--recipe` | `per_leaf` | `per_leaf`, `pooled_ft` (one student over every cell, fine-tuned per cell; model-shaping) |
| `pooled_epochs` | `--pooled-epochs` | 12 | `pooled_ft`: passes of the pooled student, >= 1 |
| `pooled_batch_gemms` | `--pooled-batch-gemms` | 64 | `pooled_ft`: GEMMs per optimizer step of the pooled student, >= 0 (0: all) |
| `pooled_lr` | `--pooled-lr` | 0.002 | `pooled_ft`: peak learning rate of the pooled student, > 0 |
| `pooled_eval_points` | `--pooled-eval-points` | 24 | `pooled_ft`: checkpoints of the pooled student measured, >= 1 |
| `pooled_threads` | `--pooled-threads` | 0 | `pooled_ft`: threads of the pooled student, >= 0 (0: every CPU the stage may use) |
| `pooled_teacher_threads` | `--pooled-teacher-threads` | 0 | `pooled_ft`: threads of the pooled teacher, >= 0 (0: `pooled_threads`) |
| `fine_tune_epochs` | `--fine-tune-epochs` | 12 | `pooled_ft`: passes of each cell's fine-tuning, >= 0 (0: every cell keeps the pooled student) |
| `fine_tune_lr` | `--fine-tune-lr` | 0.001 | `pooled_ft`: peak learning rate of the fine-tuning, > 0 |

**`validate`** (stage04b): `sel_eff_threshold` (0.95), `split_floor` (0.85;
0 disables), both numbers, `min_cell_gemms` (20), `split_min_improvement`
(0.02), `split_after_attempts` (2), `max_split_depth` (6; 0, the training
recipes' value, turns splitting off),
`structural_split_min_octaves` (0 = off), `structural_split_max_depth` (-1),
`pick_measurement` (`warmup`; `timed`, `any`, see
*Kernels and execution context*; read by stages 4.5, 5, 6, 7 and 8),
`rebench_misses` (true: stages 4.5, 6 (gate) and 8 re-bench their misses on
the GPUs and decide on the measured numbers, same section),
`rebench_runs` (3, at least 1: runs of that re-bench), `tail_floor` (0.7: a
held-out pick below it makes a tail GEMM, listed per cell for the next
round's targeted shapes), `collapse_retrain` (true: a cell that passes with
a held-out pick below `gate.tail_floor` is retrained in the same round; see
*Active learning (stage04b)*).

**`full`**: `total_rounds` (3), `final_retrain` (see *Final retrain*):
`enabled` (false), `gate` (`not-worse`; `none`), `repair` (true: a final
retrain the per-cell gate keeps out for findings in at most
`gate.repair_max_cells` base cells is repaired and judged again; see
*Final retrain*), `recipe` (a mapping of
train keys to override and stage05 options to add for the final retrain,
each named as stage05's option is, `T` included; `distill_lam` and `lam`,
or `distill_T` and `T`, together are an error).

**`audit`** (see *Audit set*): `dir` (null: no audit), `held_out` ([]: the
library's other held-out lists, audit directories or list files, whose
GEMMs no round draws; pinned only when set, with the digests of their
GEMMs), `build` (`before`;
`parallel`, `off`: when a training run builds an incomplete audit set),
`devices` (the audit's bench devices; default `runtime.devices`; with
`build: parallel` required and disjoint from them), `v2` (`rounds`; `final`,
`off`: which bundles the run measures under protocol v2), `v2_set_tolerance`
(0.1), `v2_set_tolerance_us` (0.0: no absolute floor) and `v2_set_max` (20)
(the width of the v2 sets, for the measurement's first plan, the floor for
the sustained composite's first plan too; see *The audit set in the one
command*), `canary_tolerance_us` (0.0: no absolute floor of the v2 and
sustained canaries, for a first plan; pinned only when above 0), `sustained`
(true: with v2, the sustained composite's measurement too), `official`
(`sustained`; `v2`: the protocol of the official exclusive number and of the
final retrain's decision and the per-cell gate), `shapes_per_cell` (16),
`shapes_per_context` (4, at most `shapes_per_cell`), `seed` (4242, must
differ from `seed.seed`), `target` (0.95), `bootstrap` (2000), `part_shapes`
(32: shapes per resumable bench part).

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
`<deploy root>/projects/hipblaslt/build/release`), `build_jobs`,
`provenance` (`neutral`: the model file records the library stem as its
`config_id`, the training data hash and the launch model; `full`: the
bundle's provenance as stage05 wrote it).

**`stage07`**: `build_dir` (null: the deploy build; a hipBLASLt build
whose TensileLite serves the trained format, see *Validation*), `cpus` (null;
a CPU list such as `4,5` or `8-11`), `parity_sample_trained` (0),
`parity_sample_audit` (0; needs `audit.dir`), `bench_yamls` ([]),
`timing_request_sizes` ([1]), `timing_repetitions` (3),
`parity_min_match_rate` (1.0), `parity_top_k` (8), `bootstrap` (1000),
`production_route` (true; false leaves the production route out).

**`stage08`**: `datasets` (a list of `{name, path, rocm_libraries_used}`; the
last is the checkout whose library produced a log or CSV dataset, read from
`runtime.build_dir` when it is the bench root and from its
`projects/hipblaslt/build/release` otherwise), `tie_tolerance`, `bootstrap`.

**`gate`** (the per-cell gate, see *Deployment (stage06)*; changing it
needs no `--allow-code-change`): `per_cell` (true; false leaves the deploy
gate and the final retrain's gate as they were without it),
`cell_tolerance` (0.01: a cell significantly worse than the bundle that
ships by more than this keeps a bundle out), `cell_floor` (0.85; 0 off: a
cell that confidently falls below it does), `alpha` (0.05: the cell rules'
family-wise level over their cells; `audit.bootstrap` must give at least
2 / alpha sign flips), `tail_floor` (0.5; 0 off: the tail rule's floor),
`tail_collapse` (1, in (0, 1]: a new pick below the tail floor counts
when it is at most this fraction of the shipped pick's efficiency; 1:
every step across the floor), `tail_budget` (0: the confirmed new tail
picks a bundle may add over every group; healed ones never offset them),
`groups` (`exclusive`; `all`: the audit groups
the cell rules judge as one family; the others' cells are reported, and
the tail rule judges every group), `repair` (false: a round's bundle the
per-cell gate alone keeps out is repaired in the base cells it flags,
judged again with an all-group check and deployed when it passes; see
*Repair on refusal*), `repair_max_cells` (3, at least 1: with more flagged
base cells a round's bundle or a final retrain stays out whole),
`repair_max_attempts` (3, at least 1: the attempts of a round's repair
whose repaired bundle the audit set has not measured yet, or measured under
a weaker protocol than the refusal, before the round stays out whole).

**`report`**: `origami_reference` (false; true as `--origami-reference`),
`timeout_s` (3600; seconds the end-of-run report may take, 0: no limit),
`pool_gap_fraction` (0.3: a GEMM whose true best reaches less than this
fraction of its roofline is a pool gap; see *Weak spots*).

**`correctness`** (see *Correctness gate*): `enabled` (false: no gate),
`lists` ([]: files, or names of shipped lists, of kernels that compute wrong
results), `repetitions` (3) and `trigger_repetitions` (10) (at least 3),
`first_launch` (`custom`; `all`, `none`), `timeout_s` (600: seconds a
harness process may print nothing), `devices` (null: `runtime.devices`),
`cache_dir` (null: `<run dir>/correctness/cache`), `harness` (null: build
`tools/errmap.cpp`), `compiler` (null: `amdclang++` of `ROCM_PATH`, else
`hipcc`), `static_check` (true: the static check of hand-written kernels),
`custom_kernels` (null: TensileLite's `CustomKernels` of the hipBLASLt
source the build was configured from; a directory holding the hand-written
kernels' sources). `blocklist` is set by the orchestrator: the invocation's
copy of the gate's blocklist (which holds the hazard canary's quarantine),
or of the quarantine alone without the gate, which the stages apply; a
config that sets it with the gate off (a run's snapshot with the gate turned
off) is refused, since such a blocklist names what a gate found: give such a
file in `lists`. Let through are a blocklist of the hazard canary's entries
alone with the canary on (a snapshot of an invocation whose canary ran
without the gate, whose store keeps those verdicts: the invocation names its
own) and any blocklist in `--mode audit`, whose config's correctness
section lists the blocklist of the runs the set is built for alone.

**`hazards`** (see *Hazard canary*): `enabled` (true), `timeout_s` (300:
seconds a canary bench may print nothing), `runs` (1: bench processes per
kernel), `devices` (null: the first of `runtime.devices`; several run in
parallel, one kernel each), `cache_dir` (null: `<run dir>/hazards/cache`).

## Tests

```bash
python -m pytest            # from this directory; CPU only
```

The tests use stub stages, a fake `hipblaslt-bench` and fake Tensile
libraries; tests that need the tilewright module skip without it. Each test
gets its own lease directory and an empty fake sysfs (`tests/conftest.py`),
so it never takes a lease of the host or reads its GPUs; the lease, watch and
platform tests build fake KFD and DRM trees (`tests/gpu_fakes.py`), and the
device-value tests name a fake probe in `TILEWRIGHT_DEVICE_PROBE`.

`tests/test_onecmd_e2e.py` (marker `e2e`, part of the default run; `-m "not
e2e"` leaves it out) runs the one command on a production-shaped fake world
(`tests/onecmd_fakes.py`: a gfx950 BF16 TN library with a CU-specific bank
of the same problem type, the bench that merges them): the audit set, then
`--mode full` through every round and stage with the planner, the bank
(named by `planner.bank_library_stems` and by `library.kind`), trained,
re-planned and held-out contexts, the near-winner re-reads, splits,
the deploy gate, the audit evaluations and stage08, the same command on the
finished run, and runs killed with SIGKILL in the seed bench (also of a run
written before the planner became stage02) and the bench that the same
command resumes.

`TILEWRIGHT_GFX950_LIBRARY_DIR`, set to a built gfx950 library directory,
adds the tests that read it (`tests/test_bank_pool.py`,
`tests/test_kernel_identity.py`, `tests/test_library_pool.py`); they write
nothing there.

## Layout

```
run_orchestrator.py      orchestration, config schema, resume
configs/                 run configs; several can target one library (for
                         example full, int4 and smoke variants)
stages/                  stage00a, stage00 .. stage08
lib/                     shared modules (shapes, bench, data, features,
                         model file format, evaluation, ...)
tools/errmap.cpp         the correctness gate's exact harness
tools/device_props.cpp   preflight's probe of the device values HIP reports
blocklists/              shipped lists of kernels that compute wrong results
scripts/
    run_pipeline.sh      launcher
    make_report.py       HTML report of a run (sections in report_sections.py)
    audit_eval.py        models of finished rounds on an audit set
    audit_v2.py          audit protocol v2 and the sustained composite's
                         measurement: plan, bench, status
    choose_shipped_subset.py  bank kernels to ship with the library pool
    build_whitelist_tier.py   a trained bundle's whitelist tier, built again
    repair_bundle.py     repair the base cells the per-cell gate flags in
                         a trained bundle it keeps out
                         (lib/collapse_repair.py, lib/cell_merge.py)
    gpu_run.py           run a command on one GPU under its lease
    collect_bench_failures.py
    convert_mlrec_v1_to_v2.py
    experiments/         CPU microbenchmark of the engine's ranking call
tests/                   pytest suite
```
