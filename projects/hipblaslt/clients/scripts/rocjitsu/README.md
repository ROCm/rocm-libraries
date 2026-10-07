# Packaged-library race sweeps

This hipBLASLt client tool runs selected solution indices and metadata-derived
shapes under rocJITsu. `--backend tensile` uses `tensilelite-client` with generated
INIs; `--backend bench` uses `hipblaslt-bench` with explicit algorithm indices and
YAML, exercising the hipBLASLt API. Both require matching numerical, solution and
kernel-dispatch evidence. Every race or emulator warning fails the run.

The advisory GitHub Actions sidecar invokes both backends independently against
the PR's installed artifacts. Each reconstructs its own inventory, runs up to
four workers and keeps separate reports. A failure in the first backend does
not prevent the second. CI setup, invocation and report publication belong in
`.github`; inventory, case generation, execution and result checking live here.
The Tensile metadata reader and client-option mapping are shared with existing
component tools through `Tensile.Utilities` without importing the generator.

## Selection and shapes

The installed `*.dat` / `*.dat.zlib` metadata supplies solution indices, names,
hardware predicates and size constraints. Compressed metadata uses the same
strict framing as the native loader. No maintained inventory or previous
manifest is an input. Indices are specific to the tested artifact.

CI uses the fixed seed `0` for both backends. The sample contains 100 distinct
eligible kernel names, with one solution index chosen for each. Several solution
indices can refer to the same kernel name; those aliases do not increase that
kernel's sampling weight.

Each name receives the priority `SHA256(seed + "\0" + kernel_name)`. The 100 names
with the smallest priorities are selected in priority order. Within each name,
the representative solution has the smallest
`SHA256(seed + "\0" + solution_name)`. The solution name identifies the kernel and
its launch settings, so index renumbering preserves the chosen configuration.
Identical names use metadata basename and index only to choose an executable handle.
Selection is independent of traversal order, backend and case generation.
For an unchanged eligible inventory, a fixed seed preserves the selection across
PR revisions. Increasing the count extends the same ordered sample. Selecting
every kernel name would still choose one representative index per name.

The CLI requires `--seed`; use `--seed 0` to reproduce CI or an explicit alternate
seed for local exploration. To reproduce specific indices, repeat
`--solution-index N` instead of `--kernels N`. Explicit selection retains aliases,
reports unsupported solutions and rejects missing indices.

The initial adapters support ordinary GEMM with matching C/D types, optional
bias/scaling and no activation. Grouped, fused, sparse, MX, gradient, auxiliary
output and swizzled problems are outside their scope. Inventory counts include
unsupported features and other/unknown hardware. A sampled solution that cannot
produce bounded cases remains a failure; it is never replaced by an easier one.
Explicit selection does not bypass these checks or any native predicates.

`rocjitsu_sweep_plan.py` owns the versioned `tile-depth-v1` case policy. From each
solution's macro tile `(Tm, Tn)` and DepthU `Du`, it proposes four shapes:

| Intent | M | N | K |
|---|---:|---:|---:|
| One tile | Tm | Tn | Du |
| Multiple tiles | 2 Tm | 3 Tn | 2 Du |
| M/N edge | Tm + M alignment | Tn + N alignment | 4 Du |
| K remainder | 2 Tm | 2 Tn | 4 Du + K alignment |

Batch starts at one. Common minima, multiples, equalities, upper bounds and
split-K constraints adjust the proposals. Cases must remain distinct. The
manifest records actual shapes and M/N/K tails; a label does not guarantee a
particular control-flow path. Native clients enforce the full predicate set,
and a native rejection fails the run.

Each process handles one solution and its cases. This avoids a cross product of
unrelated solutions/shapes. Explicit indices provide a route to partitioning an
entire artifact by solution index, including aliases. Broader coverage then needs
additional adapter support and case policies, with unsupported and unstarted
cases still accounted for. It does not require another runner or scheduler.

## Known failures

[`known_failures.yaml`](known_failures.yaml) is the executable skip list. After
selection and shape generation, the runner matches each case by architecture,
backend, solution name and case ID. The solution name identifies the kernel and
its launch configuration. An entry covers only that named configuration, not
every solution sharing its kernel. The current numeric index is resolved from
the artifact, so renumbering or rebuilding a metadata/code-object file does not
by itself remove the skip. File hashes remain in `manifest.json`'s `input_sha256`
map for reproduction. The original selection and all four planned cases stay in
the manifest.

Each failure has a description, tracking links and a review date. Its `cases`
list can name a subset of `tile`, `multi_tile`, `mn_edge` and `k_remainder`; listing
all four skips the solution. Matching cases are omitted from the native INI/YAML.
If all cases match, no client process is launched. A skip records `SKIPPED`, its
reason and tracking links, with zero numerical passes or dispatches. Workers
continue to the next solution. Planning errors and failures in executed cases
still stop new work.

The list is reviewed against one rocJITsu revision and case policy. Supply the
detector's full source revision with `--rocjitsu-revision`; CI obtains it from
the checkout used to build the detector. A revision/policy mismatch requires
rechecking. The `review_by` date is a maintenance reminder and does not change
execution as time passes. `--run-known-failures` disables skips and the
revision/policy guards while retaining every normal validation check.
Use it with the same seed or explicit indices to reassess the list when bumping
the detector pin. Remove passing entries and document/review remaining failures
before updating the list's revision and date. `--known-failures PATH` selects a
different list for local diagnosis; the default is the file beside the script.

The current entries cover gfx942 `s_atomic_dec` hangs, tracked by
[instruction support](https://github.com/ROCm/rocm-systems/pull/12963) and
[fatal-error propagation](https://github.com/ROCm/rocm-systems/pull/12964).
The previous LDS/FLAT wait false positives are fixed in the new pin and have no
skip entries. Newly selected indices may still expose unlisted failures.

## Evidence and bounds

TensileLite validates every output element with one target dispatch per case.
Bench enables native norm assertions and allclose measurements, with one
validation and one timed dispatch. It uses `--host_side_fill_kernel` to evaluate
the existing matrix initializers on the CPU and copy the inputs to the device,
avoiding emulated input-fill kernels. The option applies to every YAML/data
record; ordinary bench runs continue to initialize on the device by default.
Both parsers require every case sent to the client,
matching solution names/indices and exact kernel names/symbols from race-plugin
dispatch logs. A successful exit or numerical result alone cannot establish PASS.
The human-readable client output is an interface dependency; format changes must
update the parsers, and missing evidence fails the run.

Workers use separate physical cores, with emulator/OpenMP/BLAS thread budgets of
one. On failure, workers stop assigning jobs and finish their in-flight work.
Each process group has a 120-second timeout, shortened to the remaining 25-minute
suite budget. The suite clock starts before inventory preparation. Expiry kills
native process groups and records missing/unstarted cases. Preparation and report
writing are charged to the clock but are not forcibly interruptible; the outer
CI timeout still applies. Dimensions are bounded at 8192, estimated data buffers
at 128 MiB and native workspace separately at 128 MiB. These are workload bounds,
not a hard process-memory limit.

Each backend writes into a fresh report directory:

- `summary.md`: solution index, PASS/FAIL/SKIPPED/PASS WITH SKIPS, numerical passes, dispatch counts,
  elapsed time and reasons. Updated atomically after each result and published
  in the GitHub job summary. Partial reports distinguish pending/incomplete work.
- `manifest.json`: inventory, selection, policy, shapes, predicates, input hashes,
  detector revision and known-failure matches.
- `known_failures.yaml`: the exact list used for this run, hashed in the manifest.
- `settings.json`, `rocjitsu.json`: CPU assignment, environment and emulator settings.
- `batch-*`: per-solution INI/YAML, command, combined log and classified result.
- `progress.json`, `summary.json`: incremental/final accounting and first-failure
state. Setup/finalization errors also leave `setup-or-run-error.json`.

A successful run with skips reports `PASS WITH SKIPS`. In `summary.json`,
`passed` allows these tracked skips; `fully_validated` additionally requires
every planned case to have run successfully. Numerical passes keep the original
planned-case denominator; expected dispatch counts exclude skipped cases. Final
accounting includes all observed statuses, skipped cases, missing records and
unstarted cases. Skips do not establish numerical or race coverage.

The console reports each completed solution. Reduce findings using its saved
command and inputs, preserving artifact/tool revisions. Distinguish emulator
findings, native rejection, packaging problems and possible kernel races.

## Running locally

Use Linux, Python 3.11+, `msgpack`, `PyYAML`, `taskset` and matching client/runtime
artifacts. Install the Python dependencies from `requirements.txt`. CMake's
`HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` installs this tool and its tests under
`share/hipblaslt/tensilelite/rocjitsu`. Set runtime library paths as in
`.github/scripts/run_rocjitsu_hipblaslt_race_check.sh`.

```bash
export PYTHONPATH="$ROCM_PATH/share/hipblaslt/tensilelite${PYTHONPATH:+:$PYTHONPATH}"
sweep="$ROCM_PATH/share/hipblaslt/tensilelite/rocjitsu"
python3 -B "$sweep/test_rocjitsu_race_sweep.py"
python3 -B "$sweep/rocjitsu_race_sweep.py" \
  --backend tensile --seed 0 \
  --rocjitsu "$ROCJITSU_BUILD_DIR/tools/rocjitsu/rocjitsu" \
  --rocjitsu-revision "$(git -C "$ROCJITSU_SOURCE_DIR" rev-parse HEAD)" \
  --client "$ROCM_PATH/libexec/hipblaslt/tensilelite/tensilelite-client" \
  --config "$ROCJITSU_SOURCE_DIR/configs/gfx942_cdna3_kmd.json" \
  --library-dir "$ROCM_PATH/lib/hipblaslt/library" \
  --target gfx942 --reports "$RACE_REPORT_DIR/sweep-tensile" \
  --workers 4 --kernels 100 --timeout 120 --suite-timeout 1500
```

For bench, change `--backend` to `bench`, `--client` to
`$ROCM_PATH/bin/hipblaslt-bench` and use a fresh `--reports` directory. For specific
solutions, replace `--kernels 100` with e.g. `--solution-index 7 --solution-index 9`.
For source-tree development, put `projects/hipblaslt/tensilelite` on `PYTHONPATH`
and use this directory as `sweep`.

This sidecar targets customer GEMMs and their library helper kernels. It replaces
the old fixed GEMM checks; client initialization kernels and the Tensile Python
generation pipeline are outside its coverage goals. Numerical validation and
exact dispatch evidence remain required for both sweep paths.

Measurements should guide further speed work: the main candidates are repeated
metadata decoding, client/module startup and remaining runtime fill/copy work.
Bench's second GEMM dispatch remains necessary for its current validation and
timing path. The current small scheduler needs no general-purpose tuning framework.
