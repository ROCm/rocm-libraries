# Packaged-library race sweeps

This hipBLASLt client tool runs selected solution indices and metadata-derived
shapes under rocJITsu. `--backend tensile` uses `tensilelite-client` with generated
INIs; `--backend bench` uses `hipblaslt-bench` with explicit algorithm indices and
YAML converted to binary data with the active Python, exercising the hipBLASLt API. Both require matching numerical, solution and
kernel-dispatch evidence. Every race or emulator warning fails the run.

The advisory GitHub Actions sidecar invokes both backends independently against
the current run's installed artifacts. Each reconstructs its own inventory, runs up to
eight workers and keeps separate reports; CI uses four. A failure in the first backend does
not prevent the second. CI setup, invocation and report publication belong in
`.github`; inventory, case generation, execution and result checking live here.
The Tensile metadata reader and client-option mapping are shared with existing
component tools through `tensilelite.Utilities` without importing the generator.
The installed package also includes the one-problem guide at
`tensilelite/Utilities/gemm_problem_for_solution.md`.

## Selection and shapes

The installed `*.dat` / `*.dat.zlib` metadata supplies solution indices, names,
hardware predicates and size constraints. Compressed metadata uses the same
strict framing as the native loader. No maintained inventory or previous
manifest is an input. Indices are specific to the tested artifact.

CI uses the fixed seed `0` for both backends and selects 100 eligible solution
entries. A kernel name can have several solution names with different launch
settings, and a solution name can occur at several indices. Every entry is
retained, including repeated kernel names, repeated solution names and identical
configurations at different indices.

Entries are ranked by `SHA256(seed + "\0" + solution_name)`, then by solution name
and canonical JSON of the solution's configuration metadata. This includes the
kernel name, problem type, size mapping, predicates and kernel argument format;
index/logic bookkeeping and performance models are excluded. JSON keys are sorted,
so dictionary insertion order does not affect selection. The numeric index breaks
ties only when these names and configuration fields are identical. The first 100
entries are selected in this order, independently of metadata filenames,
traversal order within the eligible inventory for each case policy. Index renumbering preserves the chosen configurations; naming
or configuration changes can change the selection. Adding or removing entries
can change which fall within the selected prefix.

Increasing `--solutions N` extends the same ordered prefix for an unchanged
eligible inventory, even beyond the number of distinct kernel names.
`--all-solutions` selects every eligible index. Process and stage time limits
still apply; explicit indices can partition larger inventories into separate runs.

The CLI requires `--seed`; use `--seed 0` to reproduce CI or an explicit alternate
seed for local exploration. To reproduce specific indices, repeat
`--solution-index N` instead of `--solutions N`. Explicit selection retains aliases,
reports unsupported solutions and rejects missing indices.

The `tile-depth-v1` adapters support ordinary GEMM with matching C/D types, optional
bias/scaling and no activation. Grouped, fused, sparse, MX, gradient, auxiliary
output and swizzled problems are outside their scope. Inventory counts include
unsupported features and other/unknown hardware. A sampled solution that cannot
produce bounded cases remains a failure; it is never replaced by an easier one.
Explicit selection does not bypass these checks or any native predicates.

CI uses `--backend bench --policy regular-v1`: one full output tile from the
shared [one-problem constructor](../../../docs/how-to/choose-gemm-for-solution.md).
K is `8 * DepthU * fixed_split` for ordinary solutions and `8 * DepthU` for
Stream-K or persistent DataParallel. The native runtime assigns persistent work;
this small shape does not exercise every Stream-K scheduling/reduction path.
Optional epilogues are disabled. This supports double, complex, FNUZ and MX
operands as well as the initial real types. MX block sizes, scale datatypes and
scale layout come from metadata; the native client initializes the operands and
scales and computes the numerical reference. MX uses bounded HPL input values.
Automatic split-K and the descriptor's other
unimplemented features are counted as coverage gaps by reason; malformed
metadata fails preparation. A selected workload that exceeds the buffer budget
or fails native predicates remains selected and fails the run.

The independent TensileLite stage retains `--policy tile-depth-v1`. These two
policies have different eligible inventories and shapes. Both retain aliases,
use the same seed and report their policy explicitly. Complete gfx950 coverage
will additionally need the remaining feature adapters and virtual device profiles.

The daily `therock-multi-arch-ci-nightly.yml` calls the reusable CPU race workflow
for gfx942. It consumes BLAS/test artifacts from the same GitHub run and waits
for the complete Linux build/test workflow. An unrelated GPU test failure does
not suppress the artifact attempt. The race job is advisory during burn-in and
is excluded from the required nightly summary.

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

## Batching and shard reuse

A shard is one metadata file and its matching code-object file. The manifest
already maps each selected solution index to these files through `library` and
`code_objects`. Indices and shard ranges belong to the exact tested artifact.

Selection happens first. The bench runner then groups the selected indices by
shard, orders the groups by their first selected index, and preserves selection
order within each group. Packaged lazy-library shards occupy disjoint index
ranges, so this orders them by range. It cuts the resulting list into batches
of at most `--batch-size B` solutions, allowing a batch to cross a shard boundary.
The default is 32. Neither batching nor worker count changes the selected indices
or the problem generated for each index. Original selection IDs stay in the
manifest, alongside the ordered `batches` list and each job's `batch_id`.

Workers claim the next batch under a shared lock, release the lock, and run one
native bench process for that batch. That process receives one binary input
containing the exact per-index YAML records, executes them sequentially, and
reuses native library and loaded code-object state. There is no solution/shape
cross product. Each worker uses one physical CPU core and one emulator thread.
Workers claim batches in shard order, but operating-system scheduling and
unequal runtimes can change their start and completion order. There is no
barrier between shards. The TensileLite backend requires batch size 1 because
its INI adapter handles one solution and its cases.

For a batch within one shard, a useful cost model is `T(B) = X + B*Y`, where `X`
is reusable startup/loading cost and `Y` is incremental execution and validation
cost per index. Average cost is `Y + X/B`. Loading takes less than half the time
when `B > X/Y`, and less than 10% when `B > 9*X/Y`. A batch crossing shards pays
the loading cost of each distinct shard. These quantities vary with the library,
workload and machine; the model guides measurement rather than predicting all
indices from one shard.

Larger batches reduce repeated startup but leave fewer opportunities to balance
work between workers. The default B=32 is based on a 1,000-index gfx942 sample
with eight workers and all numerical, race and memory-wait checks enabled:

| Batch size | Native processes | Execution | Including preparation |
|---|---:|---:|---:|
| 8 | 125 | 195.43 s | 218.44 s |
| 16 | 63 | 164.59 s | 187.80 s |
| 32 | 32 | 128.81 s | 151.67 s |

Every run completed all 1,000 numerical and exact-dispatch checks. Each reproduced
the same diagnostic at index 119763; the affected batch remained failed. These
are single measurements on a shared host, not a full-library performance guarantee.

For smaller selections, reducing B can keep more workers occupied. With 200
indices and eight workers, B=8 gives 25 batches, B=16 gives 13, and B=32 gives
only seven. Use `--batch-size` to compare settings for the intended workload.
The batch result files record conversion, native execution and total elapsed
seconds. Per-solution rows display their shared batch time, not a measured
individual runtime; do not sum those rows to estimate total work.

## Known failures

[`known_failures.yaml`](known_failures.yaml) is the executable skip list. After
selection and shape generation, the runner matches each case by architecture,
backend, exact kernel name and case ID. An entry covers all selected solution
names, indices and launch configurations using that kernel. This intentionally
coarse scope keeps the list independent of solution naming and index assignment;
it can also skip configurations that would pass. Rebuilding a metadata/code-object
file does not by itself remove the skip, although a changed kernel name no longer
matches. File hashes remain in `manifest.json`'s `input_sha256` map for reproduction.
The original selection and all four planned cases stay in the manifest, with no
replacement solutions selected for skipped cases.

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
different list for local diagnosis; the default is the list for the selected policy.

Both policy lists are empty at the pinned detector revision. The 26 former
`s_atomic_dec` exclusions passed all 104 `tile-depth-v1` cases after
[instruction support](https://github.com/ROCm/rocm-systems/pull/12963) landed.
The `regular-v1` policy has its own [`known_failures_regular.yaml`](known_failures_regular.yaml).
New findings fail the run and require diagnosis before any exclusion is added.

## Evidence and bounds

TensileLite validates every output element with one target dispatch per case.
Bench enables native norm assertions and allclose measurements, with one
validation dispatch for `regular-v1`; `tile-depth-v1` also has one timed dispatch. It uses `--host_side_fill_kernel` to evaluate
the existing matrix initializers on the CPU and copy the inputs to the device,
avoiding emulated input-fill kernels. The option applies to every YAML/data
record; ordinary bench runs continue to initialize on the device by default.
Both parsers require every case sent to the client,
matching solution names/indices and exact kernel names/symbols from race-plugin
dispatch logs. A successful exit or numerical result alone cannot establish PASS.
The human-readable client output is an interface dependency; format changes must
update the parsers, and missing evidence fails the run.

Workers use separate physical cores, with emulator/OpenMP/BLAS thread budgets of
one. On failure, workers stop claiming batches and finish their in-flight batches.
`--continue-on-failure` instead processes the remaining selection while retaining
failures and an unsuccessful final exit. It is useful for complete timing samples.
A batch's time allowance is `--timeout` (default 120 seconds) multiplied by the
number of runnable solutions. Conversion and native execution share that deadline,
shortened to the remaining 25-minute suite budget. The suite clock starts before inventory preparation. Expiry kills
native process groups and records missing/unstarted cases. Preparation and report
writing are charged to the clock but are not forcibly interruptible; the outer
CI timeout still applies. The tile-depth policy bounds dimensions at 8192; both policies bound estimated data buffers
at 128 MiB and native workspace separately at 128 MiB. These are workload bounds,
not a hard process-memory limit.

Each backend writes into a fresh report directory:

- `summary.md`: solution index, PASS/FAIL/BATCH FAILED/SKIPPED/PASS WITH SKIPS, numerical passes, dispatch counts,
  elapsed time and reasons. Updated atomically after each result and published
  in the GitHub job summary. Partial reports distinguish pending/incomplete work.
- `manifest.json`: inventory, selection, policy, shapes, predicates, input hashes,
  detector revision and known-failure matches.
- `known_failures.yaml`: the exact list used for this run, hashed in the manifest.
- `settings.json`, `rocjitsu.json`: CPU assignment, environment and emulator settings.
- `batch-*`: per-process INI/YAML, binary bench input, command, combined log,
  timing breakdown and classified native results.
- `solution-*.result.json`: per-index evidence and the corresponding `batch_id`.
- `progress.json`, `summary.json`: incremental/final accounting and first-failure
state. Setup/finalization errors also leave `setup-or-run-error.json`.

A successful run with skips reports `PASS WITH SKIPS`. In `summary.json`,
`passed` allows these tracked skips; `fully_validated` additionally requires
every planned case to have run successfully. Numerical passes keep the original
planned-case denominator; expected dispatch counts exclude skipped cases. Final
accounting includes all observed statuses, skipped cases, missing records and
unstarted cases. Skips do not establish numerical or race coverage.

The console reports each solution when its batch finishes. The parser checks
numerics, index, solution name, kernel name and dispatch count separately for
each input record, including aliases that share a kernel name and shape.
A process failure, timeout, race, warning or inconsistent batch record/dispatch
count prevents the affected batch from passing. Its rows report `BATCH FAILED`,
and the report counts failed batches, not failed member indices. The structured
summary lists `failed_batch_ids`; a per-index `failed` flag includes these shared
batch checks and must not be counted as an index that produced a race or bad numerics.
Diagnostics before the first
record and during shutdown count too; they cannot reliably be attributed to
one index. Completed numerical records remain visible even when the batch fails.
Missing output in a claimed batch is reported as missing; unclaimed batches are
reported as unstarted. No automatic retry replaces the original result.

Replay the saved batch command and input to preserve the original process context.
To isolate a failure, repeat the same tool/library/configuration arguments with
`--batch-size 1 --solution-index N` and a fresh reports directory. A successful
isolated replay does not erase the original batch failure. Distinguish emulator
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
  --workers 4 --solutions 100 --timeout 120 --suite-timeout 1500
```

For the nightly bench policy, change `--backend` to `bench`, add
`--policy regular-v1 --known-failures "$sweep/known_failures_regular.yaml"`, set
`--client "$ROCM_PATH/bin/hipblaslt-bench"` and use a fresh `--reports` directory.
Use `--batch-size 32` (the bench default) to share startup across indices, or
`--batch-size 1` for a separate process per index. For specific
solutions, replace `--solutions 100` with e.g. `--solution-index 7 --solution-index 9`.
For source-tree development, put `projects/hipblaslt/tensilelite` on `PYTHONPATH`
and use this directory as `sweep`.

This sidecar targets customer GEMMs and their library helper kernels. It replaces
the old fixed GEMM checks; client initialization kernels and the Tensile Python
generation pipeline are outside its coverage goals. Numerical validation and
exact dispatch evidence remain required for both sweep paths.

Keep checks enabled when comparing batch sizes. Use the same seed, artifact,
selected count, worker count and time limits, and verify identical per-index
inputs in the manifests. Compare `summary.json` execution and suite times;
inspect missing/unstarted cases and failed results before interpreting speedups.
