# Packaged-library race sweeps

This hipBLASLt client tool runs selected solution indices and metadata-derived
shapes under rocJITsu. `--backend tensile` uses `tensilelite-client` with generated
INIs; `--backend bench` uses `hipblaslt-bench` with explicit algorithm indices and
YAML converted to binary data with the active Python, exercising the hipBLASLt API. Both require matching numerical, solution and
kernel-dispatch evidence. Every race or emulator warning fails the run.

The runner operates on supplied client, library and emulator artifacts. The
Tensile metadata reader has no generator dependency. The installed package also
includes the one-problem guide at
`tensilelite/Utilities/gemm_problem_for_solution.md`.

## Selection and shapes

The installed `*.dat` / `*.dat.zlib` metadata supplies solution indices, names,
hardware predicates and size constraints. Compressed metadata uses the same
strict framing as the native loader. No maintained inventory or previous
manifest is an input. Indices are specific to the tested artifact.

`--solution-index-range-begin BEGIN` and `--solution-index-range-end END` select
existing original solution IDs in the half-open interval `[BEGIN, END)`. Begin
defaults to zero. Omitting end selects through the catalogue's last index; omitting
both limits selects the entire catalogue. For example, `[4000, 8000)` requests
original IDs 4000 through 7999, and `[2205, 2206)` requests one index.

Selection happens before profile assignment and batching. It does not refer to
positions in an eligible or reordered list. Gaps in the catalogue do not renumber
later entries. Adjacent ranges cover disjoint sets of original IDs. An empty
selection is an error; an end beyond the catalogue is allowed. Repeated kernel
names, solution names and otherwise identical configurations at different IDs
remain separate requested entries. Changing names or file traversal order does
not change range membership.

Every existing index in the range stays in the manifest, including hardware
mismatches and unsupported features. These receive failed planning outcomes;
the runner never substitutes an index outside the range. The manifest and final
summary record the original range, and every input and result retains the original
solution index. Batch IDs are separate execution bookkeeping. There is no
solution sampling or sampling seed.

The `tile-depth-v1` adapters support ordinary GEMM with matching C/D types, optional
bias/scaling and no activation. Grouped, fused, sparse, MX, gradient, auxiliary
output and swizzled problems are outside their scope. Inventory counts include
unsupported features and other/unknown hardware. A selected solution that cannot
produce bounded cases remains a failure; it is never replaced by an easier one.
Explicit selection does not bypass these checks or any native predicates.

`--backend bench` defaults to `--policy regular-v1`: one full output tile from the
shared [one-problem constructor](https://github.com/ROCm/rocm-libraries/blob/110bcff2684645f9a2b965e8dd6620115d92e5c0/projects/hipblaslt/docs/how-to/choose-gemm-for-solution.md).
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

`--backend tensile` defaults to `--policy tile-depth-v1`. The policies have
different supported features and shapes, but both use the same original-index
range selection and retain every requested entry. Complete gfx950 execution
also needs the remaining feature adapters and compatible virtual device profiles.

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

Selection happens first. The bench runner first groups indices by their assigned hardware profile, then by
shard, orders the groups by their first selected index, and preserves selection
order within each group. Packaged lazy-library shards occupy disjoint index
ranges, so this orders them by range. It cuts the resulting list into batches
of at most `--batch-size B` solutions, allowing a batch to cross a shard boundary.
The default is 32. A batch never crosses a profile boundary. Each profile group
can finish with a smaller remainder batch; shard boundaries do not shorten batches. Neither batching nor worker count changes the selected indices
or the problem generated for each index. Original selection IDs stay in the
manifest, alongside the ordered `batches` list and each job's `batch_id`.

Workers claim the next batch under a shared lock, release the lock, and run one
native bench process for that batch. That process receives one binary input
containing the exact per-index YAML records, executes them sequentially, and
reuses native library and loaded code-object state. There is no solution/shape
cross product. Each worker uses one physical CPU core and one emulator thread.
The worker count defaults to four and supports up to 64, subject to available
physical cores.
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
The original selection and all planned cases stay in the manifest, with no
replacement solutions selected for skipped cases.

Each failure has a description, tracking links and a review date. Its `cases`
list can name `regular` for the one-GEMM policy or a subset of `tile`, `multi_tile`,
`mn_edge` and `k_remainder` for the four-case policy. Listing every case skips the
solution. Matching cases are omitted from the native INI/YAML.
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
Use it with the same original-index range to reassess the list when bumping
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
shortened to the remaining suite budget when `--suite-timeout` is supplied. There is no overall deadline by default. The optional suite clock starts before inventory preparation. Expiry kills
native process groups and records missing/unstarted cases. Preparation and report
writing are charged to the clock but are not forcibly interruptible; an external job timeout can provide an additional bound. The tile-depth policy bounds dimensions at 8192; both policies bound estimated data buffers
at 128 MiB and native workspace separately at 128 MiB. These are workload bounds,
not a hard process-memory limit.

Each backend writes into a fresh report directory:

- `summary.md`: solution index, individual outcome, numerical passes, dispatch counts,
  elapsed time and reasons. Updated atomically after each result and published
  in the GitHub job summary. Partial reports distinguish pending/incomplete work.
- `manifest.json`: inventory, selection, policy, shapes, predicates, input hashes,
  detector revision and known-failure matches.
- `known_failures.yaml`: the exact list used for this run, hashed in the manifest.
- `settings.json`, `rocjitsu.json` or `profile-*.json`: CPU assignment, environment and emulator settings.
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

The console reports each index when its batch finishes. Completed numerical records retain the norm error, tolerances, solution identity and target dispatch counts. `VALIDATED_IN_FAILED_PROCESS` means those per-index checks completed, while the native process failed a shared check. `LAUNCHED_NO_RESULT` means the target ran without producing a numerical record; `UNEXECUTED` means an aborted process never launched that input. Explicit native selection rejection reports `NATIVE_REJECTED`. Planning reports hardware exclusions, unsupported features, invalid inputs and missing code objects separately.

Race headers naming a unique selected kernel appear with that index. All race headers, warnings and process errors are also retained as process diagnostics; ambiguous or shutdown diagnostics are not assigned to a member index. `summary.json` has one `index_outcomes` entry per requested index, including unstarted work, plus separate `processes` and `failed_process_ids`. Its `outcome_counts` must not be replaced by a count of per-row `failed` flags: those flags retain process failures for the run's exit status.

Use `--recover-unexecuted` to replay each wholly unexecuted index once in its own native process. Recovery keeps the selected profile, exact YAML and checks, records both attempts, and respects any remaining suite deadline. It runs only when ordered dispatch and record evidence establishes that the original input never launched. Launched failures, native rejections and ambiguous logs are not retried automatically. The original process failure still makes the run exit unsuccessfully, even if every replay passes. Use `--continue-on-failure` as well to complete the remaining selected batches.

Replay the saved batch command and input to preserve the original process context. To investigate a launched failure, repeat the same tool/library/profile arguments with `--batch-size 1 --solution-index-range-begin N --solution-index-range-end N_PLUS_ONE` and a fresh reports directory. A successful isolated replay does not erase the original batch failure.

## Hardware profiles

`--config` retains the existing single-profile invocation. For profile selection, supply one or more complete emulator JSON configurations using repeated `--hardware-profile PATH` options. These replace `--config` as the candidate device profiles. The planner chooses the first compatible profile per solution, then orders metadata shards within each profile before batching. Profiles remain fixed within a native process, including recovery processes. A batch may cross shard boundaries within that profile.

Profile selection checks the architecture, positive integral CU resources, shader resources and the topology's total CU count. Supply configurations describing the intended device; the runner does not derive a PCI ID or change device resources from a solution predicate. Existing single-profile configurations retain their current behavior. The matcher uses the native directed chip fallback table: for example, gfx950 PCI 0x75a3 can use 0x75a0 solutions, while 0x75a0 cannot use 0x75a3 solutions. A single compatible profile can therefore cover several chip predicates. All supplied profiles and their hashes, normalized execution configurations and per-index assignments are recorded in the manifest.

## Running locally

Use Linux, Python 3.11+, `msgpack`, `PyYAML`, `taskset` and matching client/runtime
artifacts. Install the Python dependencies from `requirements.txt`. CMake's
`HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` installs the runner, guide and tests
under `share/hipblaslt/tensilelite/rocjitsu`.

Choose a complete emulator configuration for the intended hardware. Stock
configurations live in rocJITsu's `configs/` directory. `--config PATH` supplies
one profile; repeated `--hardware-profile PATH` supplies candidate profiles
without also requiring `--config`. These forms are mutually exclusive. The runner
only considers supplied profiles and does not discover or create missing variants.

Set `PYTHONPATH` to the installed TensileLite package root and ensure the client
loads the runtime matching the tested library. Set `HARDWARE_PROFILE` to a
configuration covering the requested indices; the example does not create it.

```bash
export PYTHONPATH="$ROCM_PATH/share/hipblaslt/tensilelite${PYTHONPATH:+:$PYTHONPATH}"
sweep="$ROCM_PATH/share/hipblaslt/tensilelite/rocjitsu"
python3 -B "$sweep/test_rocjitsu_race_sweep.py"
python3 -B "$sweep/rocjitsu_race_sweep.py" \
  --backend bench \
  --rocjitsu "$ROCJITSU_BUILD_DIR/tools/rocjitsu/rocjitsu" \
  --rocjitsu-revision "$(git -C "$ROCJITSU_SOURCE_DIR" rev-parse HEAD)" \
  --client "$ROCM_PATH/bin/hipblaslt-bench" \
  --config "$HARDWARE_PROFILE" \
  --library-dir "$ROCM_PATH/lib/hipblaslt/library" \
  --target gfx950 --reports "$RACE_REPORT_DIR/sweep-bench" \
  --solution-index-range-begin 4000 --solution-index-range-end 8000 \
  --workers 4 --batch-size 32 --continue-on-failure
```

For `tensilelite-client`, use `--backend tensile` and its client path. It requires
batch size one and retains the four-case policy. Bench's one-GEMM policy fixes
preferred reduction work at eight blocks; there is no work-size CLI option.

`--timeout SECONDS` is a finite positive allowance per runnable index, defaulting
to 120 seconds. It is multiplied by the runnable batch size, shared by conversion
and native execution, and enforced by terminating the process group. It is not an
individual timer for each index. There is no fixed maximum. A large local sweep
can optionally use `--suite-timeout 7200` for a two-hour overall budget. A CI job
can supply its own shorter budget. Omitting that option retains batch hang
protection while allowing the full range to run without an overall deadline.
