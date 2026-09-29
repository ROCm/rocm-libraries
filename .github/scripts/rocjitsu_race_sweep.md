# Packaged-kernel toy race sweep

The advisory `therock-rocjitsu-race-check-linux.yml` job runs this experiment
after its hipblaslt-bench and generated TensileLite smoke checks. It reuses the
job's rocJITsu build and PR-produced TensileLite client, ROCm runtime and
hipBLASLt device library. The current caller supplies **gfx942** artifacts.
The existing gfx950 lane remains disabled until presubmit supplies that target.

The scope is **four Python worker threads, 100 distinct kernel names and four
cases per solution**: M=N=256, batch=1, K=64/128/256/320, alpha=1, beta=0.
These provisional shapes do not establish four distinct control-flow paths.
No device kernels are generated for this stage.

## Selection and execution

The runner reads native MessagePack `.dat` or `.dat.zlib` shards, checks the
target/PCI/CU hardware predicates against the emulator config and selects
contiguous solution-index batches of at most ten. Selection is deterministic,
round-robin across eligible metadata groups, with distinct kernel names across
the sweep. It covers a limited set of ordinary datatypes, bias and scaling;
MX, sparse, grouped, gradient and auxiliary-output problems are excluded.
Inventory counts record these exclusions. This is a small compatibility probe,
not representative sampling or an exhaustive kernel inventory test.

Each native client loads the original metadata shard and code objects with
native problem predicates enabled. A rejected case fails the experiment; the
runner does not bypass predicates or replace failing kernels. The native
client and library must come from a compatible build.

Each worker launches one emulator/client subprocess at a time, pinned to a
distinct available physical core. ROCr helpers inherit that affinity;
OpenMP/BLAS and emulator thread budgets are one. Ten-solution subprocesses
bound retained emulator state. Each process group has a 120-second wall
timeout; the emulator tick limit is disabled so it cannot truncate this
four-case workload. Memory is not capped by the runner.

## Results and failures

The stage succeeds only when all 400 numerical records pass, the expected
solution names and target dispatch counts match, and there are no race reports
or emulator warnings. Runtime/loading-copy reports also fail; there is no
known-race allowlist. Numerical success alone does not mean race-free or
complete path coverage.

On failure, workers stop claiming batches and finish already assigned work
within its timeout. The final summary accounts for completed, missing and
unstarted cases, including failures in worker or progress callbacks. The
advisory job retains its existing `continue-on-error` behavior.

The existing `rocjitsu-race-reports-*` artifact includes `toy-sweep/`:

- `manifest.json`: exact indices, solution/kernel names, shapes, metadata,
  code objects, tool/config hashes and inventory exclusions.
- `rocjitsu.json` and `settings.json`: emulator settings, CPU assignment and
  controlled environment values.
- `batch-*.ini`, `.command.json`, `.log`, `.csv`, `.result.json`: native client
  inputs, argv, combined stdout/stderr, client CSV and classified results.
- `progress.json` and `summary.json`: incremental results and final accounting,
  with the first failing batch and number of in-flight jobs when work stopped.
- `setup-or-run-error.json`: setup or finalization errors, when writable.

For each new finding, preserve a failing batch and reduce it to one solution or
a tiny instruction-level reproducer before proposing an emulator fix. Record
the tool/artifact revisions, observed behavior, suspected component and a
regression criterion. Keep subsequent experiments within the same toy limits.

## Local reproduction

Use Python 3.11 or newer with `msgpack` (already in the CI artifact's
`requirements-test.txt`), Linux `taskset`, and at least four available physical
cores. Export the runtime library paths for the same artifact tree as the
client, as the existing shell driver does. Use a new report directory per run.

```bash
python3 -B .github/scripts/test_rocjitsu_race_sweep.py
python3 -B .github/scripts/rocjitsu_race_sweep.py \
  --rocjitsu "$ROCJITSU_BUILD_DIR/tools/rocjitsu/rocjitsu" \
  --client "$ROCM_PATH/libexec/hipblaslt/tensilelite/tensilelite-client" \
  --config "$ROCJITSU_SOURCE_DIR/configs/gfx942_cdna3_kmd.json" \
  --library-dir "$ROCM_PATH/lib/hipblaslt/library" \
  --target gfx942 --reports "$RACE_REPORT_DIR/toy-sweep" \
  --workers 4 --kernels 100 --batch-size 10 --timeout 120
```

The CLI permits smaller diagnostic runs, but caps workers at four, kernels at
100, batch size at ten and timeout at 120 seconds. To replay one saved batch,
execute its `.command.json` argv with the settings in `settings.json` and the
same compatible artifact tree; the INI contains absolute artifact paths.
