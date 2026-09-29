# Independent sampled race sweeps

The advisory `therock-rocjitsu-race-check-linux.yml` job runs two independent
packaged-kernel tests after its existing smoke checks:

- `--backend tensile`: drives the artifact's `tensilelite-client` with native
  solution metadata and generated INIs.
- `--backend bench`: drives the artifact's `hipblaslt-bench` with generated
  benchmark YAML and explicit algorithm indices, exercising hipBLASLt's API.

Each backend has its own invocation, inventory reconstruction, queue, reports
and exit status. A failure in either never prevents the other from running.
The stages run sequentially, each with up to **four workers and 100 sampled
kernels**, so concurrency stays at four. The current CI caller supplies gfx942.

## Inventory and random selection

The tested artifact is the only source of kernel inventory. At runtime, each
backend scans its `*.dat` / `*.dat.zlib` metadata, counts solutions and distinct
kernel names, and checks native hardware predicates against the emulator config.
No checked-in kernel list, exported CSV or previous run's manifest is an input.
The generated manifest is an audit/reproduction artifact, not maintained data.

`ROCJITSU_SWEEP_SEED` overrides the CI seed; otherwise the driver uses
`GITHUB_SHA` (or `local` outside CI). The CLI requires an explicit `--seed`.
Seeded hash priorities sample distinct kernel names without replacement,
independent of metadata traversal order. A separate seeded priority chooses
one representative solution when several indices reference the same kernel.
Both backends receive the same seed so their samples/cases are comparable,
without consuming each other's output. Changing the PR revision changes the
sample; the recorded seed allows replay against the same artifact.

The initial adapter scope includes ordinary supported GEMM datatypes with
matching C/D types, optional bias/scaling, and no activation. Grouped, sparse,
MX, gradient, auxiliary-output and swizzled problems are outside this adapter.
Other/unknown hardware and unsupported problem features are counted explicitly.
This is a random sample within that scope, not a whole-library coverage claim.

## Cases derived from each sampled solution

`rocjitsu_sweep_plan.py` owns the versioned `tile-depth-v1` policy. It derives
four candidates from each solution's macro tile `(Tm, Tn)` and unroll depth `Du`:

| Intent | M | N | K |
|---|---:|---:|---:|
| One tile | Tm | Tn | Du |
| Multiple tiles | 2 Tm | 3 Tn | 2 Du |
| M/N edge | Tm + M alignment | Tn + N alignment | 4 Du |
| K remainder | 2 Tm | 2 Tn | 4 Du + K alignment |

Batch starts at one. Common native minimum, multiple, equality, upper-bound and
split-K minimum constraints adjust these candidates. The manifest records the
actual shapes and whether M/N/K tails remain after adjustment; the labels do
not promise particular control-flow paths. Cases must remain distinct and fit
the dimensional/data-buffer limits. Full path coverage is a separate policy.

Selection happens before case generation. An unsupported or oversized selected
solution remains in the manifest with four `UNSUPPORTED_CASE` outcomes when
assigned; it is never replaced by an easier kernel. The policy does not
implement the complete native predicate language. All original predicates
remain enabled in the clients, and a native rejection fails the stage.

Each subprocess handles one sampled solution and its four cases. This preserves
per-solution shapes and avoids the TensileLite common-problem-list cross product.
Grouping compatible solutions for loading efficiency can be added without
changing inventory, sampling, case generation or accounting.

## Validation and execution bounds

TensileLite performs full-output numerical validation with one target dispatch
per case. Bench requests explicit indices, norm checks with native tolerance
assertions, allclose measurements, no optional warmups, and one timed iteration.
That means one validation and one timed target dispatch per bench case. Both
parsers require per-case numerical evidence, matching solution/kernel identities
and the expected target dispatch counts. Bench's YAML schema/generator must be
packaged beside its executable, and its embedded HIP initialization kernels
must support the selected architecture.

Every race report or emulator warning fails its stage, including runtime/helper
reports. Workers stop claiming jobs after failure and finish assigned work
within the timeout. Final accounting includes completed, unsupported, missing
and unstarted cases. A client's exit status alone cannot establish coverage.

Each worker is pinned to a distinct physical core; ROCr helpers inherit affinity
and OpenMP/BLAS/emulator thread budgets are one. Each process group has a
120-second timeout. Generated cases have a conservative 128 MiB estimated
buffer limit and dimensions at most 8192; client workspace is separately capped
at 128 MiB. These are workload limits, not a hard process-memory cap. The
emulator tick limit is disabled in favor of the subprocess wall timeout.

## Artifacts and follow-ups

The existing uploaded race-report artifact contains `sweep-tensile/` and
`sweep-bench/`, each with:

- `manifest.json`: reconstructed inventory counts, seed, policy version, exact
  sampled identities, size mappings/predicates, derived cases and input hashes.
- `rocjitsu.json`, `settings.json`: emulator settings, CPUs and controlled runtime
  environment. Both backends use the same artifact library directory.
- `batch-*.ini` or `.yaml`, `.command.json`, `.log`, `.result.json`: exact inputs,
  argv, combined output and classified outcomes. TensileLite also writes CSV.
- `progress.json`, `summary.json`: incremental/final accounting and first-failure
  stop state. Setup/finalization errors leave `setup-or-run-error.json` if writable.

Reduce findings to one sampled solution or a tiny instruction sequence before
proposing a rocJITsu fix. Retain tool/artifact revisions, commands, observed
behavior and regression criteria. Distinguish emulator findings from native
predicate rejection, client packaging problems and kernel race candidates.

## Reproduction

Use Linux, Python 3.11+, `msgpack`, `PyYAML`, `taskset` and matching client/runtime
artifacts. The dependencies already come from TensileLite's CI test requirements.
Export the artifact's runtime library paths as the shell driver does, and use
fresh report directories. Each command below can run independently:

```bash
python3 -B .github/scripts/test_rocjitsu_race_sweep.py
python3 -B .github/scripts/rocjitsu_race_sweep.py \
  --backend tensile --seed "$PR_REVISION" \
  --rocjitsu "$ROCJITSU_BUILD_DIR/tools/rocjitsu/rocjitsu" \
  --client "$ROCM_PATH/libexec/hipblaslt/tensilelite/tensilelite-client" \
  --config "$ROCJITSU_SOURCE_DIR/configs/gfx942_cdna3_kmd.json" \
  --library-dir "$ROCM_PATH/lib/hipblaslt/library" \
  --target gfx942 --reports "$RACE_REPORT_DIR/sweep-tensile" \
  --workers 4 --kernels 100 --timeout 120
python3 -B .github/scripts/rocjitsu_race_sweep.py \
  --backend bench --seed "$PR_REVISION" \
  --rocjitsu "$ROCJITSU_BUILD_DIR/tools/rocjitsu/rocjitsu" \
  --client "$ROCM_PATH/bin/hipblaslt-bench" \
  --config "$ROCJITSU_SOURCE_DIR/configs/gfx942_cdna3_kmd.json" \
  --library-dir "$ROCM_PATH/lib/hipblaslt/library" \
  --target gfx942 --reports "$RACE_REPORT_DIR/sweep-bench" \
  --workers 4 --kernels 100 --timeout 120
```

The CLI permits smaller diagnostic samples but caps workers at four, kernels at
100 and timeout at 120 seconds. To replay a saved solution, use its argv and
recorded environment with the same artifacts. The INI/YAML paths are absolute.
