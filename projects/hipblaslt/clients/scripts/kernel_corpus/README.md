# Kernel corpus testing

hipBLASLt ships tens of thousands of TensileLite kernels per architecture, and the client
suite reaches them only through the selection heuristic, usually its first pick, usually once.
This tool tests the corpus directly: for every solution in an installed library it builds a few
problems that fit that solution, runs each one with the solution pinned by index, and checks the
answer exactly with `fast_check`
([AIHPBLAS-3894](https://amd-hub.atlassian.net/browse/AIHPBLAS-3894)).

It needs only an installed hipBLASLt with its test client (`bin/hipblaslt-test`,
`bin/hipblaslt_common.yaml`, `lib/hipblaslt/library/<arch>/`), such as a TheRock build or the
artifacts `install_rocm_from_artifacts.py --blas --tests` fetches. Nothing is rebuilt, and no
C++ changes are involved.

## How it works

1. `corpus_index.py` reads the `TensileLibrary_*.dat` and `*.dat.zlib` files and writes one
   JSON line per solution: its runtime index, name, problem type, macro tile, DepthU, split-K
   and size predicates.
2. `plan.py` turns each solution into cases. The `smoke` profile, the only one so far, builds
   four shapes from the solution's own tile: an exact tile multiple, ragged M and N, a K tail,
   and a strided batch of 3. Each case uses `integer_exact` data, alpha 2, beta -2,
   `fast_check: 1`, `algo_method: 2` and `solution_index: <index>`, plus the scaleA, scaleB,
   scaleC and scaleD a problem type requires (each a single value of 1, or a vector for vector
   scaleA and scaleB), so the library's own
   predicate check decides whether the solution accepts the problem, and the C API packs the
   arguments as a user's call would.
3. `run.py` runs the cases in chunks, one `hipblaslt-test --yaml` process per chunk. If a
   case crashes, hangs or faults the GPU, the runner records it, runs a one-GEMM health check,
   and continues the chunk in a new process. Results are appended as cases finish, so a rerun
   resumes.
4. `report.py` merges the shards into `summary.json` and `summary.md`, and writes a one-case
   YAML for every failing case under `repro/`.

## Running

```bash
# Fetch a TheRock CI build (artifact group multi-arch-release for multi-arch CI runs).
python build_tools/install_rocm_from_artifacts.py --run-id <run-id> \
    --artifact-group multi-arch-release --amdgpu-targets gfx90a --blas --tests \
    --output-dir rocm-gfx90a

clients/scripts/kernel_corpus/run.py --rocm-dir rocm-gfx90a --out corpus-gfx90a
```

The architecture comes from the device. `--index-range lo:hi`, `--problem-filter` (a regex on
the problem-type key, such as `Half/Half`), `--per-type N` (the first N solutions of each
problem type, a quick stratified sample), `--limit` and `--dedupe` narrow the run, and
`--dry-run` plans without running. Sharding follows TheRock's test jobs: `SHARD_INDEX`
(1-based) and `TOTAL_SHARDS`, or `--shard-index` and `--total-shards`, select a contiguous
slice of the corpus, and every shard writes its own `shard-<i>-of-<n>/` directory under the
same `--out`. `report.py --out <dir>` merges whatever shards are present.

The exit code is 1 when any case failed, crashed or timed out, 3 when the GPU stopped running
even a basic GEMM, and 0 otherwise.

## Outcomes

| Outcome | Meaning |
| --- | --- |
| `pass` | fast_check found every element of D correct |
| `fail` | a wrong answer, or another gtest failure |
| `crash`, `timeout` | the process died or went silent during the case; `fault` marks a GPU memory fault |
| `refused` | the pinned solution rejected the problem (`NO solution found!`), a planner miss or a predicate that is too strict |
| `skip` | hipblaslt-test skipped the case |
| `unsupported` | fast_check cannot check this configuration (complex data, an exactness limit); hipblaslt-test reports these as failures, and the runner sets them apart |
| `deferred` | the problem type needs an epilogue or input the smoke profile does not build yet (grouped GEMM, gradients, E, swizzled inputs, MX) |
| `unreachable` | the solution requires a CU count this device does not have |

To reproduce a failure, run its YAML from `repro/` with
`hipblaslt-test --yaml <file> --gtest_filter='*pre_checkin_kc_*'`, or the `bench_cmd` from
`summary.json` with `-v`, which checks the same pinned solution against the host reference.

## Scope

The smoke profile is a first pass aimed at finding defects quickly. It does not yet cover
epilogues, scale factors, MX formats, large buffers, 4 GiB placement or repeated launches, and
it tests only TensileLite solutions: hand-written extension ops and rocRoller kernels have no
entries in the TensileLibrary files. Running the CPU-only tests:

```bash
python -m unittest discover -s clients/scripts/kernel_corpus
```
