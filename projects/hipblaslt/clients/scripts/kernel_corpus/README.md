# Kernel corpus testing

hipBLASLt ships tens of thousands of TensileLite kernels per architecture, and the client
suite reaches them only through the selection heuristic, usually its first pick, usually once.
This tool tests the corpus directly: for every solution in an installed library it builds a few
problems that fit that solution, runs each one with the solution pinned by index, and checks the
answer exactly with `fast_check`
([AIHPBLAS-3894](https://amd-hub.atlassian.net/browse/AIHPBLAS-3894)).

It needs only an installed hipBLASLt with its test client (`bin/hipblaslt-test`,
`bin/hipblaslt_common.yaml`, `lib/hipblaslt/library/<arch>/`), such as a TheRock build or the
artifacts of a TheRock CI run. Nothing is rebuilt, so the results describe the exact binaries
that CI built.

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
   arguments as a user's call would. A solution registered for exact sizes (`SizeEqual`
   predicates, from Equality logic, such as the rocKE custom kernels) gets its one registered
   problem instead, with the leading dimensions, alpha and beta its predicates demand, through
   the ext API when its problem type has no beta.
3. `run.py` runs the cases in chunks, one `hipblaslt-test --yaml` process per chunk. If a
   case crashes, hangs or faults the GPU, the runner records it, runs a one-GEMM health check,
   and continues the chunk in a new process. Results are appended as cases finish, so a rerun
   resumes.
4. `report.py` merges the shards into `summary.json` and `summary.md`, and writes a one-case
   YAML for every failing case under `repro/`.

## Running

The harness itself needs Python 3.9 or later, `msgpack` and PyYAML (`pip install msgpack
pyyaml`). To test a
CI build, take the GitHub Actions run ID of a TheRock or rocm-libraries multi-arch CI run and
fetch its artifacts with TheRock's `install_rocm_from_artifacts.py`, from a TheRock checkout
with its `requirements.txt` installed:

```bash
# In a TheRock checkout. Multi-arch CI runs use the multi-arch-release artifact group.
python build_tools/install_rocm_from_artifacts.py --run-id <run-id> \
    --artifact-group multi-arch-release --amdgpu-targets gfx90a --blas --tests \
    --output-dir rocm-gfx90a

# In this directory, on a machine with a gfx90a GPU.
./run.py --rocm-dir rocm-gfx90a --out corpus-gfx90a
```

When it finishes, `corpus-gfx90a/summary.md` lists the outcome counts and every failing case.

### MX kernels, predicate edges and repeated launches

MX FP6 kernels whose scales the host shuffles into 128x4 tiles (scale mode 1018, A in
`HIPBLASLT_ORDER_COL256_R128_EXT`) are planned at their registered size. fast_check does not
decode FP6 or swizzled operands, so these cases check D against hipblaslt-test's CPU reference:
`rand_int` data with `unit_check` and `norm_check`, or for E3M2 beyond K = 65536, where `rand_int`
sums pass 2^24, `trig_float` data with `norm_check` and `allclose_check`, as the matching
`matmul_gtest.yaml` entries do. They run one case per process with every available CPU and the
`--reference-timeout` (default 2 hours). At M = N = 4096 and K = 131584 a case needs about
12 GiB of host memory. Other MX kernels (fp4, fp8, other scale layouts) are deferred.

`--profile edges` builds, for each exact-size solution, problems one step outside each of its
predicates: K one DepthU up and down, half K, M or N one macro tile up, a batch of 2, alpha + 1,
beta = 1, padded `lda`, `ldb` or `ldd`, and B in the other order. The pinned solution must
refuse each one (`expected_refusal`); one that runs it fails, because a kernel without bounds
checks relies on those predicates to stay in bounds. hipblaslt-test cannot build the batch,
`lda` and `ldb` edges of a pre-swizzled MX kernel (its data generator needs a batch of 1 and
leading dimensions equal to the rows stored), so those report `unsupported`. Give each profile
its own `--out`.

`--refusals` asks the library the same questions without building any data: `refusal_probe.cpp`
sets up each problem through the ext API with dummy pointers, pins the solution by index, and asks
`isAlgoSupported`. Each exact-size MX FP6 solution gets its registered problem, which it must
accept, and every edges problem, which it must refuse, including the batch and stride edges
hipblaslt-test cannot build. Its results are `<index>:probe-<tag>` cases, and a probe that finds
another solution name at the index fails the case. The probe is built with the install tree's
`amdclang++` and needs the hipBLASLt and hipBLAS-common headers from TheRock's `blas_dev`
artifact in `<rocm>/include`, or give a built one with `--refusal-probe`.

`--relaunch N` then reruns the first passing case of each solution with `hipblaslt-bench
--post_bench_verify_iters N --verify_strict`, which checks the first launch against the CPU
and then that each of N back-to-back relaunches gives the same D, bit for bit. Its results are
`<index>:relaunch` cases.

```bash
# The rocKE MXFP6 kernels on an MI355X: registered sizes with 20 relaunches each, then edges.
./run.py --rocm-dir rocm-gfx950 --out rocke-mi355x --problem-filter '^B?Float6/' --relaunch 20
./run.py --rocm-dir rocm-gfx950 --out rocke-mi355x-edges --problem-filter '^B?Float6/' \
    --profile edges --refusals
```

The architecture comes from the device. `--index-range lo:hi`, `--problem-filter` (a regex on
the problem-type key, such as `Half/Half`), `--per-type N` (the first N solutions of each
problem type, a quick stratified sample), `--limit` and `--dedupe` narrow the run, and
`--dry-run` plans without running. Sharding follows TheRock's test jobs: `SHARD_INDEX`
(1-based) and `TOTAL_SHARDS`, or `--shard-index` and `--total-shards`, select a contiguous
slice of the corpus, and every shard writes its own `shard-<i>-of-<n>/` directory under the
same `--out`. `report.py --out <dir>` merges whatever shards are present.

The exit code is 1 when a case failed, crashed or timed out outside the known-bugs list, or the
list needs attention; 2 for a usage error or a malformed list; 3 when the GPU stopped running
even a basic GEMM; and 0 otherwise.

## Known bugs

[`clients/tests/data/kernel_corpus_known_bugs.yaml`](../../tests/data/kernel_corpus_known_bugs.yaml)
lists kernels known to fail, each with a ticket, the date it was added, a review date and a
reason. Listed kernels still run. A failing case that matches an entry is reported as
`known_fail` with its ticket and does not fail the run. Two things do fail it, so the list cannot
hide a fix or go stale:

- every case an entry matches passed: the bug looks fixed, so the fixing PR removes the entry;
- an entry's `review_by` date has passed: renew it with a new date and reason, or remove it.

An entry that matched no case in this run (a shard that did not reach it, or a retune that
renamed the kernel) is reported as not seen and does not fail the run. Entries name a kernel by
architecture and solution name, the TensileLite name that survives a retune; the file's header
lists the optional fields that narrow a match. `--known-bugs <file>` uses another list, and
`--no-known-bugs` reports every failure as a failure. `summary.md` lists every entry with its
age and outcome.

## Outcomes

| Outcome | Meaning |
| --- | --- |
| `pass` | fast_check, or for MX kernels the CPU reference, found D correct; for a relaunch case, every relaunch also matched the first launch |
| `fail` | a wrong answer, a relaunch that gave a different D, a solution that ran an `edges` problem it should have refused, or another gtest failure |
| `known_fail` | a failure, crash or timeout that matches an entry in the known-bugs list |
| `crash`, `timeout` | the process died or went silent during the case; `fault` marks a GPU memory fault |
| `refused` | the pinned solution rejected the problem (`NO solution found!`), a planner miss or a predicate that is too strict |
| `expected_refusal` | the pinned solution rejected an `edges` problem, as it should |
| `skip` | hipblaslt-test skipped the case |
| `unsupported` | fast_check cannot check this configuration (complex data, an exactness limit), or hipblaslt-test cannot build it (a pre-swizzled MX problem with a batch above 1, or `lda` or `ldb` other than the rows stored); hipblaslt-test reports these as failures, and the runner sets them apart |
| `deferred` | the problem type needs an epilogue or input the profile does not build yet (grouped GEMM, gradients, E, swizzled inputs outside MX FP6, MX other than FP6 with 128x4 scales), or the `edges` profile met a solution without exact sizes |
| `unreachable` | the solution requires a CU count or PCI chip ID this device does not have; a result from another device of the same architecture replaces it |
| `harness_error` | hipblaslt-test ran none of the chunk's cases twice in a row, for example because it rejected the generated YAML; retried on the next run |

To reproduce a failure, run its YAML from `repro/` with
`hipblaslt-test --yaml <file> --gtest_filter='*pre_checkin_kc_*'`, or the `bench_cmd` from
`summary.json` with `-v`, which checks the same pinned solution against the host reference.

## Scope

The smoke profile is a first pass aimed at finding defects quickly. It sets the scaleA, scaleB,
scaleC and scaleD inputs a problem type requires, and covers MX FP6 kernels with 128x4 scales at
their registered sizes, but it does not yet cover epilogues (bias, activation, gradients), other
MX formats, large buffers or 4 GiB placement. Predicate edges (`--profile edges`) and repeated
launches (`--relaunch`) cover exact-size solutions; tile-derived kernels get neither yet. It
tests only solutions in the TensileLibrary files, custom kernels registered through TensileLite
logic included: hand-written extension ops and rocRoller kernels have no entries there.

The CPU-only unit tests do not run in CI yet; they move into CI with the corpus lanes
([AIHPBLAS-3894](https://amd-hub.atlassian.net/browse/AIHPBLAS-3894)). Run them with:

```bash
python -m unittest discover -s clients/scripts/kernel_corpus
```
