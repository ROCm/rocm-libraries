# rocBLAS Testing 

## Component Overview

rocBLAS is AMD's ROCm implementation of the BLAS API (Levels 1–3 and extensions). It sits in the
math-libraries layer above HIP and the ROCr runtime, and is consumed directly and through hipBLAS,
rocSOLVER, and other libraries and frameworks.

Two architectural facts shape testing:

**GEMM performance paths are largely generated.** When `BUILD_WITH_TENSILE=ON`, Level-3 GEMM kernels
and solution libraries come from a separate library hipBLASLt (TensileLite) or the embedded Tensile (under `shared/tensile/` in the monorepo).
Correctness and performance of those paths depend on both the C++ dispatch layer in this repository and generated
kernel logic maintained separately from hand-written BLAS routines.

**Most validation requires a GPU.** Numerical correctness means agreement with a host reference BLAS
on real hardware across precisions, transposes, batching, and pointer modes. The client suite is
integration-heavy by design; only a small fraction of behavior is exercised without dispatching device
work.

Major dependencies: HIP, ROCr, host reference BLAS (OpenBLAS or AOCL-BLAS with ILP64 for clients), 
Depends on hipBLASLt and Tensile GEMM backends, unless custom build uses `BUILD_WITH_HIPBLASLT_ONLY=ON` it consumes shared monorepo infrastructure `shared/tensile` as child build step.  Other dependencies: `shared/ctest`, TheRock, Math CI.

## Development Workflow

What a developer typically does between a change and merge:

**1. Build** with clients enabled (`BUILD_CLIENTS_TESTS=ON`). Staged binaries land under
`build/release/clients/staging/`.

**2. Run tests matched to the change.**

| You changed | Run this | Needs a GPU |
| --- | --- | --- |
| Any BLAS routine or client harness | `rocblas-test --gtest_filter=*quick*-*known_bug*` or `--yaml rocblas_smoke.yaml` | Yes |
| CTest category (monorepo + `shared/ctest`) | `ctest -L quick` from build tree or `bin/rocblas/` after install | Yes |
| Broad pre-submit scope | `rocblas_rtest.py -t psdb` or `ctest -L standard` | Yes |
| Tensile / logic YAML (GEMM) | Rebuild with Tensile; run affected `*gemm*` / `*_tensile` gtest filters | Yes |
| Performance-sensitive GEMM | `rocblas-bench` on representative sizes; compare to baseline manually | Yes |
| Format / hooks only | `pre-commit run --all-files` | No |

**3. Add the right validation** — see [Coverage Expectations by Change Type](#coverage-expectations-by-change-type) and [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open a PR** targeting `develop`, following [`.github/CONTRIBUTING.rst`](.github/CONTRIBUTING.rst).

**5. Watch CI** — build plus client tests on GPU runners; see [Pre-submit / CI Gates](#pre-submit--ci-gates).

---

This remainder of this document has two layers:

1. **Testing mechanics** — how `rocblas-test`, YAML, CTest, and `rocblas_rtest.py` are wired (below).
2. **Testing strategy** — what we validate, how CI gates merge, known gaps, and what contributors should run ([strategy sections](#component-overview)).

For step-by-step instructions to add client tests, see [`clients/gtest/README.md`](clients/gtest/README.md). For client CLI and environment variables, see the [Programmer's Guide](docs/how-to/Programmers_Guide.rst) (benchmarking and testing section).



## Testing mechanics

High-level overview of how rocBLAS tests are built and executed.

## Client executables

After building with clients enabled, binaries are staged under `build/release/clients/staging/`:

| Client | Role |
|--------|------|
| `rocblas-test` | Correctness and API validation via Google Test; primary regression suite |
| `rocblas-bench` | Performance measurement and optional correctness checks; CLI and YAML driven |
| `rocblas-gemm-tune` | GEMM kernel tuning and solution selection experiments |

Both `rocblas-test` and `rocblas-bench` compare GPU results against a host reference BLAS (e.g. AOCL-BLAS with ILP64 on supported platforms). Limit host thread count with `OMP_NUM_THREADS` to avoid AOCL oversubscription hangs.

## Architecture overview

```
YAML test definitions (*.yaml)
        │
        ▼
rocblas_gentest.py  ──►  rocblas_gtest.data  (generated test parameter list)
        │
        ▼
rocblas-test  ◄──  *_gtest.cpp  (Google Test registration, type dispatch)
        │
        ▼
testing_*.hpp  (per-routine harness: setup, API call, verify)
        │
        ├── rocBLAS API under test
        └── Host reference BLAS (CBLAS / AOCL)
```

**Data-driven core.** Tests are defined as combinations of parameters in YAML, compiled into a binary data file at build time, and instantiated as parameterized Google Tests. C++ code provides typed harness logic; YAML provides coverage breadth (sizes, precisions, transposes, batching, categories).

**Shared infrastructure.** `clients/include/` holds templated harness functions (`testing_<routine>.hpp`), argument parsing (`Arguments`), memory helpers, initialization, and verification utilities. `clients/common/` is shared by `rocblas-test` and `rocblas-bench`.

## Directory layout

| Path | Purpose |
|------|---------|
| `clients/gtest/` | Google Test entry points (`*_gtest.cpp`), YAML suites (`*_gtest.yaml`), category config |
| `clients/include/` | Templated test harness headers (`blas1/`, `blas2/`, `blas3/`, `blas_ex/`), `type_dispatch.hpp`, verification helpers |
| `clients/common/` | `Arguments`, `rocblas_gentest.py`, shared client utilities |
| `clients/benchmarks/` | `rocblas-bench` source |
| `rtest.py` / `rtest.xml` | Source test orchestration scripts (copied at build/install to `rocblas_rtest.py` / `rocblas_rtest.xml`) |
| `clients/gtest/test_categories.yaml` | CTest category definitions (labels, timeouts, filters) |
| `../../shared/ctest/` | Shared monorepo CTest framework (`TestCategories.cmake`, parsers) |
| `scripts/utilities/run_tests/` | Parallel `rocblas-test` runner for long/simulation runs |
| `scripts/performance/` | YAML inputs for performance sweeps via `rocblas-bench` |

Harness headers are organized by BLAS level and variant: non-batched, `_batched`, and `_strided_batched`. ILP64 and Fortran API forms are selected via the `Arguments::api` field in the same templates.

## Test patterns

### Harness function (`testing_*.hpp`)

Each routine has a templated function taking `const Arguments&`. Typical flow:

1. Validate or handle invalid sizes (`rocblas_status_invalid_size`).
2. Allocate and initialize host/device data (`rocblas_init_*`, `device_vector`, `host_matrix`).
3. Run host reference BLAS.
4. Call rocBLAS in host and device pointer modes (`ROCBLAS_CHECK_ERROR`, `DAPI_CHECK` for C/Fortran/ILP64).
5. Compare when `arg.unit_check` is set (`UNIT_CHECK`, `NEAR_CHECK`).

Bad-argument tests use separate `testing_*_bad_arg` templates, often with programmatic setup and YAML in combination.

### Google Test glue (`*_gtest.cpp`)

Each operation suite follows a consistent pattern:

- **`rocblas_*_testing` functor** — partial specializations on supported type combinations; invalid combos derive from `rocblas_test_invalid`.
- **`type_dispatch.hpp`** — maps runtime `Arguments` types to template instantiations.
- **`RocBLAS_Test<>` (CRTP)** — provides `type_filter`, `function_filter`, and `name_suffix` for parameterized test names.
- **`TEST_P` + `INSTANTIATE_TEST_CATEGORIES`** — registers tests across YAML categories.

Test names encode category, function, precision, and parameters so `--gtest_filter` can target subsets (for example `*quick*gemm*f32_r*`).

### YAML categories

Each test entry includes a `category`:

| Category | Typical use |
|----------|-------------|
| `quick` | Fast checks and quick return unit testing |
| `pre_checkin` | PR validation breadth |
| `nightly` | Extended breadth to larger problems |
| `stress` | Large allocations / edge cases; may need `ROCBLAS_CLIENT_RAM_GB_LIMIT` |
| `known_bug` | Tracked failures; excluded from normal runs via `-*known_bug*` |

Entries matching `known_bugs.yaml` are automatically reclassified. Suite YAML files `include` each other and `rocblas_common.yaml`; the root `rocblas_gtest.yaml` aggregates all suites for code generation.

### Verification macros

| Macro / helper | Use |
|----------------|-----|
| `HIP_CHECK_ERROR` | HIP API success |
| `ROCBLAS_CHECK_ERROR` | rocBLAS success |
| `EXPECT_ROCBLAS_STATUS` | Expected error status |
| `UNIT_CHECK` / `NEAR_CHECK` | Numerical comparison vs reference |
| `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES` | Prevent SIGSEGV from aborting entire run |

## Running tests

### Direct `rocblas-test`

```bash
./build/release/clients/staging/rocblas-test --gtest_filter=*quick*-*known_bug*
./build/release/clients/staging/rocblas-test --yaml rocblas_smoke.yaml
./build/release/clients/staging/rocblas-test --gtest_filter=*quick*axpy*f32_r*
```

Filter syntax: `--gtest_filter=POSITIVE[-NEGATIVE]`.

### CTest (`shared/ctest`)

rocBLAS registers labeled CTest suites at configure time using the shared framework in [`shared/ctest/`](../../shared/ctest/). This sits above individual Google Tests: one CTest entry runs a whole category (smoke, pre-checkin, nightly, etc.) rather than each parameterized test case.

**Enablement.** `ROCBLAS_ENABLE_CTEST` (in `cmake/build-options.cmake`) defaults to **ON** when `${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake` exists. Sparse checkouts without `shared/ctest` default it **OFF**. When ON, configure requires both the shared module and `clients/gtest/test_categories.yaml`; set `-DROCBLAS_ENABLE_CTEST=OFF` to skip categorization.

**Configure-time flow:**

```
clients/gtest/test_categories.yaml
        │
        ▼
apply_test_category_labels()     ← shared/ctest/TestCategories.cmake
        │
        ▼
parse_test_categories.py         ← shared/ctest/parse_test_categories.py
        │
        ▼
build/.../test_categories.cmake  (generated add_test + set_tests_properties)
        │
        ▼
ctest -L <label>                 (build tree or install tree)
```

**rocBLAS category suites** (`clients/gtest/test_categories.yaml`):

| CTest label | Jenkins / intent | Typical driver (`rtest.xml` set) |
|-------------|-------------------|----------------------------------|
| `quick` | Smoke (~30 min) | `ctest_quick` → `rocblas_smoke.yaml` |
| `standard` | Pre-checkin / PR (~2 hr) | `ctest_standard` → `*quick*:*pre_checkin*-*known_bug*` |
| `comprehensive` | Extended / nightly (~2 hr) | `ctest_comprehensive` → `*nightly*-*known_bug*` |
| `full` | Stress / weekly (~8 hr) | `ctest_full` → quick + pre_checkin + nightly |
| `ffm-quick`, `ffm-full` | FFM simulation pipelines | FFM-specific YAML / filters |

Each category carries CTest **labels** (for `-L` filtering), a **timeout** from `execution_settings.category_timeouts`, and optional **exclude** patterns (always including `*known_bug*`).

**rtest driver (rocBLAS-specific).** Unlike most libraries, rocBLAS passes `USE_RTEST_DRIVER` to `apply_test_category_labels()`. Category suites invoke the installed driver `rocblas_rtest.py -t ctest_<category>` instead of calling `rocblas-test` directly. The source file is `rtest.py` at the project root; CMake copies it to `build/.../staging/rocblas_rtest.py` and installs it as `bin/rocblas_rtest.py` alongside `bin/rocblas_rtest.xml` (from `rtest.xml`). The matching commands live in `rtest.xml` (`ctest_quick`, `ctest_standard`, etc.). PR labels are used by Math CI to control runners; TheRock CI runs need further modification to set the `GITHUB_PR_LABELS` environment variable for use at rtest execution time.

**Generated test names:** `rocblas-test_<category>_suite` (for example `rocblas-test_quick_suite`). List them after configure:

```bash
cd build/release
ctest -N -L quick
ctest -L standard -V
```

**Install-tree CTest (TheRock / packaged builds).** An install-time `CTestTestfile.cmake` is generated with relative paths to the staged binary. Layout after install:

- `bin/rocblas-test` — test executable
- `bin/rocblas_rtest.py`, `bin/rocblas_rtest.xml`, smoke/extras YAML
- `bin/rocblas/CTestTestfile.cmake` — run CTest from this directory

```bash
cd /opt/rocm/bin/rocblas
ctest -L quick -N
ctest -L quick
```

**GPU exclusion variants.** The `exclude_gpu` system from `shared/ctest` is not used by rocBLAS.  The rocBLAS YAML test data defines what gpu are valid to test on.

Full framework documentation: [`shared/ctest/README.md`](../../shared/ctest/README.md).

### Orchestrated runs (`rtest.py` → `rocblas_rtest.py`)

The orchestration script lives in source as `rtest.py` (with `rtest.xml`). When clients are built, CMake copies both into the staging directory and the install prefix as **`rocblas_rtest.py`** and **`rocblas_rtest.xml`** so they sit next to `rocblas-test` on `PATH`.

From the **source tree** (development):

```bash
cd projects/rocblas
python3 rtest.py -t smoke
python3 rtest.py -t psdb
python3 rtest.py -t osdb
```

From the **build staging** or **install** tree:

```bash
python3 build/release/clients/staging/rocblas_rtest.py -t smoke
# or, after install:
rocblas_rtest.py -t smoke
```

The driver reads `rocblas_rtest.xml` (same content as `rtest.xml`) and selects test sets by name:

| `-t` set | Scope (from `rtest.xml`) |
|----------|--------------------------|
| `smoke` | `rocblas_smoke.yaml` (~minutes) |
| `psdb` | `*quick*:*pre_checkin*-*known_bug*` |
| `osdb` | `*nightly*-*known_bug*` |
| `cqe` | pre-checkin + nightly |

Legacy `-t` sets (`smoke`, `psdb`, `osdb`, `cqe`) remain available for ad hoc runs. CTest category suites use the parallel `ctest_*` sets in the same XML file, wired through `test_categories.yaml` and `shared/ctest` as described above.

### Parallel runner (simulation / long runs)

`scripts/utilities/run_tests/run_tests.py` splits work into BLAS level job groups with resume support and partial re-run of failed tests. See [`scripts/utilities/run_tests/run_tests.md`](scripts/utilities/run_tests/run_tests.md).

### Benchmarking

```bash
./build/release/clients/staging/rocblas-bench -f gemm -r f32_r -m 4096 -n 4096 -k 4096
./build/release/clients/staging/rocblas-bench --yaml scripts/performance/<suite>.yaml
```

Performance YAML lives under `scripts/performance/`. HPA and mixed-precision GEMM must use `gemm_ex` forms in bench, not legacy `gemm`.

## Useful environment variables

| Variable | Effect |
|----------|--------|
| `ROCBLAS_CLIENT_RAM_GB_LIMIT` | Cap host allocation for stress tests |
| `ROCBLAS_TEST_TIMEOUT` | Per-test timeout (seconds; default 600) |
| `ROCBLAS_TEST_NO_SIGACTION` | Easier debugging under `rocgdb` |
| `ROCBLAS_LAYER` / `ROCBLAS_CHECK_NUMERICS` | Logging and internal numerics checks |
| `GTEST_LISTENER=PASS_LINE_IN_LOG` | Log each passing test |
| `OMP_NUM_THREADS` | Host reference thread count |

## CI integration

- **CTest labels** align with Jenkins job types (precheckin, extended, weekly) via `test_categories.yaml` and `shared/ctest`.
- **Jenkins / multi-OS** jobs may invoke `rocblas_rtest.py` (installed name of `rtest.py`) directly or `ctest -L <label>` on build or install trees.
- **PR labels** (`--ci_labels`, `GITHUB_PR_LABELS`) are read by `rocblas_rtest.py` when CTest suites use the rtest driver.
- **Monorepo checkout:** include `shared/ctest` in sparse checkout (or use a full clone) when building with `ROCBLAS_ENABLE_CTEST=ON`.

## Adding tests (summary)

1. Add `clients/include/.../testing_<fn>.hpp` harness.
2. Add `clients/gtest/<fn>_gtest.cpp` with dispatch and `INSTANTIATE_TEST_CATEGORIES`.
3. Add `clients/gtest/<fn>_gtest.yaml` parameter matrix.
4. Include YAML in `rocblas_gtest.yaml` and list it in `clients/gtest/CMakeLists.txt` dependencies for `rocblas_gtest.data`.
5. Add the `.cpp` to the `rocblas-test` source list in CMake.

Full walkthrough: [`clients/gtest/README.md`](clients/gtest/README.md). Contributing notes: [`.github/CONTRIBUTING.rst`](.github/CONTRIBUTING.rst).

---



## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose.** Validate logic that can run without launching BLAS kernels on a device: argument
validation, handle and mode APIs, logging configuration, and other host-side paths.

**Framework and location.**

| Item | Detail |
| --- | --- |
| Framework | Google Test (`rocblas-test` binary) |
| Location | `clients/gtest/*_gtest.cpp` and `clients/include/testing_*.hpp` |
| How to run | `./rocblas-test --gtest_filter=*set_get*` (examples); most suites still require a GPU for memory setup |
| Naming | Parameterized tests encode category, function, and precision in the gtest name suffix |

**What is explicitly not unit-tested in isolation.** BLAS numerical results, Tensile kernel selection
on hardware, batched GEMM at scale, and most pointer-mode/device-memory paths are covered only
through full client integration tests. There is no separate host-only unit binary today.

**Coverage expectation.** There is no enforced unit-coverage floor on the C++ library in this
repository. The practical target for new host-reachable logic is a focused gtest (or bad-arg test)
that fails without the change. Document gaps in [Known Risks and Gaps](#known-risks-and-gaps) rather
than implying coverage that CI does not measure.

### Integration Testing Strategy

**Purpose.** Validate numerical correctness, API behavior, Fortran/ILP64 bindings, and error handling
on GPU hardware — the primary confidence signal for rocBLAS.

**What is covered.** `rocblas-test` driven by YAML (`*_gtest.yaml` → `rocblas_gentest.py` →
`rocblas_gtest.data`). Includes Level 1–3, extensions, batched and strided-batched variants, auxiliary
APIs, logging, and bad-argument cases. Results are compared to host reference BLAS when
`unit_check` is enabled.

**Tiers** (gtest name / YAML `category` and CTest — see [CTest (`shared/ctest`)](#ctest-sharedctest)):

| Tier / label | Typical contents | Duration (order of magnitude) |
| --- | --- | --- |
| `quick` / smoke | `rocblas_smoke.yaml` or `*quick*` | Minutes |
| `standard` / pre-checkin | `*quick*:*pre_checkin*` | Up to ~2 hours |
| `comprehensive` / nightly | `*quick*:*pre_checkin*:*nightly*` | TBD Hours |
| `full` / stress | Includes stress and large-memory cases | Up to ~8 hours (CTest timeout) |
| `known_bug` | Quarantined failures | Excluded via `-*known_bug*` |

`rocblas_rtest.py` offers more test set flexibility as defined in `rocblas_rtest.xml`.

**What requires GPU hardware.** Essentially all client integration tests.

**What runs on PRs.** PR pipelines run standard pre-checkin-class sets (via Math CI, or
TheRock using `rocblas_rtest.py` / CTest `standard`). Comprehensive can be run in Math CI with a label.
Stress (Math CI) runs weekly, on demand, or via label. Math CI currently exceeds most capabilities
and flexibility of that offered by TheRock CI runners.

**Parallel runner.** `scripts/utilities/run_tests/run_tests.py` splits long runs for simulation or
recovery scenarios; see [Parallel runner](#parallel-runner-simulation--long-runs).

### Performance and Benchmarking Testing

**Purpose.** Detect throughput and latency regressions on GEMM and other hot paths; validate tuning
changes.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (math library) |
| Tool | `rocblas-bench` (CLI and `--yaml` batches under `scripts/performance/`) |
| Metrics | GFLOP/s, problem timing via HIP events; optional correctness flags (`-t`, `-v`) |
| Baseline | Per-architecture; not centrally gated in this repository |
| Regression threshold | **Not automated** in rocBLAS CI — manual review of benchmark campaigns |
| Gating | Informational / manual unless a specific CI job is configured for your branch |

**Known gaps.** No repo-wide automated performance gate on PR merge; large YAML sweeps are used for
release and tuning workflows rather than blocking checks.

### Build-Time Validation

When Tensile is enabled, kernel library generation is part of the build. Invalid logic or YAML can
surface as runtime selection failures. Developers rely on successful Tensile codegen plus targeted
`*_tensile` gtest filters after logic changes. There is no separate named CI check in this tree for
“logic-only” validation equivalent to hipBLASLt’s `TensileLogic --check-all`; treat GEMM logic edits
as high-risk and expand integration coverage accordingly. Tensile component validation is handled by the
separate folder in the monorepo `shared/tensile/`, which will contain its own `TESTING.md`.

## Pre-submit / CI Gates

rocBLAS is exercised through monorepo CI (TheRock, GitHub Actions component jobs) and Math CI
(Jenkins), which post checks on the same pull request. Exact required checks depend on branch
protection and may not be fully documented in this repository.

### Validation Gates and Ownership

| Validation area | Required before merge | Owner | Notes |
| --- | --- | --- | --- |
| Build (Linux / Windows) | Yes | CI / DevOps | Client and library targets |
| Integration tests (`rocblas-test`) | Yes | Component team | Smoke / pre-checkin class on GPU runners |
| CTest categories (`shared/ctest`) | When enabled | Component team + DevOps | Requires `shared/ctest` in checkout |
| Formatting / pre-commit | Yes | CI / DevOps | |
| Code coverage floor | No | — | Optional `BUILD_CODE_COVERAGE`; not a documented merge gate |
| Address sanitizer | Varies | Component team | Build option exists; lane-dependent |
| Performance benchmarks | No | Component team | `rocblas-bench` campaigns manual |
| Release qualification | N/A | Component team + QA | Extended tiers and hardware matrix |

### PR Test Classification

| Status | Applies to |
| --- | --- |
| **Trusted gate** | Build; standard/pre-checkin client tests on supported PR hardware; formatting |
| **Quality gate** | Longer comprehensive and stress tiers require Math CI label |
| **Unstable / flaky** | Should be quarantined in `known_bugs.yaml` or fixed — not an accepted end state |

### Flaky Test Policy

Flaky tests must not be treated as permanent gates. Quarantine via `known_bugs.yaml` (reclassified to
`known_bug` category) only with a tracking ticket and intent to fix or remove the quarantine when
resolved. Prefer fixing root cause over widening filters.

### Known Bugs and Expected Failures

`clients/gtest/known_bugs.yaml` matches test parameters and forces `known_bug` category so normal tiers
exclude them. Review quarantined cases before release — each entry is a known defect shipping until
removed.

## Sanitizer Coverage

| Sanitizer | Build flag / notes | CI |
| --- | --- | --- |
| ASAN | `BUILD_ADDRESS_SANITIZER` in `cmake/build-options.cmake` | Lane-dependent; not documented as universal PR gate here |
| TSAN / UBSAN | Not documented as standard rocBLAS CI lanes | Gap if needed for concurrency / UB |

GPU-side sanitizer behavior has platform limits; host client paths benefit most from ASAN builds.

## Static Analysis

Clang format is performed.
Contributors may run `cppcheck` as described in [`.github/CONTRIBUTING.rst`](.github/CONTRIBUTING.rst).
There is no single documented mandatory clang-tidy gate for all rocBLAS PRs in this file — treat static
analysis as recommended local validation unless a specific CI job applies to your change.

## Why We Test This Way

rocBLAS correctness is inherently device-backed BLAS math. The YAML-driven gtest suite maximizes
coverage breadth (precisions, batching, API variants) with one binary and filterable tiers. CTest
integration (`shared/ctest`) aligns those tiers with monorepo automation and install-tree testing
(TheRock). Heavy integration coverage compensates for limited host-only unit testing; performance
relies on bench workflows and external CI rather than in-repo thresholds.

## Key Quality Concerns

1. **Numerical correctness** — Wrong BLAS results silently break downstream math. Validated by
   reference comparison in `rocblas-test`.
2. **API and ABI stability** — Fortran, ILP64, and pointer modes must remain consistent. Validated by
   DAPI/Fortran test paths and downstream consumers.
3. **GEMM performance and kernel selection** — Primary user-visible value for Level 3. Validated by
   Tensile rebuild + gemm gtests and manual/automated bench campaigns.
4. **Memory safety** — Workspace and buffer sizing. Partially validated by stress category and
   sanitizers when enabled.
5. **Packaging / install layout** — Installed `rocblas-test`, YAML, and `CTestTestfile.cmake` must
   remain relocatable for TheRock.

## Release Validation

Before release sign-off, expect at minimum:

- Extended client tiers (`comprehensive` / `full`) on the release hardware matrix done by QA
- Review of `known_bugs.yaml` entries done by rocBLAS team
- Performance spot-checks on representative problem sizes for architectures supported by PTS system
- QA system validation outside this repository

## Dependencies and Validation Handoffs

| Dependency | Owning team | How validated | Known gap |
| --- | --- | --- | --- |
| TheRock / shared CI | TheRock | PR and nightly lanes | Runner capacity limits architectures per PR |
| Math CI | DevOps | Multi-arch client tests | Job definitions not in this tree |
| Tensile / shared/tensile | Component + generator | Build + gemm integration tests | Logic validation not a separate named gate |
| Host reference BLAS (AOCL) | External | Linked into clients | Thread oversubscription can hang tests — use `OMP_NUM_THREADS` |
| hipBLAS / frameworks | Downstream | External integration | Pre-merge signal limited |

## Supported Configurations

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux + AMD GPU | Full client suite | PR / nightly | Primary development platform |
| Windows | Client build + tests | PR / release | Sparse checkout needs `shared/ctest` for CTest labels |
| Specific GFX | Partial on PR | PR subset; broader nightly | YAML test files encode gfx applicability |

Document unsupported combinations explicitly during release planning rather than assuming CI covered them.

## Coverage Expectations by Change Type

| Change type | Expected validation |
| --- | --- |
| New BLAS routine | `testing_*.hpp`, `*_gtest.cpp`, `*_gtest.yaml`, CMake registration |
| Bug fix | Regression gtest that fails without the fix |
| New public API / handle mode | Auxiliary gtest + YAML case |
| GEMM / Tensile logic | Rebuild Tensile; gemm / `*_tensile` filters; bench spot-check if performance-related |
| CI / CTest only | Update `test_categories.yaml` and `rtest.xml`; verify `ctest -N` |
| Packaging | Install-tree `ctest` from `bin/rocblas/` |

## Known Risks and Gaps

| Gap | Regression risk | Mitigation today |
| --- | --- | --- |
| No automated PR performance threshold | High for perf work | Manual `rocblas-bench` and external perf CI in PTS |
| Math CI job list not in this repo | Medium | Institutional knowledge; ask before assuming coverage |
| `known_bugs.yaml` quarantine without ticket discipline | Medium | Review before release |
| Sparse checkout without `shared/ctest` | Low | `ROCBLAS_ENABLE_CTEST=OFF`; no install CTest labels |
| Stress tests and OOM on small hosts | Medium | `ROCBLAS_CLIENT_RAM_GB_LIMIT`; exclude `*stress*` |

## Improvement Roadmap

Near term:

1. PTS support for performance regression detection and automated comparisons.
2. Extend TheRock CI test system to run stress test sets on demand.
3. Extend TheRock CI test system to include HMM XNACK tests where applicable.
4. Extend TheRock CI test system to perform multi-GPU tests.
...

Medium term:

1. Keep `test_categories.yaml`, `rtest.xml`, and `shared/ctest` in sync for tier names and timeouts or delete all comments and duplication of timeout information.
2. Document required GitHub checks alongside this file once TheRock CI achieves Math CI parity with branch protection.
3. Keep managing quarantine debt in `known_bugs.yaml` with tickets and removal dates.
4. Review TODO comments for ticket generation
5. Evaluate if there is added value in host-reachable unit tests for auxiliary APIs without full device.
6. Add stochastic problem-size testing.

## Owners and Review Cadence

Update this document when test tiers, CTest layout, or CI gates change. Review [Known Risks and Gaps](#known-risks-and-gaps) at least quarterly and after significant post-merge regressions.

## For New Contributors

When changing rocBLAS:

1. Read [Key Quality Concerns](#key-quality-concerns) for your area.
2. Add or extend YAML + harness tests per [`clients/gtest/README.md`](clients/gtest/README.md).
3. Run smoke/quick locally before pushing.
4. Update this document if you change tiers, CTest, or quarantine policy.

### Choosing the Right Test Type

- **Bug fix** — regression test failing before the fix.
- **GPU numerical BLAS behavior** — integration case in `*_gtest.yaml` + `testing_*.hpp`.
- **Invalid arguments / status codes** — `testing_*_bad_arg` or dedicated small gtest.  These include unit tests.
- **Handle / logging / stride APIs** — auxiliary gtest pattern (see existing `set_get_*` tests).
- **Performance** — `rocblas-bench` with sizes representative of the kernel path; note results in PR. 
- **CTest / CI tier change** — `test_categories.yaml`, `rtest.xml`, and verify `ctest -L`.

## How This Document Is Used

Living strategy artifact for onboarding, PR review, release readiness, and CI improvement planning.
Accurate gaps are preferable to aspirational claims — record what is not gated so “we assumed CI
covered it” happens less often at release time.

## Further reading

- [hipBLASLt testing strategy (sibling example)](../../hipblaslt-TESTING.md) 
- [shared/ctest README — CTest framework architecture and integration guide](../../shared/ctest/README.md)
- [Programmer's Guide — clients layout and testing](docs/how-to/Programmers_Guide.rst)
- [Linux install — building clients, CTest, and sparse checkout](docs/install/Linux_Install_Guide.rst)
- [Windows install — CTest and reference BLAS notes](docs/install/Windows_Install_Guide.rst)
- [Contributing to rocBLAS](.github/CONTRIBUTING.rst)
