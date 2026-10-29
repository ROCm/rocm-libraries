# rocALUTION Testing Strategy

**Status:** Draft
**Owner:** @doctorcolinsmith
**Technical Lead:** @ntrost57
**Last Updated:** 2026-10-29

This document describes how rocALUTION is tested today, which signals actually gate a merge, and where
the gaps are. It follows the ROCm-wide TESTING.md template and is written as a description of the
current state rather than an aspirational one. A gap that is written down is one that can be argued
about and closed.

---

## Component Overview

rocALUTION is a sparse linear algebra library for **iterative solvers and preconditioners**, built in
C++ and HIP with a portable, backend-agnostic design. It provides Krylov solvers,
direct solvers, algebraic multigrid, and a large family of preconditioners on top of the ROCm mathematics
stack.

**Where it sits in the ROCm stack:** it is an Expansion-SDK-level solver library. When GPU support is
enabled it links against **rocSPARSE**, **rocBLAS**, **rocPRIM**, and **rocRAND**
(package minimums: `rocsparse >= 1.12.10`, `rocblas >= 2.22.0`, `rocrand >= 2.1.0`,
`hip-runtime-amd >= 4.5.0`).

**Backends (selectable at configuration time):**

| Backend | Flag | Notes |
|---|---|---|
| Host | Always built | `src/base/host/` |
| OpenMP | `SUPPORT_OMP` (default ON if found) | Multi-threaded host |
| HIP / GPU | `SUPPORT_HIP` (default ON if found) | `src/base/hip/`; initializes rocBLAS/rocSPARSE |
| MPI / multi-node | `SUPPORT_MPI` (default OFF) | Compiles def `SUPPORT_MULTINODE`, links `MPI::MPI_CXX` |

**Key architectural constraint that shapes testing:** rocALUTION runs the *same* solver/preconditioner
algorithms across host, OpenMP, and HIP backends. Correctness is therefore validated by running each
algorithm on each available backend and cross-checking, which makes the suite backend-parameterized
rather than GPU-specific. Much of the solver logic *can* run on the host backend without a GPU, but the
GPU paths and the low-level `local_matrix`/`local_vector` structures require real hardware.

---

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build the library with test/benchmark/sample clients:**

```bash
cd projects/rocalution
# -c → BUILD_CLIENTS_SAMPLES + TESTS + BENCHMARKS; -d fetches dependencies (incl. googletest)
./install.sh -dc
# host-only (no GPU), OpenMP on:            ./install.sh -c --host
# MPI multi-node build:                     ./install.sh -c --host --mpi=on --no-openmp
```

Test binaries in `build/release/clients/staging/`: `rocalution-test` (the solver, preconditioner and
structure suites) and the unit tests `rocalution-unit-test` and `rocalution-unit-test-lifecycle`
(see [Unit Testing Strategy](#unit-testing-strategy)).

**2. Run the tier that matches what you touched** (tiering is driven by environment variables, not
GTest name filters):

```bash
cd build/release/clients/staging
# smoke / quick:
ROCALUTION_EMULATION_SMOKE=1      HIP_VISIBLE_DEVICES=0 gpu-run ./rocalution-test
# pre-checkin / standard:
ROCALUTION_EMULATION_REGRESSION=1 HIP_VISIBLE_DEVICES=0 gpu-run ./rocalution-test
# a single solver:
./rocalution-test --gtest_filter='cg/parameterized_cg.cg_float/*'
# full sweep (no emulation var set):
./rocalution-test
# unit tests (the emulation variables do not apply to them):
HIP_VISIBLE_DEVICES=0 gpu-run ./rocalution-unit-test
```

> When any `ROCALUTION_EMULATION_*` var is set, GPU-heavy infrastructure suites (`local_matrix_*`,
> `local_vector`, `local_stencil`, `backend`, `preconditioner`, MPI infra) call `GTEST_SKIP()`, and
> solver suites shrink their parameter lists. If no emulation variables are set, the full set of parameter combinations is exercised.

**3. Add the right kind of test:** see [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open the PR** targeting `develop`. A team member reviews and approves.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** test each layer of the library on its own, against references that are written
independently of the library. The layers are the backend objects of the host and HIP backends
(matrix formats, vectors, conversions, kernels), the `LocalMatrix` / `LocalVector` front end, the
solvers, preconditioners and multigrid hierarchies, and the initialization and shutdown of the
platform. A unit test fails at the operation that is wrong. A solver test in `rocalution-test` can
only show that some operation on the way to convergence went wrong, and a wrong operation can even
let a solver converge to the right answer more slowly.

**Location and build:** the unit tests live in `clients/unittests/` and are built together with the
other tests (`BUILD_CLIENTS_TESTS`, for example `./install.sh -c`). They link the object files of the
library (the `rocalution_objects` target) instead of the library itself, so that they can test the
internal backend classes directly. As a consequence they can only be built in the same build as the
library, not against an installed rocALUTION. They never link `roc::rocalution` as well, because
that would put a second copy of the library's global state into the process. A host-only build
(`./install.sh -c --host`) builds both binaries without the HIP-specific test files. The unit tests
are not built on Windows yet.

**Binaries:**

| Binary | Contents | CTest |
|---|---|---|
| `rocalution-unit-test` | All unit-test suites. A global environment initializes the platform once, on the device given by `--device <id>` (default 0), and stops it at the end. | One test, `rocalution-unit-test`, label `quick` |
| `rocalution-unit-test-lifecycle` | `init_rocalution` and `stop_rocalution`, device selection, thread affinity, and the host fallback without an accelerator. On Linux with HIP it also tests the error paths of the HIP backend initialization through fault injection: `hip_fault_injection.cpp` overrides the HIP, rocBLAS and rocSPARSE calls that `librocalution_hip` makes. There is no global environment, so every test starts and ends with an uninitialized platform. | One test per GoogleTest case (`lifecycle.*`, `lifecycle_hip_fault.*`), each in its own process, label `quick` |

**How to run:**

```bash
cd build/release/clients/staging
# all unit tests, the host and the GPU cases
HIP_VISIBLE_DEVICES=0 gpu-run ./rocalution-unit-test
# one suite
./rocalution-unit-test --gtest_filter='host_matrix_csr*'
# the lifecycle tests, one process per test
cd ../.. && gpu-run ctest -R '^lifecycle'
# also run the cases that known bugs break, to check whether a bug is fixed
ROCALUTION_UNIT_TEST_RUN_KNOWN_BUGS=1 gpu-run ./rocalution-unit-test
```

Without a HIP build or without a device, the accelerator cases skip. If a device is present but the
accelerator was not initialized, they fail instead, so that a broken initialization cannot pass as a
skip.

**Test structure:**

* **Typed tests** (`TYPED_TEST`) over value type and backend. The parameter is `Typed<T, Backend>`,
  and the lists are built with `WithBackends<...>`, `HostOnly<...>` and `AcceleratorOnly<...>` from
  the value-type lists `AllValueTypes`, `NumericTypes`, `FloatingTypes`, `RealTypes` and
  `IntegerTypes` (`unit_test_types.hpp`). The complex types are part of the lists only with
  `BUILD_SUPPORT_COMPLEX`, because the library instantiates them only then. Test names look like
  `cfloat_Accelerator`.
* **Value-parameterized tests** (`TEST_P`) over backend and matrix format (`BackendFormat`, names
  like `Accelerator_BCSR2`). GoogleTest has no typed value-parameterized tests, so these run their
  body once per value type with `for_each_type`.
* **Fixture and plain tests** (`TEST_F`, `TEST`) for everything with a single configuration, and
  death tests in suites ending in `_DeathTest`.
* The test files are named `unit_test_<area>.cpp`, after the part of the library they test. The
  shared helpers are in `unit_test_common.hpp` (backends, formats, death-test and known-bug macros),
  `unit_test_types.hpp` (typed tests, tolerances, comparisons), `unit_test_reference.hpp` and
  `unit_test_amg_reference.hpp` (the references) and `unit_test_local.hpp` (`LocalMatrix` and
  `LocalVector` objects built from the references).

**Conventions:**

* **References are independent of the library.** Expected results come from the reference
  implementations in `unit_test_reference.hpp` (dense and CSR products, ILU(0) and IC(0)
  factorizations, triangular solves) and `unit_test_amg_reference.hpp` (grids, strength of
  connection, prolongation of smoothed aggregation), or from values worked out by hand. They never come
  from another rocALUTION function or from another backend: agreement between the HIP and the host
  backend shows only that both compute the same, not that either is right. Where the result is not
  unique, as for aggregations, coarse/fine splittings and colorings, the test checks the properties
  that make a result valid instead of one particular answer.
* **Tests assert the intended behaviour, never the behaviour of a bug.** A case that a bug breaks
  uses `KNOWN_BUG` (below), so that it fails once known bugs are enabled and passes once the bug is
  fixed. Behaviour that is surprising but intended is asserted, with a comment that says why it is
  intended.
* **Inputs are small and chosen to reach every path.** The matrices are a few rows large and built in
  the test, for example `Tridiagonal`, a variant whose DIA storage holds zeros and whose ELL storage
  pads every other row, and an arrow matrix whose long rows go into the COO part of HYB. Complex
  values come from `from_int<T>(k)`, which gives every value a nonzero imaginary part, so that a
  missing or extra conjugation changes the result.
* **Tolerances.** `tol<T>()` is the relative tolerance of a computation in `T`: 1e-5 for `float`,
  1e-12 for `double`, and exact for integer types and `bool`. `is_near` and `expect_near` compare
  within `tol * max(1, |ref|)`, using the modulus of the difference for complex values.
  `scaled_tol<T>(t)` scales a `double` tolerance to the same number of roundings in `T`, and
  `loose_tol<T>(t)` is for tolerances with slack above the rounding. A tolerance is never widened to
  make a test pass.
* **Operations that a backend does not implement.** Several backend operations are implemented only
  for some formats or only on the host. Elsewhere they return `false` and leave their outputs
  untouched, so that `LocalMatrix` can convert to CSR or move to the host and retry. The function
  `implements(backend, format, operation)` in `unit_test_common.hpp` is the single table of which
  operations are native where. Tests of an operation's behaviour start with
  `REQUIRE_NATIVE(backend, format, operation)`, which skips where it is not native. The fallback
  tests (`unimplemented_operations_return_false_and_change_nothing`) check the other combinations.
  If a backend gains an implementation, its fallback test fails until the table is updated.
* **Known bugs.** `KNOWN_BUG(reason)` skips the rest of a case with the reason, unless
  `ROCALUTION_UNIT_TEST_RUN_KNOWN_BUGS=1` is set. It returns from the enclosing function. Inside a
  lambda, for example one value type of `for_each_type`, it ends only that call of the lambda. So
  it goes after the checks that the bug does not affect, directly before the checks that it does. A
  bug that ends the process (an illegal memory access, an abort, a hang) puts its checks into
  `expect_in_child(checks, timeout_s)` after the `KNOWN_BUG`. With known bugs enabled, the bug then
  fails one test instead of ending the run. The checks build all their objects inside the child,
  and checks that can hang get a timeout. Once the bug is fixed, the `KNOWN_BUG` and the
  `expect_in_child` wrapper are removed.
* **Preconditions and fatal errors.** Library preconditions are asserts, and they are tested with
  `EXPECT_ASSERT_DEATH(statement, regex)`. `utils/def.hpp` keeps them enabled in Release builds. An
  exit with a logged message is tested with `EXPECT_LOGGED_EXIT(statement, regex)`, and a
  `FATAL_ERROR` with `EXPECT_FATAL_ERROR(statement, file)`. Death tests use the `threadsafe` style,
  which re-executes the binary instead of forking the initialized process. All three macros start a
  watchdog in the child: a statement that hangs for more than 120 seconds ends the child with exit
  code 124, so the death test fails instead of hanging the suite.
* **Global state.** A test that changes the backend descriptor, the OpenMP thread count or the CPU
  affinity creates a `DescriptorGuard`, which restores them on destruction, also after a fatal
  failure or an exception. A file is named with `temp_file(extension)`, which is unique per test and
  parameter, so that test processes can run in parallel.
* **Table-driven loops.** A failed precondition of one case of a loop uses `ASSERT_IN_LOOP(condition)`.
  It fails the test but ends only the current case, so that the remaining cases still run.
* **Log output.** `logged(f)` returns what `f` writes to the log, for tests of warnings and of the
  fallback messages.

**Host-backend runs of `rocalution-test`:** besides the unit tests, the solver and preconditioner
suites of `rocalution-test` also run on the host backend, so their algorithms can be validated on a
machine without a GPU (`./install.sh -c --host`). These suites are parameterized
(`INSTANTIATE_TEST_CASE_P` with `TestWithParam<tuple>`, with float and double as separate `TEST_P`
cases such as `cg_float` and `cg_double`), with fixture tests (`TEST_F`) for the structures and
plain tests (`TEST`) such as `backend_init_order.backend`. Their `main` is
`clients/tests/rocalution_host_gtest_main.cpp`, which calls `init_rocalution()` before
`RUN_ALL_TESTS()` and accepts `--device <id>` and `--version`. The sources are
`clients/tests/test_<component>.cpp`, with implementation headers in
`clients/include/testing_<component>.hpp` and shared helpers in `clients/include/` (`utility.hpp`,
`random.*`, `common.hpp`, `validate.hpp`).

**What is not covered by the unit tests:** the multi-process paths of `GlobalMatrix`,
`GlobalVector` and `ParallelManager`, which the MPI suites of `rocalution-test` cover. The
deprecated AMG interface, `src/utils/` and `src/base/host/host_io.*` are tested only where other
tests reach them, and they are excluded from the unit-test coverage numbers.

**Coverage expectation:** the long-term ROCm-wide goal is >95% line coverage of hardware-independent
paths, pursued in phases. rocALUTION's host-code coverage is meaningful because a large share of the
algorithm code runs on the CPU. The GPU backend duplicates those algorithms in device code, which
host-side instrumentation does not capture. See [Coverage](#coverage).

---

### Integration Testing Strategy

**Purpose:** validate the HIP (and MPI) backends against the host reference — that each solver,
preconditioner, and structure operation produces correct results on-device, and that host–device
transfers and multi-node communication behave as expected.

**What is covered** (all as parameterized/fixture GoogleTest cases, per backend and per precision):

| Group | Examples |
|---|---|
| Krylov solvers | CG, CR, BiCGStab, BiCGStab(l), GMRES, FGMRES, FCG, IDR, QMRCGSTAB |
| Direct solvers | LU, QR, inversion |
| Algebraic multigrid | pairwise AMG, Ruge–Stüben AMG, smoothed-aggregation (SA-AMG), unsmoothed-aggregation (UA-AMG) |
| Preconditioners | Jacobi, GS/SGS, ILU/ILUT/IC, ItILU0, (AI)Chebyshev, FSAI, SPAI, TNS, AS/RAS, multi-colored GS/SGS/ILU, multi-elimination, variable |
| Structures (GPU) | `local_matrix` ops / conversions / multicoloring / itsolve / solve, `local_vector`, `local_stencil` |
| Other | Chebyshev iteration, mixed precision, iterative-solver integration (`test_itersolver`, coverage-only) |
| MPI (`SUPPORT_MPI`) | `global_matrix`, `global_vector`, `parallel_manager` |

**Tiers (environment-variable driven, mapped to CTest categories when built inside the monorepo with
`shared/ctest/`):**

| CTest category | Env var set | Labels | Timeout |
|---|---|---|---|
| `quick` | `ROCALUTION_EMULATION_SMOKE=1` | quick, pre-commit | 300 s |
| `standard` | `ROCALUTION_EMULATION_REGRESSION=1` | standard, pr | 1800 s |
| `comprehensive` | `ROCALUTION_EMULATION_EXTENDED=1` | comprehensive, nightly | 14400 s |
| `full` | *(none — full parameter sweep)* | full, weekly | 28800 s |
| `ffm-quick` / `ffm-full` | explicit filter `cg/parameterized_cg.cg_float/0` | ffm | 7200 s |

Unlike rocSPARSE/hipSPARSE, rocALUTION does **not** encode the tier in the GTest suite name — most
CTest suites use `test_patterns: ["*"]` and the subset is selected inside the binary by the emulation
env vars. The legacy `*checkin*` / `*nightly*` filters in `rtest.xml` and Jenkins do not match any
current test names.

**What requires GPU hardware:** the HIP-backend runs of every suite and the `local_*` structure
suites.

**What runs on CPU-only:** the host-backend runs (build `--host`).

**What requires MPI:** the
`global_*` and `parallel_manager` suites (`SUPPORT_MPI`).

**Managed-memory (HMM) cases:** on gfx90a, run with `HSA_XNACK=0` and `HSA_XNACK=1`.

**Test-size / coverage guidance:** solvers are validated on small representative systems (e.g.
Laplacian) across backends and both precisions; the emulation tiers shrink parameter lists rather than
running exhaustive size sweeps. The strategy prefers cross-backend correctness over large problem sizes.

---

### Performance and Benchmarking Testing

**Purpose:** detect regressions in solver/preconditioner performance on a given architecture. Absolute
numbers are not comparable across GFX targets.

| Item | Detail |
|---|---|
| Stack layer | Expansion SDK (solver library) |
| Metrics measured | Solver iteration/solve time and convergence per solver × preconditioner × format |
| How benchmarks are run | `rocalution-bench` from `clients/staging/`; single-run by default, batch/sweep mode when `--bench-x` / `--bench-o` / `--bench-n` / `--bench-std` are given |
| Solvers / preconditioners | selectable via CLI (`--iterative_solver`, `--preconditioner`, `--direct_solver`, `--format`, tolerances, `--ndim`, `--matrix_filename`, …) |
| Baseline — stored per architecture | Not stored/aggregated in this repo (manual comparison); perf regression is tracked outside this repo. Must not be aggregated across GFX |
| Where results are stored | Benchmark output files (JSON/XML); `scripts/rocalution_bench_gnuplot_helper.py` renders histograms |
| Regression threshold | Not automated in this repo; perf regression is tracked outside this repo |
| Gating approach | Manual in-repo; perf regression tracked outside this repo |
| GPU profiling | Not integrated |

**Gating:**

| Gating Level | Status | Notes |
|---|---|---|
| Public CI (TheRock) automated gate | No | No perf gate in this repo's PR or nightly CI |
| Regression tracked outside this repo | Yes | Perf regression is tracked outside this repo |
| Manual review (in-repo) | Yes | On request via the bench + gnuplot helper |
| Release qualification | Partial | Reviewed before release; not an automated sign-off in this repo |

---

## Why rocALUTION is Tested This Way

rocALUTION implements the same solver and preconditioner algorithms across host, OpenMP, and HIP
backends. The cheapest, strongest signal is cross-backend agreement: run an algorithm on the host
reference and on the GPU and require they converge to the same answer. That is why a large share of
the suite is genuinely host-testable (build `--host`) while the GPU and `local_*` structure paths need
hardware.

Tiering is driven by environment variables (`ROCALUTION_EMULATION_*`) rather than GTest name prefixes
because the same test bodies serve every tier — the tier just shrinks parameter lists and skips the
GPU-heavy infrastructure suites. This keeps one set of test definitions honest across quick, standard,
and comprehensive runs, although the selected tier is not reflected in the test name (thus a readability gap).

---

## Pre-submit / CI Gates

The presubmit gate is the monorepo **TheRock CI** GitHub Actions workflow
(`.github/workflows/therock-ci*.yml`), which runs on every pull request and push to `develop`
(`.github/scripts/therock_configure_ci.py`) and tests only changed projects
(`-DTHEROCK_ENABLE_ROCALUTION=ON` + sparse + rand). By default it runs the CTest **`standard`**
category, which sets `ROCALUTION_EMULATION_REGRESSION=1` — the regression subset in which GPU-heavy
infrastructure suites (`local_matrix_*`, `local_vector`, `local_stencil`, `backend`, MPI infra) skip
themselves and solver suites use reduced parameter sets. The other categories map to the emulation
vars: `quick` → `ROCALUTION_EMULATION_SMOKE=1`, `comprehensive` → `ROCALUTION_EMULATION_EXTENDED=1`,
`full` → complete parameter sweep (no emulation var). Scope widens per PR via labels
(`test:rocalution`, `test_type:comprehensive` / `test_type:full`); doc-only changes (`*.md`,
`docs/*`) skip CI.

Internal AMD **Jenkins** pipelines (`.jenkins/`) are legacy, running on older GPUs
(gfx900 / gfx906 / gfx908) via cron across three build variants — default (`./install.sh -c`,
HIP+OpenMP), host (`--host`), and MPI (`--host --mpi=on --no-openmp`).

> **Important nuance:** the Jenkins `runTestCommand` GTest filter is commented out, so those legacy
> lanes run the **full** `rocalution-test` suite (no emulation subset). The intended
> `*checkin*` / `*nightly*` filters match no current test names. The modern TheRock gate, by
> contrast, selects the subset through the `ROCALUTION_EMULATION_*` env var of the CTest category.

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build (HIP, host, static; MPI variant) | Yes | CI / DevOps | TheRock CI; Jenkins `precheckin`/`debug`/`staticlibrary` (legacy) |
| Unit / host-backend tests | Yes (host-backend runs); No (unit tests) | Component team | The host-backend runs of `rocalution-test` are in the CTest `standard` category (regression subset). The unit-test binaries are registered with CTest (label `quick`) and run in the Jenkins coverage job, but are not yet wired into the TheRock CTest categories |
| Integration tests | Yes | Component team | TheRock `standard` = regression subset (`ROCALUTION_EMULATION_REGRESSION=1`); legacy Jenkins runs the full suite (filter commented out) |
| Formatting | Yes | CI / DevOps | Repo-wide `pre-commit` |
| Static analysis | No (informational) | CI / DevOps | `clang-tidy`, `codeql`, Jenkins `staticanalysis.groovy` |
| Code coverage | No | Component team / CI | `codecov.groovy`, informational (Codecov, `rocALUTION`) |
| ASAN | Separate lane | CI / DevOps | TheRock ASAN workflows + `BUILD_ADDRESS_SANITIZER`; not a confirmed per-PR blocking gate |
| Shared validation infra | N/A | TheRock team | Shared build/validation infrastructure |
| Release qualification | N/A | Component team + QA + TPM | Readiness and known-gap review |

**Jenkins build variants:** default `./install.sh -c` (HIP + OpenMP); host `./install.sh -c --host`;
MPI `./install.sh -c --host --mpi=on --no-openmp`. Test path:
`build/release/clients/staging/rocalution-test`. On gfx90a, HMM tests run with `HSA_XNACK=0/1`.

### PR Test Classification

| Status | Applies to |
|---|---|
| Trusted gate | TheRock CI `standard` category (regression subset) on changed projects |
| Informational | Coverage upload (Codecov); TheRock nightly / comprehensive results; legacy Jenkins full-suite lanes |
| Unstable / flaky | No quarantine mechanism for `rocalution-test`; the unit tests skip known bugs with `KNOWN_BUG` |

**Flaky / known-bug policy:** in `rocalution-test`, there is **no `known_bugs.yaml`** and no test
uses a `known_bug` name. The `-*known_bug*` filter of the Jenkins coverage job matches nothing, and
there is no tracked quarantine list. This is a gap. The unit tests mark a case that a known bug breaks
with `KNOWN_BUG(reason)`. It skips the case with the reason in the normal run, and
`ROCALUTION_UNIT_TEST_RUN_KNOWN_BUGS=1` runs it, so whether a bug is fixed can be checked at any time
(see [Unit Testing Strategy](#unit-testing-strategy)). A `KNOWN_BUG` is for a bug of rocALUTION or of
a library it calls, never for a flaky test. A flaky test is not an acceptable permanent state; when
one appears it should be tagged and ticketed.

---

## Coverage

**Tool:** LLVM source-based coverage, as in rocSPARSE. `-DBUILD_CODE_COVERAGE=ON` requires a clang
compiler and instruments the host code of the library (the `rocalution_objects` that
`librocalution` and the unit tests are built from) and of `librocalution_hip` with
`-fprofile-instr-generate -fcoverage-mapping`; the device code and the tests themselves are not
instrumented. `install.sh
--codecoverage` selects `amdclang++`, sets `BUILD_SUPPORT_COMPLEX=OFF` and requires a Debug or
RelWithDebInfo build.

**How to build and run:**

```bash
./install.sh -cg --codecoverage
cd build/debug
make coverage_cleanup coverage GTEST_FILTER='*'
# targets: coverage_analysis (profraw) -> coverage (llvm-profdata merge, llvm-cov export,
#          scripts/filter_lcov_exclusions.py, genhtml -> coverage-report/)
```

`coverage_analysis` runs `rocalution-test` with `GTEST_FILTER`, then `rocalution-unit-test`
unfiltered, and the lifecycle tests through CTest, one process per test. The report combines all
of them. `ROCALUTION_CODE_COVERAGE=1` is set for the coverage run; it enables coverage-specific
parameter sets and is **required for `test_itersolver`** (skipped otherwise).

`scripts/filter_lcov_exclusions.py` drops `src/utils/*`, `src/base/host/host_io.*`, `clients/*`,
`/opt/*` and `/usr/*`, and applies the exclusion markers of the library sources:

* `LCOV_EXCL_START` / `LCOV_EXCL_STOP` and `LCOV_EXCL_LINE` exclude code that cannot run, such as
  the deprecated AMG interface and unreachable error branches.
* `LCOV_EXCL_BR_LINE` and `LCOV_EXCL_BR_START` / `LCOV_EXCL_BR_STOP` exclude only the branches of
  code that runs. The branches of the logging macros, whose MPI rank check is a branch at every
  call site, are excluded by default.
* Code that only runs with more than one process (the global matrix and vector, the parallel
  manager and their backend helpers) is marked with `COVERAGE_EXCL_NO_MPI_START` /
  `COVERAGE_EXCL_NO_MPI_STOP` and only excluded in builds without MPI.

The script also drops the line records of lines without code (blank lines, comments, braces and
preprocessor directives), which `llvm-cov` can report as missed. Jenkins `codecov.groovy` uploads
`coverage-report/coverage.info`.

**Code coverage vs. test coverage**:
* *Code coverage* = fraction of lines executed by the suite.
* *Test coverage* = fraction of intended functionality exercised (solvers × preconditioners × formats
  × backends × precisions × MPI). Host-only instrumentation does not capture the HIP device kernels, so measured
  code coverage understates GPU-path work; disabling complex under coverage also narrows scope.

**Scope:** measured on Linux; Windows/complex coverage not tracked separately.

---

## Nightly Validation

Beyond PR validation, the intended nightly tier is `comprehensive`
(`ROCALUTION_EMULATION_EXTENDED=1`, timeout 4 h) with broader parameter sweeps, plus additional GPU
architectures. In practice the Jenkins `extended.groovy` lane runs the full suite because the tier
filter is not applied via test names (see the note under CI Gates). The `full` sweep (no emulation
var) is the weekly / release-scope run.

---

## Supported Configurations

Default GPU targets come from `GPU_TARGETS` in the root `CMakeLists.txt`.

| Configuration | Validation Level | Frequency | Notes |
|---|---|---|---|
| Linux + HIP backend | Full | PR / Nightly / Release | Primary platform |
| Linux host / OpenMP backend | Full | PR / Nightly | `--host`; CPU-only, no GPU required |
| Linux + MPI multi-node | Partial | Nightly | `--mpi=on`; `global_*` + `parallel_manager` suites |
| gfx908 / gfx90a / gfx942 / gfx950 | Full | PR / Nightly / Release | Core CDNA |
| gfx900 / gfx906 | Partial | PR / Nightly | Older GCN/CDNA |
| gfx103x / gfx110x / gfx1200 / gfx1201 | Partial | Nightly | RDNA |
| ASAN (gfx908/gfx90a/gfx942 `xnack+`) | Partial | On demand | AddressSanitizer, xnack+ only |

**Explicitly not tested / guaranteed:**

- Complex-number paths are disabled under coverage builds
- Non-listed gfx targets
- Windows coverage
- Global (MPI) stencil tests (`test_global_stencil.cpp` is
commented out).

---

## ASAN / TSAN / Sanitizer Coverage

**AddressSanitizer:** `-DBUILD_ADDRESS_SANITIZER=ON` (`install.sh --address-sanitizer`, forces
`amdclang++`) builds with `-fsanitize=address -shared-libasan`, links with `lld`, restricts GPU
targets to `gfx908:xnack+`, `gfx90a:xnack+`, `gfx942:xnack+`, and depends on `hip-runtime-amd-asan`.
It catches host and (xnack+) device memory errors.

**What is explicitly not covered:** TSAN, UBSAN, and MSAN are not wired into rocALUTION; non-xnack
device configurations are not ASAN-covered.

---

## Choosing the Right Test Type

| Scenario | What to add |
|---|---|
| New or changed backend operation (matrix format, vector, conversion, kernel) | A typed or backend × format case in the matching `clients/unittests/unit_test_<area>.cpp`, checked against an independent reference; add the operation to `implements()` if it is not native everywhere |
| New or changed `LocalMatrix` / `LocalVector` behaviour, including moves and fallbacks | A case in `unit_test_local_matrix.cpp` or `unit_test_local_vector.cpp` |
| New or changed precondition or fatal error | An `EXPECT_ASSERT_DEATH`, `EXPECT_LOGGED_EXIT` or `EXPECT_FATAL_ERROR` case in the unit tests |
| Change to initialization, shutdown or device selection | A case in `rocalution_unit_test_lifecycle.cpp`; a HIP failure path uses `ScopedHipFault` |
| Bug found but not fixed yet | A unit-test case of the intended behaviour, behind `KNOWN_BUG(reason)` after the checks it does not affect |
| New host-backend algorithm or fix (solver/preconditioner logic) | A parameterized host-backend case in `test_<component>.cpp`; runnable with `--host`, no GPU |
| New or changed GPU-backend behavior | A parameterized case exercised on the HIP backend, cross-checked against the host reference |
| New solver / preconditioner / AMG variant | A `test_<name>.cpp` with float and double `TEST_P` cases across backends; add matching sample if useful |
| New `local_matrix` / `local_vector` operation | A `TEST_F` structure case (GPU; skipped under emulation tiers) |
| MPI / multi-node change | A `global_*` or `parallel_manager` case (guarded by `SUPPORT_MPI`) |
| Bug fix | A regression case that fails before the fix |
| Performance-sensitive change | Run `rocalution-bench` before/after; render with `scripts/rocalution_bench_gnuplot_helper.py`; note the delta in the PR |

---

## Known Gaps Summary

| Gap | Regression risk | Impact | Mitigation today |
|---|---|---|---|
| No public/TheRock perf regression gate (PR or nightly) in this repo | Medium | Medium | Perf regression is tracked outside this repo; in-repo `rocalution-bench` + gnuplot for manual checks |
| No per-architecture perf baseline reproducible from this repo | Medium | Medium | Perf regression is tracked outside this repo; in-repo output files only |
| Tier not encoded in test names; legacy `*checkin*`/`*nightly*` filters match nothing | Medium | Medium | Tiering works via `ROCALUTION_EMULATION_*`; Jenkins runs full suite |
| No `known_bugs.yaml` / quarantine mechanism for `rocalution-test` | Medium | Medium | The unit tests skip known bugs with `KNOWN_BUG`; flaky tests handled ad-hoc |
| Unit tests not wired into the TheRock CI categories | Medium | Medium | CTest label `quick`; they run in the Jenkins coverage job |
| Unit tests not built on Windows (they use POSIX calls such as `alarm`, and some helpers do not compile there) | Medium | Medium | Windows runs `rocalution-test` only |
| Device-code coverage not captured; complex disabled under coverage | Medium | Medium | Host-only LLVM coverage |
| MPI/multi-node validation is partial; global stencil test commented out | Medium | Medium | Nightly MPI variant build |
| No TSAN/UBSAN/MSAN | Low | Medium | ASAN only |

---

## Owners and Review Cadence

**Review this document and the described testing strategy when:**
* A new solver, preconditioner, backend, or CI lane is added.
* The emulation-tier or CTest-category wiring changes.
* A regression escapes to a downstream consumer or a release.
* Before a major release, alongside the known-gap review.

The effectiveness of this testing strategy is measured by the reduction of entries in the Known Gaps Summary table over time.
