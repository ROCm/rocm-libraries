# rocALUTION Testing Strategy

**Status:** Draft
**Owner:** @doctorcolinsmith
**Technical Lead:** @ntrost57
**Last Updated:** 2026-08-06

This document describes how rocALUTION is tested today, which signals actually gate a merge, and where
the gaps are. It follows the ROCm-wide TESTING.md template and is written as a description of the
current state rather than an aspirational one. A gap that is written down is one that can be argued
about and closed.

---

## Component Overview

rocALUTION is a sparse linear algebra library for **iterative solvers and preconditioners**, built in
C++ and HIP with a portable, backend-agnostic design. Version 4.1.0. It provides Krylov solvers,
direct solvers, algebraic multigrid, and a large family of preconditioners on top of the ROCm math
stack.

**Where it sits in the ROCm stack:** an Expansion-SDK-level solver library. When GPU support is
enabled it links against **rocSPARSE**, **rocBLAS**, **rocPRIM**, and **rocRAND**
(package minimums: `rocsparse >= 1.12.10`, `rocblas >= 2.22.0`, `rocrand >= 2.1.0`,
`hip-runtime-amd >= 4.5.0`).

**Backends (selectable at configure time):**

| Backend | Flag | Notes |
|---|---|---|
| Host | always built | `src/base/host/` |
| OpenMP | `SUPPORT_OMP` (default ON if found) | multi-threaded host |
| HIP / GPU | `SUPPORT_HIP` (default ON if found) | `src/base/hip/`; initializes rocBLAS/rocSPARSE |
| MPI / multi-node | `SUPPORT_MPI` (default OFF) | compile def `SUPPORT_MULTINODE`, links `MPI::MPI_CXX` |

**Key architectural constraint that shapes testing:** rocALUTION runs the *same* solver/preconditioner
algorithms across host, OpenMP, and HIP backends. Correctness is therefore validated by running each
algorithm on each available backend and cross-checking, which makes the suite backend-parameterized
rather than GPU-only. Much of the solver logic *can* run on the host backend without a GPU, but the
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

Test binary: `build/release/clients/staging/rocalution-test` (single monolithic GoogleTest binary).

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
```

> When any `ROCALUTION_EMULATION_*` var is set, GPU-heavy infrastructure suites (`local_matrix_*`,
> `local_vector`, `local_stencil`, `backend`, `preconditioner`, MPI infra) call `GTEST_SKIP()`, and
> solver suites shrink their parameter lists. Unset means the full parameter combinations.

**3. Add the right kind of test** — see [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open the PR** targeting `develop`. A team member reviews and approves.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** validate logic that can be exercised without a GPU — the **host backend** solver and
preconditioner algorithms, backend initialization/ordering, and the host implementations of the
`local_matrix` / `local_vector` structures.

Unlike GPU-bound math libraries, a large fraction of rocALUTION's logic is genuinely host-testable:
every solver and preconditioner has a host backend, so its algorithm can be run and validated on a
machine with no GPU (`./install.sh -c --host`). These cases live in the same `rocalution-test`
binary and use a mix of GoogleTest styles:

* **Parameterized** (`INSTANTIATE_TEST_CASE_P` + `TestWithParam<tuple>`) — the solver/preconditioner
  suites (float and double as separate `TEST_P` cases, e.g. `cg_float`, `cg_double`).
* **Fixture** (`TEST_F`) — structure tests such as `local_vector_test`, `local_matrix` ops.
* **Plain** (`TEST`) — e.g. `backend_init_order.backend`.

There are no `TYPED_TEST` suites.

**Framework:** GoogleTest (`find_package(GTest REQUIRED)`); main at
`clients/tests/rocalution_host_gtest_main.cpp`, which calls `init_rocalution()` before
`RUN_ALL_TESTS()` and accepts `--device <id>` / `--version`.

**Location / naming:** `clients/tests/test_<component>.cpp` with implementation headers in
`clients/include/testing_<component>.hpp`. Shared helpers in `clients/include/` (`utility.hpp`,
`random.*`, `common.hpp`, `validate.hpp`). There is no `clients/common/` directory — shared client
code is provided by rocm-cmake's `ROCMClients` module.

**How to run without a GPU:** build `--host` and run `./rocalution-test`; the host backend paths for
solvers and preconditioners execute on CPU.

**What is NOT covered as host unit tests:** the HIP backend kernels, and the GPU-only structure
suites (`local_matrix_*`, `local_vector`, `local_stencil`) which are skipped under emulation tiers and
require hardware.

**Coverage expectation:** the long-term ROCm-wide goal is >95% line coverage of hardware-independent
paths, pursued in phases. rocALUTION's host-backend coverage is meaningful because a large share of
the algorithm code runs on CPU; the GPU backend duplicates those algorithms in device code, which
host-side instrumentation does not capture. See [Coverage](#coverage).

---

### Integration Testing Strategy

**Purpose:** validate the HIP (and MPI) backends against the host reference — that each solver,
preconditioner, and structure operation produces correct results on-device, and that host↔device
transfers and multi-node communication behave.

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
suites. **What runs CPU-only:** the host-backend runs (build `--host`). **What requires MPI:** the
`global_*` and `parallel_manager` suites (`SUPPORT_MPI`).

**Managed-memory (HMM) cases:** on gfx90a, run with `HSA_XNACK=0` and `HSA_XNACK=1`.

**Test-size / coverage guidance:** solvers are validated on small representative systems (e.g.
Laplacian) across backends and both precisions; the emulation tiers shrink parameter lists rather than
running exhaustive size sweeps. Prefer cross-backend correctness over large problem sizes.

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
| Baseline — stored per architecture | Not stored/aggregated; comparison is manual. Must not be aggregated across GFX |
| Where results are stored | Benchmark output files (JSON/XML); `scripts/rocalution_bench_gnuplot_helper.py` renders histograms |
| Regression threshold | Not automated |
| Gating approach | Manual review |
| GPU profiling | Not integrated |

**Gating:**

| Gating Level | Status | Notes |
|---|---|---|
| PR-level automated gate | No | Known gap |
| Nightly automated comparison | No | Known gap |
| Manual review | Yes | On request via the bench + gnuplot helper |
| Release qualification | Partial | Reviewed before release; not an automated sign-off |

---

## Why We Test This Way

rocALUTION implements the same solver and preconditioner algorithms across host, OpenMP, and HIP
backends. The cheapest, strongest signal is cross-backend agreement: run an algorithm on the host
reference and on the GPU and require they converge to the same answer. That is why a large share of
the suite is genuinely host-testable (build `--host`) while the GPU and `local_*` structure paths need
hardware.

Tiering is driven by environment variables (`ROCALUTION_EMULATION_*`) rather than GTest name prefixes
because the same test bodies serve every tier — the tier just shrinks parameter lists and skips the
GPU-heavy infrastructure suites. This keeps one set of test definitions honest across quick, standard,
and comprehensive runs, at the cost of the tier being invisible in the test name (a readability gap).

---

## Pre-submit / CI Gates

The presubmit gate is the monorepo **TheRock CI** GitHub Actions workflow
(`.github/workflows/therock-ci*.yml`), which runs on every pull request and push to `develop`
(`.github/scripts/therock_configure_ci.py`) and tests only changed projects
(`-DTHEROCK_ENABLE_ROCALUTION=ON` + sparse + rand). By default it runs the CTest **`standard`**
category, which sets `ROCALUTION_EMULATION_REGRESSION=1` — the regression subset, where GPU-heavy
infrastructure suites (`local_matrix_*`, `local_vector`, `local_stencil`, `backend`, MPI infra) skip
themselves and solver suites use reduced parameter sets. The other categories map to the emulation
vars: `quick` → `ROCALUTION_EMULATION_SMOKE=1`, `comprehensive` → `ROCALUTION_EMULATION_EXTENDED=1`,
`full` → complete parameter sweep (no emulation var). Scope widens per PR via labels
(`test:rocalution`, `test_type:comprehensive` / `test_type:full`); doc-only changes (`*.md`,
`docs/*`) skip CI.

The rocALUTION Azure DevOps pipeline (`.azuredevops/rocm-ci.yml`) is explicitly **disabled**
(`trigger: none`, `pr: none`) because it depends on rocSPARSE. Internal AMD **Jenkins** pipelines
(`.jenkins/`) are legacy, running on older GPUs (gfx900 / gfx906 / gfx908) via cron across three build
variants — default (`./install.sh -c`, HIP+OpenMP), host (`--host`), and MPI
(`--host --mpi=on --no-openmp`).

> **Important nuance:** the Jenkins `runTestCommand` GTest filter is commented out, so those legacy
> lanes run the **full** `rocalution-test` suite (no emulation subset). The intended
> `*checkin*` / `*nightly*` filters match no current test names. The modern TheRock gate, by
> contrast, selects the subset through the `ROCALUTION_EMULATION_*` env var of the CTest category.

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build (HIP, host, static; MPI variant) | Yes | CI / DevOps | TheRock CI; Jenkins `precheckin`/`debug`/`staticlibrary` (legacy) |
| Unit / host-backend tests | Yes | Component team | In the CTest `standard` category (regression subset) |
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
| Unstable / flaky | No `known_bug` mechanism exists in rocALUTION today |

**Flaky / known-bug policy:** rocALUTION has **no `known_bugs.yaml`** and no tests use a `known_bug`
name; the `-*known_bug*` filter referenced in the Jenkins coverage job matches nothing. There is no
tracked quarantine list — this is a gap. A flaky test is not an accepted permanent state; when one
appears it should be tagged and ticketed.

---

## Coverage

**Tool:** gcov/lcov. `-DBUILD_CODE_COVERAGE=ON` compiles with `-fprofile-arcs -ftest-coverage` and
links `--coverage -lgcov` (via `install.sh --codecoverage`, which also sets `BUILD_SUPPORT_COMPLEX=OFF`
and requires a Debug or RelWithDebInfo build).

**How to build and run:**

```bash
./install.sh -cg --codecoverage
cd build/debug
make coverage_cleanup coverage GTEST_FILTER='*'
# targets: coverage_analysis -> coverage_output (lcov capture + filter + genhtml -> lcoverage/) -> coverage
```

`ROCALUTION_CODE_COVERAGE=1` is set for the coverage run; it enables coverage-specific parameter sets
and is **required for `test_itersolver`** (skipped otherwise). lcov exclusions cover `src/utils/*`,
`src/base/host/host_io.*`, `clients/*`, `build/*`, `/opt/*`, `/usr/*`. Jenkins `codecov.groovy`
uploads `lcoverage/main_coverage.info`.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by the suite.
* *Test coverage* = fraction of intended functionality exercised (solvers × preconditioners × formats
  × backends × precisions × MPI). Host-side gcov does not capture the HIP device kernels, so measured
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

**Explicitly not tested / guaranteed:** the Azure pipeline is disabled; complex-number paths are
disabled under coverage builds; non-listed gfx targets; Windows coverage; global (MPI) stencil tests
(`test_global_stencil.cpp` is commented out).

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
| No automated performance regression gate (PR or nightly) | High | High | Manual `rocalution-bench` + gnuplot review |
| No per-architecture perf baseline in a queryable format | High | High | Output files only; comparison is manual |
| Tier not encoded in test names; legacy `*checkin*`/`*nightly*` filters match nothing | Medium | Medium | Tiering works via `ROCALUTION_EMULATION_*`; Jenkins runs full suite |
| No `known_bugs.yaml` / quarantine mechanism | Medium | Medium | None; flaky tests handled ad-hoc |
| Azure CI pipeline disabled | Medium | Medium | Jenkins + TheRock provide coverage |
| Device-code coverage not captured; complex disabled under coverage | Medium | Medium | Host gcov only |
| MPI/multi-node validation is partial; global stencil test commented out | Medium | Medium | Nightly MPI variant build |
| No TSAN/UBSAN/MSAN | Low | Medium | ASAN only |

---

## Owners and Review Cadence

**Review this document when:**
* A new solver, preconditioner, backend, or CI lane is added.
* The emulation-tier or CTest-category wiring changes.
* A regression escapes to a downstream consumer or a release.
* Before a major release, alongside the known-gap review.

The measure of whether this document is working: the Known Gaps table shrinks over time.
