# hipSPARSE Testing Strategy

**Status:** Draft
**Owner:** @doctorcolinsmith
**Technical Lead:** @ntrost57
**Last Updated:** 2026-08-06

This document describes how hipSPARSE is tested today, which signals actually gate a merge, and where
the gaps are. It follows the ROCm-wide TESTING.md template and is written as a description of the
current state rather than an aspirational one. A gap that is written down is one that can be argued
about and closed.

---

## Component Overview

hipSPARSE is a **SPARSE marshalling library**. It sits between an application and a "worker" sparse
library, marshalling inputs to the backend and results back to the application, and exports a stable
interface that does not change regardless of the chosen backend. It currently supports two backends:

* **HIP backend** → [rocSPARSE](https://github.com/ROCm/rocm-libraries/tree/develop/projects/rocsparse)
  on AMD GPUs (`library/src/amd_detail/`, package `hipsparse`).
* **CUDA backend** → NVIDIA cuSPARSE (`library/src/nvidia_detail/`, package `hipsparse-alt`).

**Where it sits in the ROCm stack:** portability layer above rocSPARSE (and cuSPARSE). It is API-
compatible with cuSPARSE v2, so porting a CUDA application is largely mechanical.

**Key architectural constraint that shapes testing:** hipSPARSE is thin.
Its job is to translate the hipSPARSE API to the backend correctly, so testing is dominated by
integration tests that call real backend routines on a GPU and validate the marshalled result. The
tested surface differs by backend: the HIP backend compiles and runs additional legacy routines that
the CUDA backend does not, and the CUDA YAML set is versioned per CUDA toolkit (`cuda/12.8/`,
`cuda/13/`).

---

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build the library with test/benchmark/sample clients (HIP backend, typical):**

```bash
cd projects/hipsparse
# -c → BUILD_CLIENTS_TESTS + samples + benchmarks; -d fetches dependencies (incl. googletest)
./install.sh -dc
# equivalently, modern CMake:
cmake -B build -DHIPSPARSE_BUILD_TESTING=ON -DHIPSPARSE_ENABLE_HIP=ON
cmake --build build --parallel
```

Presets are also available: `cmake --preset default:release`, `debug`, `coverage`, `asan`.
Test binaries land in `build/<release|debug|release-debug>/clients/staging/`.

**2. Provision the test matrices** (functional suites read `.bin` matrices): the build in step 1
(`./install.sh -dc`) downloads and converts them automatically, so no separate step is required.

> **Note:** If you already have the matrices downloaded to a folder, pass `--matrices-dir <path_to_matrices>`
> to the install script to reuse them and avoid re-downloading on a rebuild:
>
> ```bash
> ./install.sh -dc --matrices-dir <path>/hipsparse_matrices
> ```
>
> To populate such a folder once (e.g. a shared location outside the build tree), use `--matrices-dir-install`:
>
> ```bash
> ./install.sh --matrices-dir-install <path>/hipsparse_matrices
> ```

**3. Run the tests that match what you touched:**

```bash
cd build/release/clients/staging
# PR scope, exclude known bugs:
HIP_VISIBLE_DEVICES=0 gpu-run ./hipsparse-test --gtest_filter='*checkin*-*known_bug*'
# a single routine:
./hipsparse-test --gtest_filter='*csrmv*-*known_bug*'
```

**4. Add the right kind of test** — see [Choosing the Right Test Type](#choosing-the-right-test-type).

**5. Open the PR** targeting `develop`. The pre-checkin GTest run (`*checkin*`, excluding
`*known_bug*`) is the merge gate on the HIP backend; the CUDA backend has its own reduced lane.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** validate hardware-independent logic — descriptor/handle construction, enum/type
marshalling, and argument validation — that can be checked without dispatching a compute kernel.

hipSPARSE does not maintain a separate host-only unit-test binary; hardware-independent checks live
inside the single `hipsparse-test` GoogleTest binary as:

* **Descriptor / bad-argument tests** — plain `TEST()` cases such as
  `clients/tests/test_dnmat_descr.cpp` (`dnmat_descr_bad_arg.*`) validate descriptor create/destroy
  and argument handling without a kernel.
* **`<routine>_bad_arg` cases** — YAML `function: <routine>_bad_arg` dispatches to
  `testing_<routine>_bad_arg`, checking status-code marshalling for invalid inputs.

**Framework:** GoogleTest (`find_package(GTest REQUIRED)`); main at
`clients/tests/hipsparse_gtest_main.cpp`, which defines `GOOGLE_TEST`.

**Location / structure:** per-routine `clients/tests/test_<routine>.cpp` (registration via
`TEST_ROUTINE` / `TEST_ROUTINE_WITH_CONFIG` in `clients/tests/test.hpp`) +
`clients/include/testing_<routine>.hpp` (implementation) + backend-specific YAML under
`clients/tests/rocm/` or `clients/tests/cuda/<ver>/`. At build time
`clients/common/hipsparse_gentest.py` expands the backend's `hipsparse_test.yaml` into
`hipsparse_test.data`, consumed at runtime.

**How to run:** `./hipsparse-test --gtest_filter='*bad_arg*'` runs the argument-validation surface.

**What is NOT covered by unit tests:** numerical correctness of marshalled routines (requires the
backend on a GPU) and anything below the hipSPARSE API in rocSPARSE/cuSPARSE.

**Coverage expectation:** the long-term ROCm-wide goal is >95% line coverage of hardware-independent
paths, pursued in phases. Because hipSPARSE is a thin marshalling layer, most of its lines are
exercised only when the backend routine actually runs, so measured coverage is dominated by
integration tests. Repo Codecov target is 80% (`codecov.yml`, flag `hipSPARSE`).

---

### Integration Testing Strategy

**Purpose:** validate that the hipSPARSE API correctly marshals to the backend and returns correct
results — for ~110 routines across sparse level 1/2/3, conversions, preconditioners, reordering, and
the generic API — by running on a GPU and comparing against a host reference.

Integration tests are overwhelmingly `TEST_P` / `INSTANTIATE_TEST_SUITE_P`, parameterized from the
binary `hipsparse_test.data`. Type dispatch uses `TEST_ROUTINE_WITH_CONFIG`. A few suites use other
GoogleTest styles: a matrix-file `TestWithParam` in `test_csrilusv.cpp`, and a typed suite
`test_spmv_csr_pytorch_compat.cpp` (**HIP backend only**).

**Runtime tiers (YAML `category:` field → GTest suite prefix):**

| Tier | Meaning | Typical use |
|---|---|---|
| `quick` | small, fast cases | local smoke |
| `pre_checkin` | PR / checkin scope | PR merge gate (Jenkins filter `*checkin*`) |
| `nightly` | extended coverage | nightly (`extended.groovy`, filter `*nightly*`) |
| `stress` | instantiated in code but no `stress` cases in YAML today | — |
| `known_bug` | auto-promoted by gentest from a YAML `Known bugs:` rule | excluded from all gating runs |

**Per-routine time budgets (rough guidelines):** each tier has a target maximum wall-clock time for
a *single routine's* cases in that tier. The budget applies to the total time of one routine's cases
at one tier — e.g. `./hipsparse-test --gtest_filter=*quick*csrmv*` should finish in under 1000 ms.

| Tier | Target max time per routine | Example filter |
|---|---|---|
| `quick` | < 1000 ms (1 s) | `--gtest_filter=*quick*csrmv*` |
| `pre_checkin` | < 10000 ms (10 s) | `--gtest_filter=*pre_checkin*csrmv*` |
| `nightly` | < 100000 ms (100 s) | `--gtest_filter=*nightly*csrmv*` |

> These budgets are rough guidelines, not hard limits: actual run time depends on the hardware. They
> exist to keep any one routine from dominating a tier's total run time; a routine that consistently
> and substantially exceeds its budget is a signal to trim redundant cases (see the test-size guidance
> below).

> Note: the Jenkins `*checkin*` filter matches the `pre_checkin` tier only; the `quick` tier is
> picked up by TheRock's CTest `standard` category (`quick` + `pre_checkin`), which is the actual PR
> scope. See [Pre-submit / CI Gates](#pre-submit--ci-gates).

**Backend effect on the tested surface:**

| Aspect | HIP / ROCm | CUDA |
|---|---|---|
| YAML root | `clients/tests/rocm/hipsparse_test.yaml` | `cuda/12.8/` (CUDA 12.8) or `cuda/13/` (CUDA 13.x) |
| Extra HIP-only C++ sources | `test_doti`, `test_dotci`, `test_csr2csc`, `test_csrgemm`, `test_csrgeam`, `test_csrmv`, `test_csrmm`, `test_hybmv`, `test_csr2hyb`, `test_hyb2csr`, `test_spmv_csr_pytorch_compat`, … | not compiled |
| CUDA version gate | n/a | 12.8 → `cuda/12.8/`; 13.x → `cuda/13/`; otherwise no test data is generated |
| Jenkins lane | `precheckin.groovy` (`*checkin*`) | `precheckin-cuda.groovy` (`./install.sh -c --cuda`, filter `*checkin*csrmv*`) |

**Test data / matrices:** `cmake/ClientMatrices.cmake` downloads **19 SuiteSparse** matrices (e.g.
`scircuit`, `nos1`–`nos7`, `amazon0312`, `webbase-1M`) as `.bin` (mirror overridable via
`HIPSPARSE_TEST_MIRROR`). Runtime resolution: installed data path `../share/hipsparse/test/` or the
executable directory; matrices via `--matrices-dir` or `HIPSPARSE_CLIENTS_MATRICES_DIR`.

**What requires GPU hardware:** all numerical-correctness cases. **What runs without a compute
kernel:** descriptor and `*bad_arg*` cases.

**What runs on PRs:** the CTest `standard` category (`quick` + `pre_checkin`, excluding `*known_bug*`)
on the HIP backend, via TheRock CI (default `test_type: standard`); a reduced `*checkin*csrmv*` on the
CUDA backend (legacy Jenkins `precheckin-cuda.groovy`). **What runs nightly:** the `comprehensive`
category (adds the `nightly` tier).

**Test-size / coverage guidance:** since hipSPARSE mostly forwards to the backend, exhaustive
numerical sweeps add little over the backend's own suite — prefer marshalling-focused cases (type/
enum coverage, descriptor variants, transpose/index-base combinations) over large size matrices.

---

### Performance and Benchmarking Testing

**Purpose:** detect throughput regressions per routine on a given architecture. Absolute numbers are
not comparable across GFX targets.

| Item | Detail |
|---|---|
| Stack layer | Portability layer above the Core SDK |
| Metrics measured | Time / GFLOP/s / bandwidth per routine (as forwarded to the backend) |
| How benchmarks are run | `hipsparse-bench` from `clients/staging/`, e.g. `./hipsparse-bench -f csrmv --bench-x -M 10 20 30 40` |
| Baseline — stored per architecture | Not stored/aggregated in this repo; comparison is manual. Backend perf regression is tracked outside this repo. Must not be aggregated across GFX |
| Where results are stored | Benchmark output files; no centralized DB in this repo |
| Regression threshold | Not automated in this repo; the backend (rocSPARSE) perf regression is tracked outside this repo |
| Gating approach | Manual in-repo; backend perf regression tracked outside this repo |
| GPU profiling | Not integrated |

**Gating:**

| Gating Level | Status | Notes |
|---|---|---|
| Public CI (TheRock) automated gate | No | No perf gate in this repo's PR or nightly CI |
| Regression tracked outside this repo | Yes (backend) | The HIP backend (rocSPARSE) perf regression is tracked outside this repo |
| Manual review (in-repo) | Yes | `hipsparse-bench` on request |
| Release qualification | Partial | Reviewed before release; not an automated sign-off in this repo |

Because hipSPARSE is a thin marshalling layer, performance is effectively that of the backend;
component-level perf tracking is intentionally light. Backend performance is owned by rocSPARSE /
cuSPARSE, whose perf regression is tracked outside this repo.

---

## Why We Test This Way

hipSPARSE owns almost no compute — it marshals to rocSPARSE or cuSPARSE. The failure modes that
matter are marshalling mistakes: wrong enum/type translation, mishandled descriptors, incorrect
status propagation, and API-compatibility drift from cuSPARSE. Those are best caught by running the
real backend routine on a GPU and comparing against a host reference, which is why the strategy is
integration-dominant and the hardware-independent surface is validated by descriptor and `*bad_arg*`
cases inside the same binary.

The backend split is deliberate: the HIP and CUDA backends expose different routine sets, so the YAML
suites are maintained per backend (and per CUDA toolkit version). This keeps each lane honest about
what its backend actually supports rather than pretending to a single unified matrix.

---

## Pre-submit / CI Gates

The presubmit gate is the monorepo **TheRock CI** GitHub Actions workflow
(`.github/workflows/therock-ci*.yml`), which runs on every pull request and push to `develop`
(`.github/scripts/therock_configure_ci.py`). It builds and tests only changed projects and, by
default, runs the CTest **`standard`** category — `quick` + `pre_checkin`, excluding `*known_bug*`
(`clients/tests/test_categories.yaml`) — on the HIP backend. Scope widens per PR via labels
(`test:hipsparse`, `test_type:comprehensive` / `test_type:full`); doc-only changes (`*.md`,
`docs/*`) skip CI. hipSPARSE and rocSPARSE share TheRock's `sparse` component
(`projects_to_test: [rocsparse, hipsparse]` in `.github/scripts/therock_matrix.py`), so a PR touching
either one builds and tests both.

Internal AMD **Jenkins** pipelines (`.jenkins/`) are legacy, running on older GPUs (gfx900 / gfx906 /
gfx908) via cron: `precheckin.groovy` builds the HIP backend (`./install.sh -c`) and runs GTest
filter `*checkin*` (the `pre_checkin` tier), `precheckin-cuda.groovy` builds `--cuda` (cuSPARSE) and
runs the narrow `*checkin*csrmv*`, `extended.groovy` runs `*nightly*`, and `codecov.groovy` /
`staticanalysis.groovy` cover coverage and format/static analysis. Build dependencies pulled include
`rocSPARSE`, `rocPRIM`, `rocBLAS`, `hipBLASLt`, `hipBLAS-common`. Repo-wide `pre-commit`, `clang-tidy`,
`codeql` apply to all components.

> The CUDA/cuSPARSE backend is validated in its own lane (`precheckin-cuda.groovy`) with a
> deliberately narrow filter; it is not part of the default TheRock HIP-backend gate.

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build (HIP backend) | Yes | CI / DevOps | TheRock CI; Jenkins `precheckin`/`static` (legacy) |
| Build (CUDA backend) | Separate lane | CI / DevOps | Jenkins `precheckin-cuda.groovy` (legacy); not in the default TheRock gate |
| Unit / descriptor / bad-arg tests | Yes | Component team | In the CTest `standard` category |
| Integration tests | Yes | Component team | HIP: `standard` (`quick`+`pre_checkin`); CUDA: narrow `*checkin*csrmv*`; both exclude `*known_bug*` |
| Formatting | Yes | CI / DevOps | Repo-wide `pre-commit` |
| Static analysis | No (informational) | CI / DevOps | `clang-tidy`, `codeql`, Jenkins `staticanalysis.groovy` |
| Code coverage | No | Component team / CI | `codecov.groovy`, informational (Codecov flag `hipSPARSE`, target 80%) |
| ASAN | Separate lane | CI / DevOps | `asan` preset build + TheRock ASAN workflows; not a confirmed per-PR blocking gate |
| Shared validation infra | N/A | TheRock team | Shared build/validation infrastructure |
| Release qualification | N/A | Component team + QA + TPM | Readiness and known-gap review |

### PR Test Classification

| Status | Applies to |
|---|---|
| Trusted gate | TheRock CI `standard` category on the HIP backend (changed projects), excluding `*known_bug*`; plus the CUDA `*checkin*csrmv*` lane |
| Informational | Coverage upload (Codecov); nightly `*nightly*` / comprehensive results |
| Unstable / flaky | `known_bug`-tagged cases (excluded from gating) |

**Flaky / known-bug policy:** hipSPARSE has no `known_bugs.yaml`. A known bug is declared as a
`Known bugs:` section inside a routine's YAML — a small number of routines carry such rules — and
`hipsparse_gentest.py` moves matching cases into the `known_bug` category. Every gating run excludes
`*known_bug*`. A known-bug tag is not an accepted permanent state; per-case owner/ticket tracking is a
gap.

---

## Coverage

**Tool:** `lcov` (`-DHIPSPARSE_ENABLE_COVERAGE=ON`, or legacy `-DBUILD_CODE_COVERAGE=ON` /
`install.sh --codecoverage`; requires a Debug or RelWithDebInfo build).

**How to build and run:**

```bash
./install.sh -kc --codecoverage        # RelWithDebInfo + clients + coverage
cd build/release-debug
make coverage_cleanup coverage GTEST_FILTER='*-*known_bug*'
# targets: coverage_analysis -> coverage_output (lcov --capture, strip /opt,/usr, genhtml) -> coverage
```

Output lands in `build/.../lcoverage/` (`main_coverage.info` + HTML). Jenkins `codecov.groovy`
uploads `lcoverage/main_coverage.info` to Codecov with flag `hipSPARSE`.

> Preset caveat: the `coverage` CMake preset sets `HIPSPARSE_BUILD_COVERAGE`, which the build does not
> define. Use `HIPSPARSE_ENABLE_COVERAGE` (or the `install.sh` flag) instead.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by the suite.
* *Test coverage* = fraction of intended functionality exercised (backends, routines, types, enums,
  platforms). A high line-coverage number on the marshalling layer can still leave backend-specific
  behavior and the CUDA lane under-tested.

**Scope:** measured on Linux; Windows coverage is not tracked separately.

---

## Nightly Validation

Beyond the PR `standard` category, nightly runs the `comprehensive` category (adds the `nightly`
tier) on the HIP backend via the TheRock nightly workflows; the legacy Jenkins `extended.groovy`
mirrors this with `*nightly*`. The CUDA backend's lane is intentionally narrow (`*checkin*csrmv*`);
broader CUDA validation is not part of the per-PR gate.

---

## Supported Configurations

| Configuration | Validation Level | Frequency | Notes |
|---|---|---|---|
| Linux + HIP backend (rocSPARSE) | Full | PR / Nightly / Release | Primary platform, package `hipsparse` |
| Linux + CUDA 12.8 backend (cuSPARSE) | Partial | PR / Nightly | package `hipsparse-alt`; PR lane is `*checkin*csrmv*` |
| Linux + CUDA 13.x backend | Partial | Nightly | `cuda/13/` YAML set |
| gfx908 / gfx90a / gfx942 (and Jenkins runners gfx900/gfx906/gfx908) | Full/Partial | PR / Nightly | via rocSPARSE backend; TheRock selects GPU families per run |
| gfx1151 (Strix Halo) | Partial | Nightly | `f64_r` / `f64_c` cases excluded (`exclude_gpu_gfx1151`) |
| Windows | Partial | — | Fortran clients off on Windows |

**Explicitly not tested / guaranteed:** enabling HIP and CUDA backends simultaneously is unsupported
(mutually exclusive); CUDA toolkit versions other than 12.8 / 13.x generate no test data; multi-GPU
validation; non-listed gfx targets.

---

## ASAN / TSAN / Sanitizer Coverage

**AddressSanitizer:** `-DHIPSPARSE_ENABLE_ASAN=ON` (legacy `-DBUILD_ADDRESS_SANITIZER=ON`,
`install.sh --address-sanitizer`, or the `asan` preset) builds `hipsparse` and the client libraries
with `-fsanitize=address -shared-libasan`, linking with `lld`. The `asan` preset also enables testing
and forces Fortran off — **ASAN and the Fortran clients are mutually exclusive**. Matrix conversion
(`mtx2csr.exe`) is built with ASAN flags when enabled.

**What is explicitly not covered:** TSAN, UBSAN, and MSAN are not wired into hipSPARSE; device-code
sanitizing follows the backend's constraints.

---

## Fortran Client Testing

Fortran is **sample-only** — there is no Fortran path in `hipsparse-test`. `HIPSPARSE_ENABLE_FORTRAN`
(default ON on non-Windows, OFF on Windows) builds the `hipsparse_fortran` object library from
`library/src/hipsparse.f90` / `hipsparse_enums.f90` and Fortran example binaries under
`clients/samples/` and `documentation_examples/` (e.g. `example_fortran_csrsv2`,
`example_fortran_spmv`). Fortran samples require the HIP backend and are disabled under ASAN. This is
a build/compile check of the Fortran bindings rather than a correctness test suite.

---

## Choosing the Right Test Type

| Scenario | What to add |
|---|---|
| New descriptor/enum/type marshalling or status path | A descriptor `TEST()` or `<routine>_bad_arg` case (host-side, no kernel) |
| New or changed marshalled routine | A YAML case in the appropriate backend directory (`rocm/` and/or `cuda/<ver>/`), validated against the host reference |
| cuSPARSE API-compatibility change | Ensure the CUDA-backend YAML covers it; keep parity with the HIP set where the routine exists on both |
| Bug fix | A regression case that fails before the fix; if the defect stays open, add a YAML `Known bugs:` rule |
| New data type / index base / transpose combination | Extend the routine's YAML / config; the parameterized suite picks it up |
| Performance-sensitive change | Run `hipsparse-bench` before/after; note the delta in the PR (backend perf is owned upstream) |

---

## Known Gaps Summary

| Gap | Regression risk | Impact | Mitigation today |
|---|---|---|---|
| No public/TheRock perf regression gate (PR or nightly) in this repo | Low | Low | Backend (rocSPARSE) perf regression is tracked outside this repo; in-repo `hipsparse-bench` for manual checks |
| CUDA-backend PR lane is narrow (`*checkin*csrmv*`) | Medium | Medium | Broader CUDA coverage only on demand |
| No tracked quarantine list (owner + ticket + expiry) for `known_bug` cases | Low | Low | `*known_bug*` excluded from gating; linkage is ad-hoc |
| `stress` tier instantiated but empty in YAML | Low | Low | No stress cases today |
| Coverage preset flag mismatch (`HIPSPARSE_BUILD_COVERAGE` vs `HIPSPARSE_ENABLE_COVERAGE`) | Low | Low | Use the enable flag / `install.sh` |
| Fortran bindings are compile-checked only, not correctness-tested | Low | Medium | Samples build; no assertion suite |
| No TSAN/UBSAN/MSAN | Low | Medium | ASAN only; ASAN excludes Fortran |

---

## Owners and Review Cadence

**Review this document when:**
* A new backend, CUDA toolkit version, test tier, or CI lane is added.
* A cuSPARSE API-compatibility change lands.
* A regression escapes to an application consumer or is traced to a marshalling error.
* Before a major release, alongside the known-gap review.

The measure of whether this document is working: the Known Gaps table shrinks over time.
