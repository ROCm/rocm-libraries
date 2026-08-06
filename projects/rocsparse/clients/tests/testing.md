# rocSPARSE Testing Strategy

**Status:** Draft
**Owner:** @doctorcolinsmith
**Technical Lead:** @ntrost57
**Last Updated:** 2026-08-06

This document describes how rocSPARSE is tested today, which signals actually gate a merge, and
where the gaps are. It follows the ROCm-wide TESTING.md template and is written as a description of
the current state rather than an aspirational one. A gap that is written down is one that can be
argued about and closed.

---

## Component Overview

rocSPARSE is the ROCm library of Basic Linear Algebra Subroutines (BLAS) for **sparse** computation,
implemented in HIP and optimized for AMD GPUs. It provides sparse level 1/2/3 routines, sparse
format conversions, preconditioner building blocks, and a set of generic API entry points, and it is
the AMD backend for the portable [hipSPARSE](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse)
marshalling library.

**Where it sits in the ROCm stack:** Core math SDK. rocSPARSE sits directly on top of the HIP
runtime and toolchain. It optionally depends on **rocBLAS** (`BUILD_WITH_ROCBLAS`, default ON) and
**rocPRIM**, and optionally uses rocTX for tracing (`BUILD_WITH_ROCTX`).

**Key architectural constraint that shapes testing:** almost every rocSPARSE routine dispatches a
GPU kernel and its result is only meaningful on real AMD hardware. The hardware-independent surface
(argument validation, descriptor/handle management, format-conversion bookkeeping, workspace-size
queries) is small relative to the device code, so confidence comes overwhelmingly from
device-executed integration tests that compare against a host reference rather than from pure host
unit tests.

---

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build the library with test and benchmark clients:**

```bash
cd projects/rocsparse
# -c builds samples + tests + benchmarks; -d fetches build dependencies (incl. googletest)
./install.sh -dc -a gfx942
# equivalently, configure directly:
CXX=/opt/rocm/bin/amdclang++ cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON ../..
```

Out-of-source builds land in `build/<release|debug|release-debug>/clients/staging/`.

**2. Provision the test matrices** (required by the functional suites that read `.csr` inputs):

```bash
# download + convert SuiteSparse matrices into <build>/matrices
./install.sh --matrices-dir-install <path>/rocsparse_matrices
./install.sh -c -a gfx942 --matrices-dir <path>/rocsparse_matrices
```

**3. Run the tests that match what you touched:**

| You changed | Run this | Needs a GPU |
|---|---|---|
| Any routine | `./clients/staging/rocsparse-test --gtest_filter='*quick*-*known_bug*'` | Yes |
| A specific routine | `./clients/staging/rocsparse-test --gtest_filter='*csrmv*'` | Yes |
| Something PR-scoped | `./clients/staging/rocsparse-test --gtest_filter='*quick*:*pre_checkin*-*known_bug*'` | Yes |
| Argument/format validation only | the same, filtered to `*bad_arg*` | No (bad-arg cases are host-side) |

On the single-GPU container, run through the serializer pinned to device 0:

```bash
HIP_VISIBLE_DEVICES=0 gpu-run ./clients/staging/rocsparse-test --gtest_filter='*quick*-*known_bug*'
```

> rocSPARSE auto-skips large configs with "Insufficient memory" on memory-constrained cards. That is
> the suite's memory guard, not a failure.

**4. Add the right kind of test** — see [Choosing the Right Test Type](#choosing-the-right-test-type).

**5. Open the PR** targeting `develop`. The pre-checkin GTest run (`*quick*:*pre_checkin*`, excluding
`*known_bug*`) is the merge gate; another rocSPARSE team member reviews and approves.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** validate hardware-independent logic that can be exercised without dispatching a kernel
to a physical device — argument validation, error-code propagation, handle/descriptor lifecycle, and
type/format bookkeeping.

rocSPARSE does not maintain a separate host-only unit-test binary. The hardware-independent checks
live inside the same GoogleTest client (`rocsparse-test`) as two recognizable classes of case:

* **Bad-argument tests** — every routine has a `testing_<routine>_bad_arg` implementation
  (`clients/testings/testing_<routine>.cpp`) exercised through YAML `function: <routine>_bad_arg`
  entries. These validate null-pointer handling, invalid sizes, unsupported types, and status-code
  propagation, and do not require a kernel launch.
* **Auxiliary API tests** — `clients/tests/test_auxiliary.cpp` (`TEST(auxiliary_pre_checkin, ...)`)
  covers handle create/destroy and descriptor management directly.

**Framework:** GoogleTest (required `1.11.0`; auto-downloaded when absent via `deps/CMakeLists.txt`
`BUILD_GTEST=ON`).

**Location / structure:** a three-layer pattern per routine —
`clients/tests/test_<routine>.yaml` (cases) → `clients/tests/test_<routine>.cpp` (registration via
the `TEST_ROUTINE` macro) → `clients/testings/testing_<routine>.cpp` (implementation, including the
`_bad_arg` variant). At build time `clients/common/rocsparse_gentest.py` expands the master
`clients/tests/rocsparse_test.yaml` into a binary `rocsparse_test.data` consumed at runtime.

**How to run:** `./clients/staging/rocsparse-test --gtest_filter='*bad_arg*'` runs the argument-
validation cases; these are the closest thing to a GPU-free unit run (a device/driver may still be
initialized, but no compute kernel is launched).

**What is NOT covered by unit tests:** numerical correctness of any routine, kernel selection, and
device-side conversions — all of these require a GPU and are covered by integration tests.

**Coverage expectation:** the long-term ROCm-wide goal is >95% line coverage of hardware-independent
paths; this is not mandated initially and will be pursued in phases. rocSPARSE's realistic near-term
target is high coverage of the argument-validation / auxiliary surface, since the dominant code paths
are device kernels whose coverage is not captured by host-side instrumentation. See
[Coverage](#coverage).

---

### Integration Testing Strategy

**Purpose:** validate behavior that requires a real GPU — numerical correctness of every sparse
routine against a host reference, format conversions, and kernel selection across data types, index
types, matrix formats, and problem sizes.

Integration tests are the bulk of rocSPARSE testing: roughly **150+ routines**, each with a
parameterized/typed GoogleTest suite driven by YAML. Cases are typed via
`clients/tests/rocsparse_test_config.hpp` (real/complex, mixed index types) and parameterized with
`TEST_P` + `INSTANTIATE_TEST_CATEGORIES`, so a new data type or index combination inherits the
standard case matrix rather than requiring hand-written variants.

**Runtime tiers (YAML `category:` field → GTest suite prefix):**

| Tier | Meaning | Typical use |
|---|---|---|
| `quick` | small, fast cases | local smoke, PR |
| `pre_checkin` | PR / checkin scope | PR merge gate |
| `nightly` | larger shapes and broader type coverage | nightly |
| `stress` | largest / longest cases | weekly / release |
| `known_bug` | cases exposing a tracked defect | excluded from all gating runs |

**Test data / matrices:** functional suites read `.csr` matrices converted from **24 SuiteSparse**
matrices (e.g. `nos1`–`nos7`, `amazon0312`, `webbase-1M`) downloaded and MD5-verified by
`cmake/ClientMatrices.cmake` (mirror overridable via `ROCSPARSE_TEST_MIRROR`). The runtime matrix
directory is set with `--matrices-dir` or `ROCSPARSE_CLIENTS_MATRICES_DIR`.

**What requires GPU hardware:** all numerical-correctness and conversion cases. **What runs without
a compute kernel:** `*bad_arg*` and auxiliary API cases.

**Managed-memory (HMM) cases:** `*csrmv_managed*` are run twice on gfx90a/gfx942, with
`HSA_XNACK=0` and `HSA_XNACK=1`.

**What runs on PRs:** build + `*quick*:*pre_checkin*` (excluding `*known_bug*`).
**What runs nightly:** the `*nightly*` tier (Jenkins `extended.groovy`, timeout ~600 min).
**What runs at release / on demand:** `stress` tier and the emulation `smoke`/`regression`/`extended`
YAML subsets driven through `rtest.py`.

**Test-size / coverage guidance:** prefer a small set of representative sizes over exhaustive
numerical variants — the typed/parameterized suites already sweep type and index combinations, so
adding many near-duplicate sizes increases run time without adding meaningful coverage. Large configs
are intentionally guarded and skipped with "Insufficient memory" on small cards.

---

### Performance and Benchmarking Testing

**Purpose:** detect throughput/bandwidth regressions per routine on a given architecture. Absolute
numbers are not comparable across GFX targets.

| Item | Detail |
|---|---|
| Stack layer | Core SDK (sparse BLAS primitive) |
| Metrics measured | Time, GFLOP/s, and memory bandwidth per routine |
| How benchmarks are run | `rocsparse-bench` from `clients/staging/`, e.g. `./rocsparse-bench -f csrmv --bench-x -M 10 20 30 40` (command-line expansion sweeps option lists) |
| Baseline — stored per architecture | Not stored/aggregated automatically; comparison is manual. Results must not be aggregated across GFX |
| Where results are stored | JSON by default (`a.json`); optional YAML export; optional `rocsparse_bench_memstat.json` with memstat |
| Regression threshold | `scripts/rocsparse-bench-regression.py` compares JSON runs with a `--tol` (default 2%); not wired into CI |
| Gating approach | Manual review |
| GPU profiling | Not integrated into the benchmark flow |

**Gating:**

| Gating Level | Status | Notes |
|---|---|---|
| PR-level automated gate | No | Known gap |
| Nightly automated comparison | No | Known gap |
| Manual review | Yes | `rocsparse-bench-regression.py` compares runs on request |
| Release qualification | Partial | Performance reviewed before release; not an automated sign-off |

**Known gaps:** no per-architecture baseline stored in a queryable format; no automated regression
threshold enforced on PRs or nightly.

---

## Why We Test This Way

rocSPARSE owns its kernels, but nearly all meaningful behavior is only observable on real AMD
hardware, so the strategy is integration-dominant: a large YAML-driven GoogleTest suite runs each
routine on-device and validates against a host reference. The hardware-independent surface (argument
validation, handle/descriptor lifecycle, auxiliary API) is exercised by the `*bad_arg*` and
auxiliary cases inside the same binary rather than by a separate host unit-test target.

The tier system (`quick` → `stress`) is a sampling strategy over a large combinatorial space (routine
× data type × index type × matrix format × size); exhaustive coverage is not achievable, so the tiers
trade breadth for time-to-signal. Typed/parameterized suites keep the matrix maintainable: adding a
type or index combination extends coverage without new hand-written cases.

---

## Pre-submit / CI Gates

CI for rocSPARSE runs primarily through internal AMD **Jenkins** pipelines (`.jenkins/`). There is no
rocSPARSE-dedicated GitHub Actions workflow in this checkout; monorepo-level TheRock workflows build
and test the `sparse` component.

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build (Linux; static and shared) | Yes | CI / DevOps | `precheckin.groovy`, `static.groovy`, `debug.groovy` |
| Unit / bad-arg tests | Yes | Component team | Part of the `*quick*:*pre_checkin*` GTest run |
| Integration tests | Yes | Component team | `*quick*:*pre_checkin*`, excluding `*known_bug*` |
| Static analysis / formatting | Yes | CI / DevOps | `staticanalysis.groovy`; repo-wide `pre-commit`, `clang-tidy`, `codeql` |
| ASAN | Yes | CI / DevOps | `asan.groovy` — `*quick*:*pre_checkin*` under AddressSanitizer |
| Code coverage | No | Component team / CI | `codecov.groovy`, informational (flag `rocSPARSE`) |
| Shared validation infra | N/A | TheRock team | Shared build/validation infrastructure |
| Release qualification | N/A | Component team + QA + TPM | Readiness and known-gap review |

**Build command pattern (pre-checkin):**

```bash
./install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices \
  && ./install.sh -c -a $gfx_arch --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices
```

### PR Test Classification

| Status | Applies to |
|---|---|
| Trusted gate | `*quick*` and `*pre_checkin*` cases (excluding `*known_bug*`) run on the PR GPU runners; ASAN quick run |
| Informational | Coverage upload (Codecov); nightly `*nightly*` results |
| Unstable / flaky | `known_bug`-tagged cases (excluded from gating) |

**Flaky / known-bug policy:** rocSPARSE has no `known_bugs.yaml`. A case that exposes a tracked
defect is tagged `category: known_bug` (either directly in the routine's YAML or reassigned by
`rocsparse_gentest.py` from a `Known bugs` note in `rocsparse_common.yaml`), and every gating run
excludes `*known_bug*`. This keeps known failures out of blocking runs but does not by itself carry an
owner or ticket per case — a tracked quarantine list with expiry is a gap (see
[Known Gaps](#known-gaps-summary)). A known-bug tag is not an accepted permanent state.

---

## Coverage

**Tool:** LLVM source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`), enabled with
`-DBUILD_CODE_COVERAGE=ON` (via `install.sh --codecoverage`, which requires a Debug `-g` or
RelWithDebInfo `-k` build). `lcov`/`gcovr` are explicitly not used because they do not work with LLVM
coverage.

**How to build and run:**

```bash
./install.sh -kc --codecoverage -a gfx942
cd build/release-debug
make coverage_cleanup coverage GTEST_FILTER='*quick*:*pre_checkin*-*known_bug*'
# targets: coverage_analysis -> coverage (llvm-profdata merge -> llvm-cov export lcov -> filter -> genhtml)
```

The report is produced under `coverage-report/`; exclusions are applied by
`scripts/filter_lcov_exclusions.py`. Jenkins `codecov.groovy` uploads `coverage.info` to Codecov with
flag `rocSPARSE`.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed (e.g. 700 of 1,000 → 70%).
* *Test coverage* = fraction of intended functionality exercised (types, index types, formats, sizes,
  platforms). Because host-side instrumentation does not capture device kernels, rocSPARSE's measured
  code coverage understates work done on-device; device-path coverage is the largest gap.

**Scope:** measured on Linux; Windows coverage is not tracked separately.

---

## Nightly Validation

Beyond PR validation, nightly (`extended.groovy`) adds:

* The full `*nightly*` tier — larger shapes and broader type/format coverage (timeout ~600 min).
* Additional GPU architectures beyond the PR runners.
* Emulation subsets (`smoke`, `regression`, `extended`) via `rtest.py --emulation`, reading
  `rocsparse_smoke.yaml`, `rocsparse_regression.yaml`, and `rocsparse_extended.yaml`.

---

## Supported Configurations

Default GPU targets come from `DEFAULT_GPU_TARGETS` in the root `CMakeLists.txt`.

| Configuration | Validation Level | Frequency | Notes |
|---|---|---|---|
| Linux (ROCm) | Full | PR / Nightly / Release | Primary platform |
| gfx908 / gfx90a / gfx942 / gfx950 | Full | PR / Nightly / Release | Core CDNA targets |
| gfx900 / gfx906 | Partial | PR / Nightly | Older CDNA/GCN |
| gfx1030 / gfx110x / gfx1200 / gfx1201 | Partial | Nightly | RDNA |
| gfx1151 (Strix Halo) | Partial | Nightly | `f64_r` / `f64_c` cases excluded (`exclude_gpu_gfx1151`) |
| ASAN (gfx908/gfx90a/gfx942 `xnack+`) | Partial | PR | AddressSanitizer build, xnack+ only |

**Explicitly not tested / guaranteed:** multi-GPU validation; non-listed gfx targets; Windows
coverage is thinner than Linux. Configurations are guarded at runtime by the "Insufficient memory"
skip on small cards.

---

## ASAN / TSAN / Sanitizer Coverage

**AddressSanitizer:** `-DBUILD_ADDRESS_SANITIZER=ON` (`install.sh --address-sanitizer`) builds with
`-fsanitize=address -shared-libasan`, links with `lld`, defines `ROCSPARSE_WITH_ASAN`, and depends on
`hip-runtime-amd-asan`. Under ASAN, GPU targets are restricted to `gfx908:xnack+`, `gfx90a:xnack+`,
`gfx942:xnack+`, and the Fortran clients are forced off. It catches host and (xnack+) device memory
errors: out-of-bounds and use-after-free.

**Runtime (Jenkins `asan.groovy`):** `ASAN_OPTIONS=detect_leaks=1`,
`LSAN_OPTIONS=suppressions=suppr.txt` (suppresses leaks in `libhsa-runtime64`, `libamd_comgr`,
`libamdhip64`, `libhsakmt`), `ASAN_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer`. The ASAN lane
runs `*quick*:*pre_checkin*`.

**What is explicitly not covered:** TSAN, UBSAN, and MSAN are not wired into rocSPARSE; non-xnack
device configurations are not ASAN-covered.

---

## Choosing the Right Test Type

| Scenario | What to add |
|---|---|
| New argument/format validation or status-code path | A `testing_<routine>_bad_arg` case (host-side, no kernel) |
| New or changed on-device routine behavior | A YAML case in `test_<routine>.yaml` at the appropriate tier, validated against the host reference |
| New data type or index type | Extend the typed config (`rocsparse_test_config*`); the parameterized suite picks up the combinations |
| Bug fix | A regression case that fails before the fix; if it exposes a still-open defect, tag `category: known_bug` with a tracking issue |
| Performance-sensitive change | Run `rocsparse-bench` before/after and compare with `scripts/rocsparse-bench-regression.py`; include the delta in the PR |
| New GPU architecture | Validate on that target and update Supported Configurations |

---

## Known Gaps Summary

| Gap | Regression risk | Impact | Mitigation today |
|---|---|---|---|
| No automated performance regression gate (PR or nightly) | High | High | Manual `rocsparse-bench-regression.py` comparison |
| No per-architecture perf baseline in a queryable format | High | High | JSON logs only; comparison is manual |
| Device-code coverage not captured (host-side instrumentation only) | Medium | Medium | Codecov reports host paths; device coverage untracked |
| No tracked quarantine list (owner + ticket + expiry) for `known_bug` cases | Medium | Medium | `*known_bug*` excluded from gating; linkage is ad-hoc |
| No separate host-only unit-test binary | Low | Medium | `*bad_arg*` / auxiliary cases cover host paths inside `rocsparse-test` |
| No TSAN/UBSAN/MSAN | Low | Medium | ASAN only |
| Windows coverage thinner than Linux; multi-GPU unvalidated | Medium | Medium | Documented as not guaranteed |

---

## Owners and Review Cadence

**Review this document when:**
* A new test tier, CTest category, or CI lane is added.
* A regression escapes to a downstream consumer (e.g. hipSPARSE) — direct evidence of a gap here.
* Before a major release, alongside the known-gap review.

The measure of whether this document is working: the Known Gaps table shrinks over time.
