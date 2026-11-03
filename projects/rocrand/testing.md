# rocRAND Testing Strategy (TESTING.md)

Status: Draft\
Owner: @RobsonRLemos\
Technical Lead: @stanleytsang-amd\
Last Updated: August 05, 2026

## Component Overview
rocRAND is the ROCm pseudorandom and quasirandom number generation library, implemented in HIP and optimized for AMD GPUs. It is a foundational math primitive that sits directly on top of HIP/ROCm and serves as the backend for `hipRAND` (the ROCm equivalent of `cuRAND`).

Three properties shape the entire test strategy:
* **Separate engines and distributions.** Each PRNG/QRNG engine and each distribution is implemented independently, so they can be unit tested in isolation.
* **Two public APIs.** rocRAND exposes a **host API** and a **device (in-kernel) API**, each with its own test and benchmark suite and independent code paths.
* **cuRAND parity.** Because rocRAND is the backend for `hipRAND` and a drop-in for `cuRAND`, it must maintain output/stream parity with `cuRAND` and expose compatible C, C++, Fortran, and Python interfaces.

**Key constraint:** most of rocRAND's logic is device code. There is relatively little hardware-independent logic, so much of the suite necessarily runs on a GPU rather than as pure host unit tests.

## Development Workflow
The sequence a developer follows from writing code to getting it merged:

1. Build with tests enabled: `cmake -B build -DBUILD_TEST=ON -DGPU_TARGETS=<gpu_arch>` then `make -j` (or `-GNinja`).
2. Run the fast tier locally against a GPU: `cd build && ctest -L quick` (target < 5 min). For focused work, run a single binary directly, e.g. `./test/test_rocrand_xorwow_prng`.
3. For engine/distribution changes, run the relevant `test/internal/` white-box tests and the matching kernel/host integration tests.
4. Run `clang-format` (clang-format 17) on changed files; a githook is available via `./.githooks/install`.
5. Open a PR. Required CI checks — **TheRock CI** and **Math CI** — must pass, and another rocRAND team member must review and approve.


# Testing Strategy and Layers

## Unit Testing Strategy
**Purpose:** validate engine and distribution logic in isolation, including behavior that does not depend on the full public API. Where practical, tests validate a single isolated piece of logic without dispatching a kernel to a physical device.

Because engines and distributions are decoupled, `test/internal/` includes library-internal headers and tests each engine's state, ordering, and skip-ahead logic directly, independent of the public API.

* **Framework:** GoogleTest.
* **Location:**
  * `test/internal/` — white-box tests that include library-internal headers: per-engine PRNG/QRNG state and ordering, distributions (uniform, normal, log-normal, Poisson), discrete distributions, config dispatch, generator-type metadata, and C++ utility helpers.
  * `test/linkage/` — multiple-translation-unit linkage / ODR checks.
* **Naming convention:** `test_rocrand_<engine>_prng.cpp` / `test_<distribution>_distribution.cpp` (e.g. `test_rocrand_threefry2x32_20_prng` tests the `threefry2x32_20` engine). GoogleTest suites use typed/parameterized suites per engine.
* **How to run:** `ctest -L quick` (fast tier) or run a binary directly, e.g. `./test/test_rocrand_xorwow_prng`.
* **Not covered by unit tests:** on-device end-to-end throughput, cross-API behavior, and packaging.

**What is unit-testable in rocRAND (hardware-independent / host-side):**
* Argument validation and error-code propagation through the host API.
* Generator-type metadata and config dispatch.
* Distribution math that can run on the host (uniform/normal/log-normal/Poisson transforms) and C++ utility helpers.
* Linkage / ODR correctness across translation units.

**What is not unit-testable (requires a GPU driver / device):**
* In-kernel generation, GPU memory allocation/copies (`hipMalloc`/`hipMemcpy`), kernel launches.
* On-device engine state, ordering, and skip-ahead as executed on hardware.


### Coverage expectation
* Long-term goal across ROCm components is **> 95%** unit-test line coverage; this is **not mandated initially** and will be pursued in phases.
* Realistic near-term target for rocRAND: **≥ 80% of hardware-independent paths where practical.**
* **Current state:** CI reports roughly **50%** line coverage because coverage instrumentation currently captures host-side code only, while most of rocRAND is device code. This is the single largest coverage gap. An initiative to adopt LLVM device-code coverage is expected to close it.

## Integration Testing Strategy
**Purpose:** validate behavior that requires a GPU, runtime layers, or cross-API interaction — behavior that cannot be validated by host-only unit tests.

| Test Type | Location | Purpose | GPU Required | Frequency |
| --- | --- | --- | --- | --- |
| Host-API (black box) | `test/test_rocrand_generate*.cpp`, `test_rocrand_host.cpp` | Validate public host generation API across engines/distributions | Yes | PR / Nightly |
| Device-API kernels | `test/test_rocrand_kernel_*.cpp` | Validate in-kernel generators per engine | Yes | PR / Nightly |
| C++ wrapper | `test/test_rocrand_cpp_wrapper.cpp`, `test_rocrand_cpp_basic.cpp`, `test/cpp_wrapper/` | Validate the `rocrand.hpp` C++ interface builds and behaves | Yes | PR / Nightly |
| cuRAND parity | `test/parity/` | Compare rocRAND output/stream against cuRAND | Yes (CUDA backend) | as available |
| hipGraph capture | `test/test_rocrand_hipgraphs.cpp` | Validate generation under HIP graph capture/replay | Yes | PR / Nightly |
| Fortran wrapper | `test/fortran/` | Validate Fortran bindings (FRUIT) | Yes | Nightly |
| Package / install | `test/package/` | Post-install smoke test via `find_package(rocrand)` | Yes | Release / packaging |

* **What requires GPU hardware:** all of the above except pure host-side compilation/linkage checks.
* **What runs on CPU-only systems:** host-side unit tests in `test/internal/` and `test/linkage/`; on the CUDA backend, cuRAND parity requires an NVIDIA GPU.
* **Test-size / coverage guidance:** engines and distributions are covered via typed/parameterized suites, so a new engine automatically inherits the standard statistical + reproducibility cases rather than requiring a large hand-written matrix. Prefer a small set of statistically meaningful sizes over exhaustive numerical variants.

## Performance & Benchmarking Testing
**Purpose:** detect regressions in generation throughput for both APIs by comparing against a per-architecture baseline over time. Absolute numbers across architectures are not comparable.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (foundational math primitive) |
| Metrics measured | Generation throughput for host-API and device-API across engines, distributions, orderings, and data types |
| How benchmarks are run | primbench-based binaries built with `-DBUILD_BENCHMARK=ON`: `benchmark/benchmark_rocrand_host_api`, `benchmark/benchmark_rocrand_device_api`; `benchmark_curand_*` for parity. AMD SMI required |
| Baseline — stored per architecture | Not formally stored/aggregated today; results are compared manually. Results must **not** be aggregated across GFX |
| Where results are stored | JSON by default (`benchmark_rocrand_*_api.json`), optional CSV |
| Regression threshold | No fixed automated threshold; regressions caught via manual review |
| Gating approach | Manual QA review |
| Test run time | Full benchmark sweep is long-running; run selectively via `--filter` during development |
| GPU profiling | AMD SMI required for benchmarks; no dedicated profiling gate |

### Gating
| Gating Level | Status | Notes |
| --- | --- | --- |
| PR-level automated gate | No | Known gap — no automated performance gate on PRs |
| Nightly automated comparison | No | Benchmarks run but are not auto-compared to a stored baseline |
| Manual nightly review | Yes | Maintainers review benchmark trends |
| Release qualification | Partial | Performance reviewed before release; not a formal automated sign-off |

### Known Gaps
* No per-architecture baselines are formally stored for automated comparison.
* No automated regression threshold or PR-level performance gate.
* Framework-level regressions in downstream consumers are not automatically traced back to rocRAND.

## Pre-submit / CI Gates

### Validation Gates and Ownership
| Validation Area | Required Before Merge | Owner | Responsibility |
| --- | --- | --- | --- |
| Build (Linux, Windows) | Yes | TheRock / Math CI | Multi-OS, multi-arch build matrix (gfx94X, gfx950, gfx11xx incl. gfx1151) |
| Unit tests | Yes | Component team /TheRock /Math CI | |
| Formatting | Yes | CI | Ran with pre-commit |
| Code coverage | No | Component team / codecov | Informational (Codecov); no enforced threshold |

### PR Test Classification
| Status | Applies to |
| --- | --- |
| Trusted gate | The full test suite is ran on multiple GPU architectures. |
| Informational | codecov |
| Unstable / flaky | None formally tagged today (see Flaky Test Policy) |

### Flaky Test Policy
* Flaky tests should be tagged clearly (e.g. `UNSTABLE`) and excluded from blocking runs until fixed.
* Every flaky test should have an owner and a tracking bug.
* A flaky test is not an accepted final state. rocRAND does not currently maintain a tagged flaky list — establishing one is a gap.


## Coverage
* **Tooling:** `CODE_COVERAGE=ON` (clang only) compiles with `-fprofile-instr-generate -fcoverage-mapping` and defines `CODE_COVERAGE_ENABLED`. No `gcovr`/`lcov` report target is wired into the repo today.
* **Target:** long-term > 95% (phased, aspirational); near-term ≥ 80% of hardware-independent paths where practical.
* **Scope / excluded paths:** Codecov ignores `benchmark/`, `build/`, `cmake/`, `docs/`, `hipRAND/`, `python/`, `scripts/`, `test/`, `tools/`. Coverage is measured on Linux/clang; Windows coverage is not tracked separately.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by tests (e.g. 700 of 1,000 lines → 70%).
* *Test coverage* = fraction of intended functionality/scenarios exercised. rocRAND can show ~50% code coverage while important scenarios (device paths, Windows, multi-GPU) remain under-tested — device coverage is currently the largest gap.

### PR Validation Summary
| Validation Area | Required Before Merge | Owner | Notes |
| --- | --- | --- | --- |
| Build | Yes | CI / DevOps (TheRock) | Multi-OS, multi-arch |
| Unit tests | Yes | Component team ||
| Integration tests | Yes | Component team ||
| Static analysis | No | CI | Not gated |
| Code coverage | No | Component team / CI | Informational |
| Formatting | Yes | CI | |

### Nightly Validation
* **standard** category (`ctest -L standard`) — full GoogleTest run (timeout budget up to 4 hours).
* **ffm-quick / ffm-full** categories (`ctest -L ffm-quick`) — Full-Feature-Matrix focused runs (timeout budget up to 2 hours).
* Slow/niche tests enabled via `RUN_SLOW_TESTS=1`.
* Additional hardware coverage: gfx94X, gfx950, and gfx11xx (specifically gfx1151).
* cuRAND parity and Fortran-wrapper suites run here rather than on PRs.

## Supported Configurations
GPU targets come from the top-level `CMakeLists.txt` default target list.

| Configuration | Validation Level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux (ROCm) | Full | PR / Nightly / Release | Primary platform |
| Windows (HIP on Windows) | Partial | PR / Nightly / Release | Built via `rmake.py` |
| CUDA backend | Partial | Nightly / as available | Used for cuRAND parity |
| gfx90a / gfx942 / gfx950 | Full | PR/ Nightly / Release | |
| gfx1030 / gfx11xx (incl. gfx1151) | Full | PR / Nightly | |
| amdgcnspirv (SPIR-V) | Partial | Build | `-DAMDGPU_TARGETS=amdgcnspirv` |

**Explicitly not guaranteed:** multi-GPU validation and non-listed gfx targets are not formally validated; Windows coverage is thinner than Linux.

## Sanitizer Coverage (ASAN / TSAN)
* **AddressSanitizer:** `BUILD_ADDRESS_SANITIZER=ON` builds an ASAN variant; when enabled, GPU targets are restricted to xnack+ variants. Catches GPU/host memory errors (out-of-bounds, use-after-free).
* **TSAN:** not currently used.
* **GPU-specific limitation:** device ASAN requires xnack+ targets and adds significant runtime cost; it is not run on every PR.
* **How to build:** `cmake -B build -DBUILD_ADDRESS_SANITIZER=ON ...`
* **Not covered:** thread-sanitizer, UB-sanitizer, and non-xnack device configurations.
