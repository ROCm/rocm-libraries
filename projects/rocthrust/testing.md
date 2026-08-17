# rocThrust Testing Strategy (TESTING.md)

Status: Draft\
Owner: @RobsonRLemos\
Technical Lead: @stanleytsang-amd\
Last Updated: August 11, 2026

## Component Overview
rocThrust is the ROCm parallel-algorithms library — a HIP port of NVIDIA's Thrust. It provides a high-level, STL-like C++ template interface (`thrust::sort`, `thrust::reduce`, `thrust::transform`, iterators, `device_vector`, execution policies, memory resources, etc.) implemented on top of **rocPRIM** and optimized for AMD GPUs. It sits one layer above rocPRIM in the ROCm math stack and is source-compatible with CUDA Thrust so existing Thrust code ports to AMD GPUs with minimal changes.

Three properties shape the entire test strategy:
* **Header-only, heavily templated library.** Almost every algorithm is a template that must work across many element types (integral, floating-point, `custom_numeric`, tuples, complex), container types, iterator categories, and input sizes. Tests are therefore typed/parameterized so a single algorithm is exercised across a large type × size matrix.
* **Thin layer over rocPRIM.** rocThrust dispatches most heavy lifting to rocPRIM kernels, so correctness depends on the backend, and nearly all behavior is only observable once a kernel runs on a device.
* **Two coexisting test suites.** A modern GoogleTest suite (`test/`) is rocThrust's primary suite, alongside the upstream Thrust `unittest` macro framework (`testing/`) inherited from NVIDIA Thrust. Both must build and pass.

**Key constraint:** rocThrust is fundamentally a device library. Algorithms dispatch kernels to the GPU, so essentially the entire suite requires a GPU to run. Hardware-independent logic is limited to compile-time metaprogramming (type traits, tag dispatch) and a small amount of host-side plumbing.

## Development Workflow
The sequence a developer follows from writing code to getting it merged:

1. Build with tests enabled: `CXX=hipcc cmake -B build -DBUILD_TEST=ON -DGPU_TARGETS=<gpu_arch>` then `make -j` (or configure with `-GNinja`). On Windows use `python rmake.py -c -a <gpu_arch>`.
2. Run the fast tier locally against a GPU: `cd build && ctest -L quick` (target < 5 min). For focused work, run a single test binary directly, e.g. `./test/sort.hip --gtest_filter="Sort*"`.
3. For changes to a specific algorithm, run both the matching `test/test_<algorithm>.cpp` (GoogleTest) and the corresponding `testing/<algorithm>.cu` (upstream) binary.
4. When running on a multi-GPU host, generate a resource spec and let CTest distribute work: `./generate_resource_spec resources.json` then `ctest --resource-spec-file ./resources.json --parallel <N>`.
5. Run `clang-format` on changed files (config in `.clang-format`; a git hook is available via `./.githooks/install`).
6. Open a PR. Required CI checks — **TheRock CI** and **Math CI** — must pass across the build matrix, and another rocThrust team member must review and approve.


# Testing Strategy and Layers

## Unit Testing Strategy
**Purpose:** validate algorithm correctness across the type/size matrix and validate library-internal behavior. Where practical, tests validate isolated logic; in rocThrust most "unit" tests still dispatch a kernel because the algorithm under test is device code.

* **Frameworks:**
  * **GoogleTest** — the primary, rocThrust-native suite in `test/`.
  * **Thrust `unittest`** — the upstream macro framework in `testing/`, using `DECLARE_UNITTEST(...)` and `ASSERT_*` macros (`ASSERT_EQUAL`, `ASSERT_THROWS`, `ASSERT_ALMOST_EQUAL`, `ASSERT_EQUAL_RANGES`, …) with type lists (`IntegralTypes`, `FloatingPointTypes`, `ThirtyTwoBitTypes`, …).
* **Location:**
  * `test/` — ~165 GoogleTest files, one binary per algorithm, e.g. `test_copy.cpp`, `test_sort.cpp`, `test_reduce.cpp`. Shared helpers: `test_utils.hpp`, `test_param_fixtures.hpp` (typed-test parameter lists), `test_real_assertions.hpp` (custom assertions), generated `test_seed.hpp`.
  * `testing/` — ~160 upstream `.cu` tests plus the framework under `testing/unittest/` (`unittest.h`, `assertions.h`, `testframework.*`, HIP backend under `testing/unittest/hip/`).
  * `test/bitwise_repro/` — bitwise-reproducibility harness backed by SQLite (`bwr_db.hpp`), driven by `test/test_reproducibility.cpp`, to confirm results are bit-for-bit stable across runs.
  * `test/hipstdpar/` — tests for C++ standard parallel algorithms offloaded via `--hipstdpar` (requires TBB; gated by `BUILD_HIPSTDPAR_TEST` / `BUILD_HIPSTDPAR_TEST_WITH_TBB`).
* **Naming convention:** GoogleTest files are `test_<algorithm>.cpp` producing a `<algorithm>.hip` binary; upstream files are `<algorithm>.cu` producing a `test_thrust_<algorithm>` binary. GoogleTest suites use `TYPED_TEST_SUITE` over parameter lists (e.g. `Params<T>`) so each algorithm runs across many element types.
* **How to run:** `ctest -L quick` (fast tier) or run a binary directly, e.g. `./test/reduce.hip`.
* **Reproducibility / seeding:** random test data is seeded at CMake configure time via `RNG_SEED_COUNT` (non-repeatable seeds from `std::random_device`) and `PRNG_SEEDS` (semicolon-delimited repeatable 32-bit seeds). On failure, GoogleTest reports the seed so the case can be reproduced.
* **Not covered by unit tests:** end-to-end throughput/performance, packaging/install, and cross-platform behavior beyond the built target.

**What is unit-testable in rocThrust (hardware-independent / host-side):**
* Compile-time metaprogramming: type traits, tag/execution-policy dispatch, iterator-category selection (e.g. `test_type_traits.cpp`, `TypeTraitsTests`, `MetaprogrammingTests`, `PreprocessorTest`).
* Host-side container/allocator plumbing and static assertions (e.g. address-stability static assertions).

**What is not unit-testable (requires a GPU driver / device):**
* Every algorithm that dispatches a kernel through rocPRIM (sort, scan, reduce, transform, set operations, …).
* `device_vector` allocation/copies (`hipMalloc`/`hipMemcpy`), kernel launches, device references/pointers, and async/event behavior as executed on hardware.


### Coverage expectation
* Long-term goal across ROCm components is **> 95%** unit-test line coverage; this is **not mandated initially** and will be pursued in phases.
* Realistic near-term target for rocThrust: **≥ 80% of hardware-independent paths where practical.**
* **Current state:** CI reports roughly **91.6%** line coverage.

## Integration Testing Strategy
**Purpose:** validate behavior that requires a GPU, the rocPRIM backend, the HIP runtime, or cross-configuration interaction — i.e. essentially all algorithm behavior, since rocThrust dispatches kernels.

| Test Type | Location | Purpose | GPU Required | Frequency |
| --- | --- | --- | --- | --- |
| Algorithm tests (GoogleTest) | `test/test_<algorithm>.cpp` | Validate each algorithm across type/size matrix on device | Yes | PR / Nightly |
| Upstream Thrust tests | `testing/<algorithm>.cu` | Validate parity with NVIDIA Thrust semantics via the legacy `unittest` framework | Yes | PR / Nightly (Linux only) |
| Bitwise reproducibility | `test/bitwise_repro/`, `test/test_reproducibility.cpp` | Confirm bit-for-bit stable results across runs (SQLite-backed) | Yes | PR / Nightly |
| HIPSTDPAR | `test/hipstdpar/` | Validate C++ parallel algorithms offloaded via `--hipstdpar` (needs TBB) | Yes | Nightly / opt-in |
| Multi-GPU distribution | CTest RESOURCE_GROUPS | Distribute the suite across multiple same-family GPUs | Yes (2+ GPUs) | as available |
| Package / install | `find_package(rocthrust)` consumer build | Post-install smoke check that headers/targets resolve | Yes | Release / packaging |

* **What requires GPU hardware:** effectively all of the above — every algorithm test launches kernels.
* **What runs on CPU-only systems:** only compile-time/metaprogramming assertions and build/link checks; there is no meaningful host-only runtime suite.
* **Windows note:** the upstream `testing/` suite is **not built on Windows** (`if (NOT WIN32 AND BUILD_TEST)` in the top-level `CMakeLists.txt`); the modern `test/` GoogleTest suite is the Windows path, driven by `rmake.py` / `rtest.py`.
* **Test-size / coverage guidance:** algorithms are covered via typed/parameterized suites, so a new algorithm inherits the standard type × size cases rather than a hand-written matrix. Prefer a small set of representative sizes (including edge sizes: 0, 1, small, and one large/`ReduceLargeTest`-style case) over exhaustive numerical variants.

## Performance & Benchmarking Testing
**Purpose:** detect regressions in algorithm throughput by comparing against a per-architecture baseline over time. Absolute numbers across architectures are not comparable.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (parallel-algorithm primitives layer, above rocPRIM) |
| Metrics measured | Per-algorithm throughput (items/s, bytes/s) across element types and input sizes |
| How benchmarks are run | Google Benchmark binaries built with `-DBUILD_BENCHMARK=ON`, organized under `benchmark/bench/<algorithm>/`; rocRAND is used to generate random input data |
| Baseline — stored per architecture | Not formally stored/aggregated today; results are compared manually. Results must **not** be aggregated across GFX |
| Where results are stored | Google Benchmark output (JSON/console); no central dashboard wired into the repo |
| Regression threshold | No fixed automated threshold; regressions caught via manual review |
| Gating approach | Manual review |
| Test run time | Full sweep is long-running; run selectively via Google Benchmark `--benchmark_filter` during development |
| GPU profiling | No dedicated profiling gate in the repo |

### Gating
| Gating Level | Status | Notes |
| --- | --- | --- |
| PR-level automated gate | No | Known gap — no automated performance gate on PRs |
| Nightly automated comparison | No | Benchmarks build/run but are not auto-compared to a stored baseline |
| Manual nightly review | Yes | Maintainers review benchmark trends |
| Release qualification | Partial | Performance reviewed before release; not a formal automated sign-off |

### Known Gaps
* No per-architecture baselines are formally stored for automated comparison.
* No automated regression threshold or PR-level performance gate.
* Downstream (framework-level) regressions are not automatically traced back to rocThrust.

### Upcoming Changes
rocThrust performance testing is getting reworked to use `Primbench` instead of `GoogleBench` for more consistent results. See PR [8499](https://github.com/ROCm/rocm-libraries/pull/8499).

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
* A flaky test is not an accepted final state. rocThrust does not currently maintain a tagged flaky list — establishing one is a gap.

## Coverage
* **Tooling:** `CODE_COVERAGE=ON` (clang only) compiles with `-O0 -fprofile-instr-generate -fcoverage-mapping`. Codecov configuration lives at the `rocm-libraries` monorepo level (`codecov.yml`), not per-component. No `gcovr`/`lcov` report target is wired into rocThrust today.
* **Target:** long-term > 95% (phased, aspirational);
* **Scope / limitations:** instrumentation is host-side and clang-only; device-code coverage is not measured. Windows coverage is not tracked separately.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by tests (e.g. 700 of 1,000 lines → 70%).
* *Test coverage* = fraction of intended functionality/scenarios exercised. rocThrust can show low host code coverage while still having broad test coverage of algorithms, because most executed logic is device code that host instrumentation does not see. Conversely, configurations such as Windows and multi-GPU remain under-exercised even where line coverage looks reasonable.

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
* **standard** category (`ctest -L standard`, gtest filter `*`) — full GoogleTest run (timeout budget up to 4 hours).
* **ffm-quick** category (`ctest -L ffm-quick`) — Full-Feature-Matrix quick tests (timeout budget up to 2 hours).
* Larger-shape cases (e.g. `ReduceLargeTest`, `StableSortLargeTests`) and the upstream `testing/` suite run beyond the PR `quick` tier.
* Additional hardware coverage across the default GPU target list.

## Supported Configurations
GPU targets come from the top-level `CMakeLists.txt` default target list (`GPU_TARGETS`/`AMDGPU_TARGETS`, default "all").

| Configuration | Validation Level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux (ROCm) | Full | PR / Nightly / Release | Primary platform; only platform that builds the `testing/` upstream suite |
| Windows (HIP on Windows) | Partial | PR / Nightly / Release | Built via `rmake.py` / `rtest.py`; upstream `testing/` suite not built |
| gfx90a / gfx942 / gfx950 | Full | PR / Nightly / Release | |
| gfx908 / gfx906 | Full | Nightly / Release | |
| gfx103x / gfx11xx (incl. gfx1151) | Full | PR / Nightly | |
| gfx120x / gfx1250 | Partial | Nightly | Newer targets in default list |

**Explicitly not guaranteed:** multi-GPU validation is opportunistic (depends on host GPU count via CTest resource allocation) and not a formal gate; non-listed gfx targets are not validated; Windows coverage is thinner than Linux (no upstream suite).

## Sanitizer Coverage (ASAN / TSAN)
* **AddressSanitizer:** `BUILD_ADDRESS_SANITIZER=ON` defines `ADDRESS_SANITIZER_BUILD`, compiles with `-fsanitize=address -fno-omit-frame-pointer`, and links with `-fsanitize=address -shared-libasan -fuse-ld=lld`. When enabled, GPU targets are restricted to xnack+ variants (e.g. `gfx908:xnack+`, `gfx90a:xnack+`, `gfx942:xnack+`, `gfx950:xnack+`). Catches host/device out-of-bounds and use-after-free. Some tests skip under ASAN via a check macro (`CHECK_ASAN_ENABLEMENT`).
* **TSAN / MSAN / UBSAN:** not currently configured.
* **GPU-specific limitation:** device ASAN requires xnack+ targets and adds significant runtime cost; it is not run on every PR.
* **How to build:** `cmake -B build -DBUILD_ADDRESS_SANITIZER=ON ...`
* **Not covered:** thread-sanitizer, UB-sanitizer, and non-xnack device configurations.
