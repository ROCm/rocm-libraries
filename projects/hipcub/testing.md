# hipCUB Testing Strategy (TESTING.md)

Status: Draft\
Owner: @RobsonRLemos\
Technical Lead: @stanleytsang-amd\
Last Updated: August 12, 2026

## Component Overview
hipCUB is the ROCm CUB-compatible parallel-primitives wrapper — a header-only library that exposes the NVIDIA CUB API on top of two interchangeable backends. On AMD/ROCm it dispatches to **rocPRIM** (`hipcub/backend/rocprim/`); on NVIDIA it forwards to **CUB** (`hipcub/backend/cub/`), selected at compile time by HIP platform detection (`__HIP_PLATFORM_AMD__` vs `__HIP_PLATFORM_NVIDIA__`). It is the drop-in replacement that lets CUB-based CUDA code build and run on AMD GPUs.

Three properties shape the entire test strategy:
* **Thin two-backend wrapper.** hipCUB contains almost no algorithm logic of its own — the heavy lifting lives in rocPRIM or CUB. Tests therefore validate the *wrapping* (correct API surface, dispatch, and type handling), not the primitive implementations, which are owned and tested by the backends.
* **Backend-agnostic test suite.** A single test suite in `test/hipcub/` compiles against whichever backend the compiler selects; there are no separate AMD/NVIDIA test trees. This repo primarily exercises the **rocPRIM backend path**.
* **Heavily templated, sharded like rocPRIM.** The correctness surface is a large type × block-size × items-per-thread matrix, so tests are typed/parameterized and a few large ones are **sharded** across binaries to bound compile time.

**Key constraint:** hipCUB is a device library. Primitives only execute meaningfully inside a kernel, so essentially the entire suite requires AMD GPU hardware. Only the version/linkage smoke test and a few host utilities are host-testable.

## Development Workflow
The sequence a developer follows from writing code to getting it merged:

1. Build with tests enabled: `CXX=hipcc cmake -B build -DBUILD_TEST=ON -DAMDGPU_TARGETS=<gpu_arch>` then `make -j` (or configure with `-GNinja`). On Windows use `python rmake.py -c -a <gpu_arch>`. The backend is chosen automatically from the compiler (HIP-clang → rocPRIM; nvcc → CUB); there is no explicit backend flag.
2. Run the fast tier locally against a GPU: `cd build && ctest -L quick` (target < 5 min). For focused work, run a single binary directly, e.g. `./test/hipcub/test_hipcub_block_scan`.
3. For a wrapped primitive change, run the matching `test/hipcub/test_hipcub_<primitive>` binary (and all of its shards — a single logical test can be split into several executables).
4. On a multi-GPU host, generate a resource spec and let CTest distribute work: `./generate_resource_spec resources.json` then `ctest --resource-spec-file ./resources.json --parallel <N>`.
5. Run `clang-format` on changed files (config in `.clang-format`; a git hook is available via `./.githooks/install`).
6. Open a PR. Required CI checks — **TheRock CI** and **Math CI** — must pass across the build matrix, and another hipCUB team member must review and approve.


# Testing Strategy and Layers

## Unit Testing Strategy
**Purpose:** validate that hipCUB correctly wraps and dispatches to its backend across the type/config matrix. In hipCUB most "unit" tests still dispatch a kernel because the wrapped primitive is device code; isolation is achieved by testing one wrapped primitive at one configuration at a time.

* **Framework:** GoogleTest (auto-downloaded if not found).
* **Location:**
  * `test/hipcub/` — ~44 primitive tests (`test_hipcub_<primitive>.cpp` and `.cpp.in` templates): block (`test_hipcub_block_scan`, `test_hipcub_block_radix_sort`, `test_hipcub_block_reduce`, …), warp (`test_hipcub_warp_reduce`, `test_hipcub_warp_scan`, …), device (`test_hipcub_device_radix_sort`, `test_hipcub_device_scan`, `test_hipcub_device_select`, …), thread/util/vector, plus iterator tests. Shared infrastructure in `common_test_header.hpp`, `test_utils*.hpp`, and custom iterators/types.
  * `test/extra/` — post-install / packaging tests (`find_package(hipcub)`).
  * `test/hipcub/detail/` — `get_hipcub_version.*` used by the linkage smoke test.
* **Naming convention:** `test_hipcub_<block|warp|device>_<primitive>.cpp` producing a `test_hipcub_<...>` binary, where `primitive` is the wrapped algorithm (e.g. `test_hipcub_device_merge_sort.cpp`). Typed suites use `TYPED_TEST_SUITE` / `INSTANTIATE_TYPED_TEST` over parameter lists spanning integral, floating-point, and reduced-precision types.
* **How to run:** `ctest -L quick` (fast tier) or run a binary directly, e.g. `./test/hipcub/test_hipcub_warp_exchange`.
* **Reproducibility / seeding:** `test/hipcub/test_seed.hpp` uses predefined seeds `{0, 1000}` plus `random_seeds_count = 2` runtime-random seeds (with a `seed_value_addition` offset for variation). On failure GoogleTest reports the seed for reproduction.
* **Special-type handling:** `rocprim::half` / `bfloat16` are covered via dedicated headers (`half.hpp`, `bfloat16.hpp`, `test_utils_bfloat16.hpp`) and instantiated into typed suites; `test_hipcub_no_half_operators.cpp` specifically guards behavior under `__HIP_NO_HALF_CONVERSIONS__`.
* **Test heritage:** the suite is adapted from rocPRIM's tests (same typed-test philosophy and utilities), not from CUB's upstream tests.
* **Not covered by unit tests:** backend primitive correctness itself (owned by rocPRIM/CUB), end-to-end throughput/performance, packaging beyond the smoke test, and the NVIDIA/CUB path (not routinely exercised in this repo's CI).

**What is unit-testable in hipCUB (hardware-independent / host-side):**
* The version/linkage smoke test (`test_hipcub_basic.cpp` + `detail/get_hipcub_version.cpp`) confirming the library links and reports version with multiple translation units.
* Type traits and small host utilities.

**What is not unit-testable (requires a GPU driver / device):**
* All wrapped block/warp/device primitives (they run as kernels) and device memory allocation/copies.


### Test sharding (inherited from rocPRIM)
To bound compile time and binary size on the largest type matrices, a few tests are **split across multiple binaries** at configure time:
* `.cpp.in` templates use `HIPCUB_TEST_SLICE`, `HIPCUB_TEST_SUITE_SLICE`, and `HIPCUB_TEST_TYPE_SLICE` directives; `add_hipcub_test_parallel()` in `test/hipcub/CMakeLists.txt` generates `SUITE_SLICE_COUNT × TYPE_SLICE_COUNT` executables per logical test.
* Currently applied to `test_hipcub_device_radix_sort`, `test_hipcub_device_segmented_radix_sort`, and `test_hipcub_device_segmented_sort`.

### Coverage expectation
* Long-term goal across ROCm components is **> 95%** line coverage; not mandated initially and pursued in phases.
* Realistic near-term target for hipCUB: **≥ 80% of hardware-independent paths where practical.**
* **Current state:** CI reports roughly **82.68%** line coverage because coverage instrumentation currently captures host-side code only, while most of rocPRIM is device code. This is the single largest coverage gap. An initiative to adopt LLVM device-code coverage is expected to close it.

## Integration Testing Strategy
**Purpose:** validate behavior that requires a real GPU, the HIP runtime, and correct backend dispatch — essentially all wrapped-primitive behavior, since hipCUB dispatches kernels through its backend.

| Test Type | Location | Purpose | GPU Required | Frequency |
| --- | --- | --- | --- | --- |
| Primitive wrapper tests (GoogleTest) | `test/hipcub/test_hipcub_<primitive>.cpp` | Validate wrapped block/warp/device primitives across type/config matrix on device | Yes | PR / Nightly |
| Linkage / version smoke test | `test/hipcub/test_hipcub_basic.cpp` (+ `detail/`) | Confirm library links across TUs and reports version | Minimal | PR / Nightly |
| Multi-GPU distribution | CTest RESOURCE_GROUPS | Distribute the suite across multiple same-family GPUs | Yes (2+ GPUs) | as available |
| Package / install | `test/extra/` | Post-install smoke check via `find_package(hipcub)` | Yes | Release / packaging |

* **What requires GPU hardware:** effectively all of the above except the linkage/version smoke test.
* **What runs on CPU-only systems:** only the linkage/version and type-trait checks plus build/link checks; there is no meaningful host-only runtime suite.
* **Two-backend note:** the same tests validate both backends by construction, but this repo's CI exercises the **rocPRIM (AMD)** path; the CUB (NVIDIA) path is validated opportunistically and is a coverage gap here.
* **Test-size / coverage guidance:** primitives are covered via typed suites, so a new wrapper inherits the standard type/config matrix. Prefer a representative set of block sizes / items-per-thread and edge sizes over exhaustive enumeration; rely on sharding rather than shrinking coverage when a matrix grows. Deep numerical validation belongs in rocPRIM, not duplicated here.

## Performance & Benchmarking Testing
**Purpose:** detect regressions in wrapped-primitive throughput against a per-architecture baseline over time. Absolute numbers across architectures are not comparable.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (CUB-compatible wrapper over rocPRIM) |
| Metrics measured | Per-primitive throughput (items/s, bytes/s) across types, block sizes, and items-per-thread |
| How benchmarks are run | Built with `-DBUILD_BENCHMARK=ON`; ~35 `benchmark_*` binaries (block/warp/device) using Google Benchmark; `.gitlab/run_benchmarks.py` drives runs with GPU temperature/noise-tolerance monitoring and batch-window sizing |
| Baseline — stored per architecture | Not formally stored/aggregated for automated comparison today. Results must **not** be aggregated across GFX |
| Where results are stored | Google Benchmark output (JSON/CSV/console); no central dashboard wired into the repo |
| Regression threshold | No fixed automated threshold; regressions caught via manual review |
| Gating approach | Manual review |
| Test run time | Full sweep is long-running; run selectively via `--benchmark_filter` |
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
* hipCUB performance largely tracks rocPRIM; regressions are typically diagnosed and fixed in the backend rather than the wrapper.

### Upcoming Changes
hipCUB performance testing is getting reworked to use `Primbench` instead of `GoogleBench` for more consistent results. See PR [8534](https://github.com/ROCm/rocm-libraries/pull/8534).

## Pre-submit / CI Gates

### Validation Gates and Ownership
| Validation Area | Required Before Merge | Owner | Responsibility |
| --- | --- | --- | --- |
| Build (Linux, Windows) | Yes | TheRock / Math CI | Multi-OS, multi-arch build matrix |
| Unit / wrapper tests | Yes | Component team / TheRock / Math CI | Create, maintain, and review |
| Formatting (clang-format) | Yes | CI / pre-commit | WebKit-based style in `.clang-format` |
| Code coverage | No | Component team / codecov | Informational; no enforced threshold |

### PR Test Classification
| Status | Applies to |
| --- | --- |
| Trusted gate | `quick` tier (`ctest -L quick`) run across multiple GPU architectures on PRs |
| Informational | Code coverage |
| Unstable / flaky | None formally tagged today (see Flaky Test Policy) |

### Flaky Test Policy
* Flaky tests should be tagged clearly (e.g. `UNSTABLE`) and excluded from blocking runs until fixed.
* Every flaky test should have an owner and a tracking bug.
* A flaky test is not an accepted final state. hipCUB does not currently maintain a tagged flaky list — establishing one is a gap.


## Coverage
* **Tooling:** `BUILD_CODE_COVERAGE=ON` (clang) compiles with `-g -O0 -fprofile-instr-generate -fcoverage-mapping`; `llvm-profdata` + `llvm-cov` (from `${ROCM_PATH}/llvm/bin`) produce HTML/LCOV reports via the `coverage_analysis` and `coverage` build targets. Test files are excluded (`--ignore-filename-regex="test_*"`). Codecov config lives at the `rocm-libraries` monorepo level.
* **Target:** long-term > 95% (phased, aspirational). Because hipCUB is a wrapper, meaningful primitive coverage is measured under rocPRIM.
* **Scope / limitations:** instrumentation is host-side and clang-only; device-code coverage is not measured. Windows coverage is not tracked separately.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by tests (e.g. 700 of 1,000 lines → 70%).
* *Test coverage* = fraction of intended functionality/scenarios exercised. hipCUB can show low host code coverage while still validating a broad API surface, because the executed logic lives in the backend. Conversely, the NVIDIA/CUB backend path, Windows, and multi-GPU remain under-exercised even where line coverage looks reasonable.

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
* **ffm-quick** category (`ctest -L ffm-quick`) — Full-Feature-Matrix tests (target < 120 min; timeout budget up to 2 hours).
* All shards of large primitives and the full type/config matrix run beyond the PR `quick` tier.
* Additional hardware coverage across the default GPU target list.

## Supported Configurations
GPU targets come from the top-level `CMakeLists.txt` default `GPU_TARGETS`/`AMDGPU_TARGETS` list.

| Configuration | Validation Level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux (ROCm, rocPRIM backend) | Full | PR / Nightly / Release | Primary platform and backend |
| Windows (HIP on Windows) | Partial | PR / Nightly / Release | Built via `rmake.py` / `rtest.py` |
| NVIDIA (CUB backend) | Partial | as available | Same tests forward to CUB; not routinely in this repo's CI |
| gfx90a / gfx942 / gfx950 | Full | PR / Nightly / Release | |
| gfx908 / gfx906 | Full | Nightly / Release | |
| gfx103x / gfx11xx (incl. gfx1151) | Full | PR / Nightly | |
| gfx120x / gfx1250 | Partial | Nightly | Newer targets in default list |

**Explicitly not guaranteed:** the NVIDIA/CUB backend path is not routinely validated in this repo's CI; multi-GPU validation is opportunistic (depends on host GPU count via CTest resource allocation) and not a formal gate; non-listed gfx targets are not validated; Windows coverage is thinner than Linux.

## Sanitizer Coverage (ASAN / TSAN)
* **AddressSanitizer:** `BUILD_ADDRESS_SANITIZER=ON` builds an ASAN variant (`-fsanitize=address -shared-libasan`, linked with `-fuse-ld=lld`); GPU targets are restricted to xnack+ variants (`gfx908:xnack+`, `gfx90a:xnack+`, `gfx942:xnack+`, `gfx950:xnack+`). Tests that cannot run under ASAN skip via `GTEST_SKIP_ASAN()`. Catches host/device out-of-bounds and use-after-free.
* **Compute Sanitizer:** `BUILD_COMPUTE_SANITIZER=ON` wraps tests with NVIDIA `compute-sanitizer` — **NVIDIA/CUB backend only**.
* **TSAN / MSAN / UBSAN:** not currently configured.
* **GPU-specific limitation:** device ASAN requires xnack+ targets and adds significant runtime cost; it is not run on every PR.
* **How to build:** `cmake -B build -DBUILD_ADDRESS_SANITIZER=ON ...`
* **Not covered:** thread-sanitizer, UB-sanitizer, and non-xnack device configurations.
