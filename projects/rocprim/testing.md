# rocPRIM Testing Strategy (TESTING.md)

Status: Draft\
Owner: @RobsonRLemos\
Technical Lead: @stanleytsang-amd\
Last Updated: August 11, 2026

## Component Overview
rocPRIM is the ROCm low-level parallel-primitives library — a header-only library providing **block-level**, **warp-level**, and **device-level** primitives (scan, reduce, sort, radix sort, etc.) optimized for AMD GPUs. It is a foundational math primitive that sits directly on top of HIP/ROCm and serves as the backend for `hipCUB` and `rocThrust` (the ROCm equivalent of `CUB and Thrust`).

Three properties shape the entire test strategy:
* **Header-only, heavily templated.** Every primitive is a template parameterized over element type, block size, items-per-thread, warp size, and algorithm variant. The correctness surface is a large combinatorial matrix (type × block size × items-per-thread × warp size), so tests are typed/parameterized and the build is deliberately **sharded** to keep compile time and binary size manageable.
* **Almost entirely device code.** Primitives only execute meaningfully inside a kernel, so nearly the whole suite requires a GPU. There is very little hardware-independent logic.
* **Foundational — everything downstream depends on it.** Because `rocThrust` and `hipCUB` build on `rocPRIM`, correctness and reproducibility here are the load-bearing guarantees for the layers above.

**Key constraint:** rocPRIM is a device library. Block/warp/device primitives run on the GPU, so essentially the entire suite requires AMD GPU hardware. Only iterator plumbing, type traits, and a few host utilities are host-testable.

## Development Workflow
The sequence a developer follows from writing code to getting it merged:

1. Build with tests enabled: `CXX=hipcc cmake -B build -DBUILD_TEST=ON -DAMDGPU_TARGETS=<gpu_arch>` then `make -j` (or configure with `-GNinja`). On Windows use `python rmake.py -c -a <gpu_arch>`.
2. Run the fast tier locally against a GPU: `cd build && ctest -L quick` (target < 5 min). For focused work, run a single binary directly, e.g. `./test/rocprim/test_block_scan` or filter with `ctest -R <regex>`.
3. For a primitive change, run the matching `test/rocprim/test_<primitive>` binary (and all of its shards — a single logical test can be split into several executables).
4. On a multi-GPU host, generate a resource spec and let CTest distribute work: `./generate_resource_spec resources.json` then `ctest --resource-spec-file ./resources.json --parallel <N>`.
5. Run `clang-format` on changed files (config in `.clang-format`; a git hook is available via `./.githooks/install`).
6. Open a PR. Required CI checks — **TheRock CI** and **Math CI** — must pass across the build matrix, and another rocPRIM team member must review and approve.


# Testing Strategy and Layers

## Unit Testing Strategy
**Purpose:** validate each primitive across its type/config matrix. In rocPRIM most "unit" tests still dispatch a kernel because the primitive under test is device code; isolation is achieved by testing one primitive at one configuration at a time.

* **Framework:** GoogleTest (auto-downloaded if not found).
* **Location:**
  * `test/rocprim/` — ~73 core primitive tests (`test_<primitive>.cpp` and `.cpp.in` templates): block (`test_block_scan`, `test_block_radix_sort`, `test_block_reduce`, …), warp (`test_warp_reduce`, `test_warp_scan`, `test_warp_sort`, …), device (`test_device_scan`, `test_device_radix_sort`, `test_device_segmented_*`, `test_device_partition`, `test_device_select`, …), iterators (`test_*_iterator.cpp`), and utilities (`test_intrinsics`, `test_thread`, `test_tuple`, `test_accumulator_t`, …).
  * `test/hip/` — low-level HIP API sanity tests with **no rocPRIM dependency**.
  * `test/hipgraph/` — HIP graph capture/replay integration tests.
  * `test/extra/` — post-install / packaging test utilities.
  * `test/common_test_header.hpp` — shared infrastructure, including `GTEST_SKIP_VALGRIND()`, `GTEST_SKIP_ASAN()`, and the `USES_ASAN` macro for conditional skipping.
* **Naming convention:** `test_block_<primitive>.cpp` or `test_device_<primitive>.cpp` or `test_warp_<primitive>.cpp` producing a `test_<block|device|warp>_<primitive>` binary, where `primitive` is the name of an algorithm or function. For example: `test_device_merge.cpp`.
* **How to run:** `ctest -L quick` (fast tier) or run a binary directly, e.g. `./test/rocprim/test_warp_exchange`.
* **Not covered by unit tests:** end-to-end throughput/performance, packaging/install correctness, and cross-platform behavior beyond the built target.

**What is unit-testable in rocPRIM (hardware-independent / host-side):**
* Iterator implementations, type traits, and small host utilities.

**What is not unit-testable (requires a GPU driver / device):**
* All block/warp/device primitives (they run as kernels), device memory allocation/copies, and warp-intrinsic codepaths (e.g. DPP vs. non-DPP shuffle paths).


### Test sharding (rocPRIM-specific)
Because the type × block-size × items-per-thread × algorithm matrix is huge, several tests are **split across multiple binaries** at configure time to bound compile time and binary size:
* `.cpp.in` templates use `ROCPRIM_TEST_SLICE`, `ROCPRIM_TEST_SUITE_SLICE`, and `ROCPRIM_TEST_TYPE_SLICE` directives; `add_rocprim_test_parallel()` in `test/rocprim/CMakeLists.txt` generates `SUITE_SLICE_COUNT × TYPE_SLICE_COUNT` executables per logical test (e.g. `test_block_radix_sort`, `test_block_scan`, `test_device_radix_sort`, `test_device_segmented_radix_sort`, `test_block_discontinuity`, `test_block_adjacent_difference`, `test_block_radix_rank`).
* Sharding is skipped in Debug builds to avoid slow compilation.
* A DPP-disabled variant (`-DROCPRIM_DISABLE_DPP=1`, via `add_rocprim_test_disable_dpp()`) tests the alternative warp-scan codepath.

### Coverage expectation
* Long-term goal across ROCm components is **> 95%** line coverage; not mandated initially and pursued in phases.
* Realistic near-term target for rocPRIM: **≥ 80% of hardware-independent paths where practical.**
* **Current state:** CI reports roughly **40%** line coverage because coverage instrumentation currently captures host-side code only, while most of rocPRIM is device code. This is the single largest coverage gap. An initiative to adopt LLVM device-code coverage is expected to close it.

## Integration Testing Strategy
**Purpose:** validate behavior that requires a real GPU, the HIP runtime, graph capture, or downstream interaction — essentially all primitive behavior, since rocPRIM dispatches kernels.

| Test Type | Location | Purpose | GPU Required | Frequency |
| --- | --- | --- | --- | --- |
| Primitive tests (GoogleTest) | `test/rocprim/test_<primitive>.cpp` | Validate block/warp/device primitives across type/config matrix | Yes | PR / Nightly |
| Low-level HIP tests | `test/hip/` | Sanity-check HIP API behavior independent of rocPRIM | Yes | PR / Nightly |
| hipGraph | `test/hipgraph/` | Validate primitives under HIP graph capture/replay | Yes | PR / Nightly |
| Multi-GPU distribution | CTest RESOURCE_GROUPS | Distribute the suite across multiple same-family GPUs | Yes (2+ GPUs) | as available |
| Package / install | `test/extra/` | Post-install smoke check via `find_package(rocprim)` | Yes | Release / packaging |

* **What requires GPU hardware:** effectively all of the above — every primitive test launches kernels.
* **What runs on CPU-only systems:** only compile-time/type-trait checks and build/link checks; there is no meaningful host-only runtime suite.
* **Windows exclusions:** `test_categories.yaml` disables specific suites on certain Windows targets — `gfx110X` skips block_discontinuity, device_merge_sort, device_reduce*, device_reduce_by_key; `gfx1151` skips device_merge_sort and device_radix_sort. These exclusions are ignored on Linux.
* **Test-size / coverage guidance:** primitives are covered via typed suites, so a new primitive inherits the standard type/config matrix. Prefer a representative set of block sizes / items-per-thread and edge sizes (0, 1, sub-warp, non-power-of-two) over exhaustive enumeration; rely on sharding rather than shrinking coverage when a matrix grows.

## Performance & Benchmarking Testing
**Purpose:** detect regressions in primitive throughput against a per-architecture baseline over time. Absolute numbers across architectures are not comparable. rocPRIM benchmarks also drive **config autotuning** — selecting the best block/algorithm configuration per architecture.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (foundational parallel-primitives layer) |
| Metrics measured | Per-primitive throughput (items/s, bytes/s) across types, block sizes, and items-per-thread |
| How benchmarks are run | Built with `-DBUILD_BENCHMARK=ON`; `benchmark_block_*`, `benchmark_warp_*`, `benchmark_device_*` binaries using the shared **primbench** framework (`shared/primbench`). AMD SMI required |
| Autotuning | `BENCHMARK_CONFIG_TUNING=ON` generates a large parameterized configuration matrix (`.parallel.cpp.in`, `add_matrix()`), used to search optimal configs; `BENCHMARK_AUTOTUNED_TYPES_ONLY` and `BUILD_NAIVE_BENCHMARK` control scope |
| Baseline — stored per architecture | Not formally stored/aggregated for automated comparison today. Results must **not** be aggregated across GFX |
| Where results are stored | Benchmark output (JSON/console); autotune scripts under `scripts/autotune*` track branch/commit for reproducibility |
| Regression threshold | No fixed automated threshold; regressions caught via manual review |
| Gating approach | Manual review |
| Test run time | Full sweep and especially `BENCHMARK_CONFIG_TUNING` are very long-running; run selectively via `--benchmark_filter` |
| GPU profiling | AMD SMI required for benchmarks; no dedicated profiling gate |

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
* Regressions in downstream consumers (rocThrust, hipCUB) are not automatically traced back to rocPRIM.

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
* A flaky test is not an accepted final state. rocPRIM does not currently maintain a tagged flaky list — establishing one is a gap. 

## Coverage
* **Tooling:** `BUILD_CODE_COVERAGE=ON` (clang) compiles with `-fprofile-instr-generate -fcoverage-mapping`; `llvm-profdata` + `llvm-cov` (from `${ROCM_PATH}/llvm/bin`) produce HTML/LCOV reports via the `coverage_analysis` and `coverage` build targets. Test files are excluded (`--ignore-filename-regex="test_*"`). Codecov config lives at the `rocm-libraries` monorepo level.
* **Target:** long-term > 95%.
* **Scope / limitations:** instrumentation is host-side and clang-only; device-code coverage is not measured. Windows coverage is not tracked separately.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by tests (e.g. 700 of 1,000 lines → 70%).
* *Test coverage* = fraction of intended functionality/scenarios exercised. rocPRIM can show low host code coverage while still having broad test coverage of primitives, because most executed logic is device code host instrumentation does not see. Conversely, configurations such as Windows and multi-GPU remain under-exercised even where line coverage looks reasonable.

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
* **ffm-quick** category (`ctest -L ffm-quick`) — Full-Feature-Matrix core-algorithm tests (timeout budget up to 2 hours).
* All shards of large primitives and the full type/config matrix run beyond the PR `quick` tier.
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

**Explicitly not guaranteed:** multi-GPU validation is opportunistic (depends on host GPU count via CTest resource allocation) and not a formal gate; non-listed gfx targets are not validated; specific Windows suites on gfx110X/gfx1151 are explicitly not run.

## Sanitizer Coverage (ASAN / TSAN)
* **AddressSanitizer:** `BUILD_ADDRESS_SANITIZER=ON` builds an ASAN variant; GPU targets are restricted to xnack+ variants (`gfx908:xnack+`, `gfx90a:xnack+`, `gfx942:xnack+`, `gfx950:xnack+`) and the package depends on `hip-runtime-amd-asan` instead of `hip-runtime-amd`. Tests that cannot run under ASAN skip via `GTEST_SKIP_ASAN()` / the `USES_ASAN` macro. Catches host/device out-of-bounds and use-after-free.
* **Valgrind:** `GTEST_SKIP_VALGRIND()` is available to skip tests incompatible with Valgrind, but Valgrind is not a GPU-path tool and is not a gate.
* **TSAN / MSAN / UBSAN:** not currently configured.
* **GPU-specific limitation:** device ASAN requires xnack+ targets and adds significant runtime cost; it is not run on every PR.
* **How to build:** `cmake -B build -DBUILD_ADDRESS_SANITIZER=ON ...`
* **Not covered:** thread-sanitizer, UB-sanitizer, and non-xnack device configurations.
