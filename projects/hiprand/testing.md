# hipRAND Testing Strategy (TESTING.md)

Status: Draft\
Owner: @RobsonRLemos\
Technical Lead: @stanleytsang-amd\
Last Updated: August 12, 2026

## Component Overview
hipRAND is the ROCm random-number-generation wrapper — a thin marshalling library that exposes a uniform RNG API on top of two interchangeable backends. On AMD/ROCm it dispatches to **rocRAND**, on NVIDIA it forwards to **cuRAND**.. It is the drop-in replacement that lets cuRAND-based CUDA code build and run on AMD GPUs, and it exposes C, C++ (`hiprand.hpp`), Fortran, and Python interfaces.

Three properties shape the entire test strategy:
* **Thin two-backend wrapper.** hipRAND contains almost no generation logic of its own — the actual PRNG/QRNG engines and distributions live in rocRAND or cuRAND. Tests validate the *wrapping* plus statistical sanity of generated output, not the engine internals, which are owned and tested by the backends.
* **Build-time backend selection.** hipRAND's backend is chosen explicitly at configure time via `BUILD_WITH_LIB` (`ROCM` or `CUDA`, default `ROCM`). The suite is built once against a single backend; this repo's CI exercises the **rocRAND (AMD) path**.
* **Multiple language surfaces.** Beyond the C API and C++ wrapper, hipRAND ships Fortran and Python bindings, each with its own (smaller) test suite in a different framework. Keeping these bindings building and functional is a distinct testing obligation.

**Key constraint:** random generation runs on the device, so the meaningful suite (host API, C++ wrapper, and device/kernel tests) requires AMD GPU hardware. Only the linkage/version tests are truly host-only. hipRAND provides a **compiled host library** plus a **header-only device API** (`hiprand_kernel.h`).

## Development Workflow
The sequence a developer follows from writing code to getting it merged:

1. Build with tests enabled: `CXX=hipcc cmake -B build -DBUILD_TEST=ON -DGPU_TARGETS=<gpu_arch>` then `make -j` (or configure with `-GNinja`). The backend defaults to rocRAND (`-DBUILD_WITH_LIB=ROCM`); rocRAND is located via `ROCRAND_FETCH_METHOD` (`PACKAGE` / `MONOREPO` / `DOWNLOAD`). On Windows use `python rmake.py -c -a <gpu_arch>`.
2. Run the tests locally against a GPU: `cd build && ctest --output-on-failure`. For focused work, run a single binary directly, e.g. `./test/test_hiprand_api` or `./test/test_hiprand_kernel`.
3. For Fortran or Python changes, enable/build the relevant binding (`-DBUILD_FORTRAN_WRAPPER=ON`) and run its suite (see below).
4. Run `clang-format` on changed files (config in `.clang-format`; a git hook is available via `./.githooks/install`).
5. Open a PR. Required CI checks — **TheRock CI** and **Math CI** — must pass across the build matrix, and another hipRAND team member must review and approve.


# Testing Strategy and Layers

## Unit Testing Strategy
**Purpose:** validate that hipRAND correctly wraps and dispatches to its backend across engine types, distributions, and interfaces. Most tests still dispatch to the device because generation is device work; isolation is achieved by testing one engine/distribution/interface at a time.

* **Frameworks:** GoogleTest (C/C++), FRUIT (Fortran), and Python `unittest`.
* **Location:**
  * `test/test_hiprand_api.cpp` — host C API across engines (XORWOW, MRG32K3A, MTGP32, MT19937, PHILOX, SOBOL32/64, scrambled Sobol) and distributions (uniform, normal, log-normal, Poisson), plus a small host-only path (`hiprand_host`, PHILOX only).
  * `test/test_hiprand_cpp_wrapper.cpp` — the `hiprand.hpp` C++ interface, using typed suites over engine types (`hiprand_cpp_wrapper`, `_32`, `_64`, `_prng`, `_qrng`, `_offset`).
  * `test/test_hiprand_kernel.cpp` — the header-only device API (in-kernel generators, state init, Sobol direction vectors); the largest suite.
  * `test/linkage/` — multiple-translation-unit linkage / version checks (host-only).
  * `test/fortran/` — FRUIT-based Fortran wrapper tests (`test_hiprand.f90`), gated by `BUILD_FORTRAN_WRAPPER`.
  * `test/package/` — post-install smoke test via `find_package(hiprand)`.
  * `python/hiprand/tests/hiprand_test.py` — Python binding tests (`unittest`): version, constructor validation, PRNG/QRNG parameter getters/setters, and generation.
  * Shared helpers in `test/test_common.hpp` (`HIP_CHECK`, `HIPRAND_CHECK`, `hipMallocHelper`).
* **Naming convention:** `test_hiprand_<area>.cpp` producing a matching binary (e.g. `test_hiprand_api`, `test_hiprand_kernel`); the Fortran runner builds `test_hiprand_fortran_wrapper`. Tests use `TYPED_TEST_SUITE` over engine types and `INSTANTIATE_TEST_SUITE_P(... ValuesIn(hiprand_rng_types))` for enum/ordering variation. There is **no `.cpp.in` sharding** (the suite is small enough not to need it).
* **How to run:** `ctest --output-on-failure`, or run a binary directly. Fortran: build with `BUILD_FORTRAN_WRAPPER=ON` and run `test_hiprand_fortran_wrapper`. Python: run `python -m unittest` against `python/hiprand/tests/`.
* **Reproducibility / seeding:** tests use fixed seeds via `hiprandSetPseudoRandomGeneratorSeed()` for determinism (and `hiprandGenerateSeeds()` where random seeding is exercised); offsets via `hiprandSetGeneratorOffset()` for engines that support them. `HIPRAND_USE_HMM=1` switches test allocations to managed memory.
* **Not covered by unit tests:** backend engine correctness/statistics themselves (owned by rocRAND/cuRAND), throughput/performance, and the NVIDIA/cuRAND path (not routinely exercised in this repo's CI).

**What is unit-testable in hipRAND (hardware-independent / host-side):**
* Linkage/version checks across translation units (`test/linkage/`).
* Argument/enum validation and status-code mapping at the C API boundary; Python constructor error handling.

**What is not unit-testable (requires a GPU driver / device):**
* All actual generation (host API, C++ wrapper, and in-kernel device API) — every generate call dispatches to the device.

### Coverage expectation
* Long-term goal across ROCm components is **> 95%** line coverage; not mandated initially and pursued in phases.
* Because hipRAND is a thin wrapper, the generation logic that dominates execution lives in the backend and is measured under rocRAND's coverage rather than hipRAND's.
* **Current state:** host-side line-coverage instrumentation (`BUILD_CODE_COVERAGE`, clang) captures only the wrapper's host code; device generation is not measured. Device-code coverage is the single largest measurement gap. See [Coverage](#coverage).

## Integration Testing Strategy
**Purpose:** validate behavior that requires a real GPU, the HIP runtime, correct backend dispatch, and cross-language bindings — essentially all generation behavior, since hipRAND dispatches to its backend on the device.

| Test Type | Location | Purpose | GPU Required | Frequency |
| --- | --- | --- | --- | --- |
| Host C API | `test/test_hiprand_api.cpp` | Validate host generation across engines/distributions | Yes | PR / Nightly |
| Device / kernel API | `test/test_hiprand_kernel.cpp` | Validate in-kernel generators and state init | Yes | PR / Nightly |
| C++ wrapper | `test/test_hiprand_cpp_wrapper.cpp` | Validate the `hiprand.hpp` interface across engine types | Yes | PR / Nightly |
| Fortran wrapper | `test/fortran/` | Validate Fortran bindings (FRUIT) | Yes | Nightly / opt-in |
| Python bindings | `python/hiprand/tests/` | Validate Python interface (unittest) | Yes | Nightly / opt-in |
| Linkage / version | `test/linkage/` | Confirm library links across TUs and reports version | Minimal | PR / Nightly |
| Package / install | `test/package/` | Post-install smoke check via `find_package(hiprand)` | Yes | Release / packaging |

* **What requires GPU hardware:** all generation paths (host, device, C++, Fortran, Python) except the linkage/version tests.
* **What runs on CPU-only systems:** linkage/version checks and host-side argument/enum validation.
* **FFM-simulator note:** `test_hiprand_api` and `test_hiprand_cpp_wrapper` are omitted from the `ffm-quick` tier because the MT19937 generator is prohibitively slow on the FFM simulator (hours); only `test_hiprand_kernel` and `test_hiprand_linkage` run there.
* **Two-backend note:** the wrapper is validated against a single backend per build; this repo's CI exercises the **rocRAND (AMD)** path. The cuRAND (NVIDIA) path is validated opportunistically and is a coverage gap here.
* **Fortran binding status:** the Fortran wrapper is **deprecated** in favor of hipfort; its tests remain to keep the legacy binding building.

## Performance & Benchmarking Testing
**Purpose:** detect regressions in generation throughput against a per-architecture baseline over time.

hipRAND does **not** ship a populated benchmark suite. The `BUILD_BENCHMARK` flag exists and the build wires up benchmark output directories, but no `benchmark/` sources are implemented — generation-throughput benchmarking lives in **rocRAND** (the backend), which hipRAND performance directly tracks.

| Item | Detail |
| --- | --- |
| Stack layer | Core SDK (cuRAND-compatible wrapper over rocRAND) |
| Metrics measured | None in-repo; measured in rocRAND |
| How benchmarks are run | N/A here — see rocRAND's benchmark suite |
| Baseline — stored per architecture | N/A here |
| Regression threshold | N/A here |
| Gating approach | N/A here |

### Gating
| Gating Level | Status | Notes |
| --- | --- | --- |
| PR-level automated gate | No | No hipRAND-level performance benchmarks |
| Nightly automated comparison | No | Performance tracked via rocRAND |
| Manual nightly review | Via rocRAND | hipRAND throughput follows the backend |
| Release qualification | Partial | Reviewed via rocRAND before release |

### Known Gaps
* No hipRAND-level benchmarks; performance regressions must be caught in rocRAND.
* No automated tracing of wrapper-introduced overhead (e.g. marshalling costs) distinct from backend performance.

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
| Trusted gate | The full test suite is ran on multiple GPU architectures. |
| Informational | codecov |
| Unstable / flaky | None formally tagged today (see Flaky Test Policy) |

### Flaky Test Policy
* Flaky tests should be tagged clearly (e.g. `UNSTABLE`) and excluded from blocking runs until fixed.
* Every flaky test should have an owner and a tracking bug.
* A flaky test is not an accepted final state. hipRAND does not currently maintain a tagged flaky list — establishing one is a gap.


## Coverage
* **Tooling:** `BUILD_CODE_COVERAGE=ON` (clang) compiles with `-fprofile-instr-generate -fcoverage-mapping`; `llvm-profdata` + `llvm-cov` (from `${ROCM_PATH}/llvm/bin`) produce HTML/LCOV reports via the `coverage_analysis` and `coverage` build targets (which honor `GTEST_FILTER`). Codecov config lives at the `rocm-libraries` monorepo level.
* **Target:** long-term > 95% (phased, aspirational). Because hipRAND is a wrapper, meaningful generation coverage is measured under rocRAND.
* **Scope / limitations:** instrumentation is host-side and clang-only; device generation is not measured. Windows coverage is not tracked separately, and the Fortran/Python bindings are outside the C++ coverage measurement.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed by tests (e.g. 700 of 1,000 lines → 70%).
* *Test coverage* = fraction of intended functionality/scenarios exercised. hipRAND can show reasonable host coverage of the wrapper while the cuRAND backend path, Fortran/Python bindings, Windows, and multi-GPU remain under-exercised.

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
* **quick** category (all tests, `.*`) — the full GoogleTest suite; this category carries the `standard`/`comprehensive`/`full` labels since hipRAND's suite is small.
* **ffm-quick** category — `test_hiprand_kernel` and `test_hiprand_linkage` only (MT19937-heavy suites excluded on the FFM simulator).
* Fortran (FRUIT) and Python (unittest) binding suites run here rather than on every PR.
* Additional hardware coverage across the default GPU target list.

Note: hipRAND's `test_categories.yaml` (at the project root) defines only `quick` and `ffm-quick`, with no per-tier timeout overrides — unlike the larger rocPRIM/rocThrust/hipCUB suites.

## Supported Configurations
GPU targets come from the top-level `CMakeLists.txt` default `GPU_TARGETS`/`AMDGPU_TARGETS` list.

| Configuration | Validation Level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux (ROCm, rocRAND backend) | Full | PR / Nightly / Release | Primary platform and backend |
| Windows (HIP on Windows) | Partial | PR / Nightly / Release | Built via `rmake.py` / `rtest.xml` |
| NVIDIA (cuRAND backend) | Partial | as available | `-DBUILD_WITH_LIB=CUDA`; not routinely in this repo's CI |
| gfx90a / gfx942 / gfx950 | Full | PR / Nightly / Release | |
| gfx908 / gfx906 | Full | Nightly / Release | |
| gfx103x / gfx11xx (incl. gfx1151) | Full | PR / Nightly | |
| gfx120x / gfx1250 | Partial | Nightly | Newer targets in default list |

**Explicitly not guaranteed:** the cuRAND (NVIDIA) backend path is not routinely validated in this repo's CI; the Fortran binding is deprecated (kept building, not actively expanded); multi-GPU validation is not a formal gate; non-listed gfx targets are not validated; Windows coverage is thinner than Linux.

## Sanitizer Coverage (ASAN / TSAN)
* **AddressSanitizer:** `BUILD_ADDRESS_SANITIZER=ON` builds an ASAN variant (`-fsanitize=address -shared-libasan`, linked with `-fuse-ld=lld`) and the package depends on `hip-runtime-amd-asan` instead of `hip-runtime-amd`. GPU targets are restricted to xnack+ variants. Catches host/device out-of-bounds and use-after-free.
* **TSAN / MSAN / UBSAN:** not currently configured.
* **GPU-specific limitation:** device ASAN requires xnack+ targets and adds significant runtime cost; it is not run on every PR.
* **How to build:** `cmake -B build -DBUILD_ADDRESS_SANITIZER=ON ...`
* **Not covered:** thread-sanitizer, UB-sanitizer, non-xnack device configurations, and the Fortran/Python binding paths.
