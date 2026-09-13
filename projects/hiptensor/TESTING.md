# hipTensor Testing Strategy

This document describes how hipTensor is tested today, which signals actually gate a merge, and
where the gaps are. It follows the ROCm-wide TESTING.md template.

It is written as an accurate description of the current state rather than an aspirational one.
Recording a gap is the point of the exercise — a gap written down is one that can be argued about
and closed.

---

## Component Overview

hipTensor is AMD's C++ library for tensor primitives accelerated by AMD GPU matrix cores (XDL/WMMA). It provides Contraction, Elementwise, and Reduction operations targeting CDNA (gfx908, gfx90a, gfx942, gfx950) and RDNA (gfx11xx, gfx12xx) architectures. hipTensor sits in the ROCm Expansion SDK layer, above the HIP runtime and directly above Composable Kernel (CK), which supplies all GPU kernels. Because every meaningful operation dispatches a CK kernel, nearly all behavior requires real AMD GPU hardware to validate; CPU-side testability is limited to argument validation, kernel selection logic, workspace calculation, and API lifecycle tests.

---

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build and install:**
```bash
cmake -B <build_dir> -DCMAKE_INSTALL_PREFIX=<prefix> .
cmake --build <build_dir> -- -j$(nproc)
cmake --install <build_dir> --component tests
```

**2. Run the tests that match what you touched:**

| You changed | Run this | Where | Needs a GPU |
|---|---|---|---|
| Any C++ source | `ctest` | Build tree | No (GPU-dependent tests will fail) |
| Contraction / elementwise / reduction logic | `ctest -L '^quick$'` | `<prefix>/bin/hiptensor` | Yes |
| Kernel selection or data types | `ctest -L '^standard$'` | `<prefix>/bin/hiptensor` | Yes |
| Performance-sensitive path | `ctest -L '^bench$' -V -O bench.log` | `<prefix>/bin/hiptensor` | Yes |

> **Run in parallel:** add `-j$(nproc)` (or `-j <N>`) to any validation run — e.g. `ctest -L '^standard$' -j$(nproc)` — to run test binaries concurrently. Do **not** parallelize the `bench` tier: concurrent runs contend for the GPU and corrupt timing measurements.

**3. Add the right kind of test.** See [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open the PR** targeting `develop`. Standard-tier tests (`-L '^standard$'`) are the integration merge gate.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** Validate hardware-independent logic without dispatching a GPU kernel.

Unit tests live in `test/00_unit/` and are built with the `add_hiptensor_unit_test()` CMake function, which links against `hiptensor::hiptensor` and GoogleTest. Most do **not** require a configured GPU to run (the exception is `plan_lifetime_test`, noted below). The test binaries are:

| Binary | What it tests |
|---|---|
| `logger_test` | Logger singleton identity, callback registration, log-level filtering, file-sink behavior |
| `yaml_test` | YAML config parsing and round-trip correctness |
| `elementwise_op_test` | CPU-side elementwise operator type traits and identity/scale op correctness |
| `util_test` | Utility helpers (type conversion, index math, stride calculation) |
| `hiptensor_options_test` | CLI/env option parsing (`--hot_runs`, `--cold_runs`, `-y`, `HIPTENSOR_*` env vars) |
| `plan_lifetime_test` | Plan object lifetime: using a permutation plan after its descriptor or preference object is destroyed. Despite living in `00_unit/`, it allocates device memory and launches a real permute kernel, so it **requires a GPU** and is registered as a device tier test |

> `interface_test` exists in the source tree but is commented out in `test/00_unit/CMakeLists.txt` — it is not currently built.

**Test framework:** GoogleTest v1.17.0 (fetched via CMake `FetchContent`; system gtest selectable with `-DHIPTENSOR_USE_SYSTEM_GOOGLETEST=ON`).

**Naming convention:** `<subsystem>_test` (e.g. `logger_test`, `yaml_test`).

**How to run:**

```bash
# From the build tree (no install needed):
ctest                          # runs all tests including unit tests
ctest -j$(nproc)               # same, running test binaries in parallel
./bin/logger_test              # run one unit test directly
./bin/hiptensor_options_test
...
```

There is no ctest filter to run only the unit tests currently.


**What is NOT covered by unit tests:**

- Kernel selection logic (`contraction_selection.cpp`, ActorCritic/DefaultPatient paths) — selection requires a live device to call `hipGetDeviceProperties`.
- Workspace size calculation for contraction — the CK template query requires a device.
- Any GPU memory allocation, data transfer, or kernel launch.
- Plan cache serialization/deserialization (reads/writes device-formatted blobs).

**Coverage expectation:** The hardware-independent surface area in hipTensor is small relative to the total codebase. The current unit test suite covers the identifiable CPU-only subsystems (logging, YAML parsing, option handling, utility math, op type traits). A target of >95% line coverage for the CPU-only subsystems is the goal; overall repository coverage is not meaningful because the dominant code paths are device kernels instantiated from CK templates. Coverage is currently not measured in CI — see the Coverage section below.

---

### Integration Testing Strategy

**Purpose:** Validate GPU-dispatch behavior: kernel selection, numerical correctness vs. CPU reference, and API contract across all supported data types and ranks.

Integration tests are device tests. They require a physical AMD GPU matching the compiled `GPU_TARGETS`. Each test binary accepts a YAML config file (`-y`) that parametrizes the type combinations, tensor ranks, shapes, and algorithms to exercise. Without `-y`, the binary uses the `validation/standard` config compiled in at build time.

**What is covered:**

| Group | Binaries | Scenarios |
|---|---|---|
| Contraction — bilinear (`C = alpha * A * B + beta * D`) | `bilinear_contraction_test_m{1-6}n{1-6}k{1-6}` | FP16/BF16/FP32/FP64/CF32/CF64; ranks 1–6; ACTOR_CRITIC and DEFAULT algorithms |
| Contraction — scale (`C = alpha * A * B`) | `scale_contraction_test_m{1-6}n{1-6}k{1-6}` | Same type matrix as bilinear |
| Contraction — complex bilinear/scale | `complex_{bilinear,scale}_contraction_test_m{1-6}...` | CF32, CF64 |
| Contraction — trinary | `trinary_bilinear_contraction_test`, `trinary_scale_contraction_test` | Three-tensor contractions |
| Contraction — unary ops | `bilinear_contraction_with_unary_ops_test`, `scale_contraction_with_unary_ops_test` | Fused unary activation ops |
| Plan cache | `plan_cache_test` | Cache write/read round-trip on device |
| Plan lifetime | `plan_lifetime_test` | Permute plan reuse after descriptor/preference destruction (launches a kernel) |
| Contraction mode | `contraction_mode_test` | CPU-side mode validation (host test) |
| Elementwise — permute | `rank{2-6}_elementwise_permute_test` | All rank/type combinations |
| Elementwise — binary op | `rank{2-6}_elementwise_binary_op_test` | Binary elementwise ops |
| Elementwise — trinary op | `rank{2-6}_elementwise_trinary_op_test` | Trinary elementwise ops |
| Elementwise — CPU impl | `elementwise_cpu_test` | CPU reference validation (host test; compiles the `elementwise_cpu_impl`, `elementwise_binary_cpu_impl`, and `elementwise_trinary_cpu_impl` sources into one binary) |
| Reduction | `rank{1-6}_reduction_test` | All rank/type combinations |
| Reduction — CPU impl | `reduction_cpu_impl_test` | CPU reference validation (host test) |

All device tests compare GPU results against a CPU reference implementation using a per-type tolerance.

**GPU-required tests:** All `bilinear_contraction_test_*`, `scale_contraction_test_*`, `complex_*`, `trinary_*`, `*_unary_ops_test`, `rank*_elementwise_*`, `rank*_reduction_test`, `plan_cache_test`, `plan_lifetime_test`.

**CPU-only (host) tests:** `contraction_mode_test`, `elementwise_cpu_test`, `reduction_cpu_impl_test`, and the `00_unit/` binaries (`logger_test`, `yaml_test`, `elementwise_op_test`, `util_test`, `hiptensor_options_test`).

**Test tiers (CTest labels):**

Tiers are applied to the installed tree only (`<prefix>/bin/hiptensor/CTestTestfile.cmake`). Labels are nested: a `quick` test also carries `standard`, `comprehensive`, and `full`.

| Label | Config dir | Duration | When run |
|---|---|---|---|
| `quick` | `validation/quick` | Short | PR smoke |
| `standard` | `validation/standard` | Medium | PR gate |
| `comprehensive` | `validation/comprehensive` | Medium | Nightly |
| `full` | `validation/full` | Long | Nightly / release |
| `ffm-quick` | `emulation/quick` | Long | FFM simulator PR |
| `ffm-full` | `emulation/full` | Long | FFM simulator nightly |
| `bench` | `bench` | Very Long | Manual / nightly perf |

> See [test_categories.yaml](test_categories.yaml) for the specific timeouts on each category.

> **Anchored label selection:** Use `-L '^quick$'` not `-L 'quick'` — an unanchored regex matches `ffm-quick` as well.

> **Parallel runs:** append `-j$(nproc)` to any tier command to run test binaries concurrently, e.g. `ctest -L '^standard$' -j$(nproc)`. The `bench` tier is the exception — run it serially so concurrent binaries do not contend for the GPU and skew timings.

**Test shape scale:** The `quick` tier runs a minimal representative shape per rank (e.g., one small tensor shape). The `standard` tier adds more data-type coverage. `comprehensive` and `full` expand both type coverage and tensor shapes. Bench configs use larger, production-representative shapes (e.g., `[256,20,128,128]`) with 5 hot runs and 1 cold run for stable timing.

**Test duration expectations:**
- Individual `quick` test case: seconds.
- Full `standard` run across all 50+ device binaries: ~20–30 min on gfx942.
- `full` tier: up to 2 h.

**What runs on PRs:** Build + unit tests + `standard` integration tier (per the CI configuration). The `ffm-quick` tier runs on PRs targeting FFM-enabled hardware queues.

**What runs nightly:** `comprehensive` and `full` tiers on all supported architectures; `bench` tier for performance tracking.

**What runs only at release qualification:** Full tier on all supported ASIC targets; manual comparison of bench results against release baseline.

---

### Performance and Benchmarking Testing

**Purpose:** Detect throughput regressions in contraction, elementwise, and reduction operations on each supported architecture.

| Item | Detail |
|---|---|
| Stack layer | Expansion SDK |
| Metrics measured | GFLOP/s per operation type; wall-clock time (hot-run average) |
| How benchmarks are run | `ctest -L '^bench$' -V -O bench.log` from `<prefix>/bin/hiptensor` (5 hot runs, 1 cold run per config; `-V` prints GFLOP/s output, `-O` saves it to a file); alternatively via `scripts/performance/Benchmark{Contraction,Permutation,Reduction}.sh` |
| Baseline stored per architecture | Yes — baselines must not be aggregated across GFX targets |
| Where results are stored | Log files written by the benchmark scripts; no centralized DB currently (known gap) |
| Regression threshold | Not automated — manual review (known gap) |
| Gating approach | Manual nightly review |
| GPU profiling | Not currently integrated |

**Gating:**

| Gating Level | Status | Notes |
|---|---|---|
| PR-level automated gate | No | Known gap |
| Nightly automated comparison | No | Known gap |
| Manual nightly review | Yes | Reviewer checks script output against prior run |
| Release qualification | Yes | Required before sign-off |

**Known gaps:**

- No per-architecture performance baseline stored in a queryable format (SQLite, dashboard, etc.).
- No automated regression threshold enforced in CI.
- Trinary contraction and unary-ops variants have no bench configs.

---

## Pre-submit / CI Gates

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build | Yes | CI / DevOps | `cmake --build` on all supported GPU_TARGETS |
| Unit tests | Yes | Component team | All `00_unit/` binaries pass |
| Integration / smoke tests | Yes | Component team | `ctest -L '^standard$'` on at least one supported ASIC |
| Static analysis | No | — | Not currently gated |
| Formatting checks | No | — | `clang-format` is available but not enforced in CI |
| Code coverage | No | — | Known gap; see Coverage section |
| Shared validation infra | N/A | TheRock team | Shared build and validation infrastructure |
| System validation | N/A | QA | System-level and release validation |
| Release qualification | N/A | Component team + QA + TPM | Readiness review |

### PR Test Classification

| Test | Status | Notes |
|---|---|---|
| `logger_test` | Trusted gate | CPU-only, reliable |
| `yaml_test` | Trusted gate | CPU-only, reliable |
| `elementwise_op_test` | Trusted gate | CPU-only, reliable |
| `util_test` | Trusted gate | CPU-only, reliable |
| `hiptensor_options_test` | Trusted gate | CPU-only, reliable |
| `plan_lifetime_test` | Trusted gate | Device test (launches a permute kernel); runs on GPU runners |
| `*standard*` device tests | Trusted gate | Run on dedicated GPU runners |
| `*comprehensive*` / `*full*` | Informational | Nightly only; not PR gates |
| `*ffm-quick*` | Trusted gate | On FFM-equipped queues |
| `*bench*` | Informational | Manual nightly review |

**Flaky test policy:**
- Flaky tests must be tagged `UNSTABLE` and excluded from blocking merge criteria.
- Every flaky test requires a tracking bug.
- A flaky test is not an accepted permanent state.

---

## Coverage

**Tool:** Clang source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`), enabled with `-DHIPTENSOR_CODE_COVERAGE=ON`.

**How to build and run:**
```bash
cmake -B build -DHIPTENSOR_CODE_COVERAGE=ON
cmake --build build -- -j$(nproc)
# Run tests, then:
llvm-profdata merge -sparse default.profraw -o coverage.profdata
llvm-cov report ./bin/<test_binary> -instr-profile=coverage.profdata
```

**Coverage targets:**

| Scope | Target | Notes |
|---|---|---|
| CPU-only (hardware-independent) paths | >95% line coverage | Achievable today with unit tests |
| Device kernel dispatch paths | Not measured | CK template instantiations are device code |
| Overall repository | Not a meaningful target | Dominant LOC is device kernels |

**Code coverage vs. test coverage distinction:**
- *Code coverage:* percentage of lines executed by the test suite.
- *Test coverage:* percentage of intended functionality exercised (data types, ranks, layouts, algorithms, edge cases).

hipTensor's test coverage surface is wide: 6 tensor ranks × multiple data types × multiple algorithms × row/col-major layouts × alpha/beta combinations. The tier system (quick → full) progressively increases test coverage. Configurations that appear only in `full` (e.g., all type × rank × shape combinations for FP64) have low code-coverage contribution because they exercise the same dispatch paths as smaller configurations.

**Known gaps:**
- Code coverage is not measured in CI today.
- No per-Linux / per-Windows coverage split.

---

## Nightly Validation

Beyond PR validation, nightly runs add:

- `comprehensive` and `full` tier validation on all supported ASIC targets (gfx908, gfx90a, gfx942, gfx950, gfx11xx, gfx12xx).
- Benchmark runs (`ctest -L '^bench$'` from `<prefix>/bin/hiptensor`) with 5 hot runs; results retained for manual comparison. The `scripts/performance/Benchmark*.sh` scripts are an alternative for running benchmarks outside CTest.
- FFM simulator full suite (`ffm-full`) for pre-silicon targets.
- Cross-component smoke: downstream consumers (e.g., libraries built on top of hipTensor) are validated separately by those components' nightly suites.

---

## Supported Configurations

| Configuration | Validation Level | Frequency | Notes |
|---|---|---|---|
| Linux (gfx942 / MI300X) | Full | PR + Nightly | Primary CI target |
| Linux (gfx90a / MI210) | Full | Nightly + Release | FP64 coverage |
| Linux (gfx950 / MI350) | Partial | Nightly + Release | |
| Linux (gfx908 / MI100) | Partial | Nightly + Release | Older CDNA; reduced type support |
| Linux (gfx11xx / RDNA3) | Partial | Nightly + Release | |
| Linux (gfx12xx / RDNA4) | Partial | Nightly + Release | |
| Windows (gfx11xx) | Partial | PR + Nightly | |
| Windows (gfx12xx) | Partial | Nightly + Release | |

**Explicitly not tested / validated:**

- Multi-GPU (single-device API only).
- ROCm versions older than 7.0.
- hipTensor compiled with non-AMD toolchains.

---

## ASAN / TSAN / Sanitizer Coverage

No sanitizer build is currently gated in CI.

**Known GPU-specific limitations:** AddressSanitizer and ThreadSanitizer are not supported for device code on AMD GPUs. Sanitizer coverage is therefore limited to the CPU-side host logic only.

**How to build with ASAN (host-side only):**
```bash
cmake -B build -DCMAKE_CXX_FLAGS="-fsanitize=address" -DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address"
cmake --build build -- -j$(nproc)
./bin/logger_test
./bin/yaml_test
```

**What is explicitly not covered:**
- Device memory access patterns (out-of-bounds GPU reads/writes).
- GPU data races (no TSAN equivalent for device code).

---

## Known Bugs and Expected Failures

There is currently one suppression mechanism: CTest `SKIP_REGULAR_EXPRESSION`. Any test binary that prints `HIPTENSOR_STATUS_ARCH_MISMATCH` or `unsupported host device` is skipped rather than failed — this is the mechanism for running a single install across multiple GPU architectures without every test failing on incompatible hardware. It is applied uniformly to all test binaries at configure time in `test/CMakeLists.txt`.

There is no client-side known-bug list (analogous to `known_bugs.yaml` in other components). Tests that expose a known defect are either disabled via empty config files (which causes CMake to omit the `add_test` entry entirely) or manually skipped until a fix lands. A tracked quarantine list with ticket references does not exist today — see the Known Gaps table.

---

## Choosing the Right Test Type

| Scenario | What to add |
|---|---|
| New CPU-side logic (logging, YAML, option parsing, utility math) | Unit test in `test/00_unit/` using GoogleTest |
| New or changed GPU operation behavior | Integration test: add a YAML config entry in the appropriate `validation/` tier |
| Bug fix | A regression test (unit if CPU-side, integration config entry if GPU-side) that fails before the fix |
| New data type or tensor rank | Config entries across all relevant validation tiers; empty config file for tiers where the test is already covered |
| Performance-sensitive change | Run `ctest -L '^bench$' -V -O bench.log` from `<prefix>/bin/hiptensor` before and after; include the comparison in the PR description. The `scripts/performance/Benchmark*.sh` scripts are an alternative. |
| New GPU architecture | Validate on that target; update the Supported Configurations table |

---

## Known Gaps Summary

| Gap | Regression risk | Impact | Mitigation today |
|---|---|---|---|
| No automated performance regression threshold or alerting in CI | High | High | Manual nightly review of benchmark script output |
| No per-architecture perf baseline in queryable format | High | High | Log files only; comparison is manual |
| No tracked quarantine list for known-failing tests | Medium | Medium | Defects are handled ad-hoc; no ticket linkage or expiry |
| Kernel selection (ActorCritic/DefaultPatient) has no unit test — requires a live device | Medium | Medium | Covered by integration tests; slow feedback |
| Code coverage not measured in CI | Medium | Medium | None |
| `interface_test` commented out and not built | Low | Medium | None |
| Trinary/unary-ops variants have no bench configs | Low | Low | Not performance-gated |
| `clang-format` and static analysis not enforced in CI | Low | Low | Manual enforcement in code review |
| No sanitizer (ASAN/TSAN) gate in CI | Low | High if hit | None; ASAN can be built manually |

---

## Owners and Review Cadence

**Review this document when:**
- A new test pattern, test lane, or CTest tier is added.
- A regression escapes to a downstream consumer — that is direct evidence of a gap in this strategy.
- Before a major release, alongside the known-gap review.

The measure of whether this document is working: the Known Gaps table shrinks over time.
