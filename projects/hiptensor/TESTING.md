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
| Performance-sensitive path | `ctest -V -O bench.log` | `<prefix>/bin/hiptensor/bench` | Yes |

> **Run in parallel:** add `-j$(nproc)` (or `-j <N>`) to any validation run — e.g. `ctest -L '^standard$' -j$(nproc)` — to run test binaries concurrently. Do **not** parallelize the `bench` tier: concurrent runs contend for the GPU and corrupt timing measurements.

**3. Add the right kind of test.** See [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open the PR** targeting `develop`. Standard-tier tests (`-L '^standard$'`) are the integration merge gate.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** Validate hardware-independent logic without dispatching a GPU kernel.

Unit tests live in `test/00_unit/` and are built with the `add_hiptensor_unit_test()` CMake function, which links against `hiptensor::hiptensor` and GoogleTest. Most do **not** require a configured GPU to run (the exceptions are `plan_lifetime_test`, `elementwise_broadcast_test`, `api_safety_test`, `large_tensor_test`, `plan_cache_unit_test`, `contraction_selection_test` and `alloc_failure_test`, noted below). The test binaries are:

| Binary | What it tests |
|---|---|
| `logger_test` | Logger singleton identity, callback registration, log-level filtering, file-sink behavior |
| `yaml_test` | YAML config parsing and round-trip correctness |
| `elementwise_op_test` | CPU-side elementwise operator type traits and identity/scale op correctness |
| `util_test` | Utility helpers (type conversion, scalar reads per compute type, index math, stride calculation) |
| `hiptensor_options_test` | CLI/env option parsing (`--hot_runs`, `--cold_runs`, `-y`, `HIPTENSOR_*` env vars) |
| `plan_lifetime_test` | Plan object lifetime: using a permutation plan after its descriptor or preference object is destroyed. Despite living in `00_unit/`, it allocates device memory and launches a real permute kernel, so it **requires a GPU** and is registered as a device tier test |
| `api_safety_test` | API misuse and documented-legal calls: operation descriptor attribute sizes and padding, compute descriptors without a scalar type, `hiptensorCreatePlan` with a null preference, one preference shared by several plans, and the contraction workspace-size query. Its plans launch permute and contraction kernels, so it **requires a GPU** and is registered as a device tier test |
| `large_tensor_test` | `hiptensorCreatePlan` returning `HIPTENSOR_STATUS_NOT_SUPPORTED` for tensors the kernels cannot address (2^31 bytes or more, from lengths and strides; 2^29 bytes or more for f64 elementwise operations with an input laid out differently from the output; the intermediate tensor of a trinary contraction), just below and at each limit, for every operation. It creates descriptors only, but needs a handle and plans small contractions, so it **requires a GPU** and is registered as a device tier test |
| `plan_cache_unit_test` | Plan cache: LRU eviction within the capacity, the key (device, output tensor E, operators, tensor ranks), the versioned cache file (round trip; files of earlier or other versions, of other devices, damaged or with more lines than the capacity are refused), and the cache staying usable after a write. Its API cases run contractions before and after writing and reading the cache file, so it **requires a GPU** and is registered as a device tier test |
| `contraction_selection_test` | `HIPTENSOR_ALGO_ACTOR_CRITIC` selection: f32 contractions with f32 and bf16 compute are planned and run correctly, a problem the actor-critic table names no kernel for is planned among the kernels that can run it, a named kernel that cannot run the problem (for a trinary contraction, either of its steps) is replaced by one that can, and (with mock kernels) the replacement is the fastest kernel that can run the problem within the workspace limit, or the plan is refused when none can. CTest also runs it as `contraction_selection_test_row_major_tables`, with `HIPTENSOR_DEFAULT_STRIDES_COL_MAJOR=OFF` and a `--gtest_filter` for the `ActorCriticRowMajorTableTest` cases, which need the actor-critic tables for row-major strides and are skipped in the plain run. It launches contraction kernels, so it **requires a GPU** and is registered as a device tier test |
| `device_support_test` | `hiptensorCreate` on a device whose architecture hipTensor does not support (simulated on any GPU with the test-only environment variable `HIPTENSOR_TEST_FORCE_UNSUPPORTED_ARCH=ON`, which can only take support away) returns `HIPTENSOR_STATUS_ARCH_MISMATCH`, and on a host without a visible GPU returns `HIPTENSOR_STATUS_HIP_ERROR`, instead of ending the process when the library is loaded. It has its own `main`, since the shared test main stops on such hosts. CTest also runs it as `device_support_test_no_visible_device`, with `HIP_VISIBLE_DEVICES=-1` and `ROCR_VISIBLE_DEVICES=-1`, which runs the no-device case; in the plain run that case is skipped. Registered as a host test |
| `elementwise_instance_selection_test` | Host-only lookup of the tuned permutation instance parameters: extents beyond the largest tuned extent use the last table entry instead of reading past the end of the table |
| `alloc_failure_test` | A device allocation failure inside the library is returned as `HIPTENSOR_STATUS_ALLOC_FAILED` and the library keeps working: it fills the device memory between two executions of a complex contraction plan. It **requires a GPU** with its own device memory (it skips on integrated GPUs) and runs serially (`RUN_SERIAL`), since it fills the device memory that other tests would use |
| `batched_contraction_test` | `hiptensorCreateContraction` / `hiptensorCreateContractionTrinary` return `HIPTENSOR_STATUS_NOT_SUPPORTED` for batched modes (a mode shared by both inputs and the output) and accept non-batched ones. No kernel launch |
| `elementwise_mode_test` | Elementwise mode validation: subset and reordered input modes are accepted; a mode missing from the output or repeated within a tensor returns `NOT_SUPPORTED`; a shared mode with mismatched extents returns `INVALID_VALUE`; a permute or trinary op whose output data type differs from its input's returns `NOT_SUPPORTED` on the Composable Kernel backend; a binary or trinary operator other than ADD, MUL, MAX or MIN returns `INVALID_VALUE`. Allocates small device buffers but launches no kernel |
| `elementwise_broadcast_test` | Elementwise broadcast results for permute, binary, and trinary (inputs carrying a subset of the output modes). Launches kernels, so it **requires a GPU** |

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

#### ABI export-surface test (`abi_exported_symbols`)

A host-only, non-GoogleTest check that guards the shared-library ABI. It runs
`scripts/check_exported_symbols.py` over the built `libhiptensor.so` and fails if any symbol
outside the public allowlist is exported. The allowlist is the public `extern "C"` C API plus
the two shared-state singletons that the test harness and the library must share
(`hiptensor::HiptensorOptions`, `hiptensor::Logger`); everything else, internal C++ classes and
Composable Kernel template instantiations, must be hidden. This enforces the hidden-visibility
baseline and the version script (`library/src/hiptensor.map`) so internal symbols cannot silently
re-leak.

- **Platform:** Linux/ELF only. On Windows the DLL export set is driven by `__declspec` and there
  is no version script, so the test is not registered there. It is also skipped if a Python 3.9+
  interpreter or an `nm`/`llvm-nm` tool is not found at configure time.
- **Requires a GPU:** No. It inspects the `.so` with `nm -D` and runs in any environment.
- **CTest label:** `symbol-visibility` (build tree). In the installed tree it is registered as
  `abi_exported_symbols quick`, so the tier nesting carries it into the `quick`, `standard`,
  `comprehensive`, and `full` label sets — it runs in every validation tier.

```bash
# Build tree:
ctest -R abi_exported_symbols            # run just the ABI check
# Or invoke the checker directly against any built/installed library:
python3 scripts/check_exported_symbols.py build/lib/libhiptensor.so
```

---

### Integration Testing Strategy

**Purpose:** Validate GPU-dispatch behavior: kernel selection, numerical correctness vs. CPU reference, and API contract across all supported data types and ranks.

Integration tests are device tests. They require a physical AMD GPU matching the compiled `GPU_TARGETS`. Each test binary accepts a YAML config file (`-y`) that parametrizes the type combinations, tensor ranks, shapes, and algorithms to exercise. Without `-y`, the binary uses the `validation/standard` config compiled in at build time.

**What is covered:**

| Group | Binaries | Scenarios |
|---|---|---|
| Contraction — bilinear (`C = alpha * A * B + beta * D`) | `bilinear_contraction_test_m{1-6}n{1-6}k{1-6}` | FP16/BF16/FP32/FP64/CF32/CF64; ranks 1–6; ACTOR_CRITIC and DEFAULT algorithms. CTest also runs `bilinear_contraction_test_m1n1k1` as `bilinear_contraction_test_m1n1k1_yaml_config`, with a one-case `-y` config and `--gtest_list_tests`, which must list only that case |
| Contraction — scale (`C = alpha * A * B`) | `scale_contraction_test_m{1-6}n{1-6}k{1-6}` | Same type matrix as bilinear |
| Contraction — complex bilinear/scale | `complex_{bilinear,scale}_contraction_test_m{1-6}...` | CF32, CF64 |
| Contraction — trinary | `trinary_bilinear_contraction_test`, `trinary_scale_contraction_test` | Three-tensor contractions |
| Contraction — layouts | `contraction_layout_test` | Binary and trinary f32 contractions with permuted mode orders, packed column-major, row-major and padded layouts, mixed layouts, and an additive input laid out differently from the output, checked bit-exactly against an independent loop reference; validation of the additive input's modes and extents |
| Contraction — trinary workspace | `trinary_contraction_workspace_test` | Trinary contractions (bilinear, bilinear with a unary op, scale; DEFAULT and ACTOR_CRITIC) with the workspace filled with NaN before every execution, checked bit-exactly against an independent loop reference |
| Contraction — test helpers | `contraction_test_utils_test` | The output comparison of the contraction tests checks every element, both parts of every complex element included; GFLOP counts for any mode order, complex types and trinary contractions. CTest also runs it as `contraction_test_utils_test_gtest_filter`, with a `--gtest_filter` flag that must reach GoogleTest |
| Contraction — unary ops | `bilinear_contraction_with_unary_ops_test`, `scale_contraction_with_unary_ops_test` | Fused unary activation ops |
| Plan cache | `plan_cache_test` | Cache write/read round-trip on device |
| Plan lifetime | `plan_lifetime_test` | Permute plan reuse after descriptor/preference destruction (launches a kernel) |
| Contraction mode | `contraction_mode_test` | CPU-side mode validation (host test) |
| Operation validation | `operation_validation_test` | Tensor descriptors without extents or with non-positive extents or negative strides; contraction modes the kernels can't address (more than six per group, no M or N mode, a mode carried by one tensor only, repeated modes, mismatched extents), binary and trinary; an outer product checked exactly against a loop; non-unary operand operators; elementwise and reduction plans either refused or executed with the loop result; the elementwise data type rows only the rocKE backend serves (declined by the Composable Kernel backend when the plan is created; with `HIPTENSOR_BACKEND=rocke` executed with the loop result, or refused by a library without the rocKE backend) |
| Workspace | `workspace_size_test` | `hiptensorEstimateWorkspaceSize` with a null, fresh and used preference, `HIPTENSOR_PLAN_REQUIRED_WORKSPACE` below a large limit, binary and trinary contractions executed with exactly the reported workspace and checked exactly against a loop; no workspace for elementwise operations and reductions |
| Elementwise — permute | `rank{2-6}_elementwise_permute_test` | FP16/BF16/FP32/FP64; FP16 and BF16 also with an FP32 scalar; ranks 2–6 |
| Elementwise — binary op | `rank{2-6}_elementwise_binary_op_test` | FP16/BF16/FP32/FP64; FP16 and BF16 also with an FP32 scalar; ranks 2–6 |
| Elementwise — trinary op | `rank{2-6}_elementwise_trinary_op_test` | FP16/BF16/FP32/FP64; FP16 and BF16 also with an FP32 scalar; ranks 2–6 |
| Elementwise — output strides | `elementwise_output_strides_test` | User-supplied (gapped and packed) output strides for permute, binary, and trinary (FP32); launches kernels |
| Elementwise — zero scaling factors | `elementwise_zero_scale_test` | Binary, trinary and permutation with a zero scaling factor and NaN, Inf, or `NULL` in that input (also through `HIPTENSOR_OP_LOG`), reused plans and a broadcast input, checked exactly against an independent loop reference |
| Elementwise — f64 unary operators | `elementwise_f64_unary_op_test` | Every unary operator in f64 binary, trinary, permutation and reduction operations, against the host `long double` math library within 64 ulp of the terms |
| Elementwise — scalars and operators | `elementwise_scalar_op_test` | Scalars with a BF16 compute descriptor, the order of `opAB` and `opABC` in trinary operations, and a permutation plan executed again with another `alpha`, checked exactly against an independent loop reference |
| Elementwise — CPU impl | `elementwise_cpu_test` | CPU reference validation against fixed expected values (host test; compiles the `elementwise_cpu_impl`, `elementwise_binary_cpu_impl`, and `elementwise_trinary_cpu_impl` sources into one binary). Permute: FP32/FP16/BF16 (with a BF16 or FP32 scalar)/FP64 (checked at FP64 precision); binary and trinary: FP32/FP16/BF16, plus FP16 and BF16 with an FP32 scalar |
| Reduction | `rank{1-6}_reduction_test` | All rank/type combinations |
| Reduction — scalars | `reduction_scalar_test` | Scalars with a BF16 compute descriptor, checked exactly against an independent loop reference |
| Reduction — semantics | `reduction_semantics_test` | C copied to a strided D on a non-blocking stream that is still writing C; zero `alpha` or `beta` with NaN or `NULL` inputs; reductions that keep every mode (rank 1, BF16, permuted output) executed twice, checked exactly against an independent loop reference |
| Reduction — CPU impl | `reduction_cpu_impl_test` | CPU reference validation (host test) |

All device tests compare GPU results against a CPU reference implementation using a per-type tolerance.

**GPU-required tests:** All `bilinear_contraction_test_*`, `scale_contraction_test_*`, `complex_*`, `trinary_*`, `*_unary_ops_test`, `rank*_elementwise_*`, `rank*_reduction_test`, `plan_cache_test`, `plan_lifetime_test`, `api_safety_test`, `large_tensor_test`, `plan_cache_unit_test`, `contraction_selection_test`, `contraction_test_utils_test`, `contraction_layout_test`, `elementwise_scalar_op_test`, `elementwise_zero_scale_test`, `elementwise_f64_unary_op_test`, `reduction_scalar_test`, `reduction_semantics_test`, `operation_validation_test`, `workspace_size_test`, `alloc_failure_test`, `elementwise_broadcast_test`, `elementwise_output_strides_test`.

**CPU-only (host) tests:** `contraction_mode_test`, `elementwise_cpu_test`, `reduction_cpu_impl_test`, and the `00_unit/` binaries (`logger_test`, `yaml_test`, `elementwise_op_test`, `util_test`, `hiptensor_options_test`, `elementwise_instance_selection_test`, `device_support_test`). `device_support_test` runs on hosts with or without a GPU, and checks the device only through `hiptensorCreate`. `batched_contraction_test` and `elementwise_mode_test` launch no kernels but create a hipTensor handle and descriptors (`elementwise_mode_test` also allocates small device buffers).

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
| `bench` | `bench` | Very Long | Manual / nightly perf (own `bench/` subdir; excluded from bare `ctest`) |

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
| How benchmarks are run | `ctest -V -O bench.log` from `<prefix>/bin/hiptensor/bench` (5 hot runs, 1 cold run per config, validation off via `-v OFF`; `-V` prints GFLOP/s output, `-O` saves it to a file); alternatively via `scripts/performance/Benchmark{Contraction,Permutation,Reduction}.sh` |
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
- Trinary contraction and unary-ops contraction variants are not in the bench tier: trinary contraction has bench configs (`test/01_contraction/configs/bench/trinary_*_test_params.yaml`) that are not registered in `_HIPTENSOR_BENCH_TESTS` (`test/CMakeLists.txt`), and unary-ops has no bench configs.

---

## Pre-submit / CI Gates

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build | Yes | CI / DevOps | `cmake --build` on all supported GPU_TARGETS |
| Unit tests | Yes | Component team | All `00_unit/` binaries pass |
| ABI export surface | Yes | Component team | `abi_exported_symbols` CTest (runs in the standard tier); Linux-only, no GPU |
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
| `abi_exported_symbols` | Trusted gate | Host-only ABI allowlist check (Linux/ELF); no GPU |
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

**Coverage test configs:** the `codecoverage` set in `rtest.xml` reads its YAML configs from `<build_dir>/code_coverage_config/`, which the build populates (`copy_code_coverage_config_files` in `cmake/Functions/CopyYamlFolder.cmake`). Contraction and reduction copy from their `configs/code_coverage/` directories; elementwise copies from `configs/emulation/quick/` (permutation, binary_op, trinary_op). `test/02_elementwise/configs/code_coverage/` is not used — edit `emulation/quick` to change elementwise coverage.

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
- Benchmark runs (`ctest` from `<prefix>/bin/hiptensor/bench`) with 5 hot runs and validation off (`-v OFF`); results retained for manual comparison. The `scripts/performance/Benchmark*.sh` scripts are an alternative for running benchmarks outside CTest.
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
| New data type or tensor rank | Config entries across all relevant validation tiers; empty config file for tiers where the test is already covered. Also add entries to `emulation/quick` **and** `emulation/full` (the FFM tiers do not nest, so `ffm-full` does not inherit `ffm-quick`), to `bench`, and a case in the op's host CPU-reference test (e.g. `elementwise_cpu_test`) where one exists |
| Performance-sensitive change | Run `ctest -V -O bench.log` from `<prefix>/bin/hiptensor/bench` before and after; include the comparison in the PR description. The `scripts/performance/Benchmark*.sh` scripts are an alternative. |
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
| Trinary and unary-ops contraction variants are not in the bench tier (trinary configs exist but are unregistered; unary-ops has none) | Low | Low | Not performance-gated |
| `clang-format` and static analysis not enforced in CI | Low | Low | Manual enforcement in code review |
| No sanitizer (ASAN/TSAN) gate in CI | Low | High if hit | None; ASAN can be built manually |
| Permute `bench` configs exceed device memory on smaller GPUs: the binary aborts with a HIP out-of-memory error partway through (gfx908 and gfx1100 already in the FP32 entries; gfx90a in the FP64 entries at ranks 2 and 4, which are listed last so the other types complete) | Low | Medium | Bench results on those targets are incomplete; run bench on MI300-class GPUs |
| `test/02_elementwise/configs/code_coverage/` is unused (elementwise coverage reads `emulation/quick`) | Low | Low | Documented in the Coverage section |

---

## Owners and Review Cadence

**Review this document when:**
- A new test pattern, test lane, or CTest tier is added.
- A regression escapes to a downstream consumer — that is direct evidence of a gap in this strategy.
- Before a major release, alongside the known-gap review.

The measure of whether this document is working: the Known Gaps table shrinks over time.
