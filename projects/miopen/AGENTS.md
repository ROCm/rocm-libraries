# Repository Guidelines

Guidance for `projects/miopen/`, supplementing the super-repo [AGENTS.md](../../AGENTS.md). Paths and commands are relative to this project unless stated otherwise.

## Project Overview

MIOpen provides AMD GPU deep-learning primitives: convolution, normalization, pooling, activation, recurrent operations, fusion, and tensor operations. Its public C API wraps a C++20 implementation and HIP kernels. Historical filenames such as `src/hip/convolutionocl.cpp` do not imply an active OpenCL backend.

## Architecture & Data Flow

```text
C API -> descriptors + Handle -> ProblemDescription + ExecutionContext
  -> applicable solver -> ConvSolution (kernel metadata + invoker factory)
  -> compiled/cached kernels -> invoker -> GPU work on the handle's stream
```

- **API boundary:** `include/miopen/miopen.h` defines public contracts; `src/*_api.cpp` bridges opaque C descriptors to internal C++ objects in `src/include/miopen/`. Keep internal types out of public headers.
- **State and extension points:** descriptors hold operation parameters; `Handle` manages resources; `ExecutionContext` carries device/search settings. Reuse solver interfaces and invoker factories. Fusion plans and RNN's stateful, multi-stream solutions have their own lifecycle rules.
- **Solver contracts:** `miopen::solver::SolverBase` implementations have no variable internal state. `IsApplicable` establishes correctness, not merely performance. Default tuning configurations must be valid without benchmarking; workspace size must not depend on the performance configuration.
- **Find versus immediate:** legacy algorithm-enum convolution execution uses invokers registered by Find. Immediate APIs select a solver ID and can prepare/cache its invoker lazily. Find 2.0 uses problem/solution objects; not every operation supports standalone discovery.
- **Persistence:** FindDb stores selected solutions/timings/workspace; PerfDb stores tuning configurations; binary and invoker caches are separate. User records override system records. Preserve the distinct key formats: `ProblemDescription::Serialize`, `VisitAll`, and `MakeNetworkConfig`.
- **Kernel backends:** `addkernels/` embeds `src/kernels/` sources; `src/hipoc/` compiles/loads HIP, assembly, and optional MLIR code. CK uses architecture-specific plugins; internal consumers link `MIOpen_with_plugins` to preserve plugin build dependencies.
- **Optional hipDNN wrapper:** `MIOPEN_ENABLE_HIPDNN_WRAPPER` splits public `libMIOpen` from `libMIOpen_private`. `src/private/wrapper.cpp` routes selected APIs through hipDNN or native `_impl` functions. Routing settings are cached; set them before first use. Forwarded failures return errors, not silent native fallback. The forwarding set is defined in `src/private/routing.cpp`.

## Key Directories

| Path | Change here for |
| --- | --- |
| `include/miopen/`, `src/include/miopen/` | Public C contracts versus internal descriptors, handles, solvers, and databases. |
| `src/solver/`, `src/conv/` | Solver applicability/solutions, convolution keys, selection, and heuristics. |
| `src/kernels/`, `src/hip/`, `src/hipoc/` | Device sources, runtime execution, compilation, caching, and DB/model assets. |
| `src/ck_impl/`, `src/hipconv/` | CK plugins and vendored hipconv integration; distinguish integration code from upstream-managed kernels. |
| `src/private/`, `include_wrapper/miopen/` | ABI split, hipDNN translation/routing, and wrapper-only declarations. |
| `test/`, `test/gtest/`, `test/public_abi/`, `driver/` | Legacy tests, GoogleTest fixtures, ABI baselines, and driver reproductions. |
| `cmake/`, `script/`, `addkernels/`, `tools/`, `docs/` | Build rules, ABI/parity utilities, kernel embedding, DB conversion, and documentation. |

Do not treat untracked logs, benchmarks, data dumps, or alternate guidance drafts as repository conventions. Do not edit generated build output such as `build/inlined_kernels/`.

## Development Commands

Use a compatible ROCm development environment; `Dockerfile` describes the toolchain/dependency setup. Match `GPU_TARGETS` to the device under test (`gfx942` below is an example). Prefer active CMake rules over historical README version/backend claims.

From the project root, using a fresh build directory. `MIOPEN_TEST_DISCRETE` defaults to ON; this example sets it OFF to build the combined `miopen_gtest` binary (see Testing & QA):

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/ClangToolChain.cmake" \
  -DROCM_CMAKE_PATH=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DMIOPEN_STANDALONE_BUILD=ON -DMIOPEN_BACKEND=HIP -DGPU_TARGETS=gfx942 \
  -DBUILD_DEV=ON -DBUILD_TESTING=ON -DMIOPEN_TEST_DISCRETE=OFF
cmake --build build --target MIOpenDriver miopen-tests
cmake --build build --target check_format
cmake --build build --target miopen-tidy
# Install only when needed; choose CMAKE_INSTALL_PREFIX at configure time.
cmake --install build
```

- Pass the toolchain on the first configure. Avoid an environment `ROCM_PATH`; use the documented CMake variables or unset it inside a container.
- `MIOPEN_STANDALONE_BUILD` enables system-first dependency fetching; missing dependencies may require network access. `MIOPEN_NO_DOWNLOAD` is not a reliable offline switch.
- CK normally uses installed packages; `MIOPEN_BUILD_CK=ON` builds sibling `../composablekernel` inline. Check compatibility and warning policy for that mode. Building `MIOpen` alone does not prove the CK plugin built.
- Feature controls include `MIOPEN_USE_COMPOSABLEKERNEL`, `MIOPEN_USE_HIPCONV`, `MIOPEN_USE_MLIR`, and `MIOPEN_ENABLE_HIPDNN_WRAPPER`. COMGR/HIPRTC settings must match. Read `CMakeLists.txt` for defaults and restrictions.
- `MIOPEN_EMBED_DB` selects database architectures; `MIOPEN_EMBED_BUILD` controls embedded-build restrictions. `MIOPEN_INSTALL_GPU_DATABASES` selects installed assets, not compiled GPU targets.
- `miopen-tests` builds tests; `miopen-check` builds dependencies and runs CTest, excluding the `*_suite` tier tests (use `ctest -L <tier>` for those). Prefer prefixed targets, including `miopen-tidy`, `miopen-cppcheck`, and `miopen-analyze`, over standalone-only aliases.

From the super-repo root:

```sh
cmake --preset miopen -DROCM_LIBS_ENABLE_ROOT_CTEST=ON -DGPU_TARGETS=gfx942
cmake --build build --target MIOpenDriver miopen-tests
pre-commit run --files projects/miopen/AGENTS.md  # Replace with changed files.
```

The `miopen-with-provider` preset also includes hipDNN and its MIOpen provider. Run pre-commit directly: CMake's wrappers can skip tooling or accept a failing pre-commit exit status. `format` modifies files; `check_format` checks them.

## Code Conventions & Common Patterns

- **Formatting:** follow `.clang-format`: four spaces, no tabs, 100 columns, custom Allman-style braces, left-aligned pointers, no automatic include sorting. Preserve local header guards/license headers; copy new-file attribution from a sibling.
- **Naming:** public functions use `miopen...`; internal types/methods use names such as `TensorDescriptor`, `Handle`, and `GetSolution` in `miopen` namespaces.
- **Errors:** `MIOPEN_THROW` carries status information. Most status-returning APIs translate exceptions with `miopen::try_` and use `miopen::deref` for checked access. Inspect the specific API before assuming exception/null-check coverage.
- **Logging:** `MIOPEN_ENABLE_LOGGING` enables API-call logging; `MIOPEN_LOG_LEVEL` independently controls severity; `MIOPEN_ENABLE_LOGGING_CMD` emits replayable driver commands. Reuse `MIOPEN_LOG_FUNCTION` and `MIOPEN_LOG_E/W/I/I2/T`.
- **Environment settings:** declare `MIOPEN_DECLARE_ENV_VAR_BOOL/UINT64/STR` in the global namespace; use `env.hpp` helpers. Values are cached on first access; declaration defaults affect unset behavior. Document user-facing settings in `docs/reference/env_variables.rst`.
- **Feature macros:** generated configuration uses numeric `#cmakedefine01`; use `#if MIOPEN_USE_...`, not `#ifdef`.
- **Ownership/async:** reuse RAII/custom deleters in `manage_ptr.hpp` and existing handle/stream machinery. Preserve resource lifetimes through asynchronous completion; do not add synchronization or shared global execution state indiscriminately.

Adding a convolution solver requires all integration points:

1. Implement under `src/solver/conv/`; add the source to `MIOpen_Source` in `src/CMakeLists.txt`.
2. Declare it in `src/include/miopen/conv/solvers.hpp`, using `ConvSolver` or `ConvTunableSolver<PerformanceConfigName>` and existing const override signatures.
3. Append registration in `src/solver.cpp`. Numeric IDs are persistent: preserve removed slots with `++id`; never renumber/reuse IDs. Preserve `SolverDbId` when renaming a class.
4. Add it to the appropriate `SolverContainer` in `src/mlo_dir_conv.cpp`; registration alone does not make it discoverable.
5. Reuse `test/gtest/unit_conv_solver*.hpp` fixtures and `unit_conv_solver_<SolverName>.cpp` naming to cover applicability and numerical correctness.

Public API changes must synchronize `miopen.h`, `src/private/miopen_impl.h`, `src/private/miopen_private_rename.h`, wrapper stubs, and mirrors under `../../dnn-providers/miopen-provider/`. Do not change ABI baselines/exclusion lists simply to conceal drift.

## Important Files

- `CMakeLists.txt`, `src/CMakeLists.txt`, `cmake/{ClangToolChain,ThirdParty,Dependencies}.cmake`: defaults, source registration, toolchain discovery, dependencies.
- `src/include/miopen/{solver,conv_solution,find_solution,errors,handle}.hpp`: solver/invoker contracts, tuning precedence, error conversion, resources.
- `test/gtest/README.md`, `test/gtest/test_categories.yaml`, `test/gtest/check_names.py`, `test/utils/gtest_formating_checks.py`: eligibility, tiers, naming rules.
- `driver/README.md`, `docs/reference/env_variables.rst`, `docs/how-to/debug-log.rst`: reproduction and diagnostics. Edit navigation template `docs/sphinx/_toc.yml.in`, not generated `_toc.yml`.
- `CONTRIBUTING.md`, `.github/pull_request_template.md`, `CHANGELOG.md`, `Jenkinsfile`, `vars/utils.groovy`: contribution expectations, release notes, current CI.

## Runtime/Tooling Preferences

- Use ROCm Clang/HIP and C++20. The top-level minimum is CMake 3.15, but standalone dependency fetching (`cmake/ThirdParty.cmake`) requires 3.25+ and the super-repo and inline-CK paths (`cmake/Dependencies.cmake`) require 3.25.2+; use 3.25.2+ in practice. Ninja and Makefiles are supported; do not share build directories across incompatible environments/generators.
- Use Python 3. Follow formatter pins in the super-repo `.pre-commit-config.yaml`, not arbitrary system versions. Preserve `.clang-tidy`'s explicit allowlist and the compiler-warning policy.
- `requirements.txt`/`dev-requirements.txt` are cget recipes, not pip manifests. `install_deps.cmake` is the legacy installer; ROCm development packages and enabled dependencies still need to be discoverable.
- GPU containers need `/dev/kfd`, `/dev/dri`, and appropriate video/render permissions. The project's Docker build uses `projects/` as context to include sibling dependencies.
- GPU database assets use DVC; configuration does not automatically download them. Pull required assets before DB-dependent validation.
- Isolate experiments with writable `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`, and, where relevant, `MIOPEN_VERIFY_CACHE_PATH`. Never clear user DBs/caches or enable destructive tuning casually.

## Testing & QA

Read `test/gtest/README.md` before changing tests. Use independent references, datatype-appropriate tolerances, explicit device eligibility, and boundary/regression coverage. Record filters, device, datatype, skips, and performance effects for solver/kernel changes.

- **GoogleTest/GoogleMock:** `MIOPEN_TEST_DISCRETE=ON` builds per-file `test_<source-stem>` binaries with individual CTest discovery. `OFF` builds `miopen_gtest`, shards, and YAML tiers. Solver-name CTest filters apply to discrete discovery, not combined shard names.
- **Naming:** suites follow `CPU_..._<TYPE>`/`GPU_..._<TYPE>`; parameterized prefixes start with `Smoke`, `Standard`, `Full`, `Perf`, or `Unit`. Keep suites unique across files. `test/gtest/check_names.py` (run via `miopen_gtest_check` on the combined test list) rejects GPU+`NONE`; `test/utils/gtest_formating_checks.py` is the source-level check.
- **Legacy tests:** `CHECK` prints `FAILED` and continues; `EXPECT`, `EXPECT_EQUAL`, and `STATUS` abort. CTest also rejects failure text. `add_custom_test(<test-name> ... COMMAND ...)` takes a positional name.
- **Selection:** ordinary discovery/shards default to Smoke FP32/NONE. `MIOPEN_TEST_ALL` broadens prefixes, not datatype selection; choose datatype flags or explicit filters. YAML categories/architecture exclusions are separate in `test_categories.yaml`; even full has exclusions.
- **Isolation:** direct binaries and YAML tier suites may not set a test DB path; export isolated DB/cache paths. Unsupported-device skips do not prove a solver works on that device.

From a configured build directory:

```sh
# Combined mode:
./bin/miopen_gtest --gtest_filter='CPU_PerfConfigGroupWrwDeterministic_NONE.*'
ctest -N -L quick   # list tier tests; labels: quick, standard, comprehensive, full
# Tier labels exist only in combined mode. -LE ex_gpu drops arch-specific suites; on an
# architecture listed in test_categories.yaml use -L '^ex_gpu_<arch>$' instead.
ctest --output-on-failure -L quick -LE ex_gpu
# Discrete mode:
./bin/test_unit_conv_solver_ConvDirectDepthwiseFwd2D --gtest_filter='Smoke/*FP32*'
# Driver; keep reference verification enabled (-V 0 disables it):
./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1
```

ABI/script checks from the project root:

```sh
python3 script/check_public_abi.py check-headers --require-provider
python3 script/check_public_abi.py check build/lib/libMIOpen.so \
  --baseline test/public_abi/public_symbols.baseline
python3 -m pytest script/test_check_public_abi.py
# test_forwarding_parity_gpu_labels lives in test/gtest/, hence PYTHONPATH; the rest are in script/.
(cd script && PYTHONPATH=../test/gtest python3 -m unittest \
  test_check_wrapper_abi test_miopen_wrapper_libs test_run_forwarding_parity \
  test_compare_forwarding_runs test_forwarding_parity_gpu_labels)
# Wrapper-enabled builds; ABI/failure checks need no GPU:
ctest --test-dir build/test/gtest --output-on-failure \
  -R 'wrapper_abi_check|forwarding_failure_modes|test_forwarding_parity_scripts'
# GPU/backend required:
ctest --test-dir build/test/gtest --output-on-failure -L forwarding_parity
```

Preserve the public SONAME/C symbol contract; wrappers must not export `_impl` names or hard-link the hipDNN backend. Source/ELF gates do not prove numerical behavior or complete ABI layout compatibility. Forwarding parity uses public APIs, `HipdnnShim` naming, and meaningful `parity_*` properties; reuse its harness without duplicate registrations or silently expanded divergence allowances.

CI requirements live in `Jenkinsfile`/`vars/utils.groovy`; local green tests do not replace hardware/datatype coverage. `codecov.yml` sets exclusions, not a fixed local coverage threshold. Update relevant docs/release notes for user-visible changes. Narrative docs use reStructuredText; Sphinx invokes Doxygen. Follow the documentation build workflow in `README.md`.
