# Repository Guidelines

Guidance for `projects/miopen/`, supplementing the super-repo [AGENTS.md](../../AGENTS.md). Paths and commands are relative to this project unless stated otherwise. Where this file names a source of truth (a CMake file, script, or YAML), that file wins if the two disagree.

## Project Overview

MIOpen provides AMD GPU deep-learning primitives: convolution, normalization, pooling, activation, recurrent operations, fusion, and tensor operations. Its public C API wraps a C++20 implementation and HIP kernels. Historical filenames such as `src/hip/convolutionocl.cpp` do not imply an active OpenCL backend.

## Architecture & Data Flow

Typical native solver path:

```text
C API -> descriptors + Handle -> ProblemDescription + ExecutionContext
  -> applicable solver -> ConvSolution (kernel metadata + invoker factory)
  -> compiled/cached kernels -> invoker -> GPU work on the handle's stream
```

- **API boundary:** `include/miopen/miopen.h` defines public contracts; `src/*_api.cpp` and `src/api/` (Find 2.0, tuning) bridge opaque C descriptors to internal C++ objects in `src/include/miopen/`. Keep internal types out of public headers.
- **State and extension points:** descriptors hold operation parameters; `Handle` manages resources; `ExecutionContext` carries device/search settings. Reuse solver interfaces and invoker factories. Fusion plans and RNN's stateful, multi-stream solutions (`src/rnn/`) have their own lifecycle rules.
- **Solver contracts:** `miopen::solver::SolverBase` implementations have no variable internal state. `IsApplicable` establishes correctness, not merely performance. Default tuning configurations must be valid without benchmarking; workspace size must not depend on the performance configuration.
- **Find versus immediate:** legacy algorithm-enum convolution execution needs an algorithm-mapped invoker, normally registered by Find. Immediate APIs select a solver ID and can prepare/cache its invoker lazily. Find 2.0 uses problem/solution objects; not every operation supports standalone discovery (see `src/problem.cpp`).
- **Persistence:** FindDb stores selected solutions/timings/workspace; PerfDb stores tuning configurations; binary and invoker caches are separate. Keep the key formats distinct: FindDb uses `ProblemDescription::Serialize`, PerfDb uses `VisitAll`, invokers use `MakeNetworkConfig`. A user FindDb record replaces the system record; PerfDb merges per solver ID, preferring user values.
- **Kernel backends:** `addkernels/` embeds `src/kernels/` sources; `src/hipoc/` compiles/loads HIP, assembly, and optional MLIR code. CK grouped convolution ships as per-architecture plugins built in `src/ck_impl/`; legacy/static CK sources (`src/legacy_composable_kernel/`, `src/kernels/static_composable_kernel/`) are embedded like other kernels. Driver and test executables link `MIOpen_with_plugins` so the plugins build with them; the plugins link the implementation target. GEMM work can be delegated to rocBLAS/hipBLASLt through `src/gemm_v2.cpp`.
- **Optional hipDNN wrapper:** `MIOPEN_ENABLE_HIPDNN_WRAPPER` splits public `libMIOpen` from `libMIOpen_private`. `src/private/wrapper.cpp` routes selected APIs through hipDNN or native `_impl` functions. The forwarded entry points are listed in `src/private/routing.cpp`; they are convolution execution APIs, while Find and immediate APIs stay native. Routing settings are cached; set them before first use. Forwarded failures return errors, not silent native fallback.

## Key Directories

| Path | Change here for |
| --- | --- |
| `include/miopen/`, `src/include/miopen/` | Public C contracts versus internal descriptors, handles, solvers, and databases. |
| `src/api/`, `src/*_api.cpp` | C API bridges; `src/api/` holds Find 2.0 and tuning entry points. |
| `src/solver/`, `src/conv/` | Solver applicability/solutions (per operation under `src/solver/<op>/`), convolution keys, selection, and heuristics. |
| `src/<op>/` (e.g. `src/batchnorm/`, `src/mha/`, `src/rnn/`) | Per-operation problem descriptions (`src/rnn/` instead holds RNN selection and multi-stream solutions). |
| `src/kernels/`, `src/hip/`, `src/hipoc/` | Device sources, runtime execution, compilation, caching, and DB/model assets. |
| `src/ck_impl/`, `src/legacy_composable_kernel/`, `src/hipconv/` | CK plugins, legacy CK sources, and vendored hipconv (see `src/hipconv/HIPCONV_VERSION`); distinguish integration code from upstream-managed kernels. |
| `src/private/`, `include_wrapper/miopen/` | ABI split, hipDNN translation/routing, and wrapper-only declarations. |
| `src/fin/` | Fin/Tuna interface. `fin_interface.cpp` is always listed in sources, but its body (including tunable-solver dispatch by ID) compiles only when `MIOPEN_ENABLE_FIN_INTERFACE` is ON (defaults to `MIOPEN_ENABLE_FIN`, OFF). |
| `test/`, `test/gtest/`, `test/public_abi/`, `driver/` | Legacy tests, GoogleTest fixtures, ABI baselines, and driver reproductions. |
| `cmake/`, `script/`, `addkernels/`, `tools/`, `docs/` | Build rules, ABI/parity utilities, kernel embedding, DB conversion, and documentation. |

Do not treat untracked logs, benchmarks, data dumps, or alternate guidance drafts as repository conventions. Do not edit generated build output such as `<build-dir>/inlined_kernels/`.

## Development Commands

Use a compatible ROCm development environment. Standalone fetching does not supply ROCm itself or ROCm libraries such as rocBLAS, hipBLASLt, and rocRAND. Match `GPU_TARGETS` to the device under test (`gfx942` below is an example). Prefer active CMake rules over historical README version/backend claims.

The recommended environment is the `rocm/miopen-dev` image, built from `Dockerfile` with ROCm, MIOpen's dependencies, and the build/analysis tooling preinstalled. Jenkins publishes `rocm/miopen-dev:latest` and `rocm/miopen-dev:multiarch_dev_<date>` from green builds; GitHub CI pins a dated tag (see `../../.github/workflows/component-ci-miopen.yml`). The image sets `ROCM_PATH`, so unset it before configuring. Pass the host's `video`/`render` group IDs, which can differ from the image's:

```sh
docker run -it --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
  --group-add "$(getent group video | cut -d: -f3)" \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v "$PWD/../..":/src -w /src/projects/miopen rocm/miopen-dev:latest bash
unset ROCM_PATH  # inside the container
```

From the project root, using a fresh build directory. `MIOPEN_TEST_DISCRETE` defaults to ON; this example sets it OFF to build the combined `miopen_gtest` binary, and sets `GTEST_PARALLEL_LEVEL` (Jenkins uses 4) so its CTest shards are valid (see Testing & QA):

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE="$PWD/cmake/ClangToolChain.cmake" \
  -DROCM_CMAKE_PATH=/opt/rocm -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DMIOPEN_STANDALONE_BUILD=ON -DMIOPEN_BACKEND=HIP -DGPU_TARGETS=gfx942 \
  -DBUILD_DEV=ON -DBUILD_TESTING=ON -DMIOPEN_TEST_DISCRETE=OFF -DGTEST_PARALLEL_LEVEL=4
cmake --build build --target MIOpenDriver miopen-tests
cmake --build build --target check_format  # clang-format only; run pre-commit separately.
cmake --build build --target miopen-tidy
# Install only when needed; choose CMAKE_INSTALL_PREFIX at configure time.
cmake --install build
```

- Pass the toolchain on the first configure. Avoid an environment `ROCM_PATH`; use the documented CMake variables or unset it inside a container.
- `MIOPEN_STANDALONE_BUILD` enables system-first dependency fetching; missing dependencies may require network access. `MIOPEN_NO_DOWNLOAD` exists only on the inline-CK path (`cmake/Dependencies.cmake`) and does not make `cmake/ThirdParty.cmake` offline.
- CK normally uses installed packages; `MIOPEN_BUILD_CK=ON` builds sibling `../composablekernel` inline. Check compatibility and warning policy for that mode. Building `MIOpen` alone does not prove a CK plugin built, and plugins are built only for supported `GPU_TARGETS`.
- Feature controls include `MIOPEN_USE_COMPOSABLEKERNEL`, `MIOPEN_USE_HIPCONV`, `MIOPEN_USE_MLIR`, and `MIOPEN_ENABLE_HIPDNN_WRAPPER`. `MIOPEN_USE_COMPOSABLEKERNEL=OFF` only skips the CK plugin build and CK-header tests; host CK solvers stay registered. COMGR/HIPRTC settings must match. Read `CMakeLists.txt` for defaults and restrictions.
- `MIOPEN_EMBED_DB` takes architecture-CU pairs (e.g. `gfx906_60`), not `GPU_TARGETS` names; see `docs/install/embed.rst`. `MIOPEN_EMBED_BUILD` forces static libraries and disables MIOpenDriver by default; configure fails unless `MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK` and `MIOPEN_ENABLE_AI_KERNEL_TUNING` are OFF (both default ON). Static builds currently also need `MIOPEN_USE_HIPCONV=OFF`, because `hipconv` is not in the install export set. `MIOPEN_INSTALL_GPU_DATABASES` selects installed assets, not compiled GPU targets.
- `miopen-tests` builds tests; `miopen-check` builds dependencies and runs CTest, excluding the `*_suite` tier tests (use `ctest -L <tier>` for those). Use the `miopen-` prefixed targets (`miopen-check`, `miopen-tidy`, `miopen-cppcheck`, `miopen-analyze`); the unprefixed aliases exist only outside the super-repo build (`ROCM_LIBS_SUPERBUILD`). Analyzer targets exist even when the tool is missing: look for `Clang tidy not found` in the configure log, and note that a missing cppcheck is not reported.

From the super-repo root:

```sh
cmake --preset miopen -DROCM_LIBS_ENABLE_ROOT_CTEST=ON -DGPU_TARGETS=gfx942
cmake --build build --target MIOpenDriver miopen-tests
pre-commit run --files projects/miopen/AGENTS.md  # Replace with changed files.
```

The `miopen-with-provider` preset also includes hipDNN and its MIOpen provider. Run pre-commit directly: the `pre_commit_checks` and `format` targets in `cmake/ClangCheck.cmake` skip hooks when `pre-commit` or the config is not found (it looks two levels above the CMake source dir, so super-repo builds miss it) and accept exit status 1. `format` modifies files; `check_format` only checks C/C++ source formatting with clang-format.

## Code Conventions & Common Patterns

- **Formatting:** follow the nearest `.clang-format`; the project-local file (not the super-repo root one) applies, and `src/hipconv/` has its own. The project style: four spaces, no tabs, 100 columns, custom Allman-style braces, left-aligned pointers, no automatic include sorting. Preserve existing header guards and license headers on edit. New files use the two-line AMD copyright and `SPDX-License-Identifier: MIT` header (see `src/solver/conv/conv_hip_dir2D11x11.cpp`), not a copied legacy MIT block.
- **Naming:** public functions use `miopen...`; internal types/methods use names such as `TensorDescriptor`, `Handle`, and `GetSolution` in `miopen` namespaces.
- **Errors:** `MIOPEN_THROW` carries status information. Most status-returning APIs translate exceptions with `miopen::try_` and use `miopen::deref` for checked access. Inspect the specific API before assuming exception/null-check coverage.
- **Logging:** `MIOPEN_ENABLE_LOGGING` enables API-call logging; `MIOPEN_LOG_LEVEL` independently controls severity; `MIOPEN_ENABLE_LOGGING_CMD` emits the associated MIOpenDriver command lines, which are incomplete for some paths (e.g. pooling with FP8 types). Reuse `MIOPEN_LOG_FUNCTION` and `MIOPEN_LOG_E/W/I/I2/T`.
- **Environment settings:** declare `MIOPEN_DECLARE_ENV_VAR_BOOL/UINT64/STR` in the global namespace; use `env.hpp` helpers. Values are cached on first access; declaration defaults affect unset behavior. Document user-facing settings in `docs/reference/env_variables.rst`.
- **Feature macros:** generated configuration (`include/miopen/config.h.in`) uses numeric `#cmakedefine01`; in host code, use `#if MIOPEN_USE_...`, not `#ifdef`. Kernel sources may use `#ifndef` defaults for build-time `-D` macros.
- **Ownership/async:** reuse RAII/custom deleters in `manage_ptr.hpp` and existing handle/stream machinery. Preserve resource lifetimes through asynchronous completion; do not add synchronization or shared global execution state indiscriminately.

Adding a convolution solver requires all integration points:

1. Implement under `src/solver/conv/`; add the source to `MIOpen_Source` in `src/CMakeLists.txt`, and any new kernel sources to `MIOPEN_KERNELS` in the same file.
2. Declare it in `src/include/miopen/conv/solvers.hpp`, using `ConvSolver` or `ConvTunableSolver<PerformanceConfigName>` and existing const override signatures.
3. Append registration in `src/solver.cpp`. Numeric IDs are persistent: when removing a solver, replace its line with `++id`; never renumber, and reuse a removed ID only deliberately (see the comment above the registration list). Preserve `SolverDbId` when renaming a class.
4. Add it to each applicable algorithm-family `SolverContainer` in `src/mlo_dir_conv.cpp` (e.g. `GetImplicitGemmSolvers` vs `GetImplicitGemmWrWSolvers`); registration alone does not make it discoverable. `test/gtest/solver_find_list_registry.cpp` only checks that it appears in at least one container.
5. For a tunable solver, add its ID case to `AnySolver` in `src/fin/fin_interface.cpp`; otherwise Fin throws `Unknown solver ID` when the interface is enabled. Build and run the Fin interface tests with `-DMIOPEN_ENABLE_FIN_INTERFACE=ON`; the default OFF neither compiles the dispatch code nor exercises those tests.
6. Document the solver's `MIOPEN_DEBUG_CONV_*` disable variable in `docs/reference/env_variables.rst` and add a `CHANGELOG.md` entry.
7. Reuse `test/gtest/unit_conv_solver*.hpp` fixtures and `unit_conv_solver_<SolverName>.cpp` naming to cover applicability and numerical correctness.

Public entry-point changes must keep the five artifacts listed at the top of `script/check_public_abi.py` in sync (public header, `_impl` declarations, private rename header, `src/private/wrapper.cpp` stubs, and the provider's `MiopenApiPrivateRename.hpp`). The `miopen-public-private-headers` pre-commit hook runs `check_public_abi.py check-headers`. Do not change ABI baselines/exclusion lists simply to conceal drift.

## Important Files

- `CMakeLists.txt`, `src/CMakeLists.txt`, `cmake/{ClangToolChain,ThirdParty,Dependencies}.cmake`: defaults, source registration, toolchain discovery, dependencies.
- `src/include/miopen/{solver,conv_solution,find_solution,errors,handle}.hpp`: solver/invoker contracts, tuning precedence, error conversion, resources.
- `test/gtest/README.md`: naming pointers and forwarding-parity conventions. `test/gtest/test_categories.yaml`: tiers and architecture exclusions. `test/gtest/check_names.py`, `test/utils/gtest_formating_checks.py`: naming validators. Device eligibility lives in each test's fixture parameters.
- `driver/README.md`, `docs/reference/env_variables.rst`, `docs/how-to/debug-log.rst`: reproduction and diagnostics. Edit navigation template `docs/sphinx/_toc.yml.in`, not generated `_toc.yml`.
- `CONTRIBUTING.md`, `.github/pull_request_template.md`, `CHANGELOG.md`: contribution expectations and release notes.
- `Jenkinsfile`, `vars/utils.groovy`: Jenkins hardware/datatype lanes and `rocm/miopen-dev` image publishing. `../../.github/workflows/component-ci-miopen.yml`: GitHub PR build variants, clang-tidy, naming, header, and ABI gates. `../../.github/workflows/miopen-wrapper-flag-on.yml`: manual-dispatch wrapper-ON lane (`wrapper_abi_check`, installed-header check, parity registration); it does not gate PRs.

## Runtime/Tooling Preferences

- Use ROCm Clang/HIP and C++20. Use CMake 3.25.2+: the super-repo and the inline-CK path require it, and the lower `cmake_minimum_required` values in individual project files are not enough for a default build. Ninja and Makefiles are supported; do not share build directories across incompatible environments/generators.
- Use Python 3. Follow formatter pins in the super-repo `.pre-commit-config.yaml`, not arbitrary system versions. Preserve `.clang-tidy`'s explicit allowlist and the compiler-warning policy.
- `requirements.txt`/`dev-requirements.txt` are cget recipes, not pip manifests: `install_deps.cmake` installs `requirements.txt` (see `README.md`), and `rbuild prepare -s develop` (`rbuild.ini`) installs `dev-requirements.txt`; the `rocm/miopen-dev` image already has both. The standalone CMake path fetches through `cmake/ThirdParty.cmake` instead. ROCm development packages and enabled dependencies still need to be discoverable.
- GPU containers need `/dev/kfd`, `/dev/dri`, and appropriate video/render permissions. The project's Docker build uses `projects/` as context to include sibling dependencies.
- GPU database assets use DVC; configuration does not automatically download them. Pull required assets before DB-dependent validation.
- Isolate experiments with writable `MIOPEN_USER_DB_PATH`, `MIOPEN_CUSTOM_CACHE_DIR`, and, where relevant, `MIOPEN_VERIFY_CACHE_PATH`. Never clear user DBs/caches or enable destructive tuning casually.

## Testing & QA

Read `test/gtest/README.md` before changing tests. Use independent references, datatype-appropriate tolerances, explicit device eligibility, and boundary/regression coverage. Record filters, device, datatype, skips, and performance effects for solver/kernel changes.

- **GoogleTest/GoogleMock:** `MIOPEN_TEST_DISCRETE=ON` builds per-file `test_<source-stem>` binaries with individual CTest discovery. `OFF` builds `miopen_gtest`; its CTest shard entries need `-DGTEST_PARALLEL_LEVEL=<n>`, otherwise they get an empty `GTEST_TOTAL_SHARDS` and fail at startup. YAML tier suites (`*_suite`) run `miopen_gtest`, so use them only in combined mode; CMake currently registers them in discrete mode too, where they cannot run. Solver-name CTest filters apply to discrete discovery, not combined shard names.
- **Naming:** suites follow `CPU_..._<TYPE>`/`GPU_..._<TYPE>`; parameterized prefixes start with `Smoke`, `Standard`, `Full`, `Perf`, or `Unit`. Keep suites unique across files. `test/gtest/check_names.py` (run via `miopen_gtest_check` on the combined test list) rejects GPU+`NONE`; `test/utils/gtest_formating_checks.py` is the source-level check.
- **Legacy tests:** `CHECK` prints `FAILED` and continues; `EXPECT`, `EXPECT_EQUAL`, and `STATUS` abort. CTest also rejects failure text. `add_custom_test(<test-name> ... COMMAND ...)` takes a positional name.
- **Selection:** ordinary discovery/shards default to Smoke FP32/NONE. `MIOPEN_TEST_ALL` broadens prefixes, not datatype selection; choose datatype flags or explicit filters. YAML tiers ignore those flags (`quick` runs all compiled `Smoke/*` datatypes plus unprefixed `CPU_*` suites) and carry their own architecture exclusions in `test_categories.yaml`; even full has exclusions.
- **Time budgets:** YAML tiers have per-tier timeouts (`execution_settings` in `test/gtest/test_categories.yaml`, including `timeout_multiplier`); `ffm-*` run a fixed list on a tight runtime budget. When adding or broadening tests in a tier, estimate the added runtime and keep it within budget. Move slow cases to a higher tier; follow the existing comment-and-ticket convention for necessary exclusions, not exclusions solely to make CI green.
- **Isolation:** direct binaries and YAML tier suites may not set a test DB path; export isolated DB/cache paths. Unsupported-device skips do not prove a solver works on that device.
- **Empty selections:** CTest exits 0 when a `-R`/`-L` selection matches nothing. Pass `--no-tests=error` to targeted runs.

From a configured build directory:

```sh
# Combined mode:
./bin/miopen_gtest --gtest_filter='CPU_PerfConfigGroupWrwDeterministic_NONE.*'
# Tier labels: quick, standard, comprehensive, full. -L is a regex; anchor it, because
# `quick` and `full` also match the ffm-quick/ffm-full suite.
ctest -N -L '^quick$'
# Generic hardware: -LE ex_gpu drops the arch-specific suites.
ctest --output-on-failure --no-tests=error -L '^quick$' -LE ex_gpu
# Architecture with its own exclusions: use the exact ex_gpu_* key from
# test_categories.yaml, often a family name (gfx1100 -> ex_gpu_gfx110X).
ctest --output-on-failure --no-tests=error -L '^quick$' -L '^ex_gpu_gfx110X$'
# Discrete mode (naive solvers apply to every GPU; many solvers skip unsupported ones):
./bin/test_unit_conv_solver_ConvDirectNaiveConvFwd --gtest_filter='Smoke/*FP32*'
# Driver; keep reference verification enabled (-V 0 disables it):
./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1
```

ABI/script checks from the project root:

```sh
python3 script/check_public_abi.py check-headers --require-provider
# Wrapper-OFF (default) shared build only; wrapper-ON builds use wrapper_abi_check below.
python3 script/check_public_abi.py check build/lib/libMIOpen.so \
  --baseline test/public_abi/public_symbols.baseline
python3 -m pytest script/test_check_public_abi.py
# test_forwarding_parity_gpu_labels lives in test/gtest/, hence PYTHONPATH; the rest are in script/.
(cd script && PYTHONPATH=../test/gtest python3 -m unittest \
  test_check_wrapper_abi test_miopen_wrapper_libs test_run_forwarding_parity \
  test_compare_forwarding_runs test_forwarding_parity_gpu_labels)
# Wrapper-ON builds, after building miopen-tests (the libraries, forwarding_failure_probe,
# and the fake hipDNN backend); no GPU needed:
ctest --test-dir build --output-on-failure --no-tests=error \
  -R '^(wrapper_abi_check|forwarding_failure_modes|test_forwarding_parity_scripts)$'
# GPU/backend required (not registered when configure detects no GPU):
ctest --test-dir build --output-on-failure --no-tests=error -L forwarding_parity
```

Preserve the public SONAME/C symbol contract; wrappers must not export `_impl` names or hard-link the hipDNN backend. Source/ELF gates do not prove numerical behavior or complete ABI layout compatibility. Forwarding parity uses public APIs, `HipdnnShim` naming, and meaningful `parity_*` properties; reuse its harness without duplicate registrations or silently expanded divergence allowances. Parameterized shim tests still need an allowed tier prefix (e.g. `Smoke`); put `HipdnnShim` in the suite name.

CI spans Jenkins and GitHub workflows (see Important Files); local green tests do not replace hardware/datatype coverage. `codecov.yml` sets exclusions, not a fixed local coverage threshold. Update relevant docs/release notes for user-visible changes. Narrative docs use reStructuredText; Sphinx invokes Doxygen. Follow the documentation build workflow in `README.md`.

## Maintaining This File

Claude Code and other agents read this file directly; do not add a `CLAUDE.md` copy or symlink. In the default mode a `CLAUDE.md` here makes Claude Code skip every `AGENTS.md`, including the super-repo one.

- Update this file in the same PR when you change something it documents (a CMake option or default, target, test label or tier, solver integration step, ABI artifact, or CI workflow).
- Verify before writing: read the source of truth or run the command in `rocm/miopen-dev` on a matching GPU. State conditions (combined vs discrete mode, wrapper ON, GPU required).
- Point to the source of truth instead of copying values that change (defaults, lists, versions). Do not restate the super-repo `AGENTS.md`, `CONTRIBUTING.md`, or PR-bot rules; link them.
- Record repository facts, not local-machine workarounds. Keep entries short; move detail to `README.md`, `test/gtest/README.md`, or `docs/`.
- If you find this file wrong while doing other work, tell the user and propose the fix; edit it only when asked or when your change already alters the documented behavior.
- Changes are reviewed by `@ROCm/miopen-reviewers` (`../../.github/CODEOWNERS`); run pre-commit on the file.
