# Repository Guidelines

## Project Overview

Composable Kernel (CK) is AMD's HIP/C++ library for GPU kernels: GEMM, convolution,
attention, reductions, normalization, and quantized operations. Its central abstraction
is **Tensor Coordinate Transformation**: compose descriptors and coordinate transforms
instead of hand-writing indexing for each layout. CK provides template headers,
precompiled operation instances, profiling tools, and kernel-generation integrations.
It is a component of the `ROCm/rocm-libraries` monorepo.

## Architecture & Data Flow

**Two distinct programming stacks; follow the stack of the code being changed.**

- **Legacy `ck::`**: a device operation builds an `Argument`, checks
  `IsSupportedArgument`, and launches through `Invoker::Run`. Concrete operations in
  `include/ck/tensor_operation/gpu/device/impl/` select grid kernels, which compose
  block/warp/thread primitives. Representative path:
  `DeviceGemm_Xdl_CShuffleV3` → `kernel_gemm_xdl_cshuffle_v3` → gridwise GEMM →
  blockwise pipeline → output epilogue. Descriptors under
  `include/ck/tensor_description/` encode padding, merging, and layout mapping.
- **`ck_tile::`**: kernels compose a **Tile Partitioner + Pipeline + Epilogue**.
  A pipeline's **Problem** carries types/layouts/shapes; its **Policy** chooses
  primitives, vector widths, and LDS allocation. Tile views/windows describe memory
  accesses; distributed tensors describe register ownership across lanes/warps.
  `UniversalGemmKernel` in `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`
  converts host arguments to device arguments and exposes grid/block/support queries.
  `make_kernel`/`launch_kernel` use the `kentry` trampoline to invoke `Kernel{}(...)`.
- **Instances and dispatch**: `library/` materializes legacy template families;
  `DeviceOperationInstanceFactory` selects dtype/layout catalogs. `tile_engine/ops/`
  generates CK Tile configurations and launch/profiling code. `dispatcher/` maps a
  runtime problem to registered compiled kernels, selects a supported candidate
  using first-fit or heuristics, and launches its backend wrapper.
- **Generation is not necessarily JIT**: `codegen/` emits host-side `Solution` source
  descriptions; HIPRTC execution helpers are under `codegen/test/rtc/`.
  `dispatcher/codegen/` generates backend wrappers and architecture metadata.
  `python/ck4inductor/` enumerates operation descriptors for downstream Inductor use.

Coordinate/view transforms change indexing metadata, not tensor payloads. Runtime
arguments carry pointers/dimensions/strides; compile-time types carry kernel structure.

## Key Directories

| Path | Use when changing |
|---|---|
| `include/ck/`, `include/ck_tile/` | Kernel algorithms, descriptors, numeric types, launch primitives |
| `library/src/tensor_operation_instance/gpu/<op>/` | Legacy instance tables and registration |
| `library/include/ck/library/` | Instance factories, host tensors, buffers, reference/verification utilities |
| `tile_engine/ops/`, `dispatcher/` | Tile instance generation, runtime selection, backend integration |
| `codegen/`, `python/ck4inductor/` | Host source generation and downstream operation enumeration |
| `profiler/` | `ckProfiler` operations and reusable `profile_<op>_impl` functions |
| `test/`, `example/ck_tile/`, `example/<number>_<op>/` | Existing fixtures and runnable kernel usage |
| `client_example/`, `tutorial/` | Installed-package checks and instructional programs |
| `cmake/`, `script/`, `test_data/` | Build helpers, selective QA/profiling, convolution datasets |
| `docs/` | Sphinx reference, installation, and contributor documentation |

## Development Commands

Run from the component root unless noted. Replace `gfx942` with the intended GPU;
choose bounded parallelism for available RAM rather than unrestricted `-j`.

```sh
# Discover configure presets; all use ./build by default.
cmake --list-presets
cmake --preset dev-gfx942 -G Ninja
cmake --build build --target test_ck_tile_gemm_pipeline_mem -j 8
ctest --test-dir build -R '^test_ck_tile_gemm_pipeline_mem$' --output-on-failure

# Build-only groups; build-and-run QA groups.
cmake --build build --target examples tests -j 8
cmake --build build --target smoke -j 8
cmake --build build --target regression -j 8
cmake --build build --target check -j 8

# Minimal configuration only: not a kernel/profiler validation build.
cmake --preset dev-minimal -G Ninja -DGPU_TARGETS=gfx942 \
  -DBUILD_TESTING=OFF -DENABLE_CLANG_CPP_CHECKS=OFF
```

Use a separate `-B <build-dir>` when switching configurations. Presets contain only
configure presets; build with `cmake --build <build-dir>`, not a build preset.
`dev-minimal` alone **does not disable tests**. Prefer direct CMake to
`script/cmake-ck-dev.sh`: that wrapper recursively deletes CMake caches below its
working directory and overrides preset GPU targets with its positional/default GPU.

```sh
# Normal dev configuration with instances/profiler enabled:
cmake --build build --target ckProfiler -j 8
./build/bin/ckProfiler gemm_universal 1 0 1 1 0 1 128 128 256 256 128 128 1
# FP16, row/row/row, verify, integer initialization, no print, time;
# M N K, stride-A/B/C, split-K. Optional warmup/iterations/buffer fields go together.

# Independent dispatcher: uses native HIP architecture configuration.
cmake -S dispatcher -B dispatcher/build -G Ninja \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx942 -DBUILD_DISPATCHER_TESTS=ON
cmake --build dispatcher/build -j 8
ctest --test-dir dispatcher/build -L dispatcher --output-on-failure
```

For selective QA, use `script/dependency-parser/README.md`: configure Ninja with
`CMAKE_EXPORT_COMPILE_COMMANDS=ON`, generate a dependency map with `main.py cmake-parse`,
then `main.py select <depmap.json> origin/develop HEAD --ctest-only --build-dir <build>`.
Use the monorepo root as `--workspace-root` during mapping. Build nonempty
`.executables` selections and run nonempty `.regex_chunks`; empty targets/regexes can
accidentally build or run everything. Mapping preprocesses sources and needs the compiler.

```sh
# From MONOREPO root, with pre-commit installed in your environment:
pre-commit run --config projects/composablekernel/.pre-commit-config.yaml \
  --files projects/composablekernel/<changed-file>

# From component root, in an activated Python environment:
(cd docs && python3 -m pip install -r sphinx/requirements.txt && \
  python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html)
```

The monorepo's default hook configuration excludes CK; select the component config
explicitly. Hooks may modify files. Documentation dependencies require network access.

## Code Conventions & Common Patterns

- **Formatting/naming**: four spaces, 100 columns, no tabs, custom next-line braces,
  include sorting disabled. Use pinned **clang-format 18.1.3** and **Ruff 0.14.0**.
  Copy neighboring names: `Device<Op>...`, `Gridwise<Op>...`, `kernel_<op>...`,
  `Blockwise...`, `Threadwise...`; Tile utilities and dispatcher methods often use
  snake_case. Preserve scoped `clang-format off` around aligned instance tables.
- **Compile-time composition**: use traits, policies, elementwise functors,
  `static constexpr`, and the stack's `Number`/`Sequence` or `number`/`sequence` types.
  Enforce static shape/layout invariants with `static_assert`; keep runtime pointers,
  dimensions, and strides in argument structs, not global mutable state.
- **Host/device boundaries**: legacy code uses HIP annotations; Tile code uses
  `CK_TILE_HOST`, `CK_TILE_DEVICE`, `CK_TILE_HOST_DEVICE`. Prefer operation/core umbrella
  headers and narrow host utility includes; do not mix stack-specific types casually.
- **Support/errors**: create arguments through the matching operation/kernel factory;
  preserve architecture, vector-width, padding, layout, and Split-K support checks.
  Unsupported candidates usually return false; invalid launch settings and dispatcher
  selection failures throw host exceptions. Reuse HIP error-check helpers; do not
  suppress errors or introduce device exceptions. Some specialized pipelines require
  additional support checks beyond the generic predicate.
- **Streams/lifetimes**: use the existing launch wrappers and caller's `StreamConfig`.
  Untimed launches enqueue work without completion synchronization and return zero
  timing; zero is neither failure nor proof of completion. Argument objects usually
  borrow device buffers/workspace: keep allocations alive until work completes.
  Timing paths synchronize and must not be substituted for asynchronous inference.
- **State/dependency injection**: template Problem/Policy composition is the kernel
  customization mechanism. Dispatcher accepts registry/heuristic dependencies and
  owns reusable Stream-K workspace; it is explicitly single-thread/single-stream and
  noncopyable/nonmovable. Use separate dispatcher instances for concurrent streams.
- **New instances**: copy a matching `device_<op>_<backend>_<dtypes>_<layout>_instance.cpp`
  and its `add_device_<op>_..._instances` function; update that operation's CMake
  `add_instance_library(...)` registration and corresponding factory where needed.
- **Utilities/headers**: reuse `Tensor<T>`, `HostTensorDescriptor`, `DeviceMem`, reference
  operations, and existing verification helpers rather than new buffer/error machinery.
  New source headers use yearless `Copyright (c) Advanced Micro Devices, Inc., or its
  affiliates.` and `SPDX-License-Identifier: MIT` with language-appropriate comments.

## Important Files

- `CMakeLists.txt`, `CMakePresets.json`: feature/architecture gates and QA targets.
- `include/ck/ck.hpp`, `include/ck_tile/core/config.hpp`: framework configuration.
- `include/ck/host_utility/kernel_launch.hpp`,
  `include/ck_tile/host/kernel_launch.hpp`: stream, timing, and launch semantics.
- `include/ck/tensor_operation/gpu/device/device_base.hpp`: legacy argument/invoker API.
- `test/CMakeLists.txt`, `test/ck_tile/gemm/CMakeLists.txt`: test registration and labels.
- `.clang-format`, `.pre-commit-config.yaml`, `cmake/EnableCompilerWarnings.cmake`:
  enforced style and warnings; analyzer checks are also configured in root CMake.
- `TERMINOLOGY.md`, `include/ck_tile/README.md`: coordinate transforms and Tile vocabulary;
  `ACRONYMS.md` is a glossary, not stronger authority than current backend code.
- `tile_engine/operation_support_matrix.md`: engine-supported/validated configurations,
  not a complete inventory of every implemented kernel.
- `dispatcher/README.md`, `dispatcher/STREAMK.md`,
  `dispatcher/codegen/ADDING_NEW_GPU.md`: dispatcher integration, workspace ABI, GPU ports.
  Update `dispatcher/codegen/arch_specs.json` and regenerate Python/C++ metadata together;
  do not hand-edit generated architecture specifications.
- `profiler/README.md`, `docs/Contributors_Guide.rst`, `CHANGELOG.md`: profiling syntax,
  contribution expectations, and release notes. Follow monorepo PR-quality guidance too.

## Runtime/Tooling Preferences

- ROCm/HIP and ROCm Clang are required for native builds; presets use
  `/opt/rocm/llvm/bin/clang++`. CMake **3.21+**, Python **3.8+**; root
  `CK_CXX_STANDARD` defaults to **20**, accepts **17/20**. Host `codegen/` uses C++20.
- Use explicit `GPU_TARGETS` for development/tests; quote semicolon lists.
  `GPU_ARCHS` selects broad library architecture lists and omits root tests/examples/
  tutorials/tile engine, but **does not itself disable the profiler**.
- Component switches are independent: `BUILD_TESTING`, `BUILD_CK_DEVICE_INSTANCES`,
  `BUILD_CK_PROFILER`, `BUILD_CK_EXAMPLES`, `BUILD_CK_TUTORIALS`, `BUILD_CK_TILE_ENGINE`.
  Disabling instances does not disable their consumers. `DTYPES="fp16;bf16"` narrows
  supported dtype-tagged builds; profiler operation/instance filters are regexes.
- Root `CK_TILE_DISPATCHER=ON` integrates generated convolution objects into the
  profiler; it is not the standalone dispatcher build. That generator uses the first
  `GPU_TARGETS` entry. Standalone dispatcher tests/examples default off.
- `BUILD_DEV` enables strict Clang warnings, including `-Werror`/`-Weverything` with
  curated exceptions. `ENABLE_CLANG_CPP_CHECKS` controls analyzers, not those warnings.
- Prefer Ninja and bounded jobs; sccache/ccache are available in CI workflows.
  Native outputs are under `<build>/bin` and `<build>/lib`.
- Python packaging uses setuptools/pip and import package `ck4inductor`; use a venv
  and purpose-specific requirements (`dispatcher/python/requirements.txt`, optional
  `dispatcher/requirements-ml.txt`, docs, AITER). Root `requirements.txt` is empty;
  `dev-requirements.txt` is not an ordinary pip requirements list. Python packaging
  does not build the native instance library. No Node/Bun runtime is required.

## Testing & QA

- CTest runs executable-level tests; GoogleTest is fetched at a pinned commit through
  `cmake/gtest.cmake` (`GOOGLETEST_DIR` supplies local sources). Reuse
  `add_gtest_executable` or `add_test_executable` in `test/CMakeLists.txt`.
- Discover coverage with `ctest --test-dir build -N` and a binary's
  `--gtest_list_tests`; use `--gtest_filter` for individual cases. GPU tests require
  matching hardware/toolchain; unsupported variants can skip. Skips are not coverage.
- Smoke/regression labels come from explicit lists, not measured runtime guarantees.
  Known-failing entries remain registered but lack those labels. Many CK Tile GEMM
  tests replace global labels with `CK_TILE_GEMM_TESTS`; use the `ck_tile_gemm_tests`
  build-and-run target for that family rather than assuming smoke covers it.
- Copy the nearest fixture: legacy typed suites call profiler implementations;
  Tile GEMM uses `TestCkTileGemmPipeline`, tuple-based kernel type lists, and shared
  `test_gemm_pipeline_ut_cases.inc`. Cover affected dtype/layout/pipeline combinations,
  partial/odd sizes, strides, support rejection, and Split-K behavior. Reuse existing
  reference computations and dtype/K/split-aware numerical thresholds; no global epsilon.
- Registered examples participate in CTest; installed `client_example/` builds test
  package exports separately. Tutorials and convolution dataset binaries are not
  automatically CTest checks. Dataset tests require generated CSVs under `test_data/`.
- Dispatcher has independent C++/Python/bridge suites; unavailable GPU bridges return
  skip code 77. `python/test/` uses unittest for `ck4inductor` descriptors and requires
  packaged instance data; it is not GPU correctness coverage. Host-codegen tests may
  also compile or execute HIP kernels.
- Run relevant existing tests and add behavioral coverage for uncovered functionality.
  Report GPU/ROCm/configuration, skipped cases, and before/after build or runtime
  performance for performance-affecting changes. Update stale docs/release notes;
  major architecture changes need design documentation. Never weaken tests for green CI.
