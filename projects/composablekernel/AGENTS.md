# Repository Guidelines

## Project Overview

Composable Kernel (CK) is AMD's HIP/C++ template library providing a
performance-portable, tile-based programming model for ML GPU kernels — GEMM,
convolution, attention/FMHA, normalization, pooling, reduction, MoE,
quantized/microscaling ops — across AMD GPU architectures (gfx908/90a/942/950/
1101/1201/1250...). The core idea ("Tensor Coordinate Transformation") encodes
tensor index math as composable coordinate transforms (`Merge`/`Unmerge`/
`Embed`/`Pad`) so new ops/layouts are built by recombining primitives instead
of hand-writing addressing logic per kernel. CK ships as: a template header
library, a static "instance" library of pre-instantiated kernels, a profiler
CLI, a runtime/JIT codegen layer, and integrations for PyTorch Inductor,
AITER, and Flash-Attention. Part of the `ROCm/rocm-libraries` monorepo at
`projects/composablekernel`.

## Architecture & Data Flow

Two parallel, independently-evolving template-metaprogramming stacks share the
same **kernel-entry → device-op → grid-op → block-op → thread-op** layering:

### 1. Legacy `ck::` (`include/ck/tensor_operation/gpu/`)
- `device/` — abstract op interfaces (`DeviceGemm`, `DeviceGemmMultipleD`, …)
  and concrete implementations under `device/impl/`, e.g.
  `struct DeviceGemmXdl : public DeviceGemm<...>`
  (`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl.hpp:62`).
  Each holds nested `Argument`/`Invoker` structs; `Invoker::Run()` selects and
  launches a grid kernel.
- `grid/` — one `__global__` kernel family per file, e.g.
  `GridwiseGemm_xdl_cshuffle_v1` / `kernel_gemm_xdl_cshuffle_v1(...)`
  (`gridwise_gemm_xdl_cshuffle_v1.hpp`), launched via
  `launch_and_time_kernel()` (`include/ck/host_utility/kernel_launch.hpp`).
- `block/` — per-workgroup primitives (`blockwise_gemm_xdlops.hpp`,
  `*_pipeline_xdlops_v3.hpp`) plus `*_selector.hpp` files that pick a pipeline
  variant via template dispatch.
- `thread/` — per-lane primitives (`threadwise_tensor_slice_transfer_v3r1..v7r3.hpp`,
  `threadwise_gemm_dlops_v3.hpp`) doing register/LDS/global-memory moves and
  MFMA/accumulation.
- `tensor_description/` — the coordinate-transform foundation
  (`multi_index_transform.hpp`, `tensor_descriptor.hpp`, `tensor_adaptor.hpp`)
  used by every layer above for index math.

### 2. Newer `ck_tile::` (`include/ck_tile/`)
- Single generic launch trampoline: `kentry<MinBlockPerCu, Kernel, Args...>`
  (`include/ck_tile/host/kernel_launch.hpp:83-87`) invokes `Kernel{}(args...)`
  on device. Every op supplies a default-constructible `Kernel` struct with
  `operator()`.
- Per-op layout under `ops/<name>/{kernel,pipeline,block,warp}/`, e.g.
  `ops/gemm/kernel/gemm_kernel.hpp` defines `GemmKernel` + `GemmHostArgs`.
- `core/` supplies containers (array/tuple/sequence), numeric types
  (`fp16_t`, `bf16_t`, `fp8_t`), coordinate-transform algorithms, and the
  tile API (`load_tile`/`store_tile`/`shuffle_tile`).
- Dual host/device functions use `CK_TILE_HOST` / `CK_TILE_DEVICE` /
  `CK_TILE_HOST_DEVICE` macros (`include/ck_tile/core/config.hpp:54-64`).
- Prefer including only the specific `ops/<op>.hpp` header needed, not the
  whole library (see `include/ck_tile/README.md`).

### Instance production & dispatch (above the template layers)
- `library/src/tensor_operation_instance/gpu/<op>/` — static enumeration of
  concrete `ck::` template specializations per dtype/layout, one
  `device_<op>_<backend>_<dtypes>_<layout>_instance.cpp` per combination,
  e.g. `device_gemm_xdl_f32_f32_f32_km_kn_mn_instance.cpp` defines
  `std::tuple<DeviceGemmXdl<...>, ...>` and an
  `add_device_gemm_xdl_..._instances(...)` registration function. **This is
  the pattern to copy when adding a new kernel instantiation.**
- `codegen/` — a separate runtime/JIT C++ codegen layer (namespace
  `ck::host::<op>`): `Problem::GetSolutions(arch, prologue, epilogue)` +
  `Operation_Xdl_CShuffle::CreateOperations(...)` factories emit `Solution`
  objects (generated C++ source) instead of static instance files — consumed
  by external integrations.
- `tile_engine/` — the `ck_tile` analogue of the instance library; per-op
  "engine" dirs under `tile_engine/ops/<op>/`. See
  `tile_engine/operation_support_matrix.md` for the authoritative dtype/
  layout/gfx-target support matrix per op.
- `dispatcher/` — runtime C++/Python kernel-selection & launch system built
  on `tile_engine` output; `dispatcher/codegen/` holds the actual Python
  jinja-style generators that emit per-arch instance C++;
  `dispatcher/heuristics/` trains ML kernel-selection models.
- `profiler/` — CLI (`ckProfiler`) with ~78 `profile_<op>.cpp` files for
  benchmarking instances.
- `python/ck4inductor/` — PyTorch Inductor integration; per-op-family
  `op.py` dataclasses (e.g. `CKGemmOperation`) + `gen_instances.py` enumerate
  parameter combos for Inductor autotuning.
- `rocm_ck/` — early-stage constexpr (non-template) description layer for
  multi-arch binary packaging (TheRock), gated by `CK_ENABLE_ROCM_CK`.

## Key Directories

| Path | Purpose |
|---|---|
| `include/ck/` | Legacy template library, namespace `ck::` |
| `include/ck_tile/` | Newer tile-programming framework, namespace `ck_tile::` |
| `library/src/tensor_operation_instance/gpu/<op>/` | Statically instantiated kernels (~90 op dirs) |
| `codegen/` | Runtime/JIT C++ codegen (`ck::host::<op>`) |
| `tile_engine/` | ck_tile kernel-instance engine + support matrix |
| `dispatcher/` | Runtime kernel dispatch/selection (C++ + Python), heuristics |
| `profiler/` | `ckProfiler` CLI benchmarking tool |
| `test/` | GoogleTest unit/functional tests, ~90 per-op subdirs |
| `example/` | Numbered correctness-check programs (`01_gemm/` … `69_gemm_add_relu/`, `ck_tile/`) |
| `client_example/` | Post-install smoke programs against an installed CK package |
| `tutorial/` | Pedagogical ck_tile walkthroughs (not part of CI test labels) |
| `test_data/` | Dataset-generation tooling for conv shape testing |
| `cmake/` | CMake modules (ClangTidy, CppCheck, gtest vendoring, sharding, …) |
| `script/` | Dev/CI tooling (build wrappers, selective-testing, build-trace analysis) |
| `docs/` | Sphinx documentation (Read the Docs) |
| `python/ck4inductor/` | PyTorch Inductor codegen integration |
| `rocm_ck/` | Constexpr multi-arch packaging layer (in progress) |

## Development Commands

Build (standard):
```
mkdir build && cd build
cmake -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -D CMAKE_BUILD_TYPE=Release -D GPU_TARGETS="gfx908;gfx90a" ..
make -j$(nproc)
```
Multi-arch, library-only (no tests/examples): use `-D GPU_ARCHS="gfx908;gfx1030;gfx1100;gfx942"`
instead of `GPU_TARGETS`.

Fast dev-iteration configure (wraps the above, clears cache, `BUILD_DEV=ON`):
```
../script/cmake-ck-dev.sh --minimal          # ~5s configure, examples/tests/profiler disabled
../script/cmake-ck-dev.sh --preset=dev-gfx942
```
Or directly via `CMakePresets.json`: `cmake --preset dev-minimal`, `dev-gfx908/90a/942/950`.

Useful CMake options: `DTYPES` (subset of `int8,fp8,bf8,fp16,fp32,tf32,fp64,bf16`;
unset = all), `DISABLE_DL_KERNELS`, `DISABLE_DPP_KERNELS`,
`CK_USE_FP8_ON_UNSUPPORTED_ARCH`, `BUILD_CK_DEVICE_INSTANCES`,
`BUILD_CK_PROFILER`, `CK_TILE_DISPATCHER`, `CK_ENABLE_ROCM_CK`.

Build & test:
```
make -j examples tests     # build only
make -j check               # build + run everything (ctest --output-on-failure)
make -j smoke                # ctest -L SMOKE_TEST  (fast tests)
make -j regression           # ctest -L REGRESSION_TEST (>60s tests)
```
Run a single test binary: `ctest -R test_gemm_add` or run the built executable directly
(e.g. `./build/bin/test_gemm_add`).

Selective/CI test targeting (maps a diff to affected tests):
```
script/dependency-parser/main.py select --ctest-only ...   # emits tests_to_run.json
script/launch_tests.sh                                      # consumes it, chunks ctest
```

Profiler (benchmark an instance):
```
./bin/ckProfiler gemm_universal 1 0 1 1 0 1 4096 4096 4096 ...
# op, dtype-enum, layout-enum, verify, init, print, time, M N K, strides, splitK, warmup, iters
```

Dispatcher subproject (separate build):
```
cd dispatcher && mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
         -DGPU_TARGETS=gfx942 -DBUILD_DISPATCHER_EXAMPLES=ON
make -j$(nproc) && make python_libs
```

Docs:
```
cd docs && pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Lint/format (pre-commit is the source of truth — `.pre-commit-config.yaml`):
```
script/install_precommit.sh     # one-time setup (venv + pre-commit install)
pre-commit run --all-files       # clang-format 18.1.3, ruff (check+format), copyright/ASCII/CRLF checks
```
clang-tidy / cppcheck are wired into CMake (`cmake/ClangTidy.cmake`,
`cmake/CppCheck.cmake`); build the `tidy` target after configuring with
`ENABLE_CLANG_CPP_CHECKS=ON` (default).

## Code Conventions & Common Patterns

- **Style**: `.clang-format` — 100-col limit, 4-space indent, left-aligned
  pointers, `AlwaysBreakTemplateDeclarations`. Pinned clang-format `18.1.3`
  everywhere (pre-commit, Docker, CI) — do not reformat with a different
  version.
- **Warnings-as-errors in dev builds**: `BUILD_DEV=ON` adds `-Werror
  -Weverything` (Clang) with curated `-Wno-*` exceptions
  (`cmake/EnableCompilerWarnings.cmake`) — new code must be clean under this.
- **Naming**: layered by role, not by casing convention alone —
  `Device<Op><Backend>` (device-op, e.g. `DeviceGemmXdl`), `Gridwise<Op>_*`
  (grid kernel struct), `kernel_<op>_*` (the `__global__` function),
  `Blockwise<Op>*`/`*_selector.hpp` (block layer), `Threadwise*` (thread
  layer). Instance files: `device_<op>_<backend>_<dtypes>_<layout>_instance.cpp`;
  registration function `add_device_<op>_..._instances(...)`.
- **Variable/dtype acronyms** (see `ACRONYMS.md`): `M,N,K` = GEMM dims;
  `Q,K,V` = attention query/key/value; `B,H,D,S,T` = batch/heads/head-dim/
  seq/time. `XDL`=MFMA-based matrix instructions (CDNA), `WMMA`=RDNA matrix
  instructions, `DL`=fallback non-matrix-instruction kernels,
  `DPP`=data-parallel-primitives kernels. Read `TERMINOLOGY.md` before editing
  kernel templates — it defines Pipeline (Problem+Policy), Tile Partitioner,
  Descriptor, Tile Window, vanilla/batched/grouped/split-K GEMM.
- **Host/device dual-compile macros**: `ck_tile` code uses
  `CK_TILE_HOST`/`CK_TILE_DEVICE`/`CK_TILE_HOST_DEVICE`
  (`include/ck_tile/core/config.hpp`); legacy `ck::` code uses plain
  `__host__ __device__` directly.
- **Kernel launch**: never call a `__global__` kernel directly from host
  code outside `launch_and_time_kernel()` (`ck::`) or `kentry<...>`
  (`ck_tile::`) — these centralize timing/warm-up/stream handling.
- **Validation utilities** (`include/ck/library/utility/README.md`): prefer
  `gpu_verify()` (`gpu_verification.hpp`, 10-100x faster) over legacy
  `check_err()`; use `Tensor<T>`/`HostTensorDescriptor` (`host_tensor.hpp`)
  and `DeviceMem` (`device_memory.hpp`) for host/device buffer management.
  Tolerances are dtype-specific (fp32 rtol 1e-5 … fp4 rtol 0.5).
- **Adding a new kernel instance**: copy an existing
  `library/src/tensor_operation_instance/gpu/<op>/device_..._instance.cpp`,
  adjust the dtype/layout/tile-size template-parameter table (kept
  `clang-format off` for column alignment), and register it in that op's
  `CMakeLists.txt` via `add_instance_library(...)`
  (`library/src/tensor_operation_instance/gpu/CMakeLists.txt:9`).
- **Copyright headers**: enforced/auto-fixed by
  `script/update_amd_copyright_headers.py` and pre-commit hooks — run it
  after adding new source files.

## Important Files

- `include/ck/ck.hpp` — legacy library root aggregator header.
- `include/ck/host_utility/kernel_launch.hpp` — `launch_and_time_kernel()`.
- `include/ck_tile/host/kernel_launch.hpp` — `kentry<...>` generic launch trampoline.
- `include/ck_tile/core/config.hpp` — host/device macros, per-arch flags.
- `CMakeLists.txt` (root) — all build options, feature-macro derivation, `check`/`smoke`/`regression` targets.
- `CMakePresets.json` — `dev`, `dev-minimal`, `dev-gfx{908,90a,942,950}` presets.
- `test/CMakeLists.txt` — `add_gtest_executable`/`add_test_executable` helpers, `REGRESSION_TESTS`/`KNOWN_FAILING_TESTS` lists.
- `.clang-format` / `.clang-tidy` / `.pre-commit-config.yaml` — style/lint enforcement.
- `TERMINOLOGY.md`, `ACRONYMS.md` — required vocabulary before reading kernel code.
- `tile_engine/operation_support_matrix.md` — per-op dtype/layout/gfx support matrix.
- `dispatcher/README.md`, `dispatcher/STREAMK.md`, `dispatcher/ADDING_NEW_GPU.md` — dispatcher subsystem docs.
- `docs/Contributors_Guide.rst` — PR rules (branch from `develop`, ~1000-line PR cap, perf numbers for perf-affecting changes).

## Runtime/Tooling Preferences

- **Toolchain**: ROCm/HIP required; compiler is `/opt/rocm/bin/hipcc` or
  `/opt/rocm/llvm/bin/clang++`, CMake ≥ 3.21, C++17 or C++20
  (`CK_CXX_STANDARD`, default 20). `find_package(hip)`, OpenMP required.
- **GPU targeting**: use `GPU_TARGETS` for similar architectures (enables
  tests/examples); use `GPU_ARCHS` for a broad/dissimilar multi-arch
  library-only build (disables tests/examples, unsets `GPU_TARGETS`).
- **Build acceleration**: `sccache`/`ccache` + `ninja` are standard in CI
  Docker images (`Dockerfile`); `script/monitor_sccache_during_build.sh` and
  `script/analyze_build/` (Clang `-ftime-trace` profiling) help diagnose slow
  builds.
- **Python**: package manager is `pip` with per-purpose requirement files —
  `requirements.txt` (root perf/build tooling), `requirements-aiter.in/.txt`
  (AITER Docker image, hash-pinned via `pip-compile`), `dev-requirements.txt`
  (rbuild tool deps), `dispatcher/requirements-ml.txt` (heuristics ML deps).
  Python lint/format is `ruff` (via pre-commit), not black/flake8.
- **Docker** (six variants, each purpose-specific):
  `Dockerfile` (base CI image, ROCm+sccache+ninja+cppcheck+clang-format),
  `Dockerfile.compiler` (optional from-source LLVM/clang build),
  `Dockerfile.manylinux` (RPM-based wheel builder),
  `Dockerfile.aiter` / `Dockerfile.fa` (splice CK into AITER / flash-attention
  for integration testing), `Dockerfile.pytorch` (validate CK against
  PyTorch's vendored copy).
- **CI**: Jenkins (`Jenkinsfile` + `groovy/vars/ck.groovy` shared lib) with a
  "smart build" mode — parses `compile_commands.json` to select only tests
  affected by a diff (5h → 30min); nightly cron runs full matrix across
  gfx908/90a/942/950/101/103/11/12/1250.

## Testing & QA

- **Framework**: GoogleTest, vendored via `cmake/gtest.cmake`
  (`FetchContent`, pinned commit). All `test/*/CMakeLists.txt` register
  binaries through `add_gtest_executable(...)` (gtest-linked) or
  `add_test_executable(...)` (plain HIP executables, used by most of
  `example/`).
- **Test classification**: every registered executable gets CTest label
  `SMOKE_TEST` or `REGRESSION_TEST`. `REGRESSION_TESTS`/`KNOWN_FAILING_TESTS`
  are explicit name lists at the top of `test/CMakeLists.txt` — add a new
  slow (>60s) test's target name there if it belongs in regression.
- **Canonical test pattern** (`test/gemm_add/test_gemm_add.cpp`): a templated
  gtest fixture (`class TestGemmAdd<Tuple> : public TestGemmD0Common<Tuple>`)
  overriding `GetImpl()` to call `ck::profiler::profile_<op>_impl<...>`,
  instantiated via `::testing::Types<std::tuple<dtype/layout combos>>` and
  driven with `TYPED_TEST_SUITE`/`TYPED_TEST`. Follow this shape (fixture +
  `_ut_cases.inc` type-list + per-backend `.cpp`) when adding a test for a
  new op/variant.
- **example/ vs test/ vs client_example/**: `example/` are numbered,
  standalone correctness-checking programs (also CTest-registered, part of
  smoke/regression); `client_example/` links against an *installed* CK
  package (`find_package(composable_kernel ...)`) to smoke-test the public
  API/install tree — built/run manually, not part of CTest; `tutorial/` is
  purely instructional and not part of any test label.
- **Running tests**: `ctest -R <name>` for one test, `make -j smoke` /
  `make -j regression` / `make -j check` for the standard tiers. In CI,
  prefer the selective `script/dependency-parser` + `script/launch_tests.sh`
  path over a full `ctest` run.
- **dispatcher/tests/** is an independent CTest tree mixing C++ gtest
  (`test_dispatcher.cpp`, `test_kernel_key.cpp`) and Python tests run via
  `python3 -m unittest`/`pytest`, labeled `dispatcher;python;bridge;gpu`
  (self-skip without a built lib/GPU).
- **Datasets**: `test_data/generate_test_dataset.sh` + `miopen_to_csv.py`
  produce real-world conv shapes consumed by dataset-driven tests via
  `test/common/csv_test_loader.hpp`.
- **Before submitting a PR** (`docs/Contributors_Guide.rst`): branch from
  `develop`; run existing tests and add new ones for uncovered behavior;
  report before/after perf numbers for build/run-time-affecting changes;
  keep PRs ≈1000 lines or split them; run pre-commit hooks.
