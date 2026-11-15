# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is RPP

ROCm Performance Primitives — a tensor based vision operations library for images (2D), voxels (3D), and audio, with HOST (CPU/AVX2) and HIP (AMD GPU) backends. The library exposes a C API (`extern "C"`) with opaque handles. Current version: 3.2.0.

## Build Commands

```bash
# HIP backend (default)
mkdir build && cd build
cmake ../ -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
sudo make install

# CPU-only backend
cmake ../ -DBACKEND=CPU && make -j$(nproc)

# With audio support (requires FFTS — configure fails if it is not found)
cmake ../ -DRPP_AUDIO_SUPPORT=ON && make -j$(nproc)
```

Key CMake options: `BACKEND=HIP|CPU`, `RPP_AUDIO_SUPPORT=ON`, `CODE_COVERAGE=ON`, `SIMD_ENABLE=1` (default 1), `BUILD_TESTING=ON` (CTest default; gates `utilities/test`), `GPU_TARGETS=gfx908;gfx90a;...` (`AMDGPU_TARGETS` is deprecated; when unset, CMakeLists uses a broad default arch list, so a targeted `GPU_TARGETS` dramatically cuts HIP build time).

`cmake/CppCheck.cmake` defines an `enable_cppcheck()` macro but nothing invokes it — there is **no** `make cppcheck` target in the current build.

RPP is also built as part of [TheRock](https://github.com/ROCm/TheRock) (`ninja -C build rpp+dist`), which is what produces the released packages. Downstream consumption is `find_package(rpp REQUIRED)` + `target_link_libraries(... rpp::rpp)`; `rpp::rpp` transitively propagates HIP includes/libs and `rpp/rpp.h` pulls in the generated `rpp_backend.h` that defines `RPP_BACKEND_HIP`.

Audio support is driven entirely by the external **FFTS** library — there is no in-house FFT fallback today. `find_package(FFTS QUIET)` at `CMakeLists.txt:196`: on success it force-enables `RPP_AUDIO_SUPPORT`, defines `AUDIO_SUPPORT`, and links `FFTS::FFTS`; on failure an explicit `-DRPP_AUDIO_SUPPORT=ON` is a **hard error** (`CMakeLists.txt:203-204`), and a default configure simply drops audio. The CPU `spectrogram` kernel includes `<ffts/ffts.h>` and calls `ffts_init_1d_real()` / `ffts_init_1d()` directly, with no alternate code path. (A CPU in-house FFT and a rocFFT GPU path are proposed in PR #11496, still open as of 2026-11-09 — re-check this section if that lands.)

## Running Tests

Under `BUILD_TESTING` the top-level `CMakeLists.txt` adds two directories: `test/` (a pair of cheap, hermetic executables) and `utilities/test/` (the GoogleTest correctness suite). `utilities/test_suite/` is **not** added to the build — it is installed as data to `$ROCM_PATH/share/rpp/test/` and configured standalone from there.

`test/` holds `test_rpp_backend_macro.cpp` (asserts `RPP_BACKEND_HIP` matches the configured `BACKEND`, a guard that regressed before) and `test_handle_lifecycle.cpp`, which interposes HIP symbols from the executable to fault-inject allocation failures — 5 cases on HOST (`rpp_handle_host_*`, `rpp_handle_null`), 10 on HIP, Linux only. Use these for quick sanity; they do not cover kernel correctness.

`utilities/test/` is the real correctness suite (`rpp_tests`): every op checked against an independent, host-computed reference model across datatypes, layouts, ROIs, and both backends.

```bash
# In-tree, as part of the library build
cmake ../ -DBUILD_TESTING=ON && make -j$(nproc)
ctest -R rpp_handle_host_mismatch -VV        # the cheap test/ cases
cd utilities/test && ctest -j8 --output-on-failure

# Standalone, against an installed RPP
cd utilities/test && mkdir build && cd build
cmake .. -DROCM_PATH=$ROCM_PATH && cmake --build . --parallel

./rpp_tests                                      # every case (~24.6k)
./rpp_tests --gtest_filter='*BrightnessTest*'    # one operator
./rpp_tests --gtest_filter='Image_Color/*'       # one category
./rpp_tests --gtest_filter='*/HIP_*'             # HIP cases only
ctest -R Rotate                                  # CTest test name == GTest suite name == operator
ctest -L Image_Geometric                         # one category
ctest -L voxel                                   # one domain: image, misc, voxel, core
ctest -L quick                                   # tiered CI label (see test_categories.yaml)
```

Case names are structured and greppable: `{Domain}_{Category}/{Op}Test.{Intent}/{Backend}_{DType}_{Layout}_{Roi}_{Size}`, e.g. `Image_Color/BrightnessTest.Correctness/HIP_U8toU8_PKD3_FullRoi_2x36x48`.

Suite details:
- Builds standalone against an installed rpp (`find_package(rpp 3.0.0)`), or in-tree reusing the `rpp` target. Uses a system GoogleTest if `find_package(GTest 1.12)` succeeds, else fetches pinned v1.17.0 (SHA256-checked). Own sources compile `-Wall -Wextra -Werror`.
- The suite includes `<rpp/...>` like an installed consumer, but the in-tree `rpp` target exports the flat `api/` dirs. An in-tree build therefore stages an installed-style `include/rpp/` layout in its binary dir and puts it on the include path `BEFORE` everything else, so a stale `$ROCM_PATH/include/rpp` can never shadow the headers being built.
- CTest registers one test per GTest suite (per operator) by default; `-DRPP_TEST_CTEST_PER_CASE=ON` registers one per case so a segfault or GPU fault takes down one case, not a whole operator.
- A custom console reporter prints one line per suite plus failure detail. `RPP_TEST_PLAIN_OUTPUT=1` restores stock GTest output (**any** value counts as set, including `0`). `--gtest_list_tests` and `--gtest_output=xml:/json:` are unaffected.
- Known kernel defects are skipped, not left red — glob patterns in `src/framework/skip_list.hpp`. Fixing a kernel means deleting its entry in the same change. `RPP_TEST_NO_SKIP_LIST=1` runs them anyway.
- `test_categories.yaml` maps patterns to TheRock's tiered-CI labels (`quick`/`smoke`, `standard`/`pr`, `comprehensive`/`nightly`, `full`). Applied via `shared/ctest/TestCategories.cmake` when `ROCM_LIBRARIES_ROOT` resolves; `rpp_tests` installs to `$ROCM_PATH/bin` (component `test`, RPATH `$ORIGIN/../lib`) alongside a generated `bin/rpp/CTestTestfile.cmake`.

The legacy QA suite under `utilities/test_suite/` (C++ executables + Python runners + `REFERENCE_OUTPUT*/` golden data) is still the source of performance numbers and golden-output comparison, configured from the install tree:

```bash
mkdir rpp-test && cd rpp-test
cmake ${ROCM_PATH}/share/rpp/test/ && make -j$(nproc) && ctest -VV

# Python runners (from utilities/test_suite/HOST/ or HIP/); --qa_mode 1 compares golden output
python3 runImageTests.py --case_list 0 21 --test_type 0 --qa_mode 1
python3 runImageTests.py --case_list 0 2 20 21 37 49 51 54 --single_image 1 --qa_mode 1
python3 runAudioTests.py --qa_mode 1
```

Image runner essentials: cases numbered 0–102 (`--case_start`/`--case_end`, or `--case_list`), `--test_type 0` = unit, `1` = performance, `--profiling YES` is HIP-only. Runs drop timestamped `QA_RESULTS_*` directories next to the runner (`--preserve_output 0` to clean up). Dependencies: Python3, OpenMP (required); OpenCV, libsndfile, NIfTI, Pandas (optional).

For review work, the GoogleTest suite is the one that catches dtype/tail-loop defects the golden QA suite passes over; run both.

## Architecture

### Call Flow

```
rppt_brightness()                              [api/rppt_tensor_color_augmentations.h]
  → rppt_tensor_color_augmentations.cpp        [src/modules/tensor/ — dispatch layer]
    → HOST: brightness_u8_u8_host_tensor()     [src/modules/tensor/cpu/kernel/brightness.cpp]
    → HIP:  hip_exec_brightness_tensor()       [src/modules/tensor/hip/kernel/brightness.cpp]
              → brightness_pkd_hip_tensor<<<>>>()
```

### Backend Selection

Every `rppt_*` entry point takes an explicit `RppBackend executionBackend` argument (`RPP_HOST_BACKEND` / `RPP_HIP_BACKEND`, a plain C enum in `rppdefs.h`). It is a *second* input to dispatch, not a replacement for the handle's own backend — both are consulted:

```cpp
rpp::Handle& handle = rpp::deref(rppHandle);
[[maybe_unused]] RppBackend handleBackend = handle.GetBackend();

if (executionBackend == RppBackend::RPP_HOST_BACKEND) { ... }    // HOST: selector alone
#ifdef GPU_SUPPORT
else if ((handleBackend     == RppBackend::RPP_HIP_BACKEND) &&   // HIP: handle must agree
         (executionBackend  == RppBackend::RPP_HIP_BACKEND)) { ... }
#endif
return RPP_ERROR_INCOMPATIBLE_BACKEND;                           // falls through otherwise
```

(`src/modules/tensor/rppt_tensor_color_augmentations.cpp:49,123` for `rppt_brightness`; the same shape repeats across all 10 `rppt_tensor_*.cpp` dispatch files.) So the HOST path runs on any handle, but a HIP selector against a HOST-created handle returns `RPP_ERROR_INCOMPATIBLE_BACKEND` rather than running — switching the argument alone is not enough. A new dispatch function must keep the `handleBackend` conjunct on its HIP branch and let the function fall through to `RPP_ERROR_INCOMPATIBLE_BACKEND`; dropping it would run HIP kernels against a handle with no stream or GPU scratch memory.

A HIP-built library therefore serves both paths from one binary; the HIP branches are compiled behind `#ifdef GPU_SUPPORT`.

### Single-Image Fast Path

The dispatch layer branches on `srcDescPtr->n == 1 && dstDescPtr->n == 1` into a separate kernel family before the datatype switch:

```
rppt_brightness(..., n == 1) → brightness_u8_u8_host_single_image()   / hip_exec_brightness_single_image()
rppt_brightness(..., n >  1) → brightness_u8_u8_host_tensor()         / hip_exec_brightness_tensor()
```

Only 8 operators have it so far — brightness, blend, flip, resize, crop, box_filter, median_filter, gaussian_filter — declared in both `host_tensor_executors.hpp` and `hip_tensor_executors.hpp` and dispatched from the color, filter, and geometric `rppt_tensor_*.cpp` files. The QA suite exercises it through separate `Tensor_single_image_{host,hip}` binaries selected by `--single_image 1`. **Both paths must stay in sync**: a change to a tensor kernel for one of these 8 ops usually needs the matching single-image change, and vice versa.

### HOST Intra-Image Threading

`get_intra_image_threads(handle, batchSize, imageHeight)` in `src/include/common/cpu/rpp_cpu_common.hpp` decides row-level parallelism for HOST kernels. It returns 1 when `batchSize > 1` (the outer batch loop already parallelizes — never nest) or when `imageHeight < numThreads * RPP_MIN_ROWS_PER_THREAD` (4); otherwise all handle threads. Kernels written for this index rows by `i` (`ptr + i * hStride`) rather than carrying a running pointer, so rows are independent, then apply `#pragma omp parallel for if(intraThreads > 1) num_threads(intraThreads)`.

Consequence for review: any SIMD store that writes past its nominal width (e.g. a 48-byte store touching 52 bytes) becomes a cross-row data race under this scheme, not merely a benign overrun.

### Key Directories

- `api/` — Public C API headers (installed to `$ROCM_PATH/include/rpp/`)
- `src/modules/tensor/rppt_tensor_*.cpp` — 10 dispatch files (one per operation category), validate inputs then switch on backend + datatype
- `src/modules/tensor/cpu/kernel/` — CPU kernel implementations (~90 files), heavy AVX2/SSE SIMD
- `src/modules/tensor/hip/kernel/` — HIP kernel implementations (~87 `.cpp` files, not `.cu`)
- `src/include/common/cpu/` — CPU SIMD helpers (load/store, math, interpolation, RNG)
- `src/include/common/hip/` — HIP device helpers (load/store, math, interpolation, ROI conversion, RNG)
- `src/include/common/rppt_validate.hpp` — Shared dispatch-layer helpers: `get_layout_params()` (channel/stride params per layout), `check_roi_out_of_bounds()`
- `src/include/tensor/host_tensor_executors.hpp` — Declares all HOST kernel function prototypes
- `src/include/tensor/hip_tensor_executors.hpp` — Declares all HIP executor function prototypes
- `test/` — Two hermetic in-tree executables (backend macro guard, handle lifecycle fault injection)
- `utilities/test/` — GoogleTest correctness suite: `src/framework/` harness, `src/reference/` independent golden models (`<op>_ref.hpp`), `src/tests/{core,image,misc,voxel}/` the tests
- `utilities/test_suite/` — Legacy QA suite with C++ test executables, Python runners, and golden reference outputs (installed as data, not built in-tree)

### 10 Operation Categories

arithmetic, audio, bitwise, color, data_exchange, effects, filter, geometric, morphological, statistical — each has a matching `api/rppt_tensor_<category>.h` header and `src/modules/tensor/rppt_tensor_<category>.cpp` dispatch file.

### Key Types (`api/rppdefs.h`)

- `Rpp8u/Rpp16f/Rpp32f/...` — Sized numeric typedefs
- `RpptDataType` — `U8`, `F16`, `F32`, `I8` (and more)
- `RpptLayout` — `NCHW`, `NHWC`, `NCDHW`, `NDHWC`, `NHW`, `NFT`, `NTF`
- `RpptDesc` / `RpptGenericDesc` — Tensor descriptors (dims, strides, layout, datatype)
- `RpptROI` — Per-image ROI (union of LTRB and XYWH formats)
- `rppHandle_t` — Opaque handle managing batch state, threads, GPU stream, scratch memory

### API Function Signature Pattern

Every function takes: source/dest `void*` pointers + descriptor structs, per-image parameter tensors (batch-sized 1D arrays), ROI tensor + ROI type, an RPP handle, and an `RppBackend executionBackend` selector.

### Handle System

`rppCreate()` / `rppDestroy()` manage lifecycle. Host handle stores batch size, thread count, CPU scratch buffers. HIP handle adds `hipStream_t`, GPU scratch memory, allocator callbacks. Internal C++ class `rpp::Handle` in `src/include/common/handle.hpp`, wrapped as opaque `rppHandle_t`.

## Adding a New Primitive

Follow any existing primitive as a template (e.g., `brightness` for a simple example). Files to create/modify:

1. **`api/rppt_tensor_<category>.h`** — Add `extern "C"` function declaration (the four datatype variants `_u8_u8`, `_f16_f16`, `_f32_f32`, `_i8_i8` are a kernel-level concern, not an API-level one)
2. **`src/modules/tensor/rppt_tensor_<category>.cpp`** — Add dispatch function: validate inputs, switch on backend + datatype, call kernel
3. **`src/include/tensor/host_tensor_executors.hpp`** — Declare HOST kernel prototypes (per datatype: `_u8_u8`, `_f16_f16`, `_f32_f32`, `_i8_i8`)
4. **`src/modules/tensor/cpu/kernel/<name>.cpp`** — Implement HOST kernel
5. **`src/include/tensor/hip_tensor_executors.hpp`** — Declare HIP executor prototypes
6. **`src/modules/tensor/hip/kernel/<name>.cpp`** — Implement HIP kernel (`__global__` kernels + `hip_exec_*` launcher)
7. **`utilities/test/src/reference/<name>_ref.hpp`** — Independent host-computed golden model for the op
8. **`utilities/test/src/tests/<domain>/<category>/`** — A short `TEST_P` exercising it across the config grid
9. **Legacy QA (optional)** — Add cases in `utilities/test_suite/` test executables, update Python runners, add `REFERENCE_OUTPUT*/` data

## Naming Conventions

- Public API: `rppt_<name>()` (C linkage, snake_case)
- HOST kernels: `<name>_<srctype>_<dsttype>_host_tensor()` / `_host_single_image()`
- HIP device kernels: `<name>_pkd_hip_tensor()` (packed/NHWC), `<name>_pln_hip_tensor()` (planar/NCHW)
- HIP launchers: `hip_exec_<name>()` / `hip_exec_<name>_single_image()`
- Types: `Rpp<bits><sign>` (e.g., `Rpp32f`), `Rppt*` for tensor types
- Macros: `RPP_*` prefix

## Conventions

- C++17 required. Compiler: `amdclang++` with `-Wall -fPIC -mavx2 -mf16c -mfma -ffast-math`
- MIT license header block on every source file
- Header guards use `#ifndef` (not `#pragma once`)
- Formatting is inherited from the monorepo root: `.clang-format` (Google base, 4-space indent, 100-col, left-aligned pointers) plus the `clang-format -i -style=file` pre-commit hook in `.pre-commit-config.yaml` (v18.1.4, over `.c/.cpp/.cc/.h/.hpp/...`). rpp is *not* in that hook's exclude list, so touched sources get reformatted — run `pre-commit run --files <changed files>` before pushing. Note older rpp sources predate the hook and are not uniformly formatted; the hook only reformats files a commit already touches, so expect formatting-only noise in the diff when editing one
- HIP kernels are `.cpp` files (not `.cu`), compiled as HIP C++
- Specific files have FMA disabled for numerical precision — audio kernels, statistical reductions, `normalize`, `slice`, `threshold`. The list is `add_fma_property()` in `src/modules/CMakeLists.txt` (`-mno-fma` on x86_64, `-ffp-contract=off` elsewhere). Adding a kernel whose golden references were generated without contraction means adding it to that list.
- HIP kernels process 8 elements per thread (`id_x * 8`), with `id_y` = row, `id_z` = batch index
- CPU kernels use vectorized batches (48/24/16/8 elements) with inline SIMD compute functions
