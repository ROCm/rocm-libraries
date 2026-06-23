# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Composable Kernel (CK) is a high-performance GPU kernel library for ML workloads on AMD GPUs using HIP C++. It has two programming paradigms:
- **Classic CK** (`include/ck/`) — original template-based approach
- **CK Tile** (`include/ck_tile/`) — newer tile-based programming model

The library is structured in four layers: Templated Tile Operators → Templated Kernel & Invoker → Instantiated Kernel & Invoker → Client API.

## Build Commands

CK must be compiled with `hipcc` or ROCm clang++. `GPU_TARGETS` is required for building tests/examples.

### Quick Start (inside Docker container with ROCm)

```bash
mkdir build && cd build

# Option A: convenience script (cleans cmake cache, sets dev defaults)
../script/cmake-ck-dev.sh                        # default: gfx908;gfx90a;gfx942
../script/cmake-ck-dev.sh .. gfx90a              # single GPU target
../script/cmake-ck-dev.sh --minimal .. gfx90a    # fast configure (~5s vs ~150s)

# Option B: cmake presets
cmake --preset dev                                # default dev preset
cmake --preset dev-minimal                        # skip instances/profiler/examples/tests
cmake --preset dev-gfx942                         # single-arch variant

# Option C: manual cmake
cmake -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc \
      -DCMAKE_BUILD_TYPE=Release \
      -DGPU_TARGETS="gfx908;gfx90a" ..
```

### Build Targets

```bash
make -j$(nproc)              # full build (expect ~2GB RAM per thread)
make -j examples tests       # only examples and tests
make -j ckProfiler           # profiler tool
make -j install              # install to prefix
```

### Accelerating Builds

- **sccache**: `sccache --start-server` then add `-DCMAKE_HIP_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache`
- **DTYPES filter**: `-DDTYPES="fp16;fp32"` to skip unneeded data types
- **DISABLE_DL_KERNELS=ON**: skip DL-specific kernels (only useful on NAVI2x)

## Running Tests

Tests use CTest. Categorized by runtime:
- **Smoke tests** (< 30s): `ctest -L SMOKE_TEST` or `make smoke`
- **Regression tests** (≥ 30s): `ctest -L REGRESSION_TEST` or `make regression`
- **All tests**: `ctest --output-on-failure` or `make check`

Test binaries are named `test_<op>_<dtype>` (e.g., `test_gemm_fp16`). Run a single test:
```bash
ctest -R test_gemm_fp16 --output-on-failure
```

## ck_dsl_c Engine (python/ck_dsl_c/)

C++20 reimplementation of the Python `ck_dsl` IR → LLVM codegen engine. Builds `libckc_core.a` with a pure C ABI (`include/ckc/`), plus pybind11 Python bindings.

Build with the canonical script:
```bash
python/ck_dsl_c/tools/ckc_build.sh              # full build (engine + provider + demos)
python/ck_dsl_c/tools/ckc_build.sh --no-demos   # engine + provider only
python/ck_dsl_c/tools/ckc_build.sh --sanitize   # ASAN+UBSAN build
```

Engine tests are in `python/ck_dsl_c/tests/` (smoke, IR round-trip, parity, differential).

## Code Style and Formatting

### C++ (clang-format v18.1.3)
- 100-char column limit, 4-space indent, no tabs
- Braces on new line for classes/structs/functions/enums/unions, same line for namespaces
- Left-aligned pointers (`int* p`)
- `SortIncludes: false`
- Template declarations always break

### Python (ruff v0.14.0)
- `ruff check --fix` and `ruff format`

### Pre-commit hooks
Install: `sudo script/install_precommit.sh`. Hooks enforce clang-format, ruff, copyright headers, and CK Tile file normalization.

### Copyright header (required on all C++, Python, shell, CMake files)
```
// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
```

## Key CMake Variables

| Variable | Purpose |
|---|---|
| `GPU_TARGETS` | Semicolon-separated GPU arch list (e.g., `gfx908;gfx90a;gfx942`) |
| `GPU_ARCHS` | Alternative for cross-family builds (e.g., `gfx908;gfx1030;gfx1100`) |
| `BUILD_DEV=ON` | Development mode (enables `-Werror -Weverything`) |
| `DTYPES` | Filter data types: `fp64;fp32;tf32;fp16;fp8;bf16;int8` |
| `CK_CXX_STANDARD` | C++ standard (17 or 20, default 20) |
| `BUILD_CK_DEVICE_INSTANCES` | Build library instances (default ON) |
| `BUILD_CK_PROFILER` | Build ckProfiler (default ON) |
| `BUILD_CK_EXAMPLES` | Build examples (default ON) |

## Architecture Notes

- `library/src/tensor_operation_instance/gpu/` contains per-operation instantiation `.cpp` files — these are the compiled GPU kernels that make up the pre-built library
- `profiler/` contains the `ckProfiler` tool for benchmarking operations
- `dispatcher/` provides a C++ and Python frontend for CK Tile GEMM/Conv dispatch
- `codegen/` has kernel code generation tooling
- `python/ck_dsl/` is the Python DSL for constructing CK IR (core, runtime, helpers, instances, analysis)
- `python/ck4inductor/` integrates CK with PyTorch Inductor

## CI

Jenkins-based. PR builds are selective (~30 min), nightly builds are full (~5 hours). Smart build detects which GPU arch to target via `rocminfo`. Force full build with `FORCE_CI=true` or `DISABLE_SMART_BUILD=true`.
