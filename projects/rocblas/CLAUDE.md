# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

rocBLAS is AMD's ROCm Basic Linear Algebra Subprograms (BLAS) library, implemented in HIP and optimized for AMD GPUs. It provides Level 1, 2, and 3 BLAS operations plus extensions, with support for multiple data types and batched operations. GEMM operations use Tensile-generated kernels built at compile time.

## Build Commands

```bash
# Linux - install dependencies (once)
./install.sh -d

# Build library + test clients (most common)
./install.sh -c

# Build only for your GPU (much faster, recommended for development)
./install.sh -c -a auto

# Debug build
./install.sh -c -g

# Windows
python rmake.py -d          # Install deps
python rmake.py -c          # Build with clients
python rmake.py -c -a auto  # Build for detected GPU
```

Binaries land in `build/release/clients/staging/`.

## Testing

```bash
# Test levels (defined in rtest.xml)
python3 rtest.py -t smoke    # ~5-10 min, quick sanity
python3 rtest.py -t psdb     # ~30-60 min, PR validation
python3 rtest.py -t osdb     # ~1.5-2 hrs, nightly regression
python3 rtest.py -t cqe      # ~3-3.5 hrs, complete QE

# Direct test execution
./build/release/clients/staging/rocblas-test --gtest_filter=*gemm*
./build/release/clients/staging/rocblas-test --gtest_filter=*quick*-*known_bug*
./build/release/clients/staging/rocblas-test --yaml clients/gtest/rocblas_smoke.yaml

# Benchmarking
./build/release/clients/staging/rocblas-bench -f gemm -r f32_r -m 4096 -n 4096 -k 4096
./build/release/clients/staging/rocblas-bench -f gemv -r f32_r -m 8192 -n 8192
```

Test filter categories: `*quick*`, `*pre_checkin*`, `*nightly*`, `*known_bug*` (excluded).

## Environment Variables

```bash
export ROCBLAS_LAYER=1         # Enable verbose logging
export ROCBLAS_LAYER=4         # Enable profiling
export ROCBLAS_CHECK_NUMERICS=4  # Enable numerical checks (1-4)
```

## Architecture

```
library/
  src/
    blas1/       # Vector-vector ops (axpy, dot, scal, nrm2, copy, ...) — O(n)
    blas2/       # Matrix-vector ops (gemv, ger, trmv, symv, ...) — O(n²)
    blas3/       # Matrix-matrix ops (gemm, trmm, symm, syrk, ...) — O(n³)
      Tensile/   # YAML configs for Tensile-generated GEMM kernels
    blas_ex/     # Extended ops (gemm_ex, gemm_strided_batched_ex, mixed precision)
    src64/       # 64-bit integer API variants
    include/     # Internal headers (handle, logging, device_malloc, ...)
clients/
  gtest/         # Google Test suite + YAML test configs
  benchmarks/    # rocblas-bench performance tool
  samples/       # Example programs
```

Each BLAS operation typically has three source files: `rocblas_<op>.cpp`, `rocblas_<op>_batched.cpp`, `rocblas_<op>_strided_batched.cpp`, plus a shared `rocblas_<op>_kernels.cpp`.

## Naming Conventions

- API functions: `rocblas_<precision><operation>` — e.g., `rocblas_sgemm`, `rocblas_daxpy`
- Precision prefixes: `s`=float, `d`=double, `c`=complex float, `z`=complex double, `h`=half, `bf16`=bfloat16
- Snake_case for all functions and variables; `SCREAMING_SNAKE_CASE` for macros
- `#pragma once` for all headers

## Critical Code Patterns

### Device Memory — Never use hipMalloc/hipFree

All device memory must go through the handle's memory manager (non-synchronizing):

```cpp
auto w_mem = handle->device_malloc(dev_bytes);
if(!w_mem)
    return rocblas_status_memory_error;
void* workspace = static_cast<void*>(w_mem);
// Freed automatically when w_mem goes out of scope
```

### `_impl` / `_launcher` Two-Tier Pattern

```cpp
// _launcher: pure computation — no error checking, logging, or allocation
template <typename API_INT, typename T>
rocblas_status rocblas_<op>_launcher(..., void* workspace) { ... }

// _impl: argument checking, logging, memory allocation, then delegates to launcher
template <typename API_INT, typename T>
rocblas_status rocblas_<op>_impl(rocblas_handle handle, ...) {
    if(!handle) return rocblas_status_invalid_handle;
    log_trace(handle, ...);
    auto w_mem = handle->device_malloc(dev_bytes);
    if(!w_mem) return rocblas_status_memory_error;
    return rocblas_<op>_launcher<API_INT, T>(..., static_cast<void*>(w_mem));
}

// C API wrapper
extern "C" rocblas_status rocblas_s<op>(...) {
    return rocblas_<op>_impl<rocblas_int, float>(handle, ...);
}
```

### Template Parameter Ordering

Non-type parameters before type parameters, so type parameters can be deduced:

```cpp
template <rocblas_int NB, typename T>   // Correct: NB first, T deduced from args
void kernel(const T* data, T* out);

template <typename T, rocblas_int NB>   // Wrong: caller must specify both explicitly
void bad_kernel(const T* data, T* out);
```

### Pointer Mode

rocBLAS scalars (`alpha`, `beta`) can be in host or device memory. Use `load_scalar()` in kernels and `handle->push_pointer_mode()` for RAII mode switching.

## Test Structure

Tests use templated infrastructure wrappers (`rocblas_gemm<T>`) that dispatch to precision-specific APIs. YAML configs in `clients/gtest/` define parameterized test cases. Tests never define `main()`.

```cpp
template <typename T>
void testing_gemm(const Arguments& arg) {
    auto rocblas_gemm_fn = arg.api & c_API_FORTRAN ? rocblas_gemm<T, true> : rocblas_gemm<T, false>;
    rocblas_local_handle handle{arg};
    HOST_MEMCHECK(host_matrix<T>, hA, (m, k, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dA, (m, k, lda));
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    DAPI_CHECK(rocblas_gemm_fn, (handle, transA, transB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
}
```

## File Header

All source files must begin with:

```cpp
/* ************************************************************************
 * Copyright (C) 2016-[CURRENT YEAR] Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */
```

## Constants

Use `static constexpr` — never macros for constants. Use `do { ... } while(0)` in macros that contain control flow.


## Changelog

- Add customer‑facing or important changes to the changelog as you commit component code.
- Add new entries under the "Since last release" heading.
- Use the standard sections: Added, Changed, Removed, Optimized, Resolved issues, Known issues, and Upcoming changes accordingly. See https://amd.atlassian.net/wiki/spaces/MLSE/pages/744167593/ROCm+component+changelogs#Process for more details.
