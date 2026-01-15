# GitHub Copilot Instructions for rocBLAS

> **Help Improve These Rules:** If you notice these instructions don't match the codebase or could be clearer, suggest updates in your responses.

## Project Overview

rocBLAS is the AMD ROCm Basic Linear Algebra Subprograms (BLAS) library, implemented in HIP and optimized for AMD GPUs. It provides highly optimized implementations of standard BLAS operations (Level 1, 2, and 3) with support for multiple data types and batched operations.

## Code Style and Conventions

### Naming
- Use `snake_case` for functions and variables (e.g., `rocblas_gemm`, `matrix_size`)
- Use `SCREAMING_SNAKE_CASE` for macros and constants
- Follow BLAS naming: `rocblas_<precision><operation>` (e.g., `rocblas_sgemm`, `rocblas_daxpy`)
- Precision prefixes: s=float, d=double, c=complex float, z=complex double, h=half, bf16=bfloat16

### File Headers
Always include the MIT license header, replacing the end year with the current year:
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

Use `#pragma once` for header include guards.

### Code Practices
- Use explicit type conversions with `static_cast<>`
- Always use braces for control flow statements
- Check return codes from rocBLAS API calls
- **Never use hipMalloc/hipFree** - use `handle->device_malloc()` for device memory (see Device Memory section below)
- Use other HIP functions as needed (hipMemcpy, hipStreamSynchronize, etc.)
- Follow const-correctness patterns

### Device Memory Allocation
**Critical:** rocBLAS code must NOT call `hipMalloc()` or `hipFree()` directly because they are synchronizing APIs.

Use the rocBLAS device memory manager instead:
```cpp
// Allocate device memory using RAII wrapper
auto w_mem = handle->device_malloc(dev_bytes);
if(!w_mem)
    return rocblas_status_memory_error;

// Use the memory
void* workspace = static_cast<void*>(w_mem);
// Memory automatically freed when w_mem goes out of scope
```

Key points:
- All device memory must be allocated upfront at the function level
- Use RAII pattern with `rocblas_device_malloc` for automatic cleanup
- Lower-level kernels receive pre-allocated memory from higher-level routines
- Use variable names like `w_mem`, `workspace`, or `w_` prefix for clarity

## Project Structure

- `library/` - Core rocBLAS implementation
  - `library/src/blas1/` - Vector-vector operations (O(n))
  - `library/src/blas2/` - Matrix-vector operations (O(n²))
  - `library/src/blas3/` - Matrix-matrix operations (O(n³)), includes Tensile
  - `library/src/blas_ex/` - Extended precision and batched operations
- `clients/` - Testing and benchmarking
  - `clients/gtest/` - Google Test test suite
  - `clients/benchmarks/` - Performance benchmarking tools
  - `clients/samples/` - Example programs
- `docs/` - Sphinx documentation
- `scripts/` - Utility scripts for performance testing

## Building and Testing

### Build Commands (Linux)
```bash
./install.sh -d              # Install dependencies
./install.sh -c              # Build with clients
./install.sh -c --architecture auto  # Build for detected GPU only (faster)
```

### Build Commands (Windows)
```powershell
$env:HIP_PATH = "C:\Program Files\AMD\ROCm\<version>"  # Set HIP path
python rmake.py -d           # Install dependencies
python rmake.py -c           # Build with clients
```

### Testing
```bash
python3 rtest.py -t smoke    # Quick smoke tests (5-10 min)
python3 rtest.py -t psdb     # Pre-submit tests (30-60 min)
python3 rtest.py -t osdb     # Nightly tests (1.5-2 hrs)

# Direct test execution
./build/release/clients/staging/rocblas-test --gtest_filter=*gemm*
./build/release/clients/staging/rocblas-bench -f gemm -r f32_r -m 4096
```

## API Patterns

### Standard BLAS API
```cpp
rocblas_status rocblas_<precision><operation>(
    rocblas_handle handle,
    // operation-specific parameters
    const <type>* input_ptr,
    <type>* output_ptr);
```

### Implementation Pattern (with device memory)
```cpp
// _impl template: handles error checking, logging, memory allocation
template <typename API_INT, typename T>
rocblas_status rocblas_<operation>_impl(rocblas_handle handle, ...)
{
    // Argument error checking
    if(!handle) return rocblas_status_invalid_handle;
    
    // Logging
    log_trace(handle, ...);
    
    // Allocate device memory
    auto w_mem = handle->device_malloc(dev_bytes);
    if(!w_mem)
        return rocblas_status_memory_error;
    
    // Call launcher
    return rocblas_<operation>_launcher<API_INT, T>(..., w_mem);
}

// _launcher template: fast computation, no checking/allocation
template <typename API_INT, typename T>
rocblas_status rocblas_<operation>_launcher(..., void* workspace)
{
    // Launch kernels using pre-allocated workspace
    // No error checking, logging, or memory allocation
}
```

### Example: GEMM
```cpp
rocblas_status rocblas_sgemm(
    rocblas_handle handle,
    rocblas_operation transA,
    rocblas_operation transB,
    rocblas_int m, rocblas_int n, rocblas_int k,
    const float* alpha,
    const float* A, rocblas_int lda,
    const float* B, rocblas_int ldb,
    const float* beta,
    float* C, rocblas_int ldc);
```

## Testing Patterns

**Note:** Tests use `rocblas_gemm<T>` as a test infrastructure wrapper. End users should call precision-specific functions directly: `rocblas_sgemm`, `rocblas_dgemm`, `rocblas_cgemm`, `rocblas_zgemm`, etc.

### Test Structure
```cpp
template <typename T>
void testing_gemm(const Arguments& arg)
{
    // Test wrapper that dispatches to precision-specific APIs
    auto rocblas_gemm_fn = rocblas_gemm<T>;
    
    rocblas_local_handle handle{arg};
    
    // Allocate memory using test infrastructure
    DEVICE_MEMCHECK(device_matrix<T>, dA, (m, k, lda));
    DEVICE_MEMCHECK(device_matrix<T>, dB, (k, n, ldb));
    DEVICE_MEMCHECK(device_matrix<T>, dC, (m, n, ldc));
    
    // Execute via test wrapper
    DAPI_CHECK(rocblas_gemm_fn, 
               (handle, transA, transB, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
}

TEST(gemm_gtest, float)
{
    Arguments arg;
    arg.M = 128; arg.N = 128; arg.K = 128;
    testing_gemm<float>(arg);
}
```

### Test Categories
- `*quick*` - Fast tests for rapid iteration
- `*pre_checkin*` - PR validation tests
- `*nightly*` - Comprehensive regression tests
- `*known_bug*` - Known issues (excluded)

## Important Notes

- **Tensile Integration:** GEMM operations use Tensile-generated kernels (build-time generation)
- **Architecture Flag:** Use `--architecture auto` for faster local builds
- **Multi-Platform:** Support Linux (primary) and Windows
- **Data Types:** Support float, double, complex, half, bfloat16, int8, int32
- **Batched Operations:** Many operations support batched and strided batched variants

## Environment Variables

- `ROCBLAS_LAYER=1` - Enable verbose logging
- `ROCBLAS_CHECK_NUMERICS=4` - Enable numerical checks
- `ROCBLAS_LAYER=4` - Enable profiling
- `HIP_PATH` - HIP SDK location (Windows)

## Common Operations

### BLAS Level 1 (Vector-Vector)
- `axpy` - y = alpha*x + y
- `dot` - dot product
- `scal` - x = alpha*x
- `nrm2` - Euclidean norm

### BLAS Level 2 (Matrix-Vector)
- `gemv` - matrix-vector multiply
- `ger` - outer product
- `trmv` - triangular matrix-vector multiply

### BLAS Level 3 (Matrix-Matrix)
- `gemm` - general matrix multiply (most performance-critical)
- `trmm` - triangular matrix multiply
- `symm` - symmetric matrix multiply
- `syrk` - symmetric rank-k update

## When Suggesting Code

1. Always include proper copyright headers
2. Follow snake_case naming conventions
3. Use appropriate BLAS naming (precision prefix + operation)
4. Include error checking for rocBLAS API calls
5. Use const-correctness for parameters
6. **Use handle->device_malloc() for device memory - NEVER hipMalloc/hipFree**
7. Follow _impl/_launcher pattern for functions needing device memory
8. Add tests in `clients/gtest/` for new functionality
9. Consider all supported data types (s, d, c, z, h, bf16)
10. Document any new APIs in `docs/reference/`

