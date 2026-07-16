// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_arguments.hpp"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <string>

/*!\file
 * \brief GPU-side correctness reference for matmul: a naive HIP reference GEMM
 *        (piece A) plus a device-side comparison reduction (piece B). Used only
 *        for testing/benchmark verification; it is not part of the GPU library.
 *
 * The reference is deliberately simple (one thread per output element, float
 * accumulate) and implemented independently of TensileLite so it can catch
 * kernel bugs. It is an opt-in accelerator for the CPU path in
 * cblas_interface.cpp -- see the `gpu_ref` argument / `--gpu_ref` bench flag.
 */

// atol/rtol candidate grid, matching allclose_check_general() in allclose.hpp so
// the GPU allclose search reports the same effective (atol, rtol).
inline constexpr int    GPU_REF_TOL_GRID_N     = 6;
inline constexpr double GPU_REF_TOL_GRID[GPU_REF_TOL_GRID_N]
    = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};

/// Aggregate result of comparing the GPU output against the device reference,
/// reduced over the valid M x N x batch region (padding rows are skipped).
struct GpuRefResult
{
    double max_abs_error = 0.0; // max |gpu - ref| over finite element pairs
    double sum_ref_sq    = 0.0; // sum(ref^2)  -> Frobenius norm of the reference
    double sum_diff_sq   = 0.0; // sum(diff^2) -> Frobenius norm of the difference
    // allclose_g[k] = max over elements of (|diff| - GPU_REF_TOL_GRID[k]*|gpu|);
    // pair (atol, rtol=GPU_REF_TOL_GRID[k]) passes iff allclose_g[k] <= atol.
    double             allclose_g[GPU_REF_TOL_GRID_N] = {0, 0, 0, 0, 0, 0};
    // finite pairs failing a 4-ULP compare (ASSERT_FLOAT_EQ equivalent), for exact unit_check
    unsigned long long num_unit_fail    = 0;
    unsigned long long num_nan_mismatch = 0; // nan/inf disagreement between gpu and ref
    unsigned long long num_elements     = 0; // element pairs compared

    /// Frobenius relative error ||gpu - ref||_F / ||ref||_F, matching
    /// norm_check_general('F', ...). Returns 0 when both norms are ~0.
    double norm_error() const;
};

/// True when `arg` describes a matmul the GPU reference path currently supports
/// (plain f32/f16 GEMM, default epilogue, no scaling/bias/aux, strided batch).
/// On false, `reason` is filled with the first unsupported feature encountered.
bool gpu_ref_supported(const Arguments& arg, std::string& reason);

/// Piece A: compute D_gold = alpha * op(A) * op(B) + beta * C on the device,
/// accumulating in float. All pointers are device pointers. Column-major, with
/// the same transpose/leading-dim/batch-stride conventions as cblas_gemm().
void run_reference_gemm_device(bool        transA_is_n,
                               bool        transB_is_n,
                               int64_t     M,
                               int64_t     N,
                               int64_t     K,
                               float       alpha,
                               float       beta,
                               const void* dA,
                               hipDataType tA,
                               int64_t     lda,
                               int64_t     strideA,
                               const void* dB,
                               hipDataType tB,
                               int64_t     ldb,
                               int64_t     strideB,
                               const void* dC,
                               hipDataType tC,
                               int64_t     ldc,
                               int64_t     strideC,
                               void*       dDgold,
                               hipDataType tD,
                               int64_t     ldd,
                               int64_t     strideD,
                               int32_t     batchCount,
                               hipStream_t stream);

/// Piece B: compare the GPU output `dGpu` against the reference `dRef` on the
/// device over the valid M x N x batch region and return the reduced result.
GpuRefResult compare_gemm_device(const void* dGpu,
                                 const void* dRef,
                                 hipDataType tD,
                                 int64_t     M,
                                 int64_t     N,
                                 int64_t     ldd,
                                 int64_t     strideD,
                                 int32_t     batchCount,
                                 hipStream_t stream);
