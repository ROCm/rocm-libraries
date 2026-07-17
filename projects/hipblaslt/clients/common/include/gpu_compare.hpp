// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

/*!\file
 * \brief Device-side comparison of a GEMM output against a reference, reduced
 *        entirely on the GPU. Used only for testing/benchmark verification.
 */

// atol/rtol candidate grid, shared with allclose_check_general() in allclose.hpp
// so the GPU allclose search reports the same effective (atol, rtol).
inline constexpr int    GPU_REF_TOL_GRID_N = 6;
inline constexpr double GPU_REF_TOL_GRID[GPU_REF_TOL_GRID_N]
    = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};

/// Reduction of a GPU output vs a reference over the valid M x N x batch region
/// (leading-dim padding skipped).
struct GpuRefResult
{
    double max_abs_error = 0.0; // max |gpu - ref| over finite element pairs
    double sum_ref_sq    = 0.0; // sum(ref^2)  -> Frobenius norm of the reference
    double sum_diff_sq   = 0.0; // sum(diff^2) -> Frobenius norm of the difference
    // allclose_g[k] = max over elements of (|diff| - GPU_REF_TOL_GRID[k]*|gpu|);
    // pair (atol, rtol=GPU_REF_TOL_GRID[k]) passes iff allclose_g[k] <= atol.
    double             allclose_g[GPU_REF_TOL_GRID_N] = {0, 0, 0, 0, 0, 0};
    double             max_ulp          = 0.0; // max per-element ULP error over finite pairs
    double             sum_ulp          = 0.0; // sum of per-element ULP error (for the average)
    unsigned long long num_unit_fail    = 0; // finite pairs failing a 4-ULP compare
    unsigned long long num_nan_mismatch = 0; // nan/inf disagreement between gpu and ref
    unsigned long long ulp_count        = 0; // finite pairs contributing to sum_ulp

    /// Frobenius relative error ||gpu - ref||_F / ||ref||_F; 0 when both norms are ~0.
    double norm_error() const;
    /// Mean per-element ULP error; 0 when no finite pairs were compared.
    double avg_ulp() const;
};

/// Compare the GPU output `dGpu` against the reference `dRef` on the device over
/// the valid M x N x batch region and return the reduced result. `ulpMantBits` is
/// the output type's mantissa width (ulp_mantissa_bits() in ulp.hpp).
GpuRefResult compare_gemm_device(const void* dGpu,
                                 const void* dRef,
                                 hipDataType tD,
                                 int64_t     M,
                                 int64_t     N,
                                 int64_t     ldd,
                                 int64_t     strideD,
                                 int32_t     batchCount,
                                 int         ulpMantBits,
                                 hipStream_t stream);
