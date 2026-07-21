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
//
// Two notes:
//  (a) Serial-float accumulation diverges from the library reduction order by more
//      than 4 ULP at large K (~71 ULP at K=16384 f32), so the exact (tol==0)
//      unit_check is only meaningful at small K.
//  (b) Matching non-finite values (matching same-signed inf, and both-nan pairs) count
//      as agreement uniformly across the unit/near/norm paths, matching the CPU
//      unit_check and near_check. This differs from CPU norm_check only, whose
//      inf-inf / nan arithmetic yields nan and fails on matching inf/nan -- an
//      intentional, more lenient but correct choice, since both references agreeing on
//      a non-finite value is genuine agreement.
inline constexpr int    GPU_REF_TOL_GRID_N = 6;
inline constexpr double GPU_REF_TOL_GRID[GPU_REF_TOL_GRID_N]
    = {1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1};

/// Reduction of a GPU output vs a reference over the valid M x N x batch region
/// (leading-dim padding skipped).
struct GpuRefResult
{
    double max_abs_error = 0.0; // max |gpu - ref| over finite element pairs
    // Batched Frobenius relative error Sum_b ||diff_b||_F / ||ref_b||_F, computed
    // per batch on the host from the device bins and summed (matches the CPU
    // norm_check_general strided branch in norm.hpp). 0-guard applied per batch.
    double norm_error_sum = 0.0;
    // allclose_g[k] = max over elements of (|diff| - GPU_REF_TOL_GRID[k]*|gpu|);
    // pair (atol, rtol=GPU_REF_TOL_GRID[k]) passes iff allclose_g[k] <= atol.
    double             allclose_g[GPU_REF_TOL_GRID_N] = {0, 0, 0, 0, 0, 0};
    double             max_ulp          = 0.0; // max per-element ULP error over finite pairs
    double             sum_ulp          = 0.0; // sum of per-element ULP error (for the average)
    unsigned long long num_unit_fail    = 0; // finite pairs failing a 4-ULP compare
    unsigned long long num_nan_mismatch = 0; // nan/inf disagreement between gpu and ref
    unsigned long long ulp_count        = 0; // finite pairs contributing to sum_ulp
    bool               valid = false; // false if a HIP error aborted the comparison

    /// Batched Frobenius relative error Sum_b ||diff_b||_F / ||ref_b||_F; each
    /// batch contributes 0 when both its norms are ~0.
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
