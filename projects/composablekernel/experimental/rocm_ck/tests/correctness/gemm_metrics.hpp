// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// GPU-computed GEMM verification metrics.
//
// All reductions run on device (data is already there).  Accumulation
// is in FP64 for precision.  Two-phase reduction: per-block partials
// on GPU, finalization on host.
//
// Metrics computed:
//   rel_frob  — ||C-R||_F / ||R||_F       (relative Frobenius norm)
//   rel_inf   — ||C-R||_∞ / ||R||_∞       (relative infinity norm)
//   max_ulp   — max ULP distance
//   rms_ulp   — sqrt(1/n Σ ULP²)          (RMS ULP distance)

#pragma once

#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rocm_ck::test {

// ============================================================================
// Result struct
// ============================================================================

struct GemmMetrics
{
    double rel_frob;    // ||C-R||_F / ||R||_F
    double rel_inf;     // ||C-R||_∞ / ||R||_∞
    int64_t max_ulp;    // max ULP distance
    double rms_ulp;     // sqrt(1/n Σ ULP²)
    int count;          // total elements
};

/// Print a formatted metrics report.
inline void printGemmMetrics(const char* label, const GemmMetrics& m)
{
    std::printf("=== %s  (%d elements) ===\n", label, m.count);
    std::printf("  rel Frobenius (E_F):   %e\n", m.rel_frob);
    std::printf("  rel inf norm  (E_inf): %e\n", m.rel_inf);
    std::printf("  max ULP:               %ld\n", static_cast<long>(m.max_ulp));
    std::printf("  RMS ULP:               %.2f\n", m.rms_ulp);
}

// ============================================================================
// Device helpers — per-element computations
// ============================================================================

namespace detail {

// ============================================================================
// ULP (Unit in the Last Place) distance
// ============================================================================

/// Compute ULP distance between two FP32 values.
///
/// IEEE 754 sign-magnitude floats are lexicographically ordered when
/// reinterpreted as integers (after mapping negatives into a linear
/// order).  The ULP distance is the number of representable floats
/// between a and b.
///
/// Special cases: if either value is NaN, returns INT64_MAX.
__device__ inline int64_t ulpDistance(float a, float b)
{
    if(isnan(a) || isnan(b))
        return INT64_MAX;

    // Reinterpret bits as signed 32-bit integers
    int32_t ia, ib;
    memcpy(&ia, &a, sizeof(float));
    memcpy(&ib, &b, sizeof(float));

    // Convert sign-magnitude to two's complement ordering.
    // Negative floats: 0x80000000 maps to 0, 0xFFFFFFFF maps to -0x7FFFFFFF
    // This makes the integer representation monotonically increasing
    // from -FLT_MAX to +FLT_MAX.
    if(ia < 0)
        ia = static_cast<int32_t>(0x80000000u) - ia;
    if(ib < 0)
        ib = static_cast<int32_t>(0x80000000u) - ib;

    int64_t diff = static_cast<int64_t>(ia) - static_cast<int64_t>(ib);
    return (diff < 0) ? -diff : diff;
}

// ============================================================================
// Partial accumulator
// ============================================================================

/// Per-thread partial accumulator (all FP64 for precision).
struct MetricsPartial
{
    double sum_sq_err;   // Σ (C_i - R_i)²        — Frobenius numerator
    double sum_sq_ref;   // Σ R_i²                 — Frobenius denominator
    double max_abs_err;  // max |C_i - R_i|        — infinity norm numerator
    double max_abs_ref;  // max |R_i|              — infinity norm denominator
    double sum_sq_ulp;   // Σ ULP_i²               — RMS ULP
    int64_t max_ulp;     // max ULP distance
};

/// Combine two partials (associative, for reduction).
__device__ inline MetricsPartial combine(const MetricsPartial& a, const MetricsPartial& b)
{
    return {a.sum_sq_err + b.sum_sq_err,
            a.sum_sq_ref + b.sum_sq_ref,
            fmax(a.max_abs_err, b.max_abs_err),
            fmax(a.max_abs_ref, b.max_abs_ref),
            a.sum_sq_ulp + b.sum_sq_ulp,
            (a.max_ulp > b.max_ulp) ? a.max_ulp : b.max_ulp};
}

/// Compute per-element contribution to metrics.
__device__ inline MetricsPartial elementMetrics(float result, float ref)
{
    double r    = static_cast<double>(result);
    double e    = static_cast<double>(ref);
    double diff = fabs(r - e);
    int64_t ulp = ulpDistance(result, ref);
    double dulp = static_cast<double>(ulp);

    return {diff * diff,      // sum_sq_err
            e * e,            // sum_sq_ref
            diff,             // max_abs_err
            fabs(e),          // max_abs_ref
            dulp * dulp,      // sum_sq_ulp
            ulp};             // max_ulp
}

// ============================================================================
// Block reduction via shared memory
// ============================================================================

constexpr int kMetricsBlock = 256;

/// Reduce a double array in shared memory (sum).
__device__ inline void blockReduceSum(double* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
}

/// Reduce a double array in shared memory (max).
__device__ inline void blockReduceMax(double* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s)
            sdata[tid] = fmax(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
}

/// Reduce an int64_t array in shared memory (max).
__device__ inline void blockReduceMaxInt64(int64_t* sdata, int tid)
{
    __syncthreads();
    for(int s = kMetricsBlock / 2; s > 0; s >>= 1)
    {
        if(tid < s && sdata[tid + s] > sdata[tid])
            sdata[tid] = sdata[tid + s];
        __syncthreads();
    }
}

// ============================================================================
// Reduction kernel
// ============================================================================

__global__ void gemmMetricsKernel(const float* __restrict__ result,
                                  const float* __restrict__ ref,
                                  int count,
                                  MetricsPartial* __restrict__ partials)
{
    int tid       = threadIdx.x;
    int globalTid = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = blockDim.x * gridDim.x;

    // --- Grid-stride accumulation ---
    MetricsPartial local = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
    for(int i = globalTid; i < count; i += gridStride)
    {
        auto em = elementMetrics(result[i], ref[i]);
        local   = combine(local, em);
    }

    // --- Block reduction ---
    // Shared memory is reused for each field.  The union ensures
    // enough space for all types without wasting memory.
    __shared__ union
    {
        double d[kMetricsBlock];
        int64_t i64[kMetricsBlock];
    } smem;

    // sum_sq_err
    smem.d[tid] = local.sum_sq_err;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_sq_err = smem.d[0];

    // sum_sq_ref
    smem.d[tid] = local.sum_sq_ref;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_sq_ref = smem.d[0];

    // max_abs_err
    smem.d[tid] = local.max_abs_err;
    blockReduceMax(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].max_abs_err = smem.d[0];

    // max_abs_ref
    smem.d[tid] = local.max_abs_ref;
    blockReduceMax(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].max_abs_ref = smem.d[0];

    // sum_sq_ulp
    smem.d[tid] = local.sum_sq_ulp;
    blockReduceSum(smem.d, tid);
    if(tid == 0)
        partials[blockIdx.x].sum_sq_ulp = smem.d[0];

    // max_ulp (int64_t)
    smem.i64[tid] = local.max_ulp;
    blockReduceMaxInt64(smem.i64, tid);
    if(tid == 0)
        partials[blockIdx.x].max_ulp = smem.i64[0];
}

} // namespace detail

// ============================================================================
// Host wrapper
// ============================================================================

/// Compute all GEMM metrics on GPU.
/// result and ref are device pointers to float arrays of length count.
inline GemmMetrics computeGemmMetrics(const float* d_result,
                                      const float* d_ref,
                                      int count)
{
    constexpr int kBlock   = detail::kMetricsBlock;
    int numBlocks          = std::min(256, (count + kBlock - 1) / kBlock);

    // Allocate partials on device
    detail::MetricsPartial* d_partials = nullptr;
    HIP_CHECK(hipMalloc(&d_partials, numBlocks * sizeof(detail::MetricsPartial)));

    // Launch reduction
    detail::gemmMetricsKernel<<<numBlocks, kBlock>>>(
        d_result, d_ref, count, d_partials);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // Download partials
    std::vector<detail::MetricsPartial> partials(numBlocks);
    HIP_CHECK(hipMemcpy(partials.data(), d_partials,
                         numBlocks * sizeof(detail::MetricsPartial),
                         hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_partials));

    // Finalize on host
    double sum_sq_err  = 0.0;
    double sum_sq_ref  = 0.0;
    double max_abs_err = 0.0;
    double max_abs_ref = 0.0;
    double sum_sq_ulp  = 0.0;
    int64_t max_ulp    = 0;

    for(int i = 0; i < numBlocks; ++i)
    {
        sum_sq_err += partials[i].sum_sq_err;
        sum_sq_ref += partials[i].sum_sq_ref;
        max_abs_err = std::fmax(max_abs_err, partials[i].max_abs_err);
        max_abs_ref = std::fmax(max_abs_ref, partials[i].max_abs_ref);
        sum_sq_ulp += partials[i].sum_sq_ulp;
        max_ulp = std::max(max_ulp, partials[i].max_ulp);
    }

    double rel_frob = (sum_sq_ref > 0.0)
                          ? std::sqrt(sum_sq_err) / std::sqrt(sum_sq_ref)
                          : 0.0;
    double rel_inf  = (max_abs_ref > 0.0)
                          ? max_abs_err / max_abs_ref
                          : 0.0;
    double rms_ulp  = std::sqrt(sum_sq_ulp / static_cast<double>(count));

    return {rel_frob,
            rel_inf,
            max_ulp,
            rms_ulp,
            count};
}

} // namespace rocm_ck::test
