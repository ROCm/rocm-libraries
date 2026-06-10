// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Split-accumulation GPU reference GEMM implementation.
//
// Tile engine dispatch:
//   - MFMA kernel for aligned sizes (M%128==0, N%128==0, K_tile%16==0)
//   - Naive fallback for non-aligned sizes
//
// CK convention: C[M,N] = sum_k A[M,K] * B[N,K]  (B is [N,K])

#include "split_gemm.hpp"

#include "ck_tile/host/hip_check_error.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

namespace rocm_ck::test {

// =========================================================================
// Workspace
// =========================================================================

SplitGemmWorkspace::SplitGemmWorkspace(int max_M, int max_N, int max_K, int min_kTile)
{
    long MN         = static_cast<long>(max_M) * max_N;
    long MK         = static_cast<long>(max_M) * max_K;
    long KN         = static_cast<long>(max_K) * max_N;
    int num_tiles   = (max_K + min_kTile - 1) / min_kTile;

    partials_bytes = static_cast<size_t>(num_tiles) * MN * sizeof(float);
    HIP_CHECK_ERROR(hipMalloc(&partials, partials_bytes));

    HIP_CHECK_ERROR(hipMalloc(&a_hi, MK * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&a_lo, MK * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&b_hi, KN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&b_lo, KN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&p0, MN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&p1, MN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&p2, MN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&p3, MN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&sum_buf, MN * sizeof(float)));
    HIP_CHECK_ERROR(hipMalloc(&err_buf, MN * sizeof(float)));

    d_A_bytes = MK * sizeof(float);
    d_B_bytes = static_cast<size_t>(std::max(MK, KN)) * sizeof(float);
    d_C_bytes = MN * sizeof(float);
    HIP_CHECK_ERROR(hipMalloc(&d_A, d_A_bytes));
    HIP_CHECK_ERROR(hipMalloc(&d_B, d_B_bytes));
    HIP_CHECK_ERROR(hipMalloc(&d_C, d_C_bytes));
}

SplitGemmWorkspace::~SplitGemmWorkspace()
{
    (void)hipFree(partials);
    (void)hipFree(a_hi);
    (void)hipFree(a_lo);
    (void)hipFree(b_hi);
    (void)hipFree(b_lo);
    (void)hipFree(p0);
    (void)hipFree(p1);
    (void)hipFree(p2);
    (void)hipFree(p3);
    (void)hipFree(sum_buf);
    (void)hipFree(err_buf);
    (void)hipFree(d_A);
    (void)hipFree(d_B);
    (void)hipFree(d_C);
}

// =========================================================================
// Naive tile kernel (fallback for non-aligned sizes)
// =========================================================================

// C[i,j] = sum_{k in tile} A[i,k] * B[j,k]   (B is [N,K])
__global__ void tile_gemm_naive_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ partial,
                                       int M,
                                       int N,
                                       int K_full,
                                       int k_offset,
                                       int K_tile)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = idx / N;
    int col = idx % N;

    if(row >= M || col >= N)
        return;

    float acc = 0.0f;
    for(int k = 0; k < K_tile; ++k)
    {
        int gk = k_offset + k;
        acc += A[row * K_full + gk] * B[col * K_full + gk];
    }
    partial[row * N + col] = acc;
}

// =========================================================================
// Tile kernel dispatch
// =========================================================================

static bool is_k4_policy(AccumKLoopPolicy p)
{
    return p == AccumKLoopPolicy::NaiveK4 ||
           p == AccumKLoopPolicy::TwoSumK4 ||
           p == AccumKLoopPolicy::KahanK4;
}

static void launch_tile_gemm(const float* A,
                              const float* B,
                              float* partial,
                              int M,
                              int N,
                              int K,
                              int k_offset,
                              int k_tile_len,
                              hipStream_t stream,
                              AccumKLoopPolicy accum_policy = AccumKLoopPolicy::Naive)
{
    if(can_use_mfma(M, N, k_tile_len))
    {
        if(accum_policy == AccumKLoopPolicy::VeltkampK4)
        {
            launch_mfma_tile_gemm_veltkamp_k4(
                A, B, partial, M, N, K, k_offset, k_tile_len, stream);
        }
        else if(accum_policy == AccumKLoopPolicy::VeltkampNaiveK4)
        {
            launch_mfma_tile_gemm_veltkamp_naive_k4(
                A, B, partial, M, N, K, k_offset, k_tile_len, stream);
        }
        else if(is_k4_policy(accum_policy))
        {
            launch_mfma_tile_gemm_k4(
                A, B, partial, M, N, K, k_offset, k_tile_len, stream, accum_policy);
        }
        else
        {
            launch_mfma_tile_gemm(
                A, B, partial, M, N, K, k_offset, k_tile_len, stream, accum_policy);
        }
    }
    else
    {
        long MN          = static_cast<long>(M) * N;
        constexpr int kB = 256;
        int grid         = static_cast<int>((MN + kB - 1) / kB);

        tile_gemm_naive_kernel<<<grid, kB, 0, stream>>>(
            A, B, partial, M, N, K, k_offset, k_tile_len);
        HIP_CHECK_ERROR(hipGetLastError());
    }
}

// =========================================================================
// split_gemm
// =========================================================================

void split_gemm(const float* A,
                const float* B,
                float* C,
                int M,
                int N,
                int K,
                int kTile,
                ReduceStrategy strategy,
                hipStream_t stream,
                SplitGemmWorkspace* ws,
                AccumKLoopPolicy accum_policy)
{
    if(M <= 0 || N <= 0 || K <= 0)
        return;

    int num_tiles = (K + kTile - 1) / kTile;
    long MN       = static_cast<long>(M) * N;

    float* partials    = nullptr;
    bool owns_partials = false;
    if(ws)
    {
        partials = ws->partials;
    }
    else
    {
        HIP_CHECK_ERROR(hipMalloc(&partials, static_cast<size_t>(num_tiles) * MN * sizeof(float)));
        owns_partials = true;
    }

    for(int t = 0; t < num_tiles; ++t)
    {
        int k_offset   = t * kTile;
        int k_tile_len = std::min(kTile, K - k_offset);

        launch_tile_gemm(
            A, B, partials + t * MN, M, N, K, k_offset, k_tile_len, stream, accum_policy);
    }

    if(num_tiles == 1)
    {
        HIP_CHECK_ERROR(
            hipMemcpyAsync(C, partials, MN * sizeof(float), hipMemcpyDeviceToDevice, stream));
    }
    else
    {
        launch_reduce(strategy, partials, C, static_cast<int>(MN), num_tiles, stream);
    }

    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    if(owns_partials)
        HIP_CHECK_ERROR(hipFree(partials));
}

// =========================================================================
// Veltkamp/Dekker splitting kernel
// =========================================================================

__global__ void veltkamp_split_kernel(const float* __restrict__ in,
                                      float* __restrict__ hi,
                                      float* __restrict__ lo,
                                      int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    constexpr float kFactor = 4097.0f;

    float a    = in[i];
    float c    = a * kFactor;
    float a_hi = c - (c - a);
    float a_lo = a - a_hi;

    hi[i] = a_hi;
    lo[i] = a_lo;
}

// =========================================================================
// Ozaki-k2 Tiled kernels (pure FP32, no FP64)
// =========================================================================

__device__ __forceinline__ void two_sum(float a, float b, float& s, float& e)
{
    s       = a + b;
    float v = s - a;
    e       = (a - (s - v)) + (b - v);
}

__global__ void accumulate_tile_ozaki_kernel(float* __restrict__ sum_buf,
                                             float* __restrict__ err_buf,
                                             const float* __restrict__ p0,
                                             const float* __restrict__ p1,
                                             const float* __restrict__ p2,
                                             const float* __restrict__ p3,
                                             int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;

    float s, e, e2;
    two_sum(p0[i], p1[i], s, e);
    two_sum(s, p2[i], s, e2);
    e += e2;
    two_sum(s, p3[i], s, e2);
    e += e2;

    float new_s, merge_e;
    two_sum(sum_buf[i], s, new_s, merge_e);
    sum_buf[i] = new_s;
    err_buf[i] += merge_e + e;
}

__global__ void finalize_ozaki_kernel(float* __restrict__ out,
                                      const float* __restrict__ sum_buf,
                                      const float* __restrict__ err_buf,
                                      int count)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count)
        return;
    out[i] = sum_buf[i] + err_buf[i];
}

static void launch_1d(int count, auto kernel, auto... args)
{
    constexpr int kBlock = 256;
    int grid             = (count + kBlock - 1) / kBlock;
    kernel<<<grid, kBlock>>>(args..., count);
}

// =========================================================================
// split_gemm_hp
// =========================================================================

void split_gemm_hp(const float* A,
                   const float* B,
                   float* C,
                   int M,
                   int N,
                   int K,
                   int kTile,
                   ReduceStrategy /*unused*/,
                   hipStream_t stream,
                   SplitGemmWorkspace* ws,
                   AccumKLoopPolicy accum_policy)
{
    if(M <= 0 || N <= 0 || K <= 0)
        return;

    long MK = static_cast<long>(M) * K;
    long KN = static_cast<long>(K) * N;
    long MN = static_cast<long>(M) * N;

    float *l_a_hi, *l_a_lo, *l_b_hi, *l_b_lo;
    float *l_p0, *l_p1, *l_p2, *l_p3;
    float *l_sum_buf, *l_err_buf;
    bool owns_buffers = false;

    if(ws)
    {
        l_a_hi = ws->a_hi; l_a_lo = ws->a_lo;
        l_b_hi = ws->b_hi; l_b_lo = ws->b_lo;
        l_p0 = ws->p0; l_p1 = ws->p1; l_p2 = ws->p2; l_p3 = ws->p3;
        l_sum_buf = ws->sum_buf; l_err_buf = ws->err_buf;
    }
    else
    {
        HIP_CHECK_ERROR(hipMalloc(&l_a_hi, MK * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_a_lo, MK * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_b_hi, KN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_b_lo, KN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_p0, MN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_p1, MN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_p2, MN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_p3, MN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_sum_buf, MN * sizeof(float)));
        HIP_CHECK_ERROR(hipMalloc(&l_err_buf, MN * sizeof(float)));
        owns_buffers = true;
    }

    HIP_CHECK_ERROR(hipMemset(l_sum_buf, 0, MN * sizeof(float)));
    HIP_CHECK_ERROR(hipMemset(l_err_buf, 0, MN * sizeof(float)));

    launch_1d(static_cast<int>(MK), veltkamp_split_kernel, A, l_a_hi, l_a_lo);
    HIP_CHECK_ERROR(hipGetLastError());
    launch_1d(static_cast<int>(KN), veltkamp_split_kernel, B, l_b_hi, l_b_lo);
    HIP_CHECK_ERROR(hipGetLastError());

    int MN_int = static_cast<int>(MN);
    constexpr int kBlock = 256;
    int grid             = (MN_int + kBlock - 1) / kBlock;

    for(int k = 0; k < K; k += kTile)
    {
        int k_len = std::min(kTile, K - k);

        launch_tile_gemm(l_a_hi, l_b_hi, l_p0, M, N, K, k, k_len, stream, accum_policy);
        launch_tile_gemm(l_a_hi, l_b_lo, l_p1, M, N, K, k, k_len, stream, accum_policy);
        launch_tile_gemm(l_a_lo, l_b_hi, l_p2, M, N, K, k, k_len, stream, accum_policy);
        launch_tile_gemm(l_a_lo, l_b_lo, l_p3, M, N, K, k, k_len, stream, accum_policy);

        accumulate_tile_ozaki_kernel<<<grid, kBlock, 0, stream>>>(
            l_sum_buf, l_err_buf, l_p0, l_p1, l_p2, l_p3, MN_int);
        HIP_CHECK_ERROR(hipGetLastError());
    }

    finalize_ozaki_kernel<<<grid, kBlock, 0, stream>>>(C, l_sum_buf, l_err_buf, MN_int);
    HIP_CHECK_ERROR(hipGetLastError());

    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    if(owns_buffers)
    {
        HIP_CHECK_ERROR(hipFree(l_a_hi));
        HIP_CHECK_ERROR(hipFree(l_a_lo));
        HIP_CHECK_ERROR(hipFree(l_b_hi));
        HIP_CHECK_ERROR(hipFree(l_b_lo));
        HIP_CHECK_ERROR(hipFree(l_p0));
        HIP_CHECK_ERROR(hipFree(l_p1));
        HIP_CHECK_ERROR(hipFree(l_p2));
        HIP_CHECK_ERROR(hipFree(l_p3));
        HIP_CHECK_ERROR(hipFree(l_sum_buf));
        HIP_CHECK_ERROR(hipFree(l_err_buf));
    }
}

// =========================================================================
// split_gemm_batched
// =========================================================================

void split_gemm_batched(const float* A,
                        const float* B,
                        float* C,
                        int M,
                        int N,
                        int K,
                        int batch_stride_A,
                        int batch_stride_B,
                        int batch_stride_C,
                        int batch_count,
                        int kTile,
                        ReduceStrategy strategy,
                        hipStream_t stream,
                        SplitGemmWorkspace* ws)
{
    for(int b = 0; b < batch_count; ++b)
    {
        split_gemm(A + b * batch_stride_A,
                   B + b * batch_stride_B,
                   C + b * batch_stride_C,
                   M, N, K, kTile, strategy, stream, ws);
    }
}

} // namespace rocm_ck::test
