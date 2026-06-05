// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Split-accumulation GPU reference GEMM.
//
// Computes C = A @ B by splitting K into tiles, computing one partial
// product per tile via a naive GPU kernel, then reducing the partials
// with a pluggable accumulation strategy.
//
// The architecture separates tile-level multiply from K-reduction so
// that different reduction strategies can be compared without changing
// the multiply engine.

#pragma once

#include "mfma_tile_gemm.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>

namespace rocm_ck::test {

enum class ReduceStrategy
{
    NaiveSum,       // FP32 sequential sum of partials
    CompensatedSum, // Kahan compensated FP32 summation
    FP64Sum,        // FP64 accumulation, cast back to FP32
};

// Pre-allocated device memory for split_gemm / split_gemm_hp.
// Create once (e.g. in test fixture SetUp), reuse across calls to avoid
// per-call hipMalloc/hipFree overhead (~10-50ms each on shared clusters).
//
// Usage:
//   SplitGemmWorkspace ws(max_M, max_N, max_K, max_kTile);
//   split_gemm(A, B, C, M, N, K, kTile, strategy, stream, &ws);
//   split_gemm_hp(A, B, C, M, N, K, kTile, strategy, stream, &ws);
//   // ws freed automatically in destructor
struct SplitGemmWorkspace
{
    // split_gemm partials: num_tiles * M * N
    float* partials = nullptr;
    size_t partials_bytes = 0;

    // split_gemm_hp buffers
    float* a_hi = nullptr;
    float* a_lo = nullptr;
    float* b_hi = nullptr;
    float* b_lo = nullptr;
    float* p0 = nullptr;
    float* p1 = nullptr;
    float* p2 = nullptr;
    float* p3 = nullptr;
    float* sum_buf = nullptr;
    float* err_buf = nullptr;

    // Test I/O buffers (reusable across test cases)
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    size_t d_A_bytes = 0;
    size_t d_B_bytes = 0;
    size_t d_C_bytes = 0;

    // Allocate for the worst-case config.
    // max_num_tiles = ceil(max_K / min_kTile) — pass the largest num_tiles you'll use.
    SplitGemmWorkspace(int max_M, int max_N, int max_K, int min_kTile);
    ~SplitGemmWorkspace();

    SplitGemmWorkspace(const SplitGemmWorkspace&) = delete;
    SplitGemmWorkspace& operator=(const SplitGemmWorkspace&) = delete;
};

/// C[M,N] = sum_k A[M,K] * B[N,K]   (CK convention: B is [N,K], not [K,N])
///
/// All pointers are device memory.
///   A is [M, K] row-major (stride = K)
///   B is [N, K] row-major (stride = K)  ← same as reference_batched_gemm
///   C is [M, N] row-major (stride = N)
///
/// K is split into tiles of kTile. Each tile produces a partial product
/// via a GPU kernel. The partials are reduced by the chosen strategy.
///
/// If ws is non-null, uses pre-allocated workspace. Otherwise allocates
/// and frees internally (slower).
void split_gemm(const float* A,
                const float* B,
                float* C,
                int M,
                int N,
                int K,
                int kTile,
                ReduceStrategy strategy,
                hipStream_t stream = nullptr,
                SplitGemmWorkspace* ws = nullptr,
                AccumKLoopPolicy accum_policy = AccumKLoopPolicy::Naive);

/// Batched variant: A, B, C are contiguous batches separated by the
/// given strides. Calls split_gemm per batch on the same stream.
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
                        hipStream_t stream = nullptr,
                        SplitGemmWorkspace* ws = nullptr);

/// High-precision GEMM via Veltkamp/Dekker splitting + Ozaki-k2 tiled
/// compensated accumulation (pure FP32, no FP64 hardware required).
///
/// If ws is non-null, uses pre-allocated workspace. Otherwise allocates
/// and frees internally (slower).
void split_gemm_hp(const float* A,
                   const float* B,
                   float* C,
                   int M,
                   int N,
                   int K,
                   int kTile,
                   ReduceStrategy tile_reduce_strategy,
                   hipStream_t stream = nullptr,
                   SplitGemmWorkspace* ws = nullptr,
                   AccumKLoopPolicy accum_policy = AccumKLoopPolicy::Naive);

// Internal: launch reduce kernel (defined in reduce_kernels.hip)
void launch_reduce(ReduceStrategy strategy,
                   const float* partials,
                   float* C,
                   int MN,
                   int num_tiles,
                   hipStream_t stream);

} // namespace rocm_ck::test
