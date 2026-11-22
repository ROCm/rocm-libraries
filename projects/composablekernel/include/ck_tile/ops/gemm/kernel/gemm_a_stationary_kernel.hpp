// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_e_permute_kernel.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_a_stationary_base.hpp"

namespace ck_tile {

/// @brief A-stationary GEMM: each workgroup keeps one A tile in LDS and walks all of N, storing
///        E either dense (GemmHostArgs) or through the [M0, M1, N0, N1] view of GemmEPermuteDesc.
///        Workgroup b covers m1 = b / ceil(M0 / MPerBlock) and MPerBlock consecutive m0. The
///        weight (b_ptr) is in the pipeline's flat layout, see GemmAStationaryWeightRepackKernel;
///        stride_B is not read.
/// @note  Launch with make_kernel<kBlockPerCu>: the pipeline sizes its registers for it.
template <typename Pipeline_>
struct GemmAStationaryKernel
{
    using Pipeline   = remove_cvref_t<Pipeline_>;
    using KernelArgs = GemmAStationaryKernelArgs;

    static constexpr index_t kBlockSize = Pipeline::BlockSize;
    static constexpr int kBlockPerCu    = Pipeline::kBlockPerCu;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat('_', "gemm_a_stationary", Pipeline::GetName());
    }

    /// Dense row-major E: M0 = M, M1 = 1, N0 = 1, N1 = N, stride_M0 = stride_E.
    CK_TILE_HOST static KernelArgs MakeKernelArgs(const GemmHostArgs& h)
    {
        return {h.a_ptr,
                h.b_ptr,
                h.e_ptr,
                h.K,
                h.stride_A,
                h.k_batch,
                GemmEPermuteDesc{h.M, 1, 1, h.N, h.stride_E, 0, 0}};
    }

    CK_TILE_HOST static KernelArgs MakeKernelArgs(const GemmEPermuteHostArgs& h)
    {
        return {h.a_ptr, h.b_ptr, h.e_ptr, h.K, h.stride_A, h.k_batch, h.e_permute};
    }

    CK_TILE_HOST static dim3 GridSize(const KernelArgs& kargs)
    {
        return dim3(kargs.e.M1 * integer_divide_ceil(kargs.e.M0, Pipeline::kMPerBlock));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        return Pipeline::IsSupportedArgument(kargs);
    }

    CK_TILE_DEVICE void operator()(KernelArgs kargs) const
    {
        alignas(16) __shared__ char smem[Pipeline::GetSmemSize()];
        const index_t m0_tiles = integer_divide_ceil(kargs.e.M0, Pipeline::kMPerBlock);
        const index_t m1       = blockIdx.x / m0_tiles;
        const index_t m0_0     = blockIdx.x % m0_tiles * Pipeline::kMPerBlock;
        Pipeline{}(kargs, m0_0, m1, min(Pipeline::kMPerBlock, kargs.e.M0 - m0_0), smem);
    }
};

/// @brief Repacks the weight W [K, N] (row-major, row stride stride_W; N a multiple of 16) into
///        the flat layout of Pipeline (Pipeline::FlatWeightOffset). One thread writes one 16 B
///        chunk: 8 consecutive k of one column. RunOnHost is the same repack on the host.
template <typename Pipeline_>
struct GemmAStationaryWeightRepackKernel
{
    using Pipeline  = remove_cvref_t<Pipeline_>;
    using BDataType = typename Pipeline::BDataType;

    static constexpr index_t kBlockSize = 256;
    static constexpr index_t kK         = Pipeline::kK;

    /// 16 B chunks of the flat weight of N columns.
    CK_TILE_HOST_DEVICE static constexpr index_t Chunks(index_t N) { return N / 16 * 16 * kK / 8; }

    CK_TILE_HOST static dim3 GridSize(index_t N)
    {
        return dim3(integer_divide_ceil(Chunks(N), kBlockSize));
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

    CK_TILE_HOST static void
    RunOnHost(const BDataType* w, BDataType* w_flat, index_t N, index_t stride_W)
    {
        for(index_t k = 0; k < kK; ++k)
            for(index_t n = 0; n < N; ++n)
                w_flat[Pipeline::FlatWeightOffset(k, n)] = w[long_index_t{k} * stride_W + n];
    }

    // Chunk c is unit c % 16 + 16 * (c / 16 % (K / 8)) of flat row c / (2 K): consecutive threads
    // read consecutive columns and write consecutive 16 B.
    CK_TILE_DEVICE void
    operator()(const BDataType* w, BDataType* w_flat, index_t N, index_t stride_W) const
    {
        const index_t c = blockIdx.x * kBlockSize + threadIdx.x;
        if(c >= Chunks(N))
        {
            return;
        }
        const index_t n = c / (kK / 8 * 16) * 16 + c % 16, k = c / 16 % (kK / 8) * 8;
        alignas(16) BDataType v[8];
#pragma unroll
        for(index_t e = 0; e < 8; ++e)
        {
            v[e] = w[long_index_t{k + e} * stride_W + n];
        }
        __builtin_memcpy(
            __builtin_assume_aligned(w_flat + Pipeline::FlatWeightOffset(k, n), 16), v, sizeof(v));
    }
};

} // namespace ck_tile
