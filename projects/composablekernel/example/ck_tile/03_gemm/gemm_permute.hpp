// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace gemm_permute {

using Bf16             = ck_tile::bf16_t;
inline constexpr int K = 384, H = 8, D = 256, N = D * H;

// The library kernel's tile per architecture; on gfx1250 the fastest TDM pipeline and tile for
// this shape, run-ahead with two LDS slots filled one K step ahead (TDM clips the M tail).
#if defined(CK_USE_GFX1250)
inline constexpr const char* Arch = "gfx1250";
inline constexpr int BM = 192, BN = 256, BK = 64, WarpsM = 2, WarpsN = 8, WarpTileM = 16,
                     WarpTileN = 16, WarpTileK = 32, TileGroups = 16, TileM01 = 8;
inline constexpr bool DoubleSmemBuffer = true; // the TDM pipelines' LDS ping-pong
template <typename Problem>
using MainPipeline = ck_tile::GemmPipelineAgBgCrCompTDMRunAhead<
    Problem,
    ck_tile::GemmPipelineAgBgCrCompTDMRunAheadPolicy<2, 1>>;
#else
inline constexpr const char* Arch = "gfx950";
inline constexpr int BM = 128, BN = 256, BK = 32, WarpsM = 2, WarpsN = 2, WarpTileM = 32,
                     WarpTileN = 32, WarpTileK = 16, TileGroups = 8, TileM01 = 16;
inline constexpr bool DoubleSmemBuffer = false;
template <typename Problem>
using MainPipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
#endif

using Shape = ck_tile::TileGemmShape<ck_tile::sequence<BM, BN, BK>,
                                     ck_tile::sequence<WarpsM, WarpsN, 1>,
                                     ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>>;
using Row   = ck_tile::tensor_layout::gemm::RowMajor;
using Traits =
    ck_tile::TileGemmUniversalTraits<false, false, false, DoubleSmemBuffer, Row, Row, Row, true>;
using Problem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, Shape, Traits>;

// Contract: the weight is consumed pre-permuted, Wp[k,h,d] = W[k,d,h] (see prepermute_weight,
// a one-time reorder of a model weight). A[t,b,k], Wp[k,h,d] -> Y[h,b,t,d]: GemmEPermuteKernel
// stores C[t*B+b, h*D+d] through the [T, B, H, D] view of Y with strides (D, T*D, B*T*D, 1).
using Epilogue =
    ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<Bf16,
                                                               Bf16,
                                                               ck_tile::tuple<>,
                                                               float,
                                                               Bf16,
                                                               ck_tile::tuple<>,
                                                               Row,
                                                               ck_tile::element_wise::PassThrough,
                                                               BM,
                                                               BN,
                                                               WarpsM,
                                                               WarpsN,
                                                               WarpTileM,
                                                               WarpTileN,
                                                               WarpTileK,
                                                               Problem::TransposeC,
                                                               1,
                                                               false,
                                                               1,
                                                               1,
                                                               DoubleSmemBuffer>>;

using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape, TileGroups, TileM01>;
using Pipeline    = MainPipeline<Problem>;
using Kernel      = ck_tile::GemmEPermuteKernel<Partitioner, Pipeline, Epilogue>;

// One-time weight reorder W[k,d,h] -> Wp[k,h,d] (dense, same size).
__global__ void prepermute_kernel(const Bf16* __restrict__ w, Bf16* __restrict__ wp)
{
    const int k = blockIdx.x, d = threadIdx.x;
    alignas(16) Bf16 heads[H];
    __builtin_memcpy(
        heads, __builtin_assume_aligned(w + (int64_t{k} * D + d) * H, 16), sizeof(heads));
#pragma unroll
    for(int h = 0; h < H; ++h)
        wp[(int64_t{k} * H + h) * D + d] = heads[h];
}

inline hipError_t prepermute_weight(const Bf16* w, Bf16* wp, hipStream_t stream)
{
    prepermute_kernel<<<K, D, 0, stream>>>(w, wp);
    return hipGetLastError();
}

// All buffers are dense and non-overlapping.
inline hipError_t
launch(const Bf16* a, const Bf16* wp, Bf16* y, int time, int batch, hipStream_t stream)
{
    const auto kargs = Kernel::MakeKernelArgs(ck_tile::GemmEPermuteHostArgs(
        a, wp, y, 1, K, K, N, {time, batch, H, D, D, time * D, batch * time * D}));
    if(!Kernel::IsSupportedArgument(kargs))
    {
        std::fprintf(stderr, "gemm_permute: T=%d B=%d is not supported\n", time, batch);
        return hipErrorInvalidValue;
    }
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<1>(
            Kernel{}, Kernel::GridSize(time * batch, N, 1), Kernel::BlockSize(), 0, kargs));
    return hipGetLastError();
}

} // namespace gemm_permute
