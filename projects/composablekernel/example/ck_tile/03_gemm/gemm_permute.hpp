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
inline constexpr int BM = 128, BN = 256, BK = 32;

using Shape   = ck_tile::TileGemmShape<ck_tile::sequence<BM, BN, BK>,
                                       ck_tile::sequence<2, 2, 1>,
                                       ck_tile::sequence<32, 32, 16>>;
using Row     = ck_tile::tensor_layout::gemm::RowMajor;
using Traits  = ck_tile::TileGemmUniversalTraits<false, false, false, false, Row, Row, Row, true>;
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
                                                               2,
                                                               2,
                                                               32,
                                                               32,
                                                               16,
                                                               Problem::TransposeC>>;

using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape, 8, 16>;
using Pipeline    = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
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
