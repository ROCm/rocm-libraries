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
inline constexpr int MaxT = 768, MaxB = 512;
inline constexpr int BM = 128, BN = 256, BK = 32, Threads = 256;

using Shape   = ck_tile::TileGemmShape<ck_tile::sequence<BM, BN, BK>,
                                       ck_tile::sequence<2, 2, 1>,
                                       ck_tile::sequence<32, 32, 16>>;
using Row     = ck_tile::tensor_layout::gemm::RowMajor;
using Traits  = ck_tile::TileGemmUniversalTraits<false, false, false, false, Row, Row, Row, true>;
using Problem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, Shape, Traits>;

// Contract: the weight is consumed pre-permuted, Wp[k,h,d] = W[k,d,h] (see prepermute_weight,
// a one-time reorder of a model weight). A[t,b,k], Wp[k,h,d] -> Y[h,b,t,d]. A tile covers one h
// and BN contiguous d, at fixed b, so it is a plain [rows, BN] window of Y with row stride D and
// the stock CShuffleEpilogue stores it.
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

template <class Pipeline>
__global__ void __launch_bounds__(Threads) kernel(const Bf16* __restrict__ a,
                                                  const Bf16* __restrict__ wp,
                                                  Bf16* __restrict__ y,
                                                  int time,
                                                  int batch)
{
    using namespace ck_tile;
    static_assert(K % BK == 0 && D % BN == 0);
    static_assert(Problem::kBlockSize == Threads);
    constexpr int Bytes = ck_tile::max(Pipeline::GetSmemSize(), Epilogue::GetSmemSize());
    alignas(16) __shared__ char lds[Bytes];

    const int time_tiles = (time + BM - 1) / BM;
    const auto [tile_m, tile_n] =
        Partitioner{time_tiles * batch * BM, N}.GetOutputTileIndex(static_cast<int>(blockIdx.x));
    const int b = tile_m / time_tiles, t0 = tile_m % time_tiles * BM, n0 = tile_n * BN;
    const int h = n0 / D, d0 = n0 % D;
    const int rows   = ck_tile::min(BM, time - t0);
    const int64_t m0 = int64_t{t0} * batch + b;
    auto av =
        make_naive_tensor_view<address_space_enum::global>(a + m0 * K,
                                                           make_tuple(rows, K),
                                                           make_tuple(batch * K, 1),
                                                           number<Pipeline::GetVectorSizeA()>{},
                                                           number<1>{});
    auto wv =
        make_naive_tensor_view<address_space_enum::global>(wp + n0,
                                                           make_tuple(K, BN),
                                                           make_tuple(N, 1),
                                                           number<Pipeline::GetVectorSizeB()>{},
                                                           number<1>{});
    // Rows past `rows` lie beyond the view's element space, so their stores are dropped.
    auto yv = make_naive_tensor_view<address_space_enum::global>(
        y + ((int64_t{h} * batch + b) * time + t0) * D + d0,
        make_tuple(rows, BN),
        make_tuple(D, 1),
        number<Epilogue::GetVectorSizeC()>{},
        number<1>{});
    auto aw        = make_tile_window(av, make_tuple(number<BM>{}, number<BK>{}), {0, 0});
    auto ww        = make_tile_window(wv, make_tuple(number<BK>{}, number<BN>{}), {0, 0});
    auto yw        = make_tile_window(yv, make_tuple(number<BM>{}, number<BN>{}), {0, 0});
    const auto acc = Pipeline{}(aw, ww, K / BK, static_cast<void*>(lds));
    Epilogue{}(yw, acc, ck_tile::tuple<>{}, static_cast<void*>(lds));
}

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

// All buffers are dense, 16-byte aligned, and non-overlapping.
inline hipError_t
launch(const Bf16* a, const Bf16* wp, Bf16* y, int time, int batch, hipStream_t stream)
{
    const auto addresses = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(wp) |
                           reinterpret_cast<uintptr_t>(y);
    if(!a || !wp || !y || (addresses & 15) || time < 1 || time > MaxT || batch < 1 || batch > MaxB)
    {
        std::fprintf(
            stderr, "gemm_permute: need aligned buffers, T in [1,%d], B in [1,%d]\n", MaxT, MaxB);
        return hipErrorInvalidValue;
    }
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
    const dim3 grid(Partitioner::GridSize((time + BM - 1) / BM * batch * BM, N));
    kernel<Pipeline><<<grid, Threads, 0, stream>>>(a, wp, y, time, batch);
    return hipGetLastError();
}

} // namespace gemm_permute
