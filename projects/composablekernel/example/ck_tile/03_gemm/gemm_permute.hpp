// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace gemm_permute {

using Bf16             = ck_tile::bf16_t;
inline constexpr int K = 256, H = 8, D = 128, N = D * H;
inline constexpr int MaxT = 256, MaxB = 128;
inline constexpr int BM = 64, BN = 128, BK = 32, Threads = 256;

using Shape   = ck_tile::TileGemmShape<ck_tile::sequence<BM, BN, BK>,
                                       ck_tile::sequence<2, 2, 1>,
                                       ck_tile::sequence<32, 32, 16>>;
using Row     = ck_tile::tensor_layout::gemm::RowMajor;
using Traits  = ck_tile::TileGemmUniversalTraits<false, false, false, false, Row, Row, Row, true>;
using Problem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, Shape, Traits>;

// A[t,b,k], W[k,d,h] -> Y[h,b,t,d]. Tile time at fixed b so output rows are D apart.
template <class Pipeline>
__global__ void __launch_bounds__(Threads) kernel(const Bf16* __restrict__ a,
                                                  const Bf16* __restrict__ w,
                                                  Bf16* __restrict__ y,
                                                  int time,
                                                  int batch)
{
    using namespace ck_tile;
    constexpr int TileD = BN / H, Vec = 8;
    static_assert(K % BK == 0 && N % BN == 0 && BN % H == 0 && TileD % Vec == 0);
    static_assert(Problem::kBlockSize == Threads);
    constexpr int Bytes = ck_tile::max(Pipeline::GetSmemSize(), int{BM * BN * sizeof(Bf16)});
    alignas(16) __shared__ uint16_t lds[Bytes / sizeof(uint16_t)];

    const int b = blockIdx.z, t0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int rows   = ck_tile::min(BM, time - t0);
    const int64_t m0 = int64_t{t0} * batch + b;
    auto av =
        make_naive_tensor_view<address_space_enum::global>(a + m0 * K,
                                                           make_tuple(rows, K),
                                                           make_tuple(batch * K, 1),
                                                           number<Pipeline::GetVectorSizeA()>{},
                                                           number<1>{});
    auto wv =
        make_naive_tensor_view<address_space_enum::global>(w + n0,
                                                           make_tuple(K, BN),
                                                           make_tuple(N, 1),
                                                           number<Pipeline::GetVectorSizeB()>{},
                                                           number<1>{});
    auto aw        = make_tile_window(av, make_tuple(number<BM>{}, number<BK>{}), {0, 0});
    auto ww        = make_tile_window(wv, make_tuple(number<BK>{}, number<BN>{}), {0, 0});
    const auto acc = Pipeline{}(aw, ww, K / BK, static_cast<void*>(lds));

    // The last K iteration can still be reading operands from this LDS in another wave.
    block_sync_lds();
    sweep_tile(acc, [&](auto i) {
        const auto rc = get_x_indices_from_distributed_indices(acc.get_tile_distribution(), i);
        const int r = rc[number<0>{}], n = rc[number<1>{}];
        lds[((n % H) * BM + r) * TileD + n / H] = bit_cast<uint16_t>(static_cast<__bf16>(acc[i]));
    });
    block_sync_lds();

    // Each lane now owns a 16-byte packet of consecutive output channels.
    for(int p = threadIdx.x; p < BM * BN / Vec; p += Threads)
    {
        const int h = p / (BM * (TileD / Vec));
        const int r = p / (TileD / Vec) % BM;
        const int d = n0 / H + p % (TileD / Vec) * Vec;
        if(r < rows)
        {
            const int64_t dst = ((int64_t{h} * batch + b) * time + t0 + r) * D + d;
            __builtin_memcpy(__builtin_assume_aligned(y + dst, 16), lds + p * Vec, 16);
        }
    }
}

// All buffers are dense, 16-byte aligned, and non-overlapping.
inline hipError_t
launch(const Bf16* a, const Bf16* w, Bf16* y, int time, int batch, hipStream_t stream)
{
    const auto addresses = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(w) |
                           reinterpret_cast<uintptr_t>(y);
    if(!a || !w || !y || (addresses & 15) || time < 1 || time > MaxT || batch < 1 || batch > MaxB)
    {
        std::fprintf(
            stderr, "gemm_permute: need aligned buffers, T in [1,%d], B in [1,%d]\n", MaxT, MaxB);
        return hipErrorInvalidValue;
    }
    using Pipeline = ck_tile::GemmPipelineAgBgCrCompV3<Problem>;
    const dim3 grid(N / BN, (time + BM - 1) / BM, batch);
    kernel<Pipeline><<<grid, Threads, 0, stream>>>(a, w, y, time, batch);
    return hipGetLastError();
}

} // namespace gemm_permute
