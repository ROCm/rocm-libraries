// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Verify the actual FMHA row distribution, paired-half exchange, and distribution fallback.
#include "example/ck_tile/01_fmha/fmha_fwd.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "gtest/gtest.h"

namespace {

template <int M, bool Mask, typename T>
using Problem = ck_tile::BlockFmhaPipelineProblem<
    T,
    T,
    T,
    float,
    float,
    T,
    uint8_t,
    float,
    T,
    float,
    T,
    ck_tile::TileFmhaShape<ck_tile::sequence<M, 64, 32, 128, 32, 128>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           ck_tile::sequence<4, 1, 1>,
                           ck_tile::sequence<16, 16, 32>,
                           true>,
    false,
    ck_tile::ComposedAttention<0>,
    ck_tile::SimplifiedGenericAttentionMask<Mask>,
    false,
    ck_tile::TileFmhaTraits<false,
                            false,
                            false,
                            false,
                            false,
                            ck_tile::BlockAttentionBiasEnum::NO_BIAS,
                            false,
                            false,
                            false,
                            ck_tile::BlockAttentionQuantScaleEnum::NO_SCALE>,
    true,
    true>;

template <int M, bool Mask, typename T>
__global__ void reduce_rows(float* out)
{
    using P     = Problem<M, Mask, T>;
    using Pipe  = ck_tile::BlockFmhaPipelineQRKSVSTdm<P>;
    auto gemm   = ck_tile::BlockFmhaPipelineQRKSVSTdmDefaultPolicy::GetQKBlockGemm<P>();
    auto scores = gemm.MakeCBlockTile();
    // Distinct M128 row slots catch accidental cross-slot mixing.
    ck_tile::static_for<0, decltype(scores)::get_thread_buffer_size(), 1>{}([&](auto i) {
        scores.get_thread_buffer()(i) = float(threadIdx.x % 32 + 1 + 100 * (i / 32));
    });
    auto add    = [](float a, float b) { return a + b; };
    auto max    = [](float a, float b) { return ck_tile::max(a, b); };
    auto sums   = ck_tile::block_tile_reduce<float>(scores, ck_tile::sequence<1>{}, add, 0.f);
    auto maxima = ck_tile::block_tile_reduce<float>(scores, ck_tile::sequence<1>{}, max, -1.f);
    Pipe::ReduceRowSync(sums, add);
    Pipe::ReduceRowSync(maxima, max);
    ck_tile::static_for<0, decltype(sums)::get_thread_buffer_size(), 1>{}([&](auto i) {
        out[threadIdx.x * 4 + 2 * i]     = sums.get_thread_buffer()[i];
        out[threadIdx.x * 4 + 2 * i + 1] = maxima.get_thread_buffer()[i];
    });
}

template <int M, bool Mask, typename T>
bool check()
{
    float* device = nullptr;
    if(hipMalloc(&device, 512 * sizeof(float)) != hipSuccess)
        return false;
    hipLaunchKernelGGL((reduce_rows<M, Mask, T>), dim3(1), dim3(128), 0, 0, device);
    std::vector<float> got(512);
    const auto launch = hipGetLastError();
    const auto copy =
        hipMemcpy(got.data(), device, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    const auto release = hipFree(device);
    if(launch != hipSuccess || copy != hipSuccess || release != hipSuccess)
        return false;
    for(int tid = 0; tid < 128; ++tid)
        for(int row = 0; row < M / 64; ++row)
        {
            const int row_lane   = tid % 16;
            const float want_sum = 32.f * (2 * row_lane + 18 + 200 * row);
            const float want_max = float(row_lane + 17 + 100 * row);
            if(got[tid * 4 + 2 * row] != want_sum || got[tid * 4 + 2 * row + 1] != want_max)
            {
                std::printf("FAIL M%d mask%d tid%d row%d: sum %g/%g max %g/%g\n",
                            M,
                            Mask,
                            tid,
                            row,
                            got[tid * 4 + 2 * row],
                            want_sum,
                            got[tid * 4 + 2 * row + 1],
                            want_max);
                return false;
            }
        }
    std::printf("PASS row reduction M%d mask%d element_bytes%zu\n", M, Mask, sizeof(T));
    return true;
}

__global__ void reduce_four_lanes(float* out)
{
    using namespace ck_tile;
    using Pipe = BlockFmhaPipelineQRKSVSTdm<Problem<64, false, bf16_t>>;
    using E    = tile_distribution_encoding<sequence<4>,
                                            tuple<sequence<4, 8, 2>>,
                                            tuple<sequence<1>, sequence<0, 1>>,
                                            tuple<sequence<0>, sequence<0, 1>>,
                                            sequence<1>,
                                            sequence<2>>;
    static_assert(E::rs_lengths_[0] == 4);
    static_assert(E::detail::ps_over_rs_derivative_[1][0] == 8);
    auto sum = make_static_distributed_tensor<float>(make_static_tile_distribution(E{}));
    auto max = sum;
    static_for<0, 2, 1>{}([&](auto i) {
        sum.get_thread_buffer()(i) =
            float(1000 * (threadIdx.x / 32) + 100 * i.value + threadIdx.x % 32 + 1);
        max.get_thread_buffer()(i) = sum.get_thread_buffer()[i];
    });
    Pipe::ReduceRowSync(sum, [](float a, float b) { return a + b; });
    Pipe::ReduceRowSync(max, [](float a, float b) { return ck_tile::max(a, b); });
    static_for<0, 2, 1>{}([&](auto i) {
        out[4 * threadIdx.x + 2 * i.value]     = sum.get_thread_buffer()[i];
        out[4 * threadIdx.x + 2 * i.value + 1] = max.get_thread_buffer()[i];
    });
}

bool check_distribution_fallback()
{
    float* device = nullptr;
    if(hipMalloc(&device, 512 * sizeof(float)) != hipSuccess)
        return false;
    hipLaunchKernelGGL(reduce_four_lanes, dim3(1), dim3(128), 0, 0, device);
    std::vector<float> got(512);
    const auto launch = hipGetLastError();
    const auto copy =
        hipMemcpy(got.data(), device, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    const auto release = hipFree(device);
    if(launch != hipSuccess || copy != hipSuccess || release != hipSuccess)
        return false;
    for(int tid = 0; tid < 128; ++tid)
        for(int slot = 0; slot < 2; ++slot)
        {
            const int row    = tid % 8;
            const int offset = 1000 * (tid / 32) + 100 * slot;
            if(got[4 * tid + 2 * slot] != float(4 * offset + 4 * row + 52) ||
               got[4 * tid + 2 * slot + 1] != float(offset + row + 25))
            {
                std::printf("FAIL distribution fallback tid%d slot%d\n", tid, slot);
                return false;
            }
        }
    std::puts("PASS four-lane distribution fallback");
    return true;
}

TEST(QrTdmRowReduction, DenseBf16UsesCorrectRowPartners)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check<64, false, ck_tile::bf16_t>()));
    EXPECT_TRUE((check<128, false, ck_tile::bf16_t>()));
}

TEST(QrTdmRowReduction, MaskedAndOtherInputTypesUseCorrectRowPartners)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE((check<64, true, ck_tile::bf16_t>()));
    EXPECT_TRUE((check<128, true, ck_tile::bf16_t>()));
    EXPECT_TRUE((check<64, false, ck_tile::half_t>()));
    EXPECT_TRUE((check<128, false, ck_tile::half_t>()));
}

TEST(QrTdmRowReduction, FourLaneDistributionUsesGenericReduction)
{
    if(!ck_tile::is_gfx125_supported())
        GTEST_SKIP();
    EXPECT_TRUE(check_distribution_fallback());
}

} // namespace
