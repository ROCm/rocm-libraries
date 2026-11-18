// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of MxGemmPipelineAgBgCrCompTDMScheduled's compile-time rules: the LDS layout of
// the hipBLASLt solutions, the lane chunks of a K step, and the policy building and rejecting
// Problems and policies.

#include <algorithm>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::MxTdmScheduledReason;

template <typename P, typename Policy>
using PipelineOf = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduled<P, Policy>;

/// A comp_tdm_scheduled Problem: types, tile, wave group, atom rows, cluster (CX N tiles by CY
/// M tiles) and traits.
template <typename AB,
          typename Out,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t AtomM,
          index_t CX      = 4,
          index_t CY      = 4,
          typename Traits = ck_tile::MxTdmScheduledGemmTraits<>>
using ProblemOf = ck_tile::MxGemmTdmScheduledProblem<
    AB,
    AB,
    Out,
    ck_tile::ClusterTileGemmShape<ck_tile::sequence<CY, CX, 1>,
                                  ck_tile::sequence<TileM, TileN, TileK>,
                                  ck_tile::sequence<WarpM, WarpN, 1>,
                                  ck_tile::sequence<AtomM, 16, 128>>,
    Traits>;

/// LdsOffsetA / ldsBaseB with the component stride, LdsOffsetMXSA / MXSB, LdsOffsetA_Blk and
/// LdsNumBytes of a Problem and policy that build and run M x N x K.
template <typename P, typename Policy>
void check_layout(index_t M, index_t N, index_t K, const index_t (&want)[8])
{
    EXPECT_EQ((Policy::template Validate<P>()), nullptr);
    const char* why = PipelineOf<P, Policy>::GetUnsupportedReason(M, N, K);
    EXPECT_EQ(why, nullptr) << why;
    constexpr auto a       = Policy::template GetLdsOperand<P, true>();
    constexpr auto b       = Policy::template GetLdsOperand<P, false>();
    constexpr auto a_scale = Policy::template GetLdsOperand<P, true, true>();
    constexpr auto b_scale = Policy::template GetLdsOperand<P, false, true>();
    const index_t got[8]   = {b.base,
                              b.stride,
                              a.base,
                              a.stride,
                              b_scale.base,
                              a_scale.base,
                              Policy::template GetLdsBufferStride<P>(),
                              Policy::template GetSmemSize<P>()};
    for(int i = 0; i < 8; ++i)
        EXPECT_EQ(got[i], want[i]) << "field " << i;
}

// The LDS layouts of the matching hipBLASLt solutions (its A is this pipeline's B).
TEST(MxGemmTdmScheduledHost, LayoutMatchesTheHipblasltSolutions)
{
    using ck_tile::fp8_t;
    using ck_tile::pk_fp4_t;
    using Default  = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using MabTu    = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false, false>;
    using MediumTu = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>;
    {
        SCOPED_TRACE("maf fp4");
        check_layout<ProblemOf<pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32>, Default>(
            4096, 4096, 65536, {0, 99584, 33024, 33280, 132608, 137216, 141312, 282624});
    }
    {
        SCOPED_TRACE("mab fp4");
        check_layout<ProblemOf<pk_fp4_t, fp8_t, 256, 16, 512, 4, 1, 16, 1, 4>, MabTu>(
            65536, 16, 65536, {0, 2176, 9216, 33280, 4608, 5120, 131072, 206848});
    }
    {
        SCOPED_TRACE("mab fp8");
        using WideA = ck_tile::MxTdmScheduledGemmTraits<false, false, true>;
        check_layout<ProblemOf<fp8_t, fp8_t, 256, 16, 256, 4, 1, 16, 1, 4, WideA>, MabTu>(
            65536, 16, 65536, {0, 2176, 6656, 33280, 4352, 4608, 131072, 204288});
    }
    {
        SCOPED_TRACE("medium fp8");
        check_layout<ProblemOf<fp8_t, fp8_t, 128, 128, 256, 2, 2, 16>, MediumTu>(
            2048, 2048, 65536, {0, 65536, 16640, 65536, 98816, 99840, 131072, 231936});
    }
}

// Lane group g reads chunk j of a row's K step at g * 16 + chunk_offset(j): (0, 32) for FP4 and
// (0, 32, 64, 96) for the 8-bit types, which tile the K step.
template <typename P>
void check_lane_chunks()
{
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    std::vector<index_t> bytes;
    for(index_t g = 0; g < Policy::kLaneGroups; ++g)
        for(index_t j = 0; j < Policy::GetChunksPerLane<P>(); ++j)
            for(index_t b = 0; b < Policy::kChunkBytes; ++b)
                bytes.push_back(g * Policy::kChunkBytes + Policy::GetChunkOffset<P>(j) + b);
    std::sort(bytes.begin(), bytes.end());
    ASSERT_EQ(static_cast<index_t>(bytes.size()), Policy::GetStepBytes<P>());
    for(index_t i = 0; i < Policy::GetStepBytes<P>(); ++i)
        EXPECT_EQ(bytes[i], i);
    EXPECT_EQ(Policy::GetChunkOffset<P>(1), 32);
}

TEST(MxGemmTdmScheduledHost, LaneChunksTileTheKStep)
{
    check_lane_chunks<ProblemOf<ck_tile::pk_fp4_t, ck_tile::fp8_t, 256, 256, 512, 2, 2, 32>>();
    check_lane_chunks<ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>>();
    check_lane_chunks<ProblemOf<ck_tile::bf8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>>();
}

// The cluster multicast masks hold exactly the workgroups sharing the coordinate, with bit
// x + y * cluster_x for workgroup (x, y), for every cluster of at most 16 workgroups.
TEST(MxGemmTdmScheduledHost, ClusterMasksHoldTheWorkgroupsSharingACoordinate)
{
    for(index_t cx = 1; cx <= 16; ++cx)
        for(index_t cy = 1; cx * cy <= 16; ++cy)
            for(index_t x = 0; x < cx; ++x)
                for(index_t y = 0; y < cy; ++y)
                {
                    uint32_t same_x = 0;
                    uint32_t same_y = 0;
                    for(index_t j = 0; j < cy; ++j)
                        for(index_t i = 0; i < cx; ++i)
                        {
                            same_x |= static_cast<uint32_t>(i == x) << (i + j * cx);
                            same_y |= static_cast<uint32_t>(j == y) << (i + j * cx);
                        }
                    EXPECT_EQ(ck_tile::cluster_mask_same_x(cx, cy, x), same_x);
                    EXPECT_EQ(ck_tile::cluster_mask_same_y(cx, y), same_y);
                }
}

// The policy derives the schedule from a Problem (its cluster) and its parameters, and the
// compile-time validator and the names follow.
using MafShape   = ck_tile::ClusterTileGemmShape<ck_tile::sequence<4, 4, 1>,
                                                 ck_tile::sequence<256, 256, 512>,
                                                 ck_tile::sequence<2, 2, 1>,
                                                 ck_tile::sequence<32, 16, 128>>;
using MafProblem = ck_tile::
    MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp8_t, MafShape>;
using MabShape   = ck_tile::ClusterTileGemmShape<ck_tile::sequence<4, 1, 1>,
                                                 ck_tile::sequence<128, 16, 512>,
                                                 ck_tile::sequence<4, 1, 1>,
                                                 ck_tile::sequence<16, 16, 128>>;
using MabProblem = ck_tile::
    MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::half_t, MabShape>;

TEST(MxGemmTdmScheduledHost, PolicyDerivesTheScheduleFromProblemAndParameters)
{
    using Default = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    static_assert(Default::GetSmemSize<MafProblem>() == 282624);
    static_assert(Default::GetReason<MafProblem>() == MxTdmScheduledReason::Ok);
    static_assert(Default::Validate<MafProblem>() == nullptr);
    // 3 scale reads and 2 chunks of 4 x 2 A rows and 8 B fragments.
    static_assert(Default::GetReadsPerStep<MafProblem>() == 35);
    EXPECT_EQ((PipelineOf<MafProblem, Default>::GetKernelName()),
              "mx_gemm_tdm_scheduled_mxfp4_t256x256x512_c4x4");
    EXPECT_EQ(Default::GetTuningName<MafProblem>(), "_c4x4");

    using Pf2 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2>;
    EXPECT_EQ(Pf2::GetTuningName<MafProblem>(), "_c4x4_pf2");

    using Mab = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<0, false, false>;
    static_assert(Mab::GetReason<MabProblem>() == MxTdmScheduledReason::Ok);
    EXPECT_EQ((PipelineOf<MabProblem, Mab>::GetKernelName()),
              "mx_gemm_tdm_scheduled_mxfp4_t128x16x512_c1x4_fp16_wg4x1_pf0_lp0_hplrnone_mma16");
    EXPECT_EQ(Mab::GetTuningName<MabProblem>(), "_c1x4_pf0_lp0_hplrnone");
}

// A's iterate-mode load steps 16 rows of K bytes per block: 2^32 bytes at K = 2^28, past 32 bits.
TEST(MxGemmTdmScheduledHost, IterateIncrementIsSixtyFourBit)
{
    using Wide   = ck_tile::MxTdmScheduledGemmTraits<false, false, true, true>;
    using P      = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 256, 64, 256, 1, 4, 16, 1, 1, Wide>;
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>;
    using Pipe   = PipelineOf<P, Policy>;
    static_assert(Policy::GetLdsOperand<P, true>().iterate_rows == 16);
    constexpr index_t K = index_t{1} << 28;
    const char* why     = Pipe::GetUnsupportedReason(256, 64, K);
    EXPECT_EQ(why, nullptr) << why;
    static_assert(Pipe::GetIterateGlobalIncrement<true>(K) == uint64_t{1} << 32);
}

// The host picks the kernel of a whole-tile count's parity: TailNumber::One for odd counts, Two
// for even ones, with the K tail excluded; each variant rejects the other parity, Full none.
TEST(MxGemmTdmScheduledHost, TailHandlerPicksTheVariantOfTheTileCount)
{
    using ck_tile::TailNumber;
    using Default = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using Full    = PipelineOf<MafProblem, Default>;
    using One     = Full::WithTailNum<TailNumber::One>;
    using Two     = Full::WithTailNum<TailNumber::Two>;
    static_assert(Full::GetTailNumber(512) == TailNumber::One);
    static_assert(Full::GetTailNumber(1024) == TailNumber::Two);
    static_assert(Full::GetTailNumber(1536 + 128) == TailNumber::One);
    const auto picked = [](index_t K) {
        const index_t tiles = K / 512;
        return Full::TailHandler([](auto, auto tail_num) { return decltype(tail_num)::value; },
                                 Full::BlockHasHotloop(tiles),
                                 Full::GetBlockLoopTailNum(tiles));
    };
#if !defined(CK_TILE_FORCE_SINGLE_TAIL_HANDLER)
    EXPECT_EQ(picked(512), TailNumber::One);
    EXPECT_EQ(picked(1024), TailNumber::Two);
    EXPECT_EQ(picked(65536), TailNumber::Two);
#else
    EXPECT_EQ(picked(512), TailNumber::Full);
#endif
    EXPECT_EQ(One::GetKernelName(), "mx_gemm_tdm_scheduled_mxfp4_t256x256x512_c4x4_odd");
    EXPECT_EQ(Two::GetKernelName(), "mx_gemm_tdm_scheduled_mxfp4_t256x256x512_c4x4_even");
    for(index_t K : {512, 1024, 1536, 2048})
    {
        const bool odd = K / 512 % 2;
        EXPECT_EQ(One::GetUnsupportedReason(1024, 1024, K) == nullptr, odd) << K;
        EXPECT_EQ(Two::GetUnsupportedReason(1024, 1024, K) == nullptr, !odd) << K;
        EXPECT_EQ(Full::GetUnsupportedReason(1024, 1024, K), nullptr) << K;
    }
}

TEST(MxGemmTdmScheduledHost, PolicyRejectsInvalidProblems)
{
    using Default  = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using Shape1x1 = ck_tile::ClusterTileGemmShape<ck_tile::sequence<4, 4, 1>,
                                                   ck_tile::sequence<16, 16, 128>,
                                                   ck_tile::sequence<2, 2, 1>,
                                                   ck_tile::sequence<32, 16, 128>>;
    using Small    = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, ck_tile::fp8_t, Shape1x1>;
    EXPECT_STREQ((Default::Validate<Small>()), "tile_atoms");
    using Fp8Atom32 = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp8_t, MafShape>;
    EXPECT_STREQ((Default::Validate<Fp8Atom32>()), "tile_k");
    using Mixed = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::fp8_t, ck_tile::fp8_t, MafShape>;
    EXPECT_STREQ((Default::Validate<Mixed>()), "dtype_width");
    using F32Out =
        ck_tile::MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t, ck_tile::pk_fp4_t, float, MafShape>;
    EXPECT_STREQ((Default::Validate<F32Out>()), "out_dtype");
    using Cluster4x8Shape = ck_tile::ClusterTileGemmShape<ck_tile::sequence<4, 8, 1>,
                                                          ck_tile::sequence<256, 256, 512>,
                                                          ck_tile::sequence<2, 2, 1>,
                                                          ck_tile::sequence<32, 16, 128>>;
    using Cluster8x4      = ck_tile::MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t,
                                                               ck_tile::pk_fp4_t,
                                                               ck_tile::fp8_t,
                                                               Cluster4x8Shape>;
    EXPECT_STREQ((Default::Validate<Cluster8x4>()), "cluster_size");
    // SplitBLdsRead without PrefetchLdsRead.
    using NoPrefetch = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, false, true>;
    EXPECT_STREQ((NoPrefetch::Validate<MafProblem>()), "split_b_read");
    // GL2 prefetch distance 1.
    using Pf1 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<1>;
    EXPECT_STREQ((Pf1::Validate<MafProblem>()), "gl2_distance");
    // 8-bit types on the 32x16 FP4 atom, with a tile_k the 8-bit types allow.
    using Fp8Atom32Shape = ck_tile::ClusterTileGemmShape<ck_tile::sequence<2, 2, 1>,
                                                         ck_tile::sequence<128, 128, 256>,
                                                         ck_tile::sequence<2, 2, 1>,
                                                         ck_tile::sequence<32, 16, 128>>;
    using Fp8Mma32       = ck_tile::
        MxGemmTdmScheduledProblem<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::fp8_t, Fp8Atom32Shape>;
    EXPECT_STREQ((Default::Validate<Fp8Mma32>()), "mma");
}

// K plus the loads' lookahead, (4 + 1) tiles of 256 at GL2 distance 4, must fit in i32: K at most
// 2^31 - 1 - 1280. M = N = 32 keeps the scales under 2^31 bytes at that K.
TEST(MxGemmTdmScheduledHost, KLimit)
{
    using P    = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 32, 32, 256, 2, 2, 16, 1, 1>;
    using Pipe = PipelineOf<P, ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>>;
    const auto* k_long = ck_tile::mx_tdm_scheduled_reason_message(MxTdmScheduledReason::KLong);
    const char* below  = Pipe::GetUnsupportedReason(32, 32, 2147482336);
    EXPECT_TRUE(below == nullptr || std::string(below) != k_long) << below;
    EXPECT_STREQ(Pipe::GetUnsupportedReason(32, 32, 2147482368), k_long);
}

// An instance's K tail has at most tile_k / 128 - 1 K steps, so no instance runs a K whose
// remainder mod tile_k is above tile_k - 128.
TEST(MxGemmTdmScheduledHost, KTailFull)
{
    using Policy       = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>;
    const auto* full   = ck_tile::mx_tdm_scheduled_reason_message(MxTdmScheduledReason::KTailFull);
    const auto is_full = [&](const char* why) { return why && std::string(why) == full; };
    using Tile256 =
        PipelineOf<ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>, Policy>;
    EXPECT_TRUE(is_full(Tile256::GetUnsupportedReason(1024, 1024, 2048 + 160)));
    EXPECT_TRUE(is_full(Tile256::GetUnsupportedReason(1024, 1024, 2048 + 224)));
    EXPECT_FALSE(is_full(Tile256::GetUnsupportedReason(1024, 1024, 2048 + 96)));
    EXPECT_EQ(Tile256::GetUnsupportedReason(1024, 1024, 2048), nullptr);
    using Tile128 =
        PipelineOf<ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 128, 2, 2, 16>, Policy>;
    EXPECT_TRUE(is_full(Tile128::GetUnsupportedReason(1024, 1024, 1024 + 32)));
}

} // namespace
