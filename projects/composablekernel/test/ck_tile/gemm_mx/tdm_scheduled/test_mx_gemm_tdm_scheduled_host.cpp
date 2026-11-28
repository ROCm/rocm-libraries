// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only checks of MxGemmPipelineAgBgCrCompTDMScheduled's compile-time rules: the LDS layout of
// the tuned instances, the lane chunks of a K step, and the policy building and rejecting Problems
// and policies.

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/epilogue.hpp"
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

/// The LDS layout of a Problem and policy that build and run M x N x K: the B and A bases and
/// component strides, the B and A scale bases, the buffer stride and the LDS size; then A's and
/// B's iterate-mode rows and stride (0 for a load the TDM pads itself).
template <typename P, typename Policy>
void check_layout(index_t M, index_t N, index_t K, const index_t (&want)[12])
{
    EXPECT_EQ((Policy::template Validate<P>()), nullptr);
    const char* why = PipelineOf<P, Policy>::GetUnsupportedReason(M, N, K);
    EXPECT_EQ(why, nullptr) << why;
    constexpr auto a       = Policy::template GetLdsOperand<P, true>();
    constexpr auto b       = Policy::template GetLdsOperand<P, false>();
    constexpr auto a_scale = Policy::template GetLdsOperand<P, true, true>();
    constexpr auto b_scale = Policy::template GetLdsOperand<P, false, true>();
    const index_t got[12]  = {b.base,
                              b.stride,
                              a.base,
                              a.stride,
                              b_scale.base,
                              a_scale.base,
                              Policy::template GetLdsBufferStride<P>(),
                              Policy::template GetSmemSize<P>(),
                              a.iterate_rows,
                              a.iterate_stride,
                              b.iterate_rows,
                              b.iterate_stride};
    for(int i = 0; i < 12; ++i)
        EXPECT_EQ(got[i], want[i]) << "field " << i;
}

// The LDS layouts of the tuned FP4 instances, FP8 variants of 256x16 and 128x128 tiles, a BF8
// variant of 256x16, the FP6 SM and MAB instances, and three-buffer layouts (FP6 SM, FP6 128x16,
// FP4 256x128), each valid for its problem. An operand is two components (one per wave pair) of
// tile / 2 rows of GetKBytes(tile_k) bytes, padded 16 B per pad block; a scale region is
// 2 x rows x K steps x 4 B. The scale regions, and B's two components when not interleaved, are
// rounded up to the region alignment (512 B for FP4 and FP6, 256 B for 8-bit). Buffer i starts i
// buffer strides in; the stride is the power of two above the buffer when the last buffer still
// fits in the 320 KiB of LDS, else the buffer; the LDS size is the last buffer's start plus the
// buffer.
TEST(MxGemmTdmScheduledHost, LdsLayoutOfTheTunedInstances)
{
    using ck_tile::bf16_t;
    using ck_tile::bf8_t;
    using ck_tile::fp8_t;
    using ck_tile::pk_fp4_t;
    using ck_tile::pk_fp6x16_t;
    using Default = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using NoSplitNoInterleave =
        ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false, false>;
    using NoSplit = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>;
    {
        // 128 rows of 256 B: A pads 16 B per 1 KiB (33280); B loads in iterate mode, 8 rows per
        // 2064 B (33024). Interleaved B0, A0, A1, B1 (B stride 33024 + 2 x 33280), then the B
        // scales (4096 B, 16 B per 256: 4352, rounded to 4608) and the A scales (4096 B). The
        // 141312 B buffer and a 256 KiB stride exceed the LDS: stride 141312, LDS 2 x 141312.
        SCOPED_TRACE("fp4 256x256x512");
        check_layout<ProblemOf<pk_fp4_t, fp8_t, 256, 256, 512, 2, 2, 32>, Default>(
            4096,
            4096,
            65536,
            {0, 99584, 33024, 33280, 132608, 137216, 141312, 282624, 0, 0, 8, 2064});
    }
    {
        // Not interleaved: B (8 rows of 256 B, 16 B per 256: 2176) twice, rounded to 4608; the B
        // scales (256 B, rounded to 512) and the A scales (4096 B); A (128 rows, 16 B per 1 KiB:
        // 33280) at 9216. Buffer 75776, stride 131072: LDS 131072 + 75776.
        SCOPED_TRACE("fp4 256x16x512");
        check_layout<ProblemOf<pk_fp4_t, fp8_t, 256, 16, 512, 4, 1, 16, 1, 4>, NoSplitNoInterleave>(
            65536, 16, 65536, {0, 2176, 9216, 33280, 4608, 5120, 131072, 206848, 0, 0, 0, 0});
    }
    {
        // As fp4 256x16x512 with 2 K steps and 256 B rounding: B twice (4352), the B scales (128 B,
        // rounded to 256), the A scales (2048 B), A at 6656. Buffer 73216: LDS 131072 + 73216.
        SCOPED_TRACE("fp8 256x16x256");
        using WideA = ck_tile::MxTdmScheduledGemmTraits<false, false, true>;
        check_layout<ProblemOf<fp8_t, fp8_t, 256, 16, 256, 4, 1, 16, 1, 4, WideA>,
                     NoSplitNoInterleave>(
            65536, 16, 65536, {0, 2176, 6656, 33280, 4352, 4608, 131072, 204288, 0, 0, 0, 0});
    }
    {
        // BF8 takes FP8's layout: 8-bit rows and 256 B region alignment.
        SCOPED_TRACE("bf8 256x16x256");
        using WideA = ck_tile::MxTdmScheduledGemmTraits<false, false, true>;
        check_layout<ProblemOf<bf8_t, fp8_t, 256, 16, 256, 4, 1, 16, 1, 4, WideA>,
                     NoSplitNoInterleave>(
            65536, 16, 65536, {0, 2176, 6656, 33280, 4352, 4608, 131072, 204288, 0, 0, 0, 0});
    }
    {
        // 64 rows of 256 B, 16 B per 1 KiB: 16640 per component. B0 and A0 share the first
        // 64 KiB segment, B1 and A1 the second (strides 65536), then the B and A scales
        // (1024 B each). Buffer 100864, stride 131072: LDS 131072 + 100864.
        SCOPED_TRACE("fp8 128x128x256");
        check_layout<ProblemOf<fp8_t, fp8_t, 128, 128, 256, 2, 2, 16>, NoSplit>(
            2048, 2048, 65536, {0, 65536, 16640, 65536, 98816, 99840, 131072, 231936, 0, 0, 0, 0});
    }
    {
        // FP6 SM: rows of 384 B. A's lanes read rows 2 apart and B's 4: 16 B after every 768 B
        // (2 rows) and 1536 B (4 rows), both in iterate mode, so A takes 16 x 784 B (12544) and B
        // 16 x 1552 B (24832). Not interleaved: B twice (49664), the B scales (2048 B) and the
        // A scales (1024 B), A at 52736. Buffer 77824, stride 131072: LDS 131072 + 77824.
        SCOPED_TRACE("fp6 64x128x512");
        using Sm = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false>;
        check_layout<ProblemOf<pk_fp6x16_t, bf16_t, 64, 128, 512, 2, 2, 16>, Sm>(
            256,
            2048,
            2048,
            {0, 24832, 52736, 12544, 49664, 51712, 131072, 208896, 2, 784, 4, 1552});
    }
    {
        // FP6 SM with 3 buffers: the same buffer, whose 77824 B are its stride (two 128 KiB
        // strides and the buffer exceed the LDS): LDS 2 x 77824 + 77824.
        SCOPED_TRACE("fp6 64x128x512, 3 buffers");
        using Sm3 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false, 3>;
        check_layout<ProblemOf<pk_fp6x16_t, bf16_t, 64, 128, 512, 2, 2, 16>, Sm3>(
            256,
            2048,
            2048,
            {0, 24832, 52736, 12544, 49664, 51712, 77824, 233472, 2, 784, 4, 1552});
    }
    {
        // FP6 128x16x512 4x1 with 3 buffers, not interleaved: A's lanes read rows 2 apart (32 x
        // 784 B, 25088) and B's 1 (4 x 784 B, 3136); B twice (6272, rounded to 6656), the B scales
        // (256 B, rounded to 512), the A scales (2048 B), A at 9216. Buffer 59392; two 64 KiB
        // strides and the buffer fit: stride 65536, LDS 2 x 65536 + 59392.
        SCOPED_TRACE("fp6 128x16x512, 3 buffers");
        using Narrow3 =
            ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<0, false, false, false, 3>;
        check_layout<ProblemOf<pk_fp6x16_t, bf16_t, 128, 16, 512, 4, 1, 16, 1, 2>, Narrow3>(
            512, 16, 2048, {0, 3136, 9216, 25088, 6656, 7168, 65536, 190464, 2, 784, 2, 784});
    }
    {
        // FP4 256x128x512 with 3 buffers, interleaved without segment alignment: B0 at 0, A0 and A1
        // (33280 B each) after it, B1 at 83200, then the B scales (2048 B) and the A scales
        // (4096 B): buffer 105984. Two 128 KiB strides and the buffer exceed the LDS: stride
        // 105984, LDS 3 x 105984.
        SCOPED_TRACE("fp4 256x128x512, 3 buffers");
        using Lb3 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, true, true, 3>;
        check_layout<ProblemOf<pk_fp4_t, fp8_t, 256, 128, 512, 2, 2, 32, 2, 2>, Lb3>(
            1024, 1024, 1536, {0, 83200, 16640, 33280, 99840, 101888, 105984, 317952, 0, 0, 0, 0});
    }
    {
        // FP6 MAB: A's lanes read rows 4 apart (32 x 1552 B, 49664) and B's 1 (16 B per 2 rows,
        // 4 x 784 B, 3136). Interleaved 4x1: A0 at 0 and A1 at 65536, B0 and B1 after A0, then
        // after A1 the B scales at 115200 (256 B, rounded to 512) and the A scales (4096 B) at
        // 115712. Buffer 119808, stride 131072: LDS 131072 + 119808.
        SCOPED_TRACE("fp6 256x16x512");
        using Mab = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<0, false, false>;
        check_layout<ProblemOf<pk_fp6x16_t, bf16_t, 256, 16, 512, 4, 1, 16, 1, 2>, Mab>(
            49152,
            16,
            32768,
            {49664, 3136, 0, 65536, 115200, 115712, 131072, 250880, 4, 1552, 2, 784});
    }
}

// Lane group g reads chunk j of a row's K step at g * chunk_bytes + offsets[j], and a fragment
// takes at least `reads` ds_loads; the chunks of both lane groups cover every byte of the K step
// exactly once.
template <typename P>
void check_lane_chunks(index_t step_bytes,
                       index_t chunk_bytes,
                       const std::vector<index_t>& offsets,
                       index_t reads)
{
    using Policy         = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    const index_t chunks = static_cast<index_t>(offsets.size());
    EXPECT_EQ(Policy::GetStepBytes<P>(), step_bytes);
    EXPECT_EQ(Policy::GetChunkBytes<P>(), chunk_bytes);
    EXPECT_EQ(Policy::GetChunksPerLane<P>(), chunks);
    EXPECT_EQ(Policy::GetReadsPerFragment<P>(), reads);
    std::vector<index_t> bytes;
    for(index_t j = 0; j < chunks; ++j)
    {
        EXPECT_EQ(Policy::GetChunkOffset<P>(j), offsets[j]) << "chunk " << j;
        for(index_t g = 0; g < Policy::kLaneGroups; ++g)
            for(index_t b = 0; b < chunk_bytes; ++b)
                bytes.push_back(g * chunk_bytes + offsets[j] + b);
    }
    std::sort(bytes.begin(), bytes.end());
    ASSERT_EQ(static_cast<index_t>(bytes.size()), step_bytes);
    for(index_t i = 0; i < step_bytes; ++i)
        EXPECT_EQ(bytes[i], i);
}

// 16-byte chunks, one ds_load each: FP4's 64-byte K step in 2 chunks per lane at (0, 32), the
// 8-bit types' 128-byte K step in 4 at (0, 32, 64, 96). FP6's 96-byte K step in 2 chunks of 24 B
// at (0, 48), read as six 8 B reads: at least 3 ds_loads of at most 16 B.
TEST(MxGemmTdmScheduledHost, LaneChunksTileTheKStep)
{
    check_lane_chunks<ProblemOf<ck_tile::pk_fp4_t, ck_tile::fp8_t, 256, 256, 512, 2, 2, 32>>(
        64, 16, {0, 32}, 2);
    check_lane_chunks<ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>>(
        128, 16, {0, 32, 64, 96}, 4);
    check_lane_chunks<ProblemOf<ck_tile::bf8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>>(
        128, 16, {0, 32, 64, 96}, 4);
    check_lane_chunks<ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 64, 128, 512, 2, 2, 16>>(
        96, 24, {0, 48}, 3);
}

// FP6 A pads, the table of test_fp6_bank_spread in dispatcher/tests/test_mx_gemm_tdm_scheduled.py:
// 16 B after lcm(lane step, 256 B), where a lane's first bank comes round again, a lane step being
// tile_m / 16 rows (A's wave repeat with a 1x4 wave group) of 3 tile_k / 4 bytes. Each block is
// whole rows, loaded in iterate mode, block + 16 B apart.
template <index_t TileM, index_t TileK>
void check_fp6_bank_spread(index_t block)
{
    SCOPED_TRACE("fp6 " + std::to_string(TileM) + "x64x" + std::to_string(TileK));
    using P      = ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, TileM, 64, TileK, 1, 4, 16>;
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>;
    EXPECT_EQ(Policy::template Validate<P>(), nullptr);
    constexpr auto a = Policy::template GetLdsOperand<P, true>();
    EXPECT_EQ(a.iterate_rows, block / (TileK / 4 * 3));
    EXPECT_EQ(a.iterate_stride, block + 16);
}

TEST(MxGemmTdmScheduledHost, Fp6BankSpreadPads)
{
    check_fp6_bank_spread<16, 512>(768);
    check_fp6_bank_spread<32, 512>(768);
    check_fp6_bank_spread<64, 512>(1536);
    check_fp6_bank_spread<128, 512>(3072);
    check_fp6_bank_spread<256, 512>(6144);
    check_fp6_bank_spread<16, 256>(768);
    check_fp6_bank_spread<256, 256>(3072);
    check_fp6_bank_spread<16, 128>(768);
    check_fp6_bank_spread<256, 128>(1536);
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
using Shape256x256x512   = ck_tile::ClusterTileGemmShape<ck_tile::sequence<4, 4, 1>,
                                                         ck_tile::sequence<256, 256, 512>,
                                                         ck_tile::sequence<2, 2, 1>,
                                                         ck_tile::sequence<32, 16, 128>>;
using Problem256x256x512 = ck_tile::MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t,
                                                              ck_tile::pk_fp4_t,
                                                              ck_tile::fp8_t,
                                                              Shape256x256x512>;
using Shape128x16x512    = ck_tile::ClusterTileGemmShape<ck_tile::sequence<4, 1, 1>,
                                                         ck_tile::sequence<128, 16, 512>,
                                                         ck_tile::sequence<4, 1, 1>,
                                                         ck_tile::sequence<16, 16, 128>>;
using Problem128x16x512  = ck_tile::MxGemmTdmScheduledProblem<ck_tile::pk_fp4_t,
                                                              ck_tile::pk_fp4_t,
                                                              ck_tile::half_t,
                                                              Shape128x16x512>;

TEST(MxGemmTdmScheduledHost, PolicyDerivesTheScheduleFromProblemAndParameters)
{
    using Default = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    static_assert(Default::GetSmemSize<Problem256x256x512>() == 282624);
    static_assert(Default::GetReason<Problem256x256x512>() == MxTdmScheduledReason::Ok);
    static_assert(Default::Validate<Problem256x256x512>() == nullptr);
    // 3 scale reads and 2 chunks of 4 x 2 A rows and 8 B fragments.
    static_assert(Default::GetReadsPerStep<Problem256x256x512>() == 35);
    EXPECT_EQ((PipelineOf<Problem256x256x512, Default>::GetKernelName()),
              "mx_gemm_tdm_scheduled_mxfp4_t256x256x512_c4x4");
    EXPECT_EQ(Default::GetTuningName<Problem256x256x512>(), "_c4x4");

    using Pf2 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2>;
    EXPECT_EQ(Pf2::GetTuningName<Problem256x256x512>(), "_c4x4_pf2");

    using Pf0Lp0 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<0, false, false>;
    static_assert(Pf0Lp0::GetReason<Problem128x16x512>() == MxTdmScheduledReason::Ok);
    EXPECT_EQ((PipelineOf<Problem128x16x512, Pf0Lp0>::GetKernelName()),
              "mx_gemm_tdm_scheduled_mxfp4_t128x16x512_c1x4_fp16_wg4x1_pf0_lp0_hplrnone_mma16");
    EXPECT_EQ(Pf0Lp0::GetTuningName<Problem128x16x512>(), "_c1x4_pf0_lp0_hplrnone");

    // FP6 SM: 2 scale reads and 3 reads for each of 2 A and 4 B fragments.
    using Fp6Sm  = ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 64, 128, 512, 2, 2, 16>;
    using SmTune = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false>;
    EXPECT_EQ(SmTune::Validate<Fp6Sm>(), nullptr);
    EXPECT_EQ(SmTune::GetReadsPerStep<Fp6Sm>(), 20);
    EXPECT_EQ((PipelineOf<Fp6Sm, SmTune>::GetKernelName()),
              "mx_gemm_tdm_scheduled_mxfp6_t64x128x512_c4x4_bf16_pf2_hplrnone_lsi0_mma16");
    // With 3 LDS buffers: lb3 in the name, three buffers of 77824 B.
    using SmTune3 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false, 3>;
    EXPECT_EQ(SmTune3::Validate<Fp6Sm>(), nullptr);
    static_assert(SmTune3::GetSmemSize<Fp6Sm>() == 233472);
    EXPECT_EQ((PipelineOf<Fp6Sm, SmTune3>::GetKernelName()),
              "mx_gemm_tdm_scheduled_mxfp6_t64x128x512_c4x4_bf16_pf2_hplrnone_lsi0_lb3_mma16");

    // FP6 MAB: 2 scale reads and 3 reads for each of 4 A fragments and 1 B fragment.
    using Fp6Mab  = ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 256, 16, 512, 4, 1, 16, 1, 2>;
    using MabTune = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<0, false, false>;
    EXPECT_EQ(MabTune::Validate<Fp6Mab>(), nullptr);
    EXPECT_EQ(MabTune::GetReadsPerStep<Fp6Mab>(), 17);

    // 256x256x128 on 4x1 waves (4 x 16 repeats): 1 + 4 scale reads and 3 (FP6) or 4 (FP8) reads
    // for each of 4 A and 16 B fragments: 65 and 85, clamped to the s_wait_dscnt immediate, 63.
    using Fp6Clamped = ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 256, 256, 128, 4, 1, 16>;
    using Fp8Clamped = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 256, 256, 128, 4, 1, 16>;
    EXPECT_EQ(Default::Validate<Fp6Clamped>(), nullptr);
    EXPECT_EQ(Default::Validate<Fp8Clamped>(), nullptr);
    EXPECT_EQ(Default::GetReadsPerStep<Fp6Clamped>(), 63);
    EXPECT_EQ(Default::GetReadsPerStep<Fp8Clamped>(), 63);
}

// The GL2 prefetch's lines of rows that straddle 256 B lines (FP6: 96, 192 and 384 B at tile_k
// 128, 256 and 512): ceil(row / 256) lines per row whatever the tile's parity, counted back from
// each tile's last byte in the row. At K = tile_k and at longer K with partial last tiles, for
// each tile the pipeline prefetches (those at multiples of tile_k below K and, past K, the last
// tile_k values of K), line l is l lines before the line of the tile's last byte (the row's last
// byte for a partial tile), every offset is inside the row, and the tiles together reach every
// line of the row.
template <index_t TileK>
void check_gl2_lines(index_t lines)
{
    SCOPED_TRACE("fp6 tile_k " + std::to_string(TileK));
    using P      = ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 64, 128, TileK, 2, 2, 16>;
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false>;
    constexpr index_t line = Policy::kGl2LineBytes;
    const index_t row      = Policy::GetKBytes<P>(TileK);
    EXPECT_TRUE(Policy::IsGl2LineStraddle<P>());
    EXPECT_EQ((Policy::GetGl2DataLines<P, 0>()), lines);
    EXPECT_EQ((Policy::GetGl2DataLines<P, 1>()), lines);
    for(const index_t K : {TileK, index_t{1056}, index_t{1440}, index_t{1568}, index_t{2208}})
    {
        const index_t packed_k = Policy::GetKBytes<P>(K);
        std::set<index_t> tiles{K - TileK};
        for(index_t k0 = 0; k0 < K; k0 += TileK)
            tiles.insert(k0);
        std::set<index_t> reached;
        for(const index_t k0 : tiles)
        {
            const index_t k_byte    = Policy::GetKBytes<P>(k0);
            const index_t last_line = (std::min(k_byte + row, packed_k) - 1) / line;
            for(index_t l = 0; l < lines; ++l)
            {
                const index_t offset = Policy::GetGl2DataOffset<P>(k_byte, packed_k, l);
                EXPECT_GE(offset, 0) << K << " " << k0 << " " << l;
                EXPECT_LT(offset, packed_k) << K << " " << k0 << " " << l;
                EXPECT_EQ(offset / line, last_line - l) << K << " " << k0 << " " << l;
                reached.insert(offset / line);
            }
        }
        EXPECT_EQ(static_cast<index_t>(reached.size()), (packed_k + line - 1) / line) << K;
    }
}

TEST(MxGemmTdmScheduledHost, Gl2LinesOfStraddlingRows)
{
    check_gl2_lines<128>(1);
    check_gl2_lines<256>(1);
    check_gl2_lines<512>(2);
    // FP4 and 8-bit rows (64 to 512 B) hold whole lines or divide one: a row of whole lines takes
    // them all for every tile, a row of half a line or less one line for even tiles only (that
    // line holds the next tile too), and the lines go forward from the tile's first byte, clamped
    // to the row's last one: at K = 2048 + 256 (FP4 rows of 1152 B, tile_k 1024) the tile at row
    // byte 512 takes 512 and 768, the partial last tile at 1024 takes 1024 and 1151.
    using Policy   = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using Fp4K128  = ProblemOf<ck_tile::pk_fp4_t, ck_tile::fp8_t, 128, 128, 128, 2, 2, 32>;
    using Fp8K128  = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 128, 2, 2, 16>;
    using Fp4K1024 = ProblemOf<ck_tile::pk_fp4_t, ck_tile::fp8_t, 128, 128, 1024, 2, 2, 32>;
    EXPECT_FALSE(Policy::IsGl2LineStraddle<Problem256x256x512>());
    EXPECT_FALSE(Policy::IsGl2LineStraddle<Fp4K128>());
    EXPECT_FALSE(Policy::IsGl2LineStraddle<Fp8K128>());
    EXPECT_FALSE(Policy::IsGl2LineStraddle<Fp4K1024>());
    EXPECT_EQ((Policy::GetGl2DataLines<Fp4K128, 0>()), 1);
    EXPECT_EQ((Policy::GetGl2DataLines<Fp4K128, 1>()), 0);
    EXPECT_EQ((Policy::GetGl2DataLines<Fp8K128, 0>()), 1);
    EXPECT_EQ((Policy::GetGl2DataLines<Fp8K128, 1>()), 0);
    EXPECT_EQ((Policy::GetGl2DataLines<Problem256x256x512, 0>()), 1);
    EXPECT_EQ((Policy::GetGl2DataLines<Problem256x256x512, 1>()), 1);
    EXPECT_EQ((Policy::GetGl2DataLines<Fp4K1024, 0>()), 2);
    EXPECT_EQ((Policy::GetGl2DataLines<Fp4K1024, 1>()), 2);
    EXPECT_EQ(Policy::GetGl2DataOffset<Fp4K1024>(512, 1152, 0), 512);
    EXPECT_EQ(Policy::GetGl2DataOffset<Fp4K1024>(512, 1152, 1), 768);
    EXPECT_EQ(Policy::GetGl2DataOffset<Fp4K1024>(1024, 1152, 0), 1024);
    EXPECT_EQ(Policy::GetGl2DataOffset<Fp4K1024>(1024, 1152, 1), 1151);
}

// Bytes of K values of an A or B row: K / 2 for FP4, 3 K / 4 for FP6 and K for the 8-bit types,
// at 0, a K step, a K tail (2048 + 160), 2^28 and 2^31 - 1568 (3 K would pass i32 there).
// MxTdmScheduledSelectTraits sizes the rows the same way on the host: at M = N = 1024, A and B
// pass 2^31 bytes above K = 2^22 for FP4, above K = 2796192 (the last multiple of 32 at most
// 2^23 / 3) for FP6 and above K = 2^21 for the 8-bit types.
TEST(MxGemmTdmScheduledHost, RowBytesOfKValues)
{
    using ck_tile::long_index_t;
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using Fp4    = Problem256x256x512;
    using Fp8    = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>;
    using Bf8    = ProblemOf<ck_tile::bf8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>;
    using Fp6    = ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 64, 128, 512, 2, 2, 16>;
    for(const index_t k :
        {index_t{0}, index_t{128}, index_t{2048 + 160}, index_t{1} << 28, index_t{2147482080}})
    {
        EXPECT_EQ(Policy::GetKBytes<Fp4>(k), k / 2) << k;
        EXPECT_EQ(Policy::GetKBytes<Fp6>(k), k / 4 * 3) << k;
        EXPECT_EQ(Policy::GetKBytes<Fp8>(k), k) << k;
        EXPECT_EQ(Policy::GetKBytes<Bf8>(k), k) << k;
    }
    const auto wide = [](auto problem, long_index_t K) {
        const auto t = ck_tile::MxTdmScheduledSelectTraits<decltype(problem)>(1024, 1024, K);
        EXPECT_EQ(t.wide_a, t.wide_b) << K;
        return t.wide_a;
    };
    EXPECT_FALSE(wide(Fp4{}, long_index_t{1} << 22));
    EXPECT_TRUE(wide(Fp4{}, (long_index_t{1} << 22) + 512));
    EXPECT_FALSE(wide(Fp8{}, long_index_t{1} << 21));
    EXPECT_TRUE(wide(Fp8{}, (long_index_t{1} << 21) + 256));
    EXPECT_FALSE(wide(Bf8{}, long_index_t{1} << 21));
    EXPECT_TRUE(wide(Bf8{}, (long_index_t{1} << 21) + 256));
    EXPECT_FALSE(wide(Fp6{}, long_index_t{2796192}));
    EXPECT_TRUE(wide(Fp6{}, long_index_t{2796192} + 32));
}

// The FP6 MAB instance on 65536 x 16 x 65536: A is 3 GiB, past 2^31 bytes, so
// MxTdmScheduledSelectTraits picks WideA alone; the instance with WideA accepts the problem and
// the same instance without it rejects it.
TEST(MxGemmTdmScheduledHost, Fp6MabWideA)
{
    using F6     = ck_tile::pk_fp6x16_t;
    using WideA  = ck_tile::MxTdmScheduledGemmTraits<false, false, true>;
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<0, false, false>;
    using Wide =
        PipelineOf<ProblemOf<F6, ck_tile::bf16_t, 256, 16, 512, 4, 1, 16, 1, 2, WideA>, Policy>;
    using Narrow = PipelineOf<ProblemOf<F6, ck_tile::bf16_t, 256, 16, 512, 4, 1, 16, 1, 2>, Policy>;
    const auto t = ck_tile::MxTdmScheduledSelectTraits<typename Wide::Problem>(65536, 16, 65536);
    EXPECT_TRUE(t.wide_a);
    EXPECT_FALSE(t.wide_b);
    EXPECT_FALSE(t.wide_d);
    const char* why = Wide::GetUnsupportedReason(65536, 16, 65536);
    EXPECT_EQ(why, nullptr) << why;
    EXPECT_STREQ(Narrow::GetUnsupportedReason(65536, 16, 65536),
                 "A's offsets pass 2^31: needs a WideA instance");
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

// Wide A and B do not imply a wide E. E's element offset is still below the i32 limit here:
// 33024 x 33024 FP16 has 1.09e9 elements, although its byte size exceeds 2 GiB.
TEST(MxGemmTdmScheduledHost, MixedWideInputsNarrowOutputIsSupported)
{
    using ck_tile::MxGemmHostArgs;
    using ck_tile::pk_fp4_t;
    using Traits = ck_tile::MxTdmScheduledGemmTraits<false, false, true, true, false, 0>;
    using P      = ProblemOf<pk_fp4_t, ck_tile::half_t, 256, 256, 512, 2, 2, 32, 1, 1, Traits>;
    using Policy = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using Pipe   = PipelineOf<P, Policy>;
    using K =
        ck_tile::MxGemmKernel<ck_tile::GemmTile2DNMajorPartitioner<typename Pipe::BlockGemmShape>,
                              Pipe,
                              ck_tile::MxTdmStoreEpilogue<Pipe>>;

    constexpr index_t M    = 33024;
    constexpr index_t N    = 33024;
    constexpr index_t Kdim = 131072;
    static_assert(Traits::kWideA && Traits::kWideB && !Traits::kWideD);

    const auto args = K::MakeKernelArgs(MxGemmHostArgs<>{{nullptr},
                                                         {nullptr},
                                                         {nullptr},
                                                         {nullptr},
                                                         {},
                                                         nullptr,
                                                         1,
                                                         M,
                                                         N,
                                                         Kdim,
                                                         {Kdim},
                                                         {Kdim},
                                                         {},
                                                         N});
    const char* why = K::GetUnsupportedReason(args);
    EXPECT_EQ(why, nullptr) << (why ? why : "");
    if(ck_tile::is_gfx125_supported())
    {
        EXPECT_TRUE(K::IsSupportedArgument(args));
    }
}

// The host picks the kernel of a whole-tile count's parity: TailNumber::One for odd counts, Two
// for even ones, with the K tail excluded; each variant rejects the other parity, Full none.
TEST(MxGemmTdmScheduledHost, TailHandlerPicksTheVariantOfTheTileCount)
{
    using ck_tile::TailNumber;
    using Default = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>;
    using Full    = PipelineOf<Problem256x256x512, Default>;
    using One     = Full::WithTailNum<TailNumber::One>;
    using Two     = Full::WithTailNum<TailNumber::Two>;
    static_assert(Full::GetTailNumber(512) == TailNumber::One);
    static_assert(Full::GetTailNumber(1024) == TailNumber::Two);
    static_assert(Full::GetTailNumber(1536 + 128) == TailNumber::One);
    const auto picked = [](index_t K) {
        const index_t tiles = Full::GetFullTiles(K);
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

// With 3 LDS buffers the host picks the kernel of a whole-tile count modulo 3: TailNumber::One and
// Two for remainders 1 and 2, Three for multiples of 3, with the K tail excluded; each variant
// rejects the other counts, Full none.
TEST(MxGemmTdmScheduledHost, TailHandlerPicksTheVariantOfTheTileCountModuloThree)
{
    using ck_tile::TailNumber;
    using Sm3 = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false, 3>;
    using Full =
        PipelineOf<ProblemOf<ck_tile::pk_fp6x16_t, ck_tile::bf16_t, 64, 128, 512, 2, 2, 16>, Sm3>;
    using One   = Full::WithTailNum<TailNumber::One>;
    using Two   = Full::WithTailNum<TailNumber::Two>;
    using Three = Full::WithTailNum<TailNumber::Three>;
    static_assert(Full::PrefetchStages == 3);
    static_assert(Full::GetTailNumber(512) == TailNumber::One);
    static_assert(Full::GetTailNumber(1024) == TailNumber::Two);
    static_assert(Full::GetTailNumber(1536) == TailNumber::Three);
    static_assert(Full::GetTailNumber(2048 + 128) == TailNumber::One);
    const auto picked = [](index_t K) {
        const index_t tiles = Full::GetFullTiles(K);
        return Full::TailHandler([](auto, auto tail_num) { return decltype(tail_num)::value; },
                                 Full::BlockHasHotloop(tiles),
                                 Full::GetBlockLoopTailNum(tiles));
    };
#if !defined(CK_TILE_FORCE_SINGLE_TAIL_HANDLER)
    EXPECT_EQ(picked(512), TailNumber::One);
    EXPECT_EQ(picked(1024), TailNumber::Two);
    EXPECT_EQ(picked(1536), TailNumber::Three);
    EXPECT_EQ(picked(3584), TailNumber::One);
#else
    EXPECT_EQ(picked(512), TailNumber::Full);
#endif
    const std::string base =
        "mx_gemm_tdm_scheduled_mxfp6_t64x128x512_c4x4_bf16_pf2_hplrnone_lsi0_lb3_mma16";
    EXPECT_EQ(One::GetKernelName(), base + "_r1");
    EXPECT_EQ(Two::GetKernelName(), base + "_r2");
    EXPECT_EQ(Three::GetKernelName(), base + "_r0");
    for(index_t tiles = 1; tiles <= 7; ++tiles)
    {
        const index_t K = tiles * 512;
        EXPECT_EQ(One::GetUnsupportedReason(256, 2048, K) == nullptr, tiles % 3 == 1) << K;
        EXPECT_EQ(Two::GetUnsupportedReason(256, 2048, K) == nullptr, tiles % 3 == 2) << K;
        EXPECT_EQ(Three::GetUnsupportedReason(256, 2048, K) == nullptr, tiles % 3 == 0) << K;
        EXPECT_EQ(Full::GetUnsupportedReason(256, 2048, K), nullptr) << K;
    }
}

/// The code of a GetUnsupportedReason result: "ok" for nullptr, the reason whose message it is,
/// else the message (a shape path the traits lack).
const char* reason_code_of(const char* why)
{
    if(why == nullptr)
        return "ok";
    for(int r = 0; r <= static_cast<int>(MxTdmScheduledReason::LdsCapacity); ++r)
    {
        const auto reason = static_cast<MxTdmScheduledReason>(r);
        if(std::string(why) == ck_tile::mx_tdm_scheduled_reason_message(reason))
            return ck_tile::mx_tdm_scheduled_reason_code(reason);
    }
    return why;
}

template <typename P, typename Policy>
void check_instance(const char* name, const char* want)
{
    const char* got = Policy::template Validate<P>();
    EXPECT_STREQ(got ? got : "ok", want) << name;
}

template <typename P, typename Policy>
void check_problem(const char* name, index_t M, index_t N, index_t K, const char* want)
{
    EXPECT_STREQ(reason_code_of(PipelineOf<P, Policy>::GetUnsupportedReason(M, N, K)), want)
        << name;
}

// The rows of parity_cases.py, which dispatcher/tests/test_mx_gemm_tdm_scheduled.py checks
// against tile_engine's mx_tdm_scheduled.py: both rule sets give each row the same reason.
TEST(MxGemmTdmScheduledHost, ParityWithPythonSpec)
{
#include "parity_cases.inc"
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

// With 3 LDS buffers the refill looks 3 tiles ahead: at GL2 distance 2, (3 + 1) tiles of 256 take K
// to at most 2^31 - 1 - 1024, where 2 buffers' (2 + 1) tiles allow 2^31 - 1 - 768.
TEST(MxGemmTdmScheduledHost, KLimitThreeBuffers)
{
    using P     = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 32, 32, 256, 2, 2, 16, 1, 1>;
    using Lb2   = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false>;
    using Lb3   = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<2, true, false, false, 3>;
    using Pipe2 = PipelineOf<P, Lb2>;
    using Pipe3 = PipelineOf<P, Lb3>;
    const auto* k_long = ck_tile::mx_tdm_scheduled_reason_message(MxTdmScheduledReason::KLong);
    const char* below  = Pipe3::GetUnsupportedReason(32, 32, 2147482592);
    EXPECT_TRUE(below == nullptr || std::string(below) != k_long) << below;
    EXPECT_STREQ(Pipe3::GetUnsupportedReason(32, 32, 2147482624), k_long);
    const char* two = Pipe2::GetUnsupportedReason(32, 32, 2147482624);
    EXPECT_TRUE(two == nullptr || std::string(two) != k_long) << two;
}

// A K tail has at most tile_k / 128 - 1 K steps: a partial tile of every K step (K mod tile_k
// above tile_k - 128) runs in the loop as one more tile, on an instance without a K tail.
TEST(MxGemmTdmScheduledHost, LongKTailRunsInTheLoop)
{
    using ck_tile::TailNumber;
    using Policy      = ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<4, true, false>;
    using P256        = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16>;
    using P128        = ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 128, 2, 2, 16>;
    using Tile256     = PipelineOf<P256, Policy>;
    using Tile128     = PipelineOf<P128, Policy>;
    using Tail1Traits = ck_tile::MxTdmScheduledGemmTraits<false, false, false, false, false, 1>;
    using Tail1       = PipelineOf<
              ProblemOf<ck_tile::fp8_t, ck_tile::fp8_t, 128, 128, 256, 2, 2, 16, 4, 4, Tail1Traits>,
              Policy>;
    const auto tail_steps = [](auto problem, index_t K) {
        return ck_tile::MxTdmScheduledSelectTraits<decltype(problem)>(1024, 1024, K).tail_steps;
    };
    EXPECT_EQ(tail_steps(P256{}, 2048 + 96), 1);
    EXPECT_EQ(tail_steps(P256{}, 2048 + 128), 1);
    EXPECT_EQ(tail_steps(P256{}, 2048 + 160), 0);
    EXPECT_EQ(tail_steps(P256{}, 2048 + 224), 0);
    EXPECT_EQ(tail_steps(P128{}, 1024 + 32), 0);
    static_assert(Tile256::GetFullTiles(2048 + 128) == 8);
    static_assert(Tile256::GetFullTiles(2048 + 160) == 9);
    static_assert(Tile256::GetTailNumber(2048 + 160) == TailNumber::One);
    static_assert(Tile256::GetTailNumber(1792 + 160) == TailNumber::Two);
    static_assert(Tile128::GetFullTiles(1024 + 32) == 9);
    for(index_t K : {2048, 2048 + 160, 2048 + 224})
        EXPECT_EQ(Tile256::GetUnsupportedReason(1024, 1024, K), nullptr) << K;
    EXPECT_EQ(Tile128::GetUnsupportedReason(1024, 1024, 1024 + 32), nullptr);
    EXPECT_NE(Tail1::GetUnsupportedReason(1024, 1024, 2048 + 160), nullptr);
    EXPECT_EQ(Tail1::GetUnsupportedReason(1024, 1024, 2048 + 96), nullptr);
}

} // namespace
