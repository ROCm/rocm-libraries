// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGeometry.hpp"

/**
 * @file TestUnifiedAttentionGeometry.cpp
 * @brief The arch-neutral unified-attention geometry: what each architecture's pack
 *        inherits by supplying its own UnifiedAttentionArchTraits.
 *
 * The gfx950 values are pinned in TestGfx950AttentionUnified2DGeometry /
 * TestGfx950AttentionUnified3DGeometry; this suite pins what the traits change -- the wave
 * width, the grid order, the accepted warp counts -- and that the gfx950 wrappers are the
 * neutral functions at the gfx950 traits, so the two cannot drift. Literals are worked out
 * by hand from `AttentionTuningSpec.launch_grid`, not recomputed.
 *
 * No device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

/// A q-major, wave32 target that builds 1-4 warps only: everything the traits can vary.
constexpr UnifiedAttentionArchTraits Q_MAJOR_WAVE32{
    "gfxtest", 32, true, 0x7FFF0000, {16, 32, 64}, 8, 8, {1, 2, 4, 4}, {16, 32}};

/// Block sizes listed out of order, smallest last.
constexpr UnifiedAttentionArchTraits UNSORTED_BLOCKS{
    "gfxtest", 64, false, 0x7FFF0000, {64, 32, 8}, 16, 8, {1, 2, 4, 8}, {16, 32}};

/// Two block sizes in the three-slot array: the unused slot is zero.
constexpr UnifiedAttentionArchTraits TWO_BLOCKS{
    "gfxtest", 64, false, 0x7FFF0000, {16, 32}, 16, 8, {1, 2, 4, 8}, {16, 32}};

} // namespace

/// D128 4 warps x 16 rows, Hq 8 / Hkv 2 (block_q 16), batch 2 x Sq 512: total_q 1024,
/// qblocks 1024 / 16 + 2 = 66. kv-major on gfx950, 256 threads.
TEST(TestUnifiedAttentionGeometry, Gfx950IsKvMajorWave64)
{
    const auto g = unifiedAttention2DLaunchGeometry(
        GFX950_UNIFIED_ATTENTION_TRAITS, 4, 16, 8, 2, 512, 2, "k");
    EXPECT_EQ(g, (UnifiedAttentionLaunchGeometry{2, 66, 1, 256}));
}

/// gfx942 launches the same grid and CTA as gfx950: both are wave64 and kv-major.
TEST(TestUnifiedAttentionGeometry, Gfx942IsKvMajorWave64)
{
    const auto g = unifiedAttention2DLaunchGeometry(
        GFX942_UNIFIED_ATTENTION_TRAITS, 4, 16, 8, 2, 512, 2, "k");
    EXPECT_EQ(g, (UnifiedAttentionLaunchGeometry{2, 66, 1, 256}));
    EXPECT_EQ(GFX942_UNIFIED_ATTENTION_TRAITS.kvBufferBytes,
              GFX950_UNIFIED_ATTENTION_TRAITS.kvBufferBytes);
}

/// The smallest block size bounds which Skv any candidate can cut into whole blocks.
TEST(TestUnifiedAttentionGeometry, SmallestBlockSizeIsTheMinimumOfTheTraits)
{
    static_assert(unifiedAttentionSmallestBlockSize(GFX950_UNIFIED_ATTENTION_TRAITS) == 16);
    static_assert(unifiedAttentionSmallestBlockSize(GFX942_UNIFIED_ATTENTION_TRAITS) == 16);
    static_assert(unifiedAttentionSmallestBlockSize(UNSORTED_BLOCKS) == 8);
    static_assert(unifiedAttentionSmallestBlockSize(TWO_BLOCKS) == 16);
    EXPECT_EQ(unifiedAttentionSmallestBlockSize(GFX950_UNIFIED_ATTENTION_TRAITS),
              GFX950_UNIFIED_2D_BLOCK_SIZE_SMALL);
}

/// The unused slot of a two-size array is zero, and a zero is no block size: the predicate
/// must not admit it, or a caller could reach `tileSize % blockSize` with 0.
TEST(TestUnifiedAttentionGeometry, BlockSizePredicateRejectsTheUnusedZeroSlotAndNegatives)
{
    EXPECT_TRUE(isSupportedUnifiedAttentionBlockSize(TWO_BLOCKS, 16));
    EXPECT_TRUE(isSupportedUnifiedAttentionBlockSize(TWO_BLOCKS, 32));
    EXPECT_FALSE(isSupportedUnifiedAttentionBlockSize(TWO_BLOCKS, 64));
    EXPECT_FALSE(isSupportedUnifiedAttentionBlockSize(TWO_BLOCKS, 0));
    EXPECT_FALSE(isSupportedUnifiedAttentionBlockSize(TWO_BLOCKS, -16));
    EXPECT_FALSE(isSupportedUnifiedAttentionBlockSize(GFX942_UNIFIED_ATTENTION_TRAITS, 0));
    EXPECT_FALSE(isSupportedUnifiedAttentionBlockSize(GFX950_UNIFIED_ATTENTION_TRAITS, 0));
}

/// Only gfx950 ships LSE-on kernels: gfx942's graph match declines a stats output.
TEST(TestUnifiedAttentionGeometry, OnlyGfx950ServesAStatsOutput)
{
    static_assert(GFX950_UNIFIED_ATTENTION_TRAITS.servesStats);
    static_assert(!GFX942_UNIFIED_ATTENTION_TRAITS.servesStats);
    EXPECT_TRUE(Q_MAJOR_WAVE32.servesStats);
}

/// The same launch on a q-major wave32 target: the axes swap and the CTA halves.
TEST(TestUnifiedAttentionGeometry, TraitsSwapTheGridAndSizeTheBlock)
{
    const auto g = unifiedAttention2DLaunchGeometry(Q_MAJOR_WAVE32, 4, 16, 8, 2, 512, 2, "k");
    EXPECT_EQ(g, (UnifiedAttentionLaunchGeometry{66, 2, 1, 128}));
}

/// A warp count one target builds and another does not, refused with the target's name.
TEST(TestUnifiedAttentionGeometry, TraitsDecideTheWarpCountsALaunchMayDeclare)
{
    EXPECT_NO_THROW(unifiedAttention2DLaunchGeometry(
        GFX950_UNIFIED_ATTENTION_TRAITS, 8, 16, 8, 2, 512, 2, "k"));
    try
    {
        unifiedAttention2DLaunchGeometry(Q_MAJOR_WAVE32, 8, 16, 8, 2, 512, 2, "k");
        FAIL() << "num_warps 8 is not a gfxtest geometry";
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
    {
        EXPECT_NE(std::string(e.what()).find("gfxtest"), std::string::npos) << e.what();
    }
}

/// The gfx950 wrappers are the neutral functions at the gfx950 traits.
TEST(TestUnifiedAttentionGeometry, Gfx950WrappersAreTheNeutralFunctions)
{
    EXPECT_EQ(gfx950Unified2DLaunchGeometry(2, 32, 16, 4, 300, 3, "k"),
              unifiedAttention2DLaunchGeometry(
                  GFX950_UNIFIED_ATTENTION_TRAITS, 2, 32, 16, 4, 300, 3, "k"));
    EXPECT_EQ(gfx950Unified2DWorkspaceLayout(3, 512, 16),
              unifiedAttentionWorkspaceLayout(3, 512, 16));
    EXPECT_EQ(gfx950Unified2DWorkspaceImage(3, 7, 512, 16),
              unifiedAttentionWorkspaceImage(3, 7, 512, 16));
    EXPECT_EQ(gfx950Unified3DWorkspaceLayout(4, 1, 1024, 16, 32, 128, 16),
              unifiedAttention3DWorkspaceLayout(4, 1, 1024, 16, 32, 128, 16));
    EXPECT_EQ(GFX950_UNIFIED_3D_CTA_THREADS, 64U);
}

/// The 3D scratch follows the 2D identity prefix: batch 4, Skv 1024 at block 16 gives a
/// 64 x 4 int32 block table (1024 B), seq_lens 16 B -> 256, query_start_len 20 B -> 256;
/// segm_output 4 x 32 x 16 x 128 f32 = 1 MiB at offset 1536.
TEST(TestUnifiedAttentionGeometry, TheSplitKvScratchFollowsTheIdentityPrefix)
{
    const auto layout = unifiedAttention3DWorkspaceLayout(4, 1, 1024, 16, 32, 128, 16);
    ASSERT_TRUE(layout.has_value());
    EXPECT_EQ(layout->identity, *unifiedAttentionWorkspaceLayout(4, 1024, 16));
    EXPECT_EQ(layout->segmOutputOffset, 1536U);
    EXPECT_EQ(layout->segmOutputBytes, 1048576U);
    EXPECT_EQ(layout->segmMaxOffset, 1536U + 1048576U);
    EXPECT_EQ(layout->segmMaxBytes, 8192U);
}

/// The kernels index segm_output [total_q, Hq, nseg, D] with i32 element offsets. A tensor
/// of N elements reaches offset N - 1, so exactly 2^31 elements still addresses and one more
/// does not. Batch 1, Sq 16384, Hq 32, D 256, 16 segments is 16384 * 32 * 16 * 256 = 2^31.
/// The layout takes any Sq; the shape the gfx950 matcher serves (Sq 1) is the next test.
TEST(TestUnifiedAttentionGeometry, TheSplitKvScratchAdmitsTwoToThe31ElementsAndRefusesOneMore)
{
    EXPECT_TRUE(unifiedAttention3DWorkspaceLayout(1, 16384, 1024, 16, 32, 256, 16).has_value());
    EXPECT_FALSE(unifiedAttention3DWorkspaceLayout(1, 16385, 1024, 16, 32, 256, 16).has_value());
}

/// The same boundary at the shape the gfx950 matcher serves: one query token per sequence,
/// so the batch carries the size. Batch 16384, Hq 32, D 256, 16 segments is 2^31 elements.
TEST(TestUnifiedAttentionGeometry, TheSplitKvScratchAdmitsTwoToThe31ElementsAtSqOne)
{
    EXPECT_TRUE(unifiedAttention3DWorkspaceLayout(16384, 1, 1024, 16, 32, 256, 16).has_value());
    EXPECT_FALSE(unifiedAttention3DWorkspaceLayout(16385, 1, 1024, 16, 32, 256, 16).has_value());
}

/// 274177 * 67280421310721 = 2^64 + 1, which wraps to exactly 1 in 64 bits: the true scratch
/// is far past the limit, and a product taken without an overflow check would not see it.
TEST(TestUnifiedAttentionGeometry, TheSplitKvScratchRefusesAProductThatWrapsToOne)
{
    EXPECT_FALSE(
        unifiedAttention3DWorkspaceLayout(274177, 67280421310721, 1024, 16, 1, 1, 1).has_value());
    EXPECT_FALSE(
        unifiedAttention3DWorkspaceLayout(1, 274177, 1024, 16, 67280421310721, 1, 1).has_value());
    EXPECT_FALSE(
        unifiedAttention3DWorkspaceLayout(1, 1, 1024, 16, 274177, 1, 67280421310721).has_value());
    EXPECT_FALSE(
        unifiedAttention3DWorkspaceLayout(1, 1, 1024, 16, 1, 274177, 67280421310721).has_value());
}

/// The shared helper places the three f32 sections, each on a 256-byte boundary: total_q 4,
/// Hq 32, 16 segments, D 128 is 2048 rows, so segm_output is 2048 * 128 * 4 B = 1 MiB and
/// segm_max and segm_expsum are 2048 * 4 B = 8 KiB each.
TEST(TestUnifiedAttentionGeometry, SegmentScratchLayoutPlacesTheThreeSections)
{
    const auto s = unifiedAttentionSegmentScratchLayout(256, 4, 32, 16, 128);
    ASSERT_TRUE(s.has_value());
    EXPECT_EQ(s->segmOutputOffset, 256U);
    EXPECT_EQ(s->segmOutputBytes, 1048576U);
    EXPECT_EQ(s->segmMaxOffset, 256U + 1048576U);
    EXPECT_EQ(s->segmMaxBytes, 8192U);
    EXPECT_EQ(s->segmExpsumOffset, 256U + 1048576U + 8192U);
    EXPECT_EQ(s->segmExpsumBytes, 8192U);
    EXPECT_EQ(s->endOffset, 256U + 1048576U + 8192U + 8192U);
}

/// The helper owns the bound, so a pack cannot lay out a scratch without it. Exactly 2^31
/// elements addresses (largest offset 2^31 - 1); one more row does not.
TEST(TestUnifiedAttentionGeometry, SegmentScratchLayoutRefusesWhatTheKernelsCannotIndex)
{
    EXPECT_TRUE(unifiedAttentionSegmentScratchLayout(0, 16384, 32, 16, 256).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 16385, 32, 16, 256).has_value());
    // Every dimension is a factor: growing any one of them alone crosses the bound.
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 16384, 33, 16, 256).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 16384, 32, 17, 256).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 16384, 32, 16, 257).has_value());
    // One element past 2^31: 3 * 715827883 = 2^31 + 1.
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 3, 715827883, 1, 1).has_value());
}

TEST(TestUnifiedAttentionGeometry, SegmentScratchLayoutRefusesNonPositiveAndWrappingDimensions)
{
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 0, 32, 16, 128).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, -1, 32, 16, 128).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 4, 0, 16, 128).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 4, -1, 16, 128).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 4, 32, 0, 128).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 4, 32, -1, 128).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 4, 32, 16, 0).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 4, 32, 16, -1).has_value());
    // 274177 * 67280421310721 = 2^64 + 1 wraps to 1 in each position a product can take.
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 274177, 67280421310721, 1, 1).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 1, 274177, 67280421310721, 1).has_value());
    EXPECT_FALSE(unifiedAttentionSegmentScratchLayout(0, 1, 1, 274177, 67280421310721).has_value());
}

/// The predicate a matcher can call without laying anything out holds the same rule.
TEST(TestUnifiedAttentionGeometry, SegmentScratchFitsI32IsTheLayoutsBound)
{
    EXPECT_TRUE(unifiedAttentionSegmentScratchFitsI32(16384, 32, 16, 256));
    EXPECT_FALSE(unifiedAttentionSegmentScratchFitsI32(16385, 32, 16, 256));
    EXPECT_FALSE(unifiedAttentionSegmentScratchFitsI32(0, 32, 16, 256));
    EXPECT_FALSE(unifiedAttentionSegmentScratchFitsI32(274177, 67280421310721, 1, 1));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
