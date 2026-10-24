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
 * @brief The arch-neutral unified-attention geometry: what another architecture's pack
 *        (gfx942, AICK-2544) inherits by supplying its own UnifiedAttentionArchTraits.
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

} // namespace

/// D128 4 warps x 16 rows, Hq 8 / Hkv 2 (block_q 16), batch 2 x Sq 512: total_q 1024,
/// qblocks 1024 / 16 + 2 = 66. kv-major on gfx950, 256 threads.
TEST(TestUnifiedAttentionGeometry, Gfx950IsKvMajorWave64)
{
    const auto g = unifiedAttention2DLaunchGeometry(
        GFX950_UNIFIED_ATTENTION_TRAITS, 4, 16, 8, 2, 512, 2, "k");
    EXPECT_EQ(g, (UnifiedAttentionLaunchGeometry{2, 66, 1, 256}));
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
    EXPECT_EQ(gfx950Unified2DBatchCapacity(511), unifiedAttentionBatchCapacity(511));
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

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
