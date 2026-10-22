// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"

/**
 * @file TestGfx942AttentionUnified3dGeometry.cpp
 * @brief Pins the gfx942 unified 3D launch geometry, workspace layout and tile rule
 *        against their Python originals.
 *
 * Every expected number is read off the Python (cited per test) and written as a
 * literal; deriving one from the expression under test would assert only that the
 * expression equals itself.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

// _get_3d_pipeline: seg grid (num_seqs [decode grid], num_kv_heads, num_segments),
// reduce grid (total_q, num_query_heads, 1), block 64 for both.
TEST(TestGfx942AttentionUnified3dGeometry, GridsMatchGet3dPipeline)
{
    const auto g = gfx942AttentionUnified3dGeometry(4, 32, 8, 16);
    EXPECT_EQ(g.segment, (Gfx942AttentionUnified3dLaunch{4, 8, 16, 64}));
    EXPECT_EQ(g.reduce, (Gfx942AttentionUnified3dLaunch{4, 32, 1, 64}));

    const auto one = gfx942AttentionUnified3dGeometry(1, 32, 8, 64);
    EXPECT_EQ(one.segment, (Gfx942AttentionUnified3dLaunch{1, 8, 64, 64}));
    EXPECT_EQ(one.reduce, (Gfx942AttentionUnified3dLaunch{1, 32, 1, 64}));
}

TEST(TestGfx942AttentionUnified3dGeometry, GridRefusesNonPositiveInputs)
{
    EXPECT_THROW(gfx942AttentionUnified3dGeometry(0, 32, 8, 16),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx942AttentionUnified3dGeometry(1, 32, 8, 0),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// _attention_3d_workspace_specs + the strided seqused_k [B] / cu_seqlens_q [B+1] int32
// metadata, 256-byte aligned. B=1, Hq=32, nseg=64, D=128:
//   segm_output 1*32*64*128*4 = 1,048,576 at 0
//   segm_max    1*32*64*4     =     8,192 at 1,048,576
//   segm_expsum                     8,192 at 1,056,768
//   seqused_k   1*4                         at 1,064,960
//   cu_seqlens_q 2*4                        at 1,065,216, end 1,065,224 -> 1,065,472
TEST(TestGfx942AttentionUnified3dGeometry, WorkspaceWorkedExample)
{
    const auto w = gfx942AttentionUnified3dWorkspace(1, 32, 64, 128);
    EXPECT_EQ(w.segmOutput, 0);
    EXPECT_EQ(w.segmMax, 1048576);
    EXPECT_EQ(w.segmExpsum, 1056768);
    EXPECT_EQ(w.sequsedK, 1064960);
    EXPECT_EQ(w.cuSeqlensQ, 1065216);
    EXPECT_EQ(w.totalBytes, 1065472);
}

// Unaligned scratch: B=3, Hq=1, nseg=3, D=64 -> 9 rows.
//   segm_output 9*64*4 = 2304 at 0; segm_max 36 at 2304; segm_expsum 36 at 2560;
//   seqused_k 12 at 2816; cu_seqlens_q 16 at 3072, end 3088 -> 3328.
TEST(TestGfx942AttentionUnified3dGeometry, WorkspaceAlignsEveryRegion)
{
    const auto w = gfx942AttentionUnified3dWorkspace(3, 1, 3, 64);
    EXPECT_EQ(w.segmMax, 2304);
    EXPECT_EQ(w.segmExpsum, 2560);
    EXPECT_EQ(w.sequsedK, 2816);
    EXPECT_EQ(w.cuSeqlensQ, 3072);
    EXPECT_EQ(w.totalBytes, 3328);
}

// _gfx942_3d_tile_size_override: block_size // 2 iff head_size >= 128 and
// block_size >= 32, else None (one block per tile).
TEST(TestGfx942AttentionUnified3dGeometry, TileOverrideTable)
{
    struct Row
    {
        int64_t d;
        int64_t block;
        std::optional<int64_t> override;
        int64_t tile;
    };
    const Row rows[] = {
        {64, 16, std::nullopt, 16},
        {64, 32, std::nullopt, 32},
        {64, 64, std::nullopt, 64},
        {128, 16, std::nullopt, 16},
        {128, 32, 16, 16},
        {128, 64, 32, 32},
        {256, 16, std::nullopt, 16},
        {256, 32, 16, 16},
        {256, 64, 32, 32},
    };
    for(const auto& r : rows)
    {
        SCOPED_TRACE(::testing::Message() << "D=" << r.d << " block=" << r.block);
        EXPECT_EQ(gfx942TileSizeOverride(r.d, r.block), r.override);
        EXPECT_EQ(gfx942ResolvedTileSize(r.d, r.block), r.tile);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
