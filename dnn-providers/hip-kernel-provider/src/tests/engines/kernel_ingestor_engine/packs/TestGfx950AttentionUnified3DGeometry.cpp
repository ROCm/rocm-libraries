// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"

/**
 * @file TestGfx950AttentionUnified3DGeometry.cpp
 * @brief The pure functions hipkernel:Gfx950AttentionUnified3D restates from its Python
 *        builders: geometry legality, the two launch grids, and the shared workspace.
 *
 * Every expectation is worked out by hand from the Python, so a C++ rule that drifts from
 * it fails here rather than as wrong numerics on a device. The workspace is pinned byte for
 * byte: the reduce kernel indexes the scratch by its own descriptors, and an offset that is
 * off by a section reads another array with no fault.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

int32_t int32At(const std::vector<unsigned char>& image, std::size_t offset, std::size_t index)
{
    int32_t value = 0;
    std::memcpy(&value, image.data() + offset + index * sizeof(int32_t), sizeof(value));
    return value;
}

} // namespace

// =============================================================================
// Geometry legality
// =============================================================================

TEST(TestGfx950AttentionUnified3DGeometry, AcceptsEveryShippedGeometry)
{
    for(const int64_t headSize : {64, 128, 256})
    {
        for(const int64_t segments : {16, 32, 64, 128})
        {
            EXPECT_TRUE(isSupportedGfx950Unified3DGeometry(headSize, 16, 8, 1, segments))
                << headSize << " / " << segments;
        }
    }
}

TEST(TestGfx950AttentionUnified3DGeometry, RefusesWhatTheBuildersRefuse)
{
    // head_size outside {64, 128, 256}; the reduce kernel needs it divisible by 64.
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(96, 16, 8, 1, 16));
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(32, 16, 8, 1, 16));
    // block_size outside {16, 32, 64}.
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 8, 8, 1, 16));
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 128, 8, 1, 16));
    // A group that does not divide BLOCK_M 16, or exceeds it, or does not divide at all.
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 16, 12, 2, 16));
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 16, 32, 1, 16));
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 16, 9, 2, 16));
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 16, 8, 0, 16));
    // Segments: at least one, and no more than the reduce kernel's LDS factor cache holds.
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 16, 8, 1, 0));
    EXPECT_FALSE(isSupportedGfx950Unified3DGeometry(128, 16, 8, 1, 4097));
    EXPECT_TRUE(isSupportedGfx950Unified3DGeometry(128, 16, 8, 1, 4096));
}

// =============================================================================
// Launch geometry
// =============================================================================

/// Segment grid (num_seqs, Hkv, num_segments) on the decode grid; reduce grid
/// (total_q, Hq, 1); one wave64 CTA each. B2, Hq 8 over Hkv 2, 16 segments.
TEST(TestGfx950AttentionUnified3DGeometry, MirrorsThePythonLaunchers)
{
    const auto geometry = gfx950Unified3DLaunchGeometry(8, 2, 16, 2, "k");
    Gfx950Unified3DLaunchGeometry expected;
    expected.segmentGridX = 2;
    expected.segmentGridY = 2;
    expected.segmentGridZ = 16;
    expected.reduceGridX = 2;
    expected.reduceGridY = 8;
    expected.blockX = 64;
    EXPECT_EQ(geometry, expected);
}

/// The axes are not interchangeable: swapping the head counts launches a reduce CTA per KV
/// head, leaving most query heads unreduced.
TEST(TestGfx950AttentionUnified3DGeometry, KeepsQueryAndKvHeadsOnTheirOwnAxes)
{
    const auto geometry = gfx950Unified3DLaunchGeometry(16, 1, 4, 5, "k");
    EXPECT_EQ(geometry.segmentGridY, 1U);
    EXPECT_EQ(geometry.reduceGridY, 16U);
    EXPECT_EQ(geometry.segmentGridX, 5U);
    EXPECT_EQ(geometry.reduceGridX, 5U);
    EXPECT_EQ(geometry.segmentGridZ, 4U);
}

TEST(TestGfx950AttentionUnified3DGeometry, RefusesALaunchItCannotDescribe)
{
    using hipdnn_plugin_sdk::HipdnnPluginException;
    EXPECT_THROW(gfx950Unified3DLaunchGeometry(8, 2, 16, 0, "k"), HipdnnPluginException);
    EXPECT_THROW(gfx950Unified3DLaunchGeometry(0, 2, 16, 2, "k"), HipdnnPluginException);
    EXPECT_THROW(gfx950Unified3DLaunchGeometry(8, 2, 0, 2, "k"), HipdnnPluginException);
    EXPECT_THROW(gfx950Unified3DLaunchGeometry(12, 2, 16, 2, "k"), HipdnnPluginException);
    EXPECT_THROW(gfx950Unified3DLaunchGeometry(8, 2, 16, int64_t{1} << 32, "k"),
                 HipdnnPluginException);
}

// =============================================================================
// Workspace
// =============================================================================

/// B2, Sq 1, Skv 256, block 16, Hq 8, D128, 16 segments:
///   block_tables     32 x int32 = 128 B   at 0
///   seq_lens          2 x int32 =   8 B   at 256
///   query_start_len   3 x int32 =  12 B   at 512
///   segm_output   2*8*16*128 f32 = 131072 B at 768
///   segm_max      2*8*16 f32     =   1024 B at 131840
///   segm_expsum   2*8*16 f32     =   1024 B at 132864
///   total 133888 -- the identity's 768 plus num_segments * B * Hq * (D + 2) * 4.
TEST(TestGfx950AttentionUnified3DGeometry, LaysOutTheWorkspaceByteForByte)
{
    const auto layout = gfx950Unified3DWorkspaceLayout(2, 1, 256, 16, 8, 128, 16);
    ASSERT_TRUE(layout.has_value());
    EXPECT_EQ(layout->identity.blocksPerSeq, 16);
    EXPECT_EQ(layout->identity.blockTablesOffset, 0U);
    EXPECT_EQ(layout->identity.seqLensOffset, 256U);
    EXPECT_EQ(layout->identity.queryStartLenOffset, 512U);
    EXPECT_EQ(layout->segmOutputOffset, 768U);
    EXPECT_EQ(layout->segmOutputBytes, 131072U);
    EXPECT_EQ(layout->segmMaxOffset, 131840U);
    EXPECT_EQ(layout->segmMaxBytes, 1024U);
    EXPECT_EQ(layout->segmExpsumOffset, 132864U);
    EXPECT_EQ(layout->segmExpsumBytes, 1024U);
    EXPECT_EQ(layout->totalBytes, 133888U);
    EXPECT_EQ(layout->totalBytes - layout->segmOutputOffset, 16U * 2U * 8U * (128U + 2U) * 4U);
}

/// The identity prefix is Unified2D's, exactly: both engines stage the same three arrays for
/// the same graph, so one cannot be fixed without the other.
TEST(TestGfx950AttentionUnified3DGeometry, SharesUnified2DsIdentityLayout)
{
    const auto layout = gfx950Unified3DWorkspaceLayout(3, 1, 1024, 32, 16, 64, 8);
    const auto identity = gfx950Unified2DWorkspaceLayout(3, 1024, 32);
    ASSERT_TRUE(layout.has_value());
    ASSERT_TRUE(identity.has_value());
    EXPECT_EQ(layout->identity, *identity);
    EXPECT_EQ(layout->segmOutputOffset, identity->totalBytes);
}

/// The scratch follows the segment count and the query heads, not the KV heads.
TEST(TestGfx950AttentionUnified3DGeometry, ScratchFollowsSegmentsAndQueryHeads)
{
    const auto base = gfx950Unified3DWorkspaceLayout(1, 1, 64, 16, 8, 128, 16);
    const auto moreSegments = gfx950Unified3DWorkspaceLayout(1, 1, 64, 16, 8, 128, 32);
    const auto moreHeads = gfx950Unified3DWorkspaceLayout(1, 1, 64, 16, 16, 128, 16);
    ASSERT_TRUE(base && moreSegments && moreHeads);
    EXPECT_EQ(moreSegments->segmOutputBytes, 2 * base->segmOutputBytes);
    EXPECT_EQ(moreHeads->segmOutputBytes, 2 * base->segmOutputBytes);
    EXPECT_EQ(moreSegments->segmMaxBytes, 2 * base->segmMaxBytes);
}

TEST(TestGfx950AttentionUnified3DGeometry, RefusesAWorkspaceItCannotLayOut)
{
    // A block that does not divide Skv.
    EXPECT_FALSE(gfx950Unified3DWorkspaceLayout(2, 1, 100, 16, 8, 128, 16).has_value());
    // Non-positive scratch dimensions.
    EXPECT_FALSE(gfx950Unified3DWorkspaceLayout(2, 0, 256, 16, 8, 128, 16).has_value());
    EXPECT_FALSE(gfx950Unified3DWorkspaceLayout(2, 1, 256, 16, 0, 128, 16).has_value());
    EXPECT_FALSE(gfx950Unified3DWorkspaceLayout(2, 1, 256, 16, 8, 128, 0).has_value());
    // segm_output past 2^31 - 1 elements, which the kernels index in 32 bits.
    EXPECT_FALSE(gfx950Unified3DWorkspaceLayout(4096, 1, 256, 16, 128, 256, 128).has_value());
}

/// The bytes launch() copies, for a batch above one: B3, Skv 64, block 16 is four blocks per
/// sequence. Sequence b owns blocks 4b..4b+3, every sequence has 64 keys, and with one query
/// token each query_start_len counts 0, 1, 2, 3. A table that restarted at 0 per sequence, or
/// a query_start_len stepping by Skv, would read the wrong sequence's keys with no fault.
TEST(TestGfx950AttentionUnified3DGeometry, StagesTheIdentityTablesForADecodeBatch)
{
    const auto layout = gfx950Unified3DWorkspaceLayout(3, 1, 64, 16, 8, 128, 16);
    const auto image = gfx950Unified3DWorkspaceImage(3, 1, 64, 16);
    ASSERT_TRUE(layout.has_value());
    ASSERT_TRUE(image.has_value());
    ASSERT_EQ(image->size(), layout->segmOutputOffset);

    for(std::size_t entry = 0; entry < 12; ++entry)
    {
        EXPECT_EQ(int32At(*image, layout->identity.blockTablesOffset, entry),
                  static_cast<int32_t>(entry));
    }
    for(std::size_t b = 0; b < 3; ++b)
    {
        EXPECT_EQ(int32At(*image, layout->identity.seqLensOffset, b), 64);
    }
    for(std::size_t b = 0; b <= 3; ++b)
    {
        EXPECT_EQ(int32At(*image, layout->identity.queryStartLenOffset, b),
                  static_cast<int32_t>(b));
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
