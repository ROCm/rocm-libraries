// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

namespace
{

constexpr const char* KERNEL_NAME = "hipkernel:Gfx950AttentionUnified2D/unit";

Gfx950AttentionUnified2DGeometry geometryFor(int64_t blockMPerWarp,
                                             int64_t numWarps,
                                             int64_t numQueryHeads,
                                             int64_t numKvHeads,
                                             int64_t seqLenQ,
                                             int64_t batch)
{
    return gfx950AttentionUnified2DGeometry(
        blockMPerWarp, numWarps, numQueryHeads, numKvHeads, seqLenQ, batch, KERNEL_NAME);
}

Gfx950AttentionUnified2DWorkspace
    workspaceFor(int64_t batch, int64_t seqLenQ, int64_t seqLenKv, int64_t blockSize)
{
    return gfx950AttentionUnified2DWorkspace(batch, seqLenQ, seqLenKv, blockSize, KERNEL_NAME);
}

} // namespace

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DGeometry — grid / block
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DGeometry, GridXIsNumKvHeads)
{
    // gridX = numKvHeads (the KV-head index is the outermost CTA loop dim)
    const auto g = geometryFor(16, 1, 32, 8, 512, 2);
    EXPECT_EQ(g.gridX, 8U);
}

TEST(TestGfx950AttentionUnified2DGeometry, GridYIsQBlocksFloorPlusBatch)
{
    // block_m = blockMPerWarp * numWarps = 16 * 1 = 16
    // num_queries_per_kv = 32 / 8 = 4
    // block_q = 16 / 4 = 4
    // total_q = 2 * 512 = 1024;  qblocks = 1024 / 4 + 2 = 258
    const auto g = geometryFor(16, 1, 32, 8, 512, 2);
    EXPECT_EQ(g.gridY, 258U);
}

TEST(TestGfx950AttentionUnified2DGeometry, GridZIsAlwaysOne)
{
    const auto g = geometryFor(16, 1, 32, 8, 512, 2);
    EXPECT_EQ(g.gridZ, 1U);
}

TEST(TestGfx950AttentionUnified2DGeometry, BlockXIsWaveLanesTimesNumWarps)
{
    // blockX = 64 * numWarps
    const auto g1 = geometryFor(16, 1, 32, 8, 512, 2);
    EXPECT_EQ(g1.blockX, 64U);
    const auto g2 = geometryFor(16, 4, 32, 8, 512, 2);
    EXPECT_EQ(g2.blockX, 256U);
}

TEST(TestGfx950AttentionUnified2DGeometry, MFQAHeadSizeOneQueriesPerKv)
{
    // num_queries_per_kv = 8/8 = 1; block_m = 16; block_q = 16/1 = 16
    // total_q = 1 * 1024 = 1024; qblocks = 1024/16 + 1 = 65
    const auto g = geometryFor(16, 1, 8, 8, 1024, 1);
    EXPECT_EQ(g.gridY, 65U);
    EXPECT_EQ(g.gridX, 8U);
}

TEST(TestGfx950AttentionUnified2DGeometry, GqaBlockQClampedToOneWhenNqkExceedsBlockM)
{
    // block_m = 16 * 1 = 16; num_queries_per_kv = 32/1 = 32 > 16 => block_q = 1
    // total_q = 4 * 256 = 1024; qblocks = 1024/1 + 4 = 1028
    const auto g = geometryFor(16, 1, 32, 1, 256, 4);
    EXPECT_EQ(g.gridY, 1028U);
}

TEST(TestGfx950AttentionUnified2DGeometry, NonPositiveBlockMPerWarpThrows)
{
    EXPECT_THROW(geometryFor(0, 1, 8, 8, 512, 1), hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(geometryFor(-1, 1, 8, 8, 512, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionUnified2DGeometry, NonPositiveNumWarpsThrows)
{
    EXPECT_THROW(geometryFor(16, 0, 8, 8, 512, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionUnified2DGeometry, NonPositiveSeqLenQThrows)
{
    EXPECT_THROW(geometryFor(16, 1, 8, 8, 0, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionUnified2DGeometry, NonDivisibleGqaThrows)
{
    // numQueryHeads % numKvHeads != 0
    EXPECT_THROW(geometryFor(16, 1, 7, 4, 512, 1), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DWorkspace — derived paged-attention layout
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DWorkspace, BlockTablesOffsetIsZero)
{
    const auto ws = workspaceFor(2, 512, 512, 16);
    EXPECT_EQ(ws.blockTablesOffset, 0);
}

TEST(TestGfx950AttentionUnified2DWorkspace, BlockTablesBytesIsCorrect)
{
    // batch=2, seqlenKv=512, blockSize=16 -> 32 pages per seq -> 2*32*4 = 256 bytes
    const auto ws = workspaceFor(2, 512, 512, 16);
    EXPECT_EQ(ws.blocksPerSequence, 32);
    EXPECT_EQ(ws.blockTablesBytes, 2 * 32 * 4);
}

TEST(TestGfx950AttentionUnified2DWorkspace, SectionsAre256ByteAligned)
{
    // seqLenKv=16, blockSize=16 -> 1 block/seq -> blockTablesBytes = batch*1*4
    // For batch=1: blockTablesBytes=4, seqLensOffset should be 256
    const auto ws = workspaceFor(1, 512, 16, 16);
    EXPECT_EQ(ws.seqLensOffset % 256, 0);
    EXPECT_EQ(ws.queryStartLenOffset % 256, 0);
    EXPECT_EQ(ws.totalBytes % 256, 0);
}

TEST(TestGfx950AttentionUnified2DWorkspace, SeqLensBytesIsOncePerSequence)
{
    // seq_lens[batch] -- one int32 per sequence
    const auto ws = workspaceFor(3, 512, 512, 16);
    EXPECT_EQ(ws.seqLensBytes, 3 * 4);
}

TEST(TestGfx950AttentionUnified2DWorkspace, QueryStartLenBytesIsBatchPlusOne)
{
    // query_start_len[batch+1]
    const auto ws = workspaceFor(3, 512, 512, 16);
    EXPECT_EQ(ws.queryStartLenBytes, 4 * 4);
}

TEST(TestGfx950AttentionUnified2DWorkspace, NonDivisibleSeqLenKvThrows)
{
    EXPECT_THROW(workspaceFor(2, 512, 513, 16), hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionUnified2DWorkspace, NonPositiveBatchThrows)
{
    EXPECT_THROW(workspaceFor(0, 512, 512, 16), hipdnn_plugin_sdk::HipdnnPluginException);
}

// ---------------------------------------------------------------------------
// TestGfx950AttentionUnified2DSearchBound — binary-search capacity
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionUnified2DSearchBound, NumSeqs256Gives9Iters)
{
    // ceil(log2(257)) = 9
    EXPECT_EQ(gfx950AttentionUnified2DSearchIterations(256), 9);
}

TEST(TestGfx950AttentionUnified2DSearchBound, MaxBatchForNumSeqs256Is511)
{
    // 2^9 - 1 = 511
    EXPECT_EQ(gfx950AttentionUnified2DMaxBatch(256), 511);
}

TEST(TestGfx950AttentionUnified2DSearchBound, NumSeqs0GivesFallback)
{
    EXPECT_EQ(gfx950AttentionUnified2DSearchIterations(0), GFX950_UNIFIED_2D_FALLBACK_SEARCH_ITERS);
}

TEST(TestGfx950AttentionUnified2DSearchBound, NumSeqs1Gives1Iter)
{
    // ceil(log2(2)) = 1; serves batch <= 1
    EXPECT_EQ(gfx950AttentionUnified2DSearchIterations(1), 1);
    EXPECT_EQ(gfx950AttentionUnified2DMaxBatch(1), 1);
}

TEST(TestGfx950AttentionUnified2DSearchBound, NumSeqs2Gives2Iters)
{
    // ceil(log2(3)) = 2; serves batch <= 3
    EXPECT_EQ(gfx950AttentionUnified2DSearchIterations(2), 2);
    EXPECT_EQ(gfx950AttentionUnified2DMaxBatch(2), 3);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
