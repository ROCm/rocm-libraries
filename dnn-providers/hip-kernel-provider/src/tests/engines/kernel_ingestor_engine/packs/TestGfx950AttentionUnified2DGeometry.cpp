// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"

/**
 * @file TestGfx950AttentionUnified2DGeometry.cpp
 * @brief Pins what hipkernel:Gfx950AttentionUnified2D restates from its Python builder: the
 *        geometry rules, the launch grid, the batch capacity and the paged-KV metadata.
 *
 * Every expectation is a literal worked out from the Python (attention_tiled_2d.py,
 * `AttentionTuningSpec.launch_grid`), not computed the way the header computes it: a helper
 * repeating the formula would assert only that the two agree. A mismatch here is not a fault
 * on the device -- the kernel computes something else, or reads another sequence's lengths.
 *
 * No device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using Fields = Gfx950Unified2DGeometryFields;

/// The shipped baseline: D128, block 16, tile 64, 4 warps x 16 rows, narrow 16x16 MFMA.
Fields baseline()
{
    return Fields{128, 16, 64, 4, 16, false, false};
}

/// The int32 at element @p index of the section starting at byte @p offset of @p image.
int32_t int32At(const std::vector<unsigned char>& image, std::size_t offset, std::size_t index)
{
    int32_t value = 0;
    std::memcpy(&value, image.data() + offset + index * sizeof(int32_t), sizeof(value));
    return value;
}

} // namespace

// =============================================================================
// Geometry rules
// =============================================================================

/// The positive control every refusal below depends on. Worked out from the Python at 4
/// warps (256 threads): D128 admits every dispatcher tile at every block size, and D64 loses
/// only block 16 / tile 16, whose slab (16 * 64 = 1024 elements) is below the 4 * 64 * 8 =
/// 2048 one DMA call carries.
TEST(TestGfx950AttentionUnified2DGeometry, AcceptsTheShippedBaselineAndTheDispatcherTiles)
{
    EXPECT_TRUE(isSupportedGfx950Unified2DGeometry(baseline()));

    for(const int64_t headSize : {64, 128})
    {
        for(const int64_t blockSize : {16, 32, 64})
        {
            for(const int64_t multiplier : {1, 2, 4, 8})
            {
                auto fields = baseline();
                fields.headSize = headSize;
                fields.blockSize = blockSize;
                fields.tileSize = blockSize * multiplier;
                const bool slabTooSmall = headSize == 64 && fields.tileSize == 16;
                EXPECT_EQ(isSupportedGfx950Unified2DGeometry(fields), !slabTooSmall)
                    << "head " << headSize << " block " << blockSize << " tile " << fields.tileSize;
            }
        }
    }
}

TEST(TestGfx950AttentionUnified2DGeometry, RefusesAnUnbuiltWarpCountOrRowCount)
{
    for(const int64_t warps : {0, 3, 5, 6, 7, 9, 16, -4})
    {
        auto fields = baseline();
        fields.numWarps = warps;
        EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields)) << "num_warps " << warps;
    }
    for(const int64_t rows : {0, 8, 24, 48, -16})
    {
        auto fields = baseline();
        fields.blockMPerWarp = rows;
        EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields)) << "block_m_per_warp " << rows;
    }
}

/// 32 rows per warp at 8 warps would be 512 rows and a CTA the 1024-thread cap cannot hold:
/// the Python raises, and a catalog record asking for it is malformed.
TEST(TestGfx950AttentionUnified2DGeometry, RefusesThirtyTwoRowsAtEightWarps)
{
    auto fields = baseline();
    fields.numWarps = 8;
    fields.blockMPerWarp = 32;
    EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields));
    fields.numWarps = 4;
    EXPECT_TRUE(isSupportedGfx950Unified2DGeometry(fields));
}

TEST(TestGfx950AttentionUnified2DGeometry, RefusesABlockOrTileTheKernelIsNotBuiltFor)
{
    for(const int64_t blockSize : {0, 8, 24, 48, 128, -16})
    {
        auto fields = baseline();
        fields.blockSize = blockSize;
        fields.tileSize = 64;
        EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields)) << "block_size " << blockSize;
    }
    // Not a multiple of the block, non-positive, or past the dispatcher's 8x policy.
    for(const int64_t tileSize : {0, -64, 24, 70, 144})
    {
        auto fields = baseline();
        fields.tileSize = tileSize;
        EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields)) << "tile_size " << tileSize;
    }
}

/// head_size must be a positive multiple of 32 (`supports_tiled_2d`); the engine ships 64 and
/// 128 and the matcher narrows to those, but a record outside the kernel's own range is
/// malformed before that.
TEST(TestGfx950AttentionUnified2DGeometry, RefusesAHeadSizeOutsideTheKernelsRange)
{
    for(const int64_t headSize : {0, -64, 16, 48, 100})
    {
        auto fields = baseline();
        fields.headSize = headSize;
        EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields)) << "head_size " << headSize;
    }
}

/// `tile_size * head_size >= threads * 8`: one DMA call carries 8 halves per thread, and a
/// slab smaller than that under-fills and corrupts the partial buffer.
TEST(TestGfx950AttentionUnified2DGeometry, RefusesATileTooSmallForTheCtaWidth)
{
    auto fields = baseline();
    fields.headSize = 64;
    fields.blockSize = 16;
    fields.tileSize = 16; // 1024 elements ...
    fields.numWarps = 4; // ... against 4 * 64 * 8 = 2048
    EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields));
    fields.numWarps = 2; // 1024 needed
    EXPECT_TRUE(isSupportedGfx950Unified2DGeometry(fields));
}

TEST(TestGfx950AttentionUnified2DGeometry, TheThirtyTwoByThirtyTwoPathsKeepTheirPreconditions)
{
    Fields fields{128, 16, 64, 4, 32, true, false};
    EXPECT_TRUE(isSupportedGfx950Unified2DGeometry(fields));

    // 32x32 needs 32 rows per warp.
    fields.blockMPerWarp = 16;
    EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(fields));

    // The transposed QK variant exists only on top of 32x32.
    const Fields transposedAlone{128, 16, 64, 4, 32, false, true};
    EXPECT_FALSE(isSupportedGfx950Unified2DGeometry(transposedAlone));
    const Fields transposed{128, 16, 64, 4, 32, true, true};
    EXPECT_TRUE(isSupportedGfx950Unified2DGeometry(transposed));
}

// =============================================================================
// Batch capacity
// =============================================================================

/// The kernel bisects [0, num_seqs] in `iters` unrolled steps, which resolves 2^iters - 1
/// sequences. Table values are ceil(log2(num_seqs + 1)) worked out by hand.
TEST(TestGfx950AttentionUnified2DGeometry, BinarySearchItersIsCeilLogTwoOfNumSeqsPlusOne)
{
    const std::vector<std::pair<int64_t, int64_t>> table{{1, 1},
                                                         {2, 2},
                                                         {3, 2},
                                                         {4, 3},
                                                         {7, 3},
                                                         {8, 4},
                                                         {15, 4},
                                                         {16, 5},
                                                         {255, 8},
                                                         {256, 9},
                                                         {511, 9},
                                                         {512, 10},
                                                         {1023, 10},
                                                         {1024, 11}};
    for(const auto& [numSeqs, iters] : table)
    {
        EXPECT_EQ(gfx950Unified2DBinarySearchIters(numSeqs), iters) << "num_seqs " << numSeqs;
    }
}

/// A spec that declares no num_seqs gets the kernel's conservative 32 iterations.
TEST(TestGfx950AttentionUnified2DGeometry, AnUndeclaredNumSeqsIsTheThirtyTwoIterationFallback)
{
    EXPECT_EQ(gfx950Unified2DBinarySearchIters(0), 32);
    EXPECT_EQ(gfx950Unified2DBinarySearchIters(-7), 32);
}

/// Capacity is 2^iters - 1, so the baked 511 serves exactly 511 and not 512. The boundary is
/// the whole point: one past it reads another sequence's seq_lens with no fault.
TEST(TestGfx950AttentionUnified2DGeometry, BatchCapacityIsTheLargestBatchTheSearchResolves)
{
    EXPECT_EQ(gfx950Unified2DBatchCapacity(511), 511);
    EXPECT_EQ(gfx950Unified2DBatchCapacity(512), 1023);
    EXPECT_EQ(gfx950Unified2DBatchCapacity(1), 1);
    EXPECT_EQ(gfx950Unified2DBatchCapacity(2), 3);
    EXPECT_EQ(gfx950Unified2DBatchCapacity(3), 3);
    EXPECT_EQ(gfx950Unified2DBatchCapacity(4), 7);
}

// =============================================================================
// Launch geometry
// =============================================================================

/// Hq 8 over Hkv 1 is 8 queries per KV head; block_m is 4 * 16 = 64, so block_q is 8 and each
/// CTA covers 8 query tokens. 2 * 512 = 1024 tokens -> 128 blocks, plus one per sequence for
/// the partial blocks the floor drops: 130.
TEST(TestGfx950AttentionUnified2DGeometry, GridIsKvHeadsByQueryBlocksPlusOnePerSequence)
{
    const auto geometry = gfx950Unified2DLaunchGeometry(4, 16, 8, 1, 512, 2, "k");
    EXPECT_EQ(geometry.gridX, 1U);
    EXPECT_EQ(geometry.gridY, 130U);
    EXPECT_EQ(geometry.gridZ, 1U);
    EXPECT_EQ(geometry.blockX, 256U);
}

/// MHA at block_m 64 has block_q 64: 300 tokens floor to 4 blocks, +3 sequences = 7.
TEST(TestGfx950AttentionUnified2DGeometry, TheQueryBlockCountFloorsAndAddsTheBatch)
{
    const auto geometry = gfx950Unified2DLaunchGeometry(4, 16, 4, 4, 100, 3, "k");
    EXPECT_EQ(geometry.gridX, 4U);
    EXPECT_EQ(geometry.gridY, 7U);
}

/// A group of 7 does not divide block_m: block_q is the floor, 64 // 7 = 9.
TEST(TestGfx950AttentionUnified2DGeometry, ANonPowerOfTwoGroupUsesTheFlooredBlockQ)
{
    // Hq 28 over Hkv 4 is a group of 7; 512 tokens // 9 = 56, + 1 sequence.
    const auto geometry = gfx950Unified2DLaunchGeometry(4, 16, 28, 4, 512, 1, "k");
    EXPECT_EQ(geometry.gridX, 4U);
    EXPECT_EQ(geometry.gridY, 57U);
}

TEST(TestGfx950AttentionUnified2DGeometry, BlockIsSixtyFourLanesPerWarp)
{
    for(const auto& [warps, threads] :
        std::vector<std::pair<int64_t, unsigned>>{{1, 64}, {2, 128}, {4, 256}, {8, 512}})
    {
        EXPECT_EQ(gfx950Unified2DLaunchGeometry(warps, 16, 8, 8, 64, 1, "k").blockX, threads)
            << "num_warps " << warps;
    }
}

/// block_m is rows per warp times warps, so 2 warps x 32 rows covers as many tokens per CTA as
/// 4 warps x 16: the same grid, half the threads.
TEST(TestGfx950AttentionUnified2DGeometry, RowsPerWarpEnterThroughBlockMNotTheThreadCount)
{
    const auto narrow = gfx950Unified2DLaunchGeometry(4, 16, 8, 1, 512, 2, "k");
    const auto stacked = gfx950Unified2DLaunchGeometry(2, 32, 8, 1, 512, 2, "k");
    EXPECT_EQ(narrow.gridY, stacked.gridY);
    EXPECT_EQ(stacked.blockX, 128U);
}

TEST(TestGfx950AttentionUnified2DGeometry, ALaunchThatCouldNotBeDescribedThrowsByName)
{
    const auto throws = [](int64_t warps,
                           int64_t rows,
                           int64_t heads,
                           int64_t kvHeads,
                           int64_t seqLenQ,
                           int64_t batch) {
        try
        {
            gfx950Unified2DLaunchGeometry(warps, rows, heads, kvHeads, seqLenQ, batch, "kernel-x");
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
        {
            return std::string(error.what()).find("kernel-x") != std::string::npos;
        }
        return false;
    };

    EXPECT_TRUE(throws(3, 16, 8, 8, 64, 1)) << "num_warps 3";
    EXPECT_TRUE(throws(4, 24, 8, 8, 64, 1)) << "block_m_per_warp 24";
    EXPECT_TRUE(throws(4, 16, 0, 8, 64, 1)) << "no query heads";
    EXPECT_TRUE(throws(4, 16, 8, 0, 64, 1)) << "no kv heads";
    EXPECT_TRUE(throws(4, 16, 8, 8, 0, 1)) << "no query tokens";
    EXPECT_TRUE(throws(4, 16, 8, 8, 64, 0)) << "no sequences";
    EXPECT_TRUE(throws(4, 16, 9, 2, 64, 1)) << "a group of 4.5";
    EXPECT_TRUE(throws(4, 16, 34, 2, 64, 1)) << "a group of 17";
    EXPECT_TRUE(throws(4, 16, 8, 8, std::numeric_limits<int64_t>::max(), 2)) << "overflow";
    // A grid past 2^31 - 1 blocks: 2^40 tokens // 64 + 1.
    EXPECT_TRUE(throws(4, 16, 8, 8, int64_t{1} << 40, 1)) << "a grid wider than 32 bits";
}

// =============================================================================
// Paged-KV metadata
// =============================================================================

/// B2, Skv 96, block 16: 6 blocks a sequence, a 12-entry table (48 bytes). seq_lens starts at
/// the next 256 boundary, query_start_len 256 bytes after it, and the total is three 256-byte
/// sections.
TEST(TestGfx950AttentionUnified2DGeometry, EachSectionStartsOnA256ByteBoundary)
{
    const auto layout = gfx950Unified2DWorkspaceLayout(2, 96, 16);
    ASSERT_TRUE(layout.has_value());
    EXPECT_EQ(layout->blocksPerSeq, 6);
    EXPECT_EQ(layout->blockTablesOffset, 0U);
    EXPECT_EQ(layout->blockTablesBytes, 48U);
    EXPECT_EQ(layout->seqLensOffset, 256U);
    EXPECT_EQ(layout->seqLensBytes, 8U);
    EXPECT_EQ(layout->queryStartLenOffset, 512U);
    EXPECT_EQ(layout->queryStartLenBytes, 12U);
    EXPECT_EQ(layout->totalBytes, 768U);
}

/// A table that fills its section exactly does not push the next one a section further, and
/// one byte over does: B3, Skv 1024, block 16 is 192 entries = 768 bytes = three sections.
TEST(TestGfx950AttentionUnified2DGeometry, ALargeTableMovesTheFollowingSections)
{
    const auto layout = gfx950Unified2DWorkspaceLayout(3, 1024, 16);
    ASSERT_TRUE(layout.has_value());
    EXPECT_EQ(layout->blockTablesBytes, 768U);
    EXPECT_EQ(layout->seqLensOffset, 768U);
    EXPECT_EQ(layout->queryStartLenOffset, 1024U);
    EXPECT_EQ(layout->totalBytes, 1280U);

    // 65 entries is 260 bytes: past one section, so seq_lens starts at 512.
    const auto over = gfx950Unified2DWorkspaceLayout(1, int64_t{65} * 16, 16);
    ASSERT_TRUE(over.has_value());
    EXPECT_EQ(over->blockTablesBytes, 260U);
    EXPECT_EQ(over->seqLensOffset, 512U);
}

TEST(TestGfx950AttentionUnified2DGeometry, NoLayoutExistsWhenTheCacheIsNotWholeBlocks)
{
    EXPECT_FALSE(gfx950Unified2DWorkspaceLayout(2, 100, 16).has_value()) << "100 % 16";
    EXPECT_FALSE(gfx950Unified2DWorkspaceLayout(2, 96, 64).has_value()) << "96 % 64";
    EXPECT_TRUE(gfx950Unified2DWorkspaceLayout(2, 96, 32).has_value());
    EXPECT_FALSE(gfx950Unified2DWorkspaceLayout(0, 96, 16).has_value());
    EXPECT_FALSE(gfx950Unified2DWorkspaceLayout(2, 0, 16).has_value());
    EXPECT_FALSE(gfx950Unified2DWorkspaceLayout(2, 96, 0).has_value());
    EXPECT_FALSE(gfx950Unified2DWorkspaceLayout(-1, 96, 16).has_value());
}

/// The kernel computes `num_seqs * block_table_stride` in 32 bits: a table of 2^31 entries or
/// more is one it cannot bound-check, and an overflowing product is one nothing can lay out.
TEST(TestGfx950AttentionUnified2DGeometry, NoLayoutExistsForATableTheKernelCannotIndex)
{
    EXPECT_FALSE(
        gfx950Unified2DWorkspaceLayout(int64_t{1} << 20, int64_t{1} << 31, 16).has_value());
    EXPECT_FALSE(
        gfx950Unified2DWorkspaceLayout(std::numeric_limits<int64_t>::max(), 16, 16).has_value());
}

/// The exact bytes launch() copies, for a batch of 3 with Sq != Skv: Sq 32 over Skv 64 at
/// block 16, so 4 blocks a sequence. A wrong identity table gives wrong numerics with no
/// fault: sequence 1 would read sequence 0's keys, or a query_start_len stepping by Skv would
/// hand sequence 1 the wrong query rows.
TEST(TestGfx950AttentionUnified2DGeometry, TheImageIsTheIdentityBlockTableAndPerSequenceLengths)
{
    const auto image = gfx950Unified2DWorkspaceImage(3, 32, 64, 16);
    ASSERT_TRUE(image.has_value());
    const auto layout = gfx950Unified2DWorkspaceLayout(3, 64, 16);
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(image->size(), layout->totalBytes);

    // block_tables[b][j] = b * 4 + j: 0..11 laid out row by row.
    for(int32_t entry = 0; entry < 12; ++entry)
    {
        EXPECT_EQ(int32At(*image, layout->blockTablesOffset, static_cast<std::size_t>(entry)),
                  entry)
            << "block_tables entry " << entry;
    }
    // seq_lens is Skv for every sequence -- not Sq, which would truncate the cache.
    for(std::size_t b = 0; b < 3; ++b)
    {
        EXPECT_EQ(int32At(*image, layout->seqLensOffset, b), 64) << "seq_lens[" << b << "]";
    }
    // query_start_len steps by Sq -- not Skv -- and has batch + 1 entries.
    const std::vector<int32_t> starts{0, 32, 64, 96};
    for(std::size_t b = 0; b < starts.size(); ++b)
    {
        EXPECT_EQ(int32At(*image, layout->queryStartLenOffset, b), starts.at(b))
            << "query_start_len[" << b << "]";
    }
}

/// Everything outside the three arrays is zero, so a stale byte in a recycled workspace cannot
/// be read as a table entry past the end of one.
TEST(TestGfx950AttentionUnified2DGeometry, TheImagePaddingIsZero)
{
    const auto image = gfx950Unified2DWorkspaceImage(3, 32, 64, 16);
    const auto layout = gfx950Unified2DWorkspaceLayout(3, 64, 16);
    ASSERT_TRUE(image.has_value() && layout.has_value());

    const auto zeroBetween = [&image](std::size_t from, std::size_t to) {
        for(std::size_t i = from; i < to; ++i)
        {
            if(image->at(i) != 0)
            {
                return false;
            }
        }
        return true;
    };
    EXPECT_TRUE(zeroBetween(layout->blockTablesBytes, layout->seqLensOffset));
    EXPECT_TRUE(
        zeroBetween(layout->seqLensOffset + layout->seqLensBytes, layout->queryStartLenOffset));
    EXPECT_TRUE(
        zeroBetween(layout->queryStartLenOffset + layout->queryStartLenBytes, layout->totalBytes));
}

/// One sequence with one block is the smallest image there is.
TEST(TestGfx950AttentionUnified2DGeometry, TheSmallestImageIsOneBlockForOneSequence)
{
    const auto image = gfx950Unified2DWorkspaceImage(1, 16, 16, 16);
    ASSERT_TRUE(image.has_value());
    EXPECT_EQ(image->size(), 768U);
    EXPECT_EQ(int32At(*image, 0, 0), 0);
    EXPECT_EQ(int32At(*image, 256, 0), 16);
    EXPECT_EQ(int32At(*image, 512, 0), 0);
    EXPECT_EQ(int32At(*image, 512, 1), 16);
}

TEST(TestGfx950AttentionUnified2DGeometry, NoImageExistsWhereNoLayoutDoes)
{
    EXPECT_FALSE(gfx950Unified2DWorkspaceImage(2, 32, 100, 16).has_value());
    EXPECT_FALSE(gfx950Unified2DWorkspaceImage(2, 0, 96, 16).has_value());
    EXPECT_FALSE(gfx950Unified2DWorkspaceImage(0, 32, 96, 16).has_value());
    // batch * seqLenQ past int32: query_start_len would wrap.
    EXPECT_FALSE(
        gfx950Unified2DWorkspaceImage(int64_t{1} << 16, int64_t{1} << 16, 16, 16).has_value());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
