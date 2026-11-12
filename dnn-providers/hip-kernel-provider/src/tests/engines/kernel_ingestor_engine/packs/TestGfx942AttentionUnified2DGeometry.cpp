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

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified2DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified3dGeometry.hpp"

/**
 * @file TestGfx942AttentionUnified2DGeometry.cpp
 * @brief Pins what hipkernel:Gfx942AttentionUnified2D restates from its Python builder: the
 *        geometry rules, the launch grid, the paged-KV metadata and the decode handoff.
 *
 * Every expectation is a literal worked out from the Python (gfx942 attention_tiled_2d.py,
 * `AttentionTuningSpec.launch_grid`), not computed the way the header computes it: a helper
 * repeating the formula would assert only that the two agree. A mismatch here is not a fault
 * on the device: the kernel computes something else, or reads another sequence's lengths.
 *
 * No device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{
namespace
{

using Fields = Gfx942Unified2DGeometryFields;

/// The D128 cells' default: block 16, tile 32, 2 warps x 32 rows, transposed 32x32x8.
Fields d128Default()
{
    return Fields{128, 16, 32, 2, 32, true, true};
}

/// A narrow 16x16 geometry: D128, block 16, tile 64, 4 warps x 16 rows.
Fields narrow()
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

/// The positive control every refusal below depends on. Worked out from the Python at 4 warps
/// (256 threads): gfx942's async K/V copy moves 2 halves per lane, so the smallest slab a
/// call needs is 4 * 64 * 2 = 512 elements, below the smallest legal slab (tile 16 x D64 =
/// 1024): every dispatcher tile at every block size is admitted at both head sizes.
TEST(TestGfx942AttentionUnified2DGeometry, AcceptsTheShippedDefaultsAndTheDispatcherTiles)
{
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(d128Default()));
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(Fields{64, 16, 64, 4, 32, true, true}));

    for(const int64_t headSize : {64, 128})
    {
        for(const int64_t blockSize : {16, 32, 64})
        {
            for(const int64_t multiplier : {1, 2, 4, 8})
            {
                auto fields = narrow();
                fields.headSize = headSize;
                fields.blockSize = blockSize;
                fields.tileSize = blockSize * multiplier;
                EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields))
                    << "head " << headSize << " block " << blockSize << " tile " << fields.tileSize;
            }
        }
    }
}

TEST(TestGfx942AttentionUnified2DGeometry, RefusesAnUnbuiltWarpCountOrRowCount)
{
    for(const int64_t warps : {0, 3, 5, 6, 7, 9, 16, -4})
    {
        auto fields = narrow();
        fields.numWarps = warps;
        EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(fields)) << "num_warps " << warps;
    }
    for(const int64_t rows : {0, 8, 24, 48, -16})
    {
        auto fields = narrow();
        fields.blockMPerWarp = rows;
        EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(fields)) << "block_m_per_warp " << rows;
    }
}

/// gfx942 builds 32 rows per warp at 8 warps (gfx950 does not): the builder then needs
/// direct-Q and conflict-free V stores to fit the LDS, which the catalog's spec carries.
TEST(TestGfx942AttentionUnified2DGeometry, AcceptsThirtyTwoRowsAtEightWarps)
{
    auto fields = d128Default();
    fields.numWarps = 8;
    fields.tileSize = 64;
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields));
    fields.useMfma32x32x8 = false;
    fields.useTransposedQk32x32 = false;
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields)) << "narrow, 8 warps x 32 rows";
}

TEST(TestGfx942AttentionUnified2DGeometry, RefusesABlockOrTileTheKernelIsNotBuiltFor)
{
    for(const int64_t blockSize : {0, 8, 24, 48, 128, -16})
    {
        auto fields = narrow();
        fields.blockSize = blockSize;
        fields.tileSize = 64;
        EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(fields)) << "block_size " << blockSize;
    }
    // Not a multiple of the block, non-positive, or past the dispatcher's 8x policy.
    for(const int64_t tileSize : {0, -64, 24, 70, 144})
    {
        auto fields = narrow();
        fields.tileSize = tileSize;
        EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(fields)) << "tile_size " << tileSize;
    }
}

/// head_size must be one of {64, 128, 192, 256} (`supports_tiled_2d`); the engine ships 64
/// and 128 and the matcher narrows to those, but a record outside the kernel's own range is
/// malformed before that. 32 and 96 are multiples of 32 the gfx942 builder still refuses.
TEST(TestGfx942AttentionUnified2DGeometry, RefusesAHeadSizeOutsideTheKernelsRange)
{
    for(const int64_t headSize : {0, -64, 16, 32, 48, 96, 100, 512})
    {
        auto fields = narrow();
        fields.headSize = headSize;
        EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(fields)) << "head_size " << headSize;
    }
    for(const int64_t headSize : {192, 256})
    {
        auto fields = narrow();
        fields.headSize = headSize;
        EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields)) << "head_size " << headSize;
    }
}

/// `tile_size * head_size >= threads * 2`: one async copy moves 2 halves per lane on gfx942.
/// The tightest legal geometry, D64 at block 16, tile 16 and 8 warps, meets it exactly
/// (16 * 64 = 1024 = 8 * 64 * 2), so no legal head size and tile falls below it.
TEST(TestGfx942AttentionUnified2DGeometry, TheTightestSlabMeetsTheCopyWidthExactly)
{
    const Fields fields{64, 16, 16, 8, 16, false, false};
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields));
}

TEST(TestGfx942AttentionUnified2DGeometry, TheThirtyTwoByThirtyTwoByEightPathKeepsItsPreconditions)
{
    Fields fields{128, 16, 32, 2, 32, true, false};
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(fields));

    // 32x32x8 needs 32 rows per warp.
    fields.blockMPerWarp = 16;
    EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(fields));

    // ... and a tile that is a multiple of 32: 48 is three whole blocks of 16, but not that.
    const Fields tile48{128, 16, 48, 2, 32, true, false};
    EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(tile48));

    // The transposed QK variant exists only on top of 32x32x8.
    const Fields transposedAlone{128, 16, 32, 2, 32, false, true};
    EXPECT_FALSE(isSupportedGfx942Unified2DGeometry(transposedAlone));
    EXPECT_TRUE(isSupportedGfx942Unified2DGeometry(d128Default()));
}

// =============================================================================
// Launch geometry
// =============================================================================

/// Hq 8 over Hkv 1 is 8 queries per KV head; block_m is 4 * 16 = 64, so block_q is 8 and each
/// CTA covers 8 query tokens. 2 * 512 = 1024 tokens -> 128 blocks, plus one per sequence for
/// the partial blocks the floor drops: 130.
TEST(TestGfx942AttentionUnified2DGeometry, GridIsKvHeadsByQueryBlocksPlusOnePerSequence)
{
    const auto geometry = gfx942Unified2DLaunchGeometry(4, 16, 8, 1, 512, 2, "k");
    EXPECT_EQ(geometry.gridX, 1U);
    EXPECT_EQ(geometry.gridY, 130U);
    EXPECT_EQ(geometry.gridZ, 1U);
    EXPECT_EQ(geometry.blockX, 256U);
}

/// The D128 default, 2 warps x 32 rows, is block_m 64 too: Hq 32 over Hkv 8 (a group of 4) has
/// block_q 16; 3 * 200 = 600 tokens // 16 = 37, + 3 sequences = 40, over 8 KV heads, 128
/// threads.
TEST(TestGfx942AttentionUnified2DGeometry, TheD128DefaultLaunchesHalfTheThreadsForTheSameRows)
{
    const auto geometry = gfx942Unified2DLaunchGeometry(2, 32, 32, 8, 200, 3, "k");
    EXPECT_EQ(geometry.gridX, 8U);
    EXPECT_EQ(geometry.gridY, 40U);
    EXPECT_EQ(geometry.blockX, 128U);
}

/// MHA at block_m 64 has block_q 64: 300 tokens floor to 4 blocks, +3 sequences = 7.
TEST(TestGfx942AttentionUnified2DGeometry, TheQueryBlockCountFloorsAndAddsTheBatch)
{
    const auto geometry = gfx942Unified2DLaunchGeometry(4, 16, 4, 4, 100, 3, "k");
    EXPECT_EQ(geometry.gridX, 4U);
    EXPECT_EQ(geometry.gridY, 7U);
}

/// The head counts are the graph's, not the catalog's: two head pairs of one ratio get the
/// same block_q and differ only in the KV-head axis of the grid.
TEST(TestGfx942AttentionUnified2DGeometry, HeadPairsOfOneRatioDifferOnlyInTheKvHeadAxis)
{
    const auto small = gfx942Unified2DLaunchGeometry(2, 32, 8, 2, 512, 2, "k");
    const auto large = gfx942Unified2DLaunchGeometry(2, 32, 64, 16, 512, 2, "k");
    EXPECT_EQ(small.gridX, 2U);
    EXPECT_EQ(large.gridX, 16U);
    EXPECT_EQ(small.gridY, large.gridY);
    EXPECT_EQ(small.blockX, large.blockX);
}

TEST(TestGfx942AttentionUnified2DGeometry, BlockIsSixtyFourLanesPerWarp)
{
    for(const auto& [warps, threads] :
        std::vector<std::pair<int64_t, unsigned>>{{1, 64}, {2, 128}, {4, 256}, {8, 512}})
    {
        EXPECT_EQ(gfx942Unified2DLaunchGeometry(warps, 16, 8, 8, 64, 1, "k").blockX, threads)
            << "num_warps " << warps;
    }
}

TEST(TestGfx942AttentionUnified2DGeometry, ALaunchThatCouldNotBeDescribedThrowsByName)
{
    const auto throws = [](int64_t warps,
                           int64_t rows,
                           int64_t heads,
                           int64_t kvHeads,
                           int64_t seqLenQ,
                           int64_t batch) {
        try
        {
            gfx942Unified2DLaunchGeometry(warps, rows, heads, kvHeads, seqLenQ, batch, "kernel-x");
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
TEST(TestGfx942AttentionUnified2DGeometry, EachSectionStartsOnA256ByteBoundary)
{
    const auto layout = gfx942Unified2DWorkspaceLayout(2, 96, 16);
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

TEST(TestGfx942AttentionUnified2DGeometry, NoLayoutExistsWhenTheCacheIsNotWholeBlocks)
{
    EXPECT_FALSE(gfx942Unified2DWorkspaceLayout(2, 100, 16).has_value()) << "100 % 16";
    EXPECT_FALSE(gfx942Unified2DWorkspaceLayout(2, 96, 64).has_value()) << "96 % 64";
    EXPECT_TRUE(gfx942Unified2DWorkspaceLayout(2, 96, 32).has_value());
    EXPECT_FALSE(gfx942Unified2DWorkspaceLayout(0, 96, 16).has_value());
    EXPECT_FALSE(gfx942Unified2DWorkspaceLayout(2, 0, 16).has_value());
    EXPECT_FALSE(gfx942Unified2DWorkspaceLayout(2, 96, 0).has_value());
    EXPECT_FALSE(gfx942Unified2DWorkspaceLayout(-1, 96, 16).has_value());
}

/// The kernel computes `num_seqs * block_table_stride` in 32 bits: a table of 2^31 entries or
/// more is one it cannot bound-check, and an overflowing product is one nothing can lay out.
TEST(TestGfx942AttentionUnified2DGeometry, NoLayoutExistsForATableTheKernelCannotIndex)
{
    EXPECT_FALSE(
        gfx942Unified2DWorkspaceLayout(int64_t{1} << 20, int64_t{1} << 31, 16).has_value());
    EXPECT_FALSE(
        gfx942Unified2DWorkspaceLayout(std::numeric_limits<int64_t>::max(), 16, 16).has_value());
}

/// The exact bytes launch() copies, for a batch of 3 with Sq != Skv: Sq 32 over Skv 64 at
/// block 16, so 4 blocks a sequence. A wrong identity table gives wrong numerics with no
/// fault: sequence 1 would read sequence 0's keys, or a query_start_len stepping by Skv would
/// hand sequence 1 the wrong query rows.
TEST(TestGfx942AttentionUnified2DGeometry, TheImageIsTheIdentityBlockTableAndPerSequenceLengths)
{
    const auto image = gfx942Unified2DWorkspaceImage(3, 32, 64, 16);
    ASSERT_TRUE(image.has_value());
    const auto layout = gfx942Unified2DWorkspaceLayout(3, 64, 16);
    ASSERT_TRUE(layout.has_value());
    ASSERT_EQ(image->size(), layout->totalBytes);

    // block_tables[b][j] = b * 4 + j: 0..11 laid out row by row.
    for(int32_t entry = 0; entry < 12; ++entry)
    {
        EXPECT_EQ(int32At(*image, layout->blockTablesOffset, static_cast<std::size_t>(entry)),
                  entry)
            << "block_tables entry " << entry;
    }
    // seq_lens is Skv for every sequence: not Sq, which would truncate the cache.
    for(std::size_t b = 0; b < 3; ++b)
    {
        EXPECT_EQ(int32At(*image, layout->seqLensOffset, b), 64) << "seq_lens[" << b << "]";
    }
    // query_start_len steps by Sq, not Skv, and has batch + 1 entries.
    const std::vector<int32_t> starts{0, 32, 64, 96};
    for(std::size_t b = 0; b < starts.size(); ++b)
    {
        EXPECT_EQ(int32At(*image, layout->queryStartLenOffset, b), starts.at(b))
            << "query_start_len[" << b << "]";
    }
}

TEST(TestGfx942AttentionUnified2DGeometry, NoImageExistsWhereNoLayoutDoes)
{
    EXPECT_FALSE(gfx942Unified2DWorkspaceImage(2, 32, 100, 16).has_value());
    EXPECT_FALSE(gfx942Unified2DWorkspaceImage(2, 0, 96, 16).has_value());
    EXPECT_FALSE(gfx942Unified2DWorkspaceImage(0, 32, 96, 16).has_value());
    // batch * seqLenQ past int32: query_start_len would wrap.
    EXPECT_FALSE(
        gfx942Unified2DWorkspaceImage(int64_t{1} << 16, int64_t{1} << 16, 16, 16).has_value());
}

// =============================================================================
// Decode handoff
// =============================================================================

/// The split-KV gfx942 engine ships GQA ratios 1, 2, 4, 8 and 16; decode at those is its, and
/// any other group (or a non-dividing pair, which has none) stays with this engine's matcher.
/// Both engines read gfx942Unified3dServesGqaGroup, so this pins the one shared rule.
TEST(TestGfx942AttentionUnified2DGeometry, TheSplitKvEngineServesDecodeAtTheRatiosItShips)
{
    for(const auto& [hq, hkv] : std::vector<std::pair<int64_t, int64_t>>{
            {8, 8}, {16, 8}, {32, 8}, {64, 8}, {128, 8}, {24, 6}, {16, 1}})
    {
        EXPECT_TRUE(gfx942Unified3dServesGqaGroup(hq, hkv)) << "Hq " << hq << " Hkv " << hkv;
    }
    for(const auto& [hq, hkv] :
        std::vector<std::pair<int64_t, int64_t>>{{24, 8}, {40, 8}, {12, 1}, {9, 2}, {8, 0}})
    {
        EXPECT_FALSE(gfx942Unified3dServesGqaGroup(hq, hkv)) << "Hq " << hq << " Hkv " << hkv;
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
