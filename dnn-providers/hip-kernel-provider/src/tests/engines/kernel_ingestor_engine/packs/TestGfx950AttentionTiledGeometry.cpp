// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionTiledGeometry.hpp"

/**
 * @file TestGfx950AttentionTiledGeometry.cpp
 * @brief The launch contract this engine restates from Python, per shape family.
 *
 * The engine relaunches a rocKE binary from C++, recomputing by hand what the Python
 * launch path computes. Nothing in the build, the packer, the validator or the rest of
 * this suite compares the two, and a mismatch does not fail -- the kernel runs,
 * returns, and leaves part of the output untouched.
 *
 * These are the SECOND, independent statement of the rule. The assertions below spell
 * the expected grid out per shape family rather than calling a shared helper: a test
 * that recomputes with the same expression as the code under test agrees with it by
 * construction, including when both are wrong. Where the arithmetic is restated it is
 * restated from the PYTHON (`_get_2d_launch_meta`,
 * kernels/common/attention_unified.py:4079-4102), which is the actual source of truth.
 *
 * THREE THINGS DIFFER FROM THE DENSE TWIN, and each is why this file is not that one
 * with the numbers changed:
 *
 *  1. **The `+ num_seqs` VARLEN SLACK TERM has no dense analogue.** One reserved
 *     padding q-block per sequence, so a ragged batch needs no exact-division block
 *     count. Drop it and the kernel under-launches, leaving tail blocks unwritten --
 *     which the harness reports as `allClose=false` with ZERO finite mismatches, a
 *     shape that reads like a tolerance problem and is not one. The families below
 *     assert the term is PRESENT and scales with the sequence count.
 *  2. **grid.x is num_kv_heads, not num_query_heads.** The dense engine keys grid.y on
 *     the query heads; here the CTA owns one KV head and the GQA ratio is folded into
 *     `block_q` instead. Copying the dense expression would over-launch by up to 16x,
 *     every extra CTA indexing past the KV cache. The GQA families are what catch it.
 *  3. **Division is FLOOR, not ceil.** The dense grid ceils because its final query
 *     block may be partial; here the per-sequence padding block covers the tail, so a
 *     ceil PLUS the slack would over-launch. `FloorsRatherThanCeilingsTheQueryBlocks`
 *     is the case that pins the distinction.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

/// Restated as literals rather than reusing the header's constants: this file is the
/// independent statement, and importing the values under test would make the
/// comparison circular.
constexpr int64_t WAVE_LANES = 64;

/// One shape family: the inputs `_get_2d_launch_meta` reads, and the grid it must
/// produce. `expected*` are computed by hand from the Python, per family.
struct GeometryCase
{
    std::string name;
    int64_t numKvHeads;
    int64_t numQueryHeads;
    int64_t totalQ;
    int64_t numSeqs;
    int64_t numWarps;
    int64_t blockMPerWarp;
    unsigned expectedGridX;
    unsigned expectedGridY;
    unsigned expectedBlockX;
};

std::vector<GeometryCase> geometryCases()
{
    return {
        // Hq == Hkv (no GQA): num_queries_per_kv = 1, so block_q = block_m = 4*16 = 64.
        // gridY = 256/64 + 1 = 5. gridX = 8. blockX = 64*4 = 256.
        {"mha_single_sequence", 8, 8, 256, 1, 4, 16, 8, 5, 256},

        // The same shape across 4 sequences. THE SLACK TERM IS THE WHOLE POINT: the
        // query rows are unchanged, so a launcher without `+ num_seqs` would produce
        // the identical gridY and silently drop three sequences' padding blocks.
        // gridY = 256/64 + 4 = 8.
        {"mha_four_sequences", 8, 8, 256, 4, 4, 16, 8, 8, 256},

        // GQA 8:1. num_queries_per_kv = 8, block_m = 64, block_q = 64/8 = 8.
        // gridY = 256/8 + 2 = 34. gridX = 2 -- the KV heads, NOT the 16 query heads.
        {"gqa_eight_to_one", 2, 16, 256, 2, 4, 16, 2, 34, 256},

        // GQA 16:1 at the ratio ceiling supports_tiled_2d admits.
        // block_m = 64, block_q = 64/16 = 4. gridY = 256/4 + 1 = 65.
        {"gqa_sixteen_to_one", 1, 16, 256, 1, 4, 16, 1, 65, 256},

        // num_queries_per_kv EXCEEDS block_m, so the Python falls back to block_q = 1
        // rather than dividing to zero. block_m = 1*16 = 16, ratio = 32 > 16.
        // gridY = 128/1 + 2 = 130. A guard that only checked for division by zero
        // would compute block_q = 0 here and produce an infinite or negative grid.
        {"gqa_ratio_exceeds_block_m", 1, 32, 128, 2, 1, 16, 1, 130, 64},

        // Decode: one query row per sequence, many sequences. The slack term DOMINATES
        // -- gridY = 64/8 + 64 = 72, of which 64 are padding blocks that early-return.
        // block_m = 64, ratio 8, block_q = 8.
        {"decode_sixty_four_sequences", 2, 16, 64, 64, 4, 16, 2, 72, 256},

        // block_m_per_warp = 32 with num_warps = 4: block_m = 128, ratio 8,
        // block_q = 16. gridY = 4096/16 + 2 = 258. blockX = 64*4 = 256.
        {"wide_tile_prefill", 2, 16, 4096, 2, 4, 32, 2, 258, 256},

        // Single warp: blockX = 64, block_m = 16, ratio 8, block_q = 2.
        // gridY = 1024/2 + 3 = 515.
        {"single_warp", 2, 16, 1024, 3, 1, 16, 2, 515, 64},

        // Two warps at 32 rows: block_m = 64, ratio 2, block_q = 32.
        // gridY = 2048/32 + 1 = 65. blockX = 128.
        {"two_warps_wide_rows", 8, 16, 2048, 1, 2, 32, 8, 65, 128},
    };
}

TEST(Gfx950AttentionTiledGeometry, MatchesThePythonLaunchMetaPerShapeFamily)
{
    for(const auto& testCase : geometryCases())
    {
        const auto geometry = gfx950AttentionTiledGeometry(testCase.numKvHeads,
                                                           testCase.numQueryHeads,
                                                           testCase.totalQ,
                                                           testCase.numSeqs,
                                                           testCase.numWarps,
                                                           testCase.blockMPerWarp,
                                                           testCase.name);
        EXPECT_EQ(geometry.gridX, testCase.expectedGridX) << testCase.name << " grid.x";
        EXPECT_EQ(geometry.gridY, testCase.expectedGridY) << testCase.name << " grid.y";
        EXPECT_EQ(geometry.gridZ, 1U) << testCase.name << " grid.z is always 1";
        EXPECT_EQ(geometry.blockX, testCase.expectedBlockX) << testCase.name << " block.x";
    }
}

/// The single most likely transcription error: omitting the slack term. Two graphs
/// with IDENTICAL query rows and different sequence counts must produce different
/// grids, and the difference must be exactly the sequence count.
TEST(Gfx950AttentionTiledGeometry, GridGrowsByExactlyOneBlockPerSequence)
{
    const auto one = gfx950AttentionTiledGeometry(8, 8, 256, 1, 4, 16, "one");
    const auto four = gfx950AttentionTiledGeometry(8, 8, 256, 4, 4, 16, "four");
    EXPECT_EQ(four.gridY - one.gridY, 3U)
        << "the `+ num_seqs` term reserves one padding q-block per sequence; without it "
           "these two launches would be identical and three sequences' tails unwritten";
}

/// FLOOR, not ceil -- the opposite of the dense grid. The per-sequence padding block
/// covers the partial tail, so ceiling here would over-launch.
TEST(Gfx950AttentionTiledGeometry, FloorsRatherThanCeilingsTheQueryBlocks)
{
    // block_m = 64, ratio 1, block_q = 64. total_q = 100 -> 100/64 = 1 (floor), + 1 seq.
    const auto geometry = gfx950AttentionTiledGeometry(8, 8, 100, 1, 4, 16, "partial");
    EXPECT_EQ(geometry.gridY, 2U)
        << "floor(100/64) + 1 = 2; a ceil would give 3 and over-launch on top of the "
           "slack block that already covers the tail";
}

/// grid.x keys on the KV heads. A dense-style `numQueryHeads` here would launch the
/// GQA ratio too many CTAs, every extra one indexing past the KV cache.
TEST(Gfx950AttentionTiledGeometry, KeysGridXOnKvHeadsNotQueryHeads)
{
    const auto geometry = gfx950AttentionTiledGeometry(2, 16, 256, 1, 4, 16, "gqa");
    EXPECT_EQ(geometry.gridX, 2U);
    EXPECT_NE(geometry.gridX, 16U) << "16 would be the query-head count -- 8x too many CTAs";
}

/// The CTA is `wave_size * num_warps`, and num_warps VARIES per shape (measured over
/// {1,2,4} across the 52 dispatcher-resolved shapes). A hardcoded 256 would be right
/// for the four-warp majority and silently wrong for the rest.
TEST(Gfx950AttentionTiledGeometry, BlockSizeTracksNumWarps)
{
    for(const int64_t numWarps : {1, 2, 4})
    {
        const auto geometry = gfx950AttentionTiledGeometry(8, 8, 256, 1, numWarps, 16, "warps");
        EXPECT_EQ(geometry.blockX, static_cast<unsigned>(WAVE_LANES * numWarps))
            << "num_warps " << numWarps;
    }
}

// ===========================================================================
// Guards. Each of these would otherwise launch a degenerate grid, which returns
// cleanly having written nothing -- the silent failure this header exists to
// prevent. prepare() is the last place a named failure is cheap.
// ===========================================================================

TEST(Gfx950AttentionTiledGeometry, ThrowsOnNonPositiveExtents)
{
    EXPECT_THROW(gfx950AttentionTiledGeometry(0, 8, 256, 1, 4, 16, "kv"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionTiledGeometry(8, 0, 256, 1, 4, 16, "q"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionTiledGeometry(8, 8, 0, 1, 4, 16, "totalq"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionTiledGeometry(8, 8, 256, 0, 4, 16, "numseqs"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(Gfx950AttentionTiledGeometry, ThrowsOnGeometryTheBuilderRefusesToEmit)
{
    // num_warps outside {1,2,4,8}.
    EXPECT_THROW(gfx950AttentionTiledGeometry(8, 8, 256, 1, 3, 16, "warps3"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    // block_m_per_warp outside {16,32}.
    EXPECT_THROW(gfx950AttentionTiledGeometry(8, 8, 256, 1, 4, 64, "bmpw64"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    // The 1024-thread CTA cap: 8 warps x 32 rows exceeds it, and the predicate
    // refuses the pair (attention_tiled_2d.py:970-975).
    EXPECT_THROW(gfx950AttentionTiledGeometry(8, 8, 256, 1, 8, 32, "cap"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    // 8 warps at 16 rows is 128 rows in a 512-lane CTA -- legal, and must NOT throw.
    EXPECT_NO_THROW(gfx950AttentionTiledGeometry(8, 8, 256, 1, 8, 16, "eight_narrow"));
}

TEST(Gfx950AttentionTiledGeometry, ThrowsOnNonDivisibleHeadCounts)
{
    EXPECT_THROW(gfx950AttentionTiledGeometry(3, 16, 256, 1, 4, 16, "gqa"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

// ===========================================================================
// The paged-KV derivation. THE HIGHEST-RISK ARITHMETIC IN THE INTEGRATION:
// wrong here indexes the KV cache with the wrong stride and returns silently
// wrong numbers rather than faulting.
// ===========================================================================

/// `block_size` is the CONTAINER's page size -- K's sequence axis -- never a property
/// of the page table. Asserted directly so a future "derive it from the table
/// instead" refactor fails here rather than on a device.
TEST(Gfx950AttentionTiledPagedGeometry, TakesBlockSizeFromTheContainerNotTheTable)
{
    const auto geometry = gfx950TiledPagedKvGeometry(/*kSeqAxisExtent=*/32,
                                                     /*pageTableInnerExtent=*/64);
    EXPECT_EQ(geometry.blockSize, 32) << "the page size is K.dims[SEQ_AXIS]";
    EXPECT_NE(geometry.blockSize, 64) << "64 is the table's max-blocks-per-seq, not a page size";
}

/// The stride is in ELEMENTS. A byte stride would be a 2x indexing error into the KV
/// cache for a 2-byte dtype -- silently wrong numbers, not a fault.
TEST(Gfx950AttentionTiledPagedGeometry, ReportsBlockTableStrideInElements)
{
    const auto geometry = gfx950TiledPagedKvGeometry(16, 128);
    EXPECT_EQ(geometry.blockTableStride, 128)
        << "the host computes this as a torch .stride(0), which is an element count";
}

TEST(Gfx950AttentionTiledPagedGeometry, AcceptsExactlyTheLegalPageSizes)
{
    EXPECT_TRUE(gfx950TiledBlockSizeIsLegal(16));
    EXPECT_TRUE(gfx950TiledBlockSizeIsLegal(32));
    EXPECT_TRUE(gfx950TiledBlockSizeIsLegal(64));
    // The neighbours. Neither may be rounded to a legal value: a clamp is a wrong
    // stride, not a smaller tile.
    EXPECT_FALSE(gfx950TiledBlockSizeIsLegal(8));
    EXPECT_FALSE(gfx950TiledBlockSizeIsLegal(128));
    EXPECT_FALSE(gfx950TiledBlockSizeIsLegal(0));
    EXPECT_FALSE(gfx950TiledBlockSizeIsLegal(-16));
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
