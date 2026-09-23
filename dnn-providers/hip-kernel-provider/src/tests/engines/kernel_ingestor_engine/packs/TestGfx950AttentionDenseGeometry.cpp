// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"

/**
 * @file TestGfx950AttentionDenseGeometry.cpp
 * @brief Pins gfx950AttentionDenseGeometry() term for term against its Python original.
 *
 * The engine relaunches a binary Python launched; a geometry that disagrees with the
 * Python does not fault, it computes something else. Nothing else in the suite compares
 * the two halves, so every expected number below is read off the Python and written as a
 * literal. Deriving one from the C++ expression under test would assert only that the
 * expression equals itself.
 *
 * Sources for the expectations:
 *   attention_dense_grid  (rocke/library/kernels/gfx950/attention_dense.py:2046-2054)
 *   attention_dense_block (rocke/library/kernels/gfx950/attention_dense.py:2057-2059)
 *   num_waves, block_m    (rocke/library/kernels/common/attention_dense_spec.py:25,213-214)
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

/// Only reaches the diagnostic string; no case asserts on it.
constexpr const char* KERNEL_NAME = "hipkernel:Gfx950AttentionDense/unit";

/// DEFAULT_DENSE_TILE_GEOMETRY pins block_m = 256 for every shipped dense variant
/// (attention_dense_spec.py:25). The query grid is ceil(seqlen_q / 256).
constexpr int64_t PYTHON_BLOCK_M = 256;

/// attention_dense_block is `(spec.num_waves * 64, 1, 1)` and `num_waves = block_m // 32`
/// (attention_dense_spec.py:213-214), so at block_m 256 it is (256 / 32) * 64 = 8 * 64.
constexpr unsigned PYTHON_BLOCK_X = 512U;

Gfx950AttentionDenseGeometry geometryFor(int64_t seqLenQ, int64_t numQueryHeads, int64_t batch)
{
    return gfx950AttentionDenseGeometry(seqLenQ, numQueryHeads, batch, KERNEL_NAME);
}

} // namespace

// =============================================================================
// gridX -- ceil(seqlen_q / block_m). Both sides of the divisibility boundary are
// pinned: truncation agrees with the ceiling on the aligned side and disagrees on
// every ragged one, so only the ragged cases below distinguish the two.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, AlignedSeqLenQIsOneBlockPerWholeTile)
{
    // 256 / 256 = 1, 512 / 256 = 2, 4096 / 256 = 16 -- exact, no partial block.
    EXPECT_EQ(geometryFor(256, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(512, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(4096, 1, 1).gridX, 16U);
}

TEST(TestGfx950AttentionDenseGeometry, RaggedSeqLenQKeepsThePartialFinalBlock)
{
    // ceil(257 / 256) = 2: one whole tile plus a single-row tail block. Truncating
    // gives 1 and the tail rows are never written.
    EXPECT_EQ(geometryFor(257, 1, 1).gridX, 2U);
    // ceil(513 / 256) = 3 and ceil(769 / 256) = 4 -- same one-row tail, further out.
    EXPECT_EQ(geometryFor(513, 1, 1).gridX, 3U);
    EXPECT_EQ(geometryFor(769, 1, 1).gridX, 4U);
    // ceil(384 / 256) = 2: a half-full tail block, not a one-row one.
    EXPECT_EQ(geometryFor(384, 1, 1).gridX, 2U);
    // ceil(255 / 256) = 1 and ceil(1 / 256) = 1. Truncating gives 0 here, which is an
    // empty grid: the kernel returns having written nothing and reports success.
    EXPECT_EQ(geometryFor(255, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(1, 1, 1).gridX, 1U);
}

// =============================================================================
// The full triple -- (nqb, num_query_heads, batch), in that order.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, GridYIsQueryHeadsAndGridZIsBatch)
{
    // ceil(1024 / 256) = 4 query blocks; heads and batch pass through untouched and
    // are mutually distinct, so an exchanged pair cannot read as correct.
    const auto geometry = geometryFor(1024, 16, 3);
    EXPECT_EQ(geometry.gridX, 4U);
    EXPECT_EQ(geometry.gridY, 16U);
    EXPECT_EQ(geometry.gridZ, 3U);
}

TEST(TestGfx950AttentionDenseGeometry, RaggedShapeCarriesHeadsAndBatchAlongside)
{
    // ceil(1025 / 256) = 5, with heads 5 and batch 2. gridX equals gridY here only by
    // arithmetic; the case exists so the ceiling and the passthrough are pinned together
    // on one call, as prepare() reads them.
    const Gfx950AttentionDenseGeometry expected{5U, 5U, 2U, PYTHON_BLOCK_X};
    EXPECT_TRUE(geometryFor(1025, 5, 2) == expected);
}

TEST(TestGfx950AttentionDenseGeometry, SingleHeadSingleBatchIsAOneDeepGrid)
{
    const auto geometry = geometryFor(512, 1, 1);
    EXPECT_EQ(geometry.gridX, 2U);
    EXPECT_EQ(geometry.gridY, 1U);
    EXPECT_EQ(geometry.gridZ, 1U);
}

// =============================================================================
// blockX -- the CTA. One number for the whole shipped set, because block_m is.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, BlockXIsEightWave64Waves)
{
    // (256 / 32) * 64 = 512 threads. Not seqlen-dependent: the same CTA serves an
    // aligned shape, a ragged one, and a shape shorter than a single tile.
    EXPECT_EQ(geometryFor(4096, 8, 4).blockX, PYTHON_BLOCK_X);
    EXPECT_EQ(geometryFor(257, 8, 4).blockX, PYTHON_BLOCK_X);
    EXPECT_EQ(geometryFor(1, 1, 1).blockX, PYTHON_BLOCK_X);
}

TEST(TestGfx950AttentionDenseGeometry, BlockMIsTheTileTheCtaAndTheGridAgreeOn)
{
    // The same 256 divides the query grid and sizes the CTA. A shape of exactly one
    // tile is the shape where both readings of block_m are visible on one call:
    // one query block, 512 lanes.
    const auto geometry = geometryFor(PYTHON_BLOCK_M, 1, 1);
    EXPECT_EQ(geometry.gridX, 1U);
    EXPECT_EQ(geometry.blockX, PYTHON_BLOCK_X);
}

// =============================================================================
// The guard. An empty or negative launch is rejected by name rather than turned
// into a zero-CTA grid that returns cleanly having written nothing.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, NonPositiveSeqLenQThrows)
{
    EXPECT_THROW(geometryFor(0, 8, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(-1, 8, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(-256, 8, 4), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, NonPositiveQueryHeadsThrows)
{
    EXPECT_THROW(geometryFor(512, 0, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(512, -1, 4), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, NonPositiveBatchThrows)
{
    EXPECT_THROW(geometryFor(512, 8, 0), HipdnnPluginException);
    EXPECT_THROW(geometryFor(512, 8, -1), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, AllThreeNonPositiveThrows)
{
    EXPECT_THROW(geometryFor(0, 0, 0), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, SmallestPositiveLaunchIsAccepted)
{
    // The other side of the guard: 1 is positive, so the smallest real launch must
    // pass and produce a single CTA of 512 lanes.
    EXPECT_NO_THROW(geometryFor(1, 1, 1));
    const Gfx950AttentionDenseGeometry expected{1U, 1U, 1U, PYTHON_BLOCK_X};
    EXPECT_TRUE(geometryFor(1, 1, 1) == expected);
}

TEST(TestGfx950AttentionDenseGeometry, RejectionNamesTheKernel)
{
    // The diagnostic exists so a failure identifies the descriptor that declared the
    // shape; a message without the name leaves the caller nothing to look up.
    try
    {
        geometryFor(0, 8, 4);
        FAIL() << "expected a non-positive seqlen_q to be rejected";
    }
    catch(const HipdnnPluginException& e)
    {
        EXPECT_NE(std::string(e.what()).find(KERNEL_NAME), std::string::npos);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
