// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <set>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"

/**
 * @file TestGfx950AttentionDenseGeometry.cpp
 * @brief Pins gfx950AttentionDenseGeometry() and the tile rules term for term against
 *        their Python originals.
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
 *   tile rules            (attention_dense_spec.py:88, :399; attention_dense.py:301-313)
 *   LDS budget            (build_attention_dense's K/V slabs; gfx950 163840 bytes)
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

/// Only reaches the diagnostic string; no case asserts on it except the one that pins
/// that the diagnostic names the kernel.
constexpr const char* KERNEL_NAME = "hipkernel:Gfx950AttentionDense/unit";

/// The two block_m values gfx950 builds (DENSE_TILE_GEOMETRIES).
constexpr int64_t BM256 = 256;
constexpr int64_t BM128 = 128;

/// attention_dense_block is `(spec.num_waves * 64, 1, 1)` and `num_waves = block_m // 32`
/// (attention_dense_spec.py:213-214): (256 / 32) * 64 = 512 and (128 / 32) * 64 = 256.
constexpr unsigned PYTHON_BLOCK_X_BM256 = 512U;
constexpr unsigned PYTHON_BLOCK_X_BM128 = 256U;

Gfx950AttentionDenseGeometry
    geometryFor(int64_t blockM, int64_t seqLenQ, int64_t numQueryHeads, int64_t batch)
{
    return gfx950AttentionDenseGeometry(blockM, seqLenQ, numQueryHeads, batch, KERNEL_NAME);
}

} // namespace

// =============================================================================
// gridX -- ceil(seqlen_q / block_m), as the Python writes it. kernel_match only serves
// multiples of block_m, where the ceiling is the quotient; the non-multiple cases pin
// that the function still matches the Python outside that contract rather than
// truncating to a grid that skips rows.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, AlignedSeqLenQIsOneBlockPerWholeTile)
{
    // 256 / 256 = 1, 512 / 256 = 2, 4096 / 256 = 16 -- exact, no partial block.
    EXPECT_EQ(geometryFor(BM256, 256, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(BM256, 512, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM256, 4096, 1, 1).gridX, 16U);
    // The same lengths at block_m 128 are twice as many blocks.
    EXPECT_EQ(geometryFor(BM128, 256, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM128, 512, 1, 1).gridX, 4U);
    EXPECT_EQ(geometryFor(BM128, 4096, 1, 1).gridX, 32U);
    // 384 is a whole number of 128-row tiles and not of 256-row ones.
    EXPECT_EQ(geometryFor(BM128, 384, 1, 1).gridX, 3U);
}

TEST(TestGfx950AttentionDenseGeometry, NonMultipleSeqLenQKeepsThePartialFinalBlock)
{
    // ceil(257 / 256) = 2: one whole tile plus a single-row partial block. Truncating
    // gives 1 and the last row is never written.
    EXPECT_EQ(geometryFor(BM256, 257, 1, 1).gridX, 2U);
    // ceil(513 / 256) = 3 and ceil(769 / 256) = 4 -- same one-row remainder, further out.
    EXPECT_EQ(geometryFor(BM256, 513, 1, 1).gridX, 3U);
    EXPECT_EQ(geometryFor(BM256, 769, 1, 1).gridX, 4U);
    // ceil(384 / 256) = 2: a half-full final block, not a one-row one.
    EXPECT_EQ(geometryFor(BM256, 384, 1, 1).gridX, 2U);
    // ceil(255 / 256) = 1 and ceil(1 / 256) = 1. Truncating gives 0 here, which is an
    // empty grid: the kernel returns having written nothing and reports success.
    EXPECT_EQ(geometryFor(BM256, 255, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(BM256, 1, 1, 1).gridX, 1U);
    // block_m 128: ceil(129 / 128) = 2, ceil(197 / 128) = 2, ceil(127 / 128) = 1.
    EXPECT_EQ(geometryFor(BM128, 129, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM128, 197, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM128, 127, 1, 1).gridX, 1U);
}

// =============================================================================
// The full triple -- (nqb, num_query_heads, batch), in that order.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, GridYIsQueryHeadsAndGridZIsBatch)
{
    // ceil(1024 / 256) = 4 query blocks; heads and batch pass through untouched and
    // are mutually distinct, so an exchanged pair cannot read as correct.
    const auto geometry = geometryFor(BM256, 1024, 16, 3);
    EXPECT_EQ(geometry.gridX, 4U);
    EXPECT_EQ(geometry.gridY, 16U);
    EXPECT_EQ(geometry.gridZ, 3U);
}

TEST(TestGfx950AttentionDenseGeometry, SingleHeadSingleBatchIsAOneDeepGrid)
{
    const auto geometry = geometryFor(BM256, 512, 1, 1);
    EXPECT_EQ(geometry.gridX, 2U);
    EXPECT_EQ(geometry.gridY, 1U);
    EXPECT_EQ(geometry.gridZ, 1U);
}

/// The launches the plan's witnesses name, read off attention_dense_grid/_block for each
/// block_m. The same graph launches differently per selected tile, so a prepare() that
/// used one tile's block_m for another's binary produces exactly one of these mismatches.
TEST(TestGfx950AttentionDenseGeometry, WitnessLaunchesDependOnTheSelectedBlockM)
{
    // BF16/D128/H32/8 causal, B3, Sq=Skv=1536.
    const Gfx950AttentionDenseGeometry causalBm256{6U, 32U, 3U, PYTHON_BLOCK_X_BM256};
    const Gfx950AttentionDenseGeometry causalBm128{12U, 32U, 3U, PYTHON_BLOCK_X_BM128};
    EXPECT_TRUE(geometryFor(BM256, 1536, 32, 3) == causalBm256);
    EXPECT_TRUE(geometryFor(BM128, 1536, 32, 3) == causalBm128);

    // BF16/D128/H9/9 noncausal, B1, Sq=1024 (Skv does not enter the launch).
    const Gfx950AttentionDenseGeometry crossBm256{4U, 9U, 1U, PYTHON_BLOCK_X_BM256};
    const Gfx950AttentionDenseGeometry crossBm128{8U, 9U, 1U, PYTHON_BLOCK_X_BM128};
    EXPECT_TRUE(geometryFor(BM256, 1024, 9, 1) == crossBm256);
    EXPECT_TRUE(geometryFor(BM128, 1024, 9, 1) == crossBm128);

    // Sq=384 fits block_m 128 exactly: three blocks of 256 lanes.
    const Gfx950AttentionDenseGeometry q384Bm128{3U, 9U, 1U, PYTHON_BLOCK_X_BM128};
    EXPECT_TRUE(geometryFor(BM128, 384, 9, 1) == q384Bm128);
}

// =============================================================================
// blockX -- the CTA. One number per block_m, independent of the shape.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, BlockXIsNumWavesWave64Waves)
{
    // (256 / 32) * 64 = 512 and (128 / 32) * 64 = 256 threads. Not seqlen-dependent:
    // the CTA is the same whatever the query length.
    for(const int64_t seqLenQ : {4096, 257, 1})
    {
        EXPECT_EQ(geometryFor(BM256, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM256) << seqLenQ;
        EXPECT_EQ(geometryFor(BM128, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM128) << seqLenQ;
    }
}

TEST(TestGfx950AttentionDenseGeometry, BlockMIsTheTileTheCtaAndTheGridAgreeOn)
{
    // The same block_m divides the query grid and sizes the CTA. A shape of exactly one
    // tile is the shape where both readings are visible on one call: one query block,
    // num_waves * 64 lanes.
    const auto large = geometryFor(BM256, BM256, 1, 1);
    EXPECT_EQ(large.gridX, 1U);
    EXPECT_EQ(large.blockX, PYTHON_BLOCK_X_BM256);

    const auto small = geometryFor(BM128, BM128, 1, 1);
    EXPECT_EQ(small.gridX, 1U);
    EXPECT_EQ(small.blockX, PYTHON_BLOCK_X_BM128);
}

// =============================================================================
// The guard. An empty or negative launch, or a block_m gfx950 does not build, is
// rejected by name rather than turned into a zero-CTA grid or a division by it.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, NonPositiveSeqLenQThrows)
{
    EXPECT_THROW(geometryFor(BM256, 0, 8, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM256, -1, 8, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM128, -256, 8, 4), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, NonPositiveQueryHeadsThrows)
{
    EXPECT_THROW(geometryFor(BM256, 512, 0, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM128, 512, -1, 4), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, NonPositiveBatchThrows)
{
    EXPECT_THROW(geometryFor(BM256, 512, 8, 0), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM128, 512, 8, -1), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, AllThreeNonPositiveThrows)
{
    EXPECT_THROW(geometryFor(BM256, 0, 0, 0), HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, UnbuiltBlockMThrowsBeforeDividingByIt)
{
    // Zero would divide by zero; a negative or other value would size a CTA and a grid
    // for a binary that does not exist. 64 and 512 are real multiples of 32 that gfx950
    // does not build, so the guard is membership, not a sign or granularity test.
    for(const int64_t blockM :
        {int64_t{0}, int64_t{-128}, int64_t{32}, int64_t{64}, int64_t{192}, int64_t{512}})
    {
        EXPECT_THROW(geometryFor(blockM, 512, 8, 4), HipdnnPluginException) << blockM;
    }
}

TEST(TestGfx950AttentionDenseGeometry, SmallestPositiveLaunchIsAccepted)
{
    // The other side of the guard: 1 is positive, so the smallest real launch must
    // pass and produce a single CTA at either tile.
    EXPECT_NO_THROW(geometryFor(BM256, 1, 1, 1));
    const Gfx950AttentionDenseGeometry expectedBm256{1U, 1U, 1U, PYTHON_BLOCK_X_BM256};
    EXPECT_TRUE(geometryFor(BM256, 1, 1, 1) == expectedBm256);
    const Gfx950AttentionDenseGeometry expectedBm128{1U, 1U, 1U, PYTHON_BLOCK_X_BM128};
    EXPECT_TRUE(geometryFor(BM128, 1, 1, 1) == expectedBm128);
}

TEST(TestGfx950AttentionDenseGeometry, RejectionNamesTheKernel)
{
    // The diagnostic exists so a failure identifies the descriptor that declared the
    // shape; a message without the name leaves the caller nothing to look up.
    for(const int64_t blockM : {BM256, int64_t{0}})
    {
        try
        {
            geometryFor(blockM, 0, 8, 4);
            FAIL() << "expected block_m " << blockM << " with seqlen_q 0 to be rejected";
        }
        catch(const HipdnnPluginException& e)
        {
            EXPECT_NE(std::string(e.what()).find(KERNEL_NAME), std::string::npos);
        }
    }
}

// =============================================================================
// The tile rules. Exhaustive over a window wide enough to contain every legal tile
// and each rule's nearest violators, so a dropped or loosened clause admits a tile.
// =============================================================================

TEST(TestGfx950AttentionDenseTile, AdmitsExactlyTheBuildableTilesAtEachHeadSize)
{
    // Read off the Python rules plus the LDS budget: D128 block_n 256 needs 303104 bytes
    // against gfx950's 163840 and is the only rule-satisfying tile the budget removes.
    const std::set<std::pair<int64_t, int64_t>> d64{
        {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}, {256, 256}};
    const std::set<std::pair<int64_t, int64_t>> d128{
        {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}};

    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        const auto& expected = headSize == 64 ? d64 : d128;
        std::set<std::pair<int64_t, int64_t>> admitted;
        for(int64_t blockM = -64; blockM <= 544; blockM += 16)
        {
            for(int64_t blockN = -64; blockN <= 544; blockN += 16)
            {
                if(isSupportedGfx950AttentionDenseTile(headSize, blockM, blockN))
                {
                    admitted.emplace(blockM, blockN);
                }
            }
        }
        EXPECT_EQ(admitted, expected) << "head_size " << headSize;
    }
}

TEST(TestGfx950AttentionDenseTile, RefusesNamedNeighboursOfTheLegalSet)
{
    // One violator per rule, each otherwise closest to a legal tile.
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(64, 128, 256)); // block_m % block_n
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 128, 256)); // block_m % block_n
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 256, 256)); // LDS budget
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 256, 0)); // block_n positive
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 256, -64)); // block_n positive
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 256, 48)); // block_n % 32
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 0, 64)); // block_m membership
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(128, 512, 64)); // block_m membership
    EXPECT_FALSE(isSupportedGfx950AttentionDenseTile(96, 256, 64)); // head_size
    // And the positive neighbours those are closest to.
    EXPECT_TRUE(isSupportedGfx950AttentionDenseTile(64, 256, 256));
    EXPECT_TRUE(isSupportedGfx950AttentionDenseTile(128, 128, 128));
    EXPECT_TRUE(isSupportedGfx950AttentionDenseTile(128, 256, 64));
}

TEST(TestGfx950AttentionDenseTile, StaticLdsMatchesTheBuilderSlabs)
{
    // 528 * BN at D64 and 1184 * BN at D128 -- the slab sizes build_attention_dense
    // allocates at K pad 8, V pad 32 and two buffers.
    EXPECT_EQ(gfx950AttentionDenseStaticLdsBytes(64, 32), 16896);
    EXPECT_EQ(gfx950AttentionDenseStaticLdsBytes(64, 256), 135168);
    EXPECT_EQ(gfx950AttentionDenseStaticLdsBytes(128, 64), 75776);
    EXPECT_EQ(gfx950AttentionDenseStaticLdsBytes(128, 128), 151552);
    EXPECT_EQ(gfx950AttentionDenseStaticLdsBytes(128, 256), 303104);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
