// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"

/**
 * @file TestAttentionDenseGeometry.cpp
 * @brief Pins gfx942AttentionDenseGeometry(), gfx950AttentionDenseGeometry() and each arch's
 *        tile rules against their Python originals, plus this engine's own gfx942
 *        sliding-window guard.
 *
 * A geometry that disagrees with the Python does not fault, it computes something else,
 * and nothing else in the suite compares the two halves. Every expected tile number below
 * is read off `kernels/gfx94x/attention_dense.py` (`attention_dense_grid`,
 * `attention_dense_block`) and written as a literal; deriving one from the expression
 * under test would assert only that the expression equals itself. The window-guard
 * witnesses are literals too, but they pin this pack's own decline policy: the Python no
 * longer has a counterpart.
 *
 * The two arches keep separate suites: they build different block_m sets under different
 * tile rules (gfx942 by quantum and CTA size, gfx950 by membership and LDS budget), so
 * no case reads the same literals on both.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

// =============================================================================
// gfx942
// =============================================================================

namespace gfx942
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

constexpr const char* KERNEL_NAME = "hipkernel:Gfx942AttentionDense/unit";

constexpr int64_t BM64 = 64;
constexpr int64_t BM256 = 256;
constexpr int64_t BM512 = 512;

/// attention_dense_block is `(spec.num_waves * 64, 1, 1)` and `num_waves = block_m // 32`.
constexpr unsigned PYTHON_BLOCK_X_BM64 = 128U;
constexpr unsigned PYTHON_BLOCK_X_BM256 = 512U;
constexpr unsigned PYTHON_BLOCK_X_BM512 = 1024U;

Gfx942AttentionDenseGeometry
    geometryFor(int64_t blockM, int64_t seqLenQ, int64_t numQueryHeads, int64_t batch)
{
    return gfx942AttentionDenseGeometry(blockM, seqLenQ, numQueryHeads, batch, KERNEL_NAME);
}

} // namespace

// =============================================================================
// The launch -- (ceil(seqlen_q / block_m), num_query_heads, batch) and
// (block_m / 32 * 64, 1, 1).
// =============================================================================

TEST(TestGfx942AttentionDenseGeometry, GridIsQueryBlocksThenHeadsThenBatch)
{
    // Heads and batch are mutually distinct, so an exchanged pair cannot read as correct.
    const Gfx942AttentionDenseGeometry bm256{4U, 16U, 3U, PYTHON_BLOCK_X_BM256};
    EXPECT_TRUE(geometryFor(BM256, 1024, 16, 3) == bm256);
    const Gfx942AttentionDenseGeometry bm64{16U, 16U, 3U, PYTHON_BLOCK_X_BM64};
    EXPECT_TRUE(geometryFor(BM64, 1024, 16, 3) == bm64);
    const Gfx942AttentionDenseGeometry bm512{2U, 16U, 3U, PYTHON_BLOCK_X_BM512};
    EXPECT_TRUE(geometryFor(BM512, 1024, 16, 3) == bm512);
}

TEST(TestGfx942AttentionDenseGeometry, NonMultipleSeqLenQKeepsThePartialFinalBlock)
{
    // ceil, as the Python writes it; truncation would drop the last rows unwritten.
    EXPECT_EQ(geometryFor(BM256, 257, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM256, 255, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(BM64, 65, 1, 1).gridX, 2U);
}

TEST(TestGfx942AttentionDenseGeometry, BlockXFollowsTheSelectedBlockMOnly)
{
    for(const int64_t seqLenQ : {4096, 257, 1})
    {
        EXPECT_EQ(geometryFor(BM64, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM64) << seqLenQ;
        EXPECT_EQ(geometryFor(BM256, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM256) << seqLenQ;
        EXPECT_EQ(geometryFor(BM512, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM512) << seqLenQ;
    }
}

TEST(TestGfx942AttentionDenseGeometry, NonPositiveExtentsThrow)
{
    EXPECT_THROW(geometryFor(BM256, 0, 8, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM256, 512, 0, 4), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM256, 512, 8, 0), HipdnnPluginException);
    EXPECT_THROW(geometryFor(BM256, -512, 8, 4), HipdnnPluginException);
}

TEST(TestGfx942AttentionDenseGeometry, UnbuildableBlockMThrowsBeforeDividingByIt)
{
    // Zero divides by zero; 48 floors to a wave count that leaves rows uncovered; 1024
    // needs a 2048-thread CTA, past the workgroup limit; 2^62 is 32-aligned metadata whose
    // thread count would overflow int64_t if it were multiplied out.
    for(const int64_t blockM :
        {int64_t{0}, int64_t{-256}, int64_t{48}, int64_t{1024}, int64_t{1} << 62})
    {
        EXPECT_THROW(geometryFor(blockM, 512, 8, 4), HipdnnPluginException) << blockM;
    }
}

TEST(TestGfx942AttentionDenseGeometry, RejectionNamesTheKernel)
{
    try
    {
        geometryFor(BM256, 0, 8, 4);
        FAIL() << "expected seqlen_q 0 to be rejected";
    }
    catch(const HipdnnPluginException& e)
    {
        EXPECT_NE(std::string(e.what()).find(KERNEL_NAME), std::string::npos);
    }
}

// =============================================================================
// The tile rules. Exhaustive over a window wide enough to contain every legal tile and
// each rule's nearest violators.
// =============================================================================

TEST(TestGfx942AttentionDenseTile, AdmitsThePythonTilesPlusOnlyThoseTheLdsBudgetRemoves)
{
    // supports_attention_dense's accepted tiles, each built at seqlens the tile divides,
    // plus the tiles it rejects ONLY on the LDS budget, which this predicate does not
    // restate (no such tile can be packed).
    const std::set<std::pair<int64_t, int64_t>> d64{{32, 32},
                                                    {64, 32},
                                                    {64, 64},
                                                    {96, 96},
                                                    {128, 32},
                                                    {128, 64},
                                                    {128, 128},
                                                    {160, 160},
                                                    {192, 96},
                                                    {192, 192},
                                                    {224, 224},
                                                    {256, 32},
                                                    {256, 64},
                                                    {256, 128},
                                                    {320, 160},
                                                    {384, 96},
                                                    {384, 192},
                                                    {448, 224},
                                                    {512, 32},
                                                    {512, 64},
                                                    {512, 128},
                                                    // LDS budget only:
                                                    {256, 256},
                                                    {288, 288},
                                                    {320, 320},
                                                    {352, 352},
                                                    {384, 384},
                                                    {416, 416},
                                                    {448, 448},
                                                    {480, 480},
                                                    {512, 256},
                                                    {512, 512}};
    const std::set<std::pair<int64_t, int64_t>> d128{{32, 32},
                                                     {64, 32},
                                                     {64, 64},
                                                     {96, 96},
                                                     {128, 32},
                                                     {128, 64},
                                                     {192, 96},
                                                     {256, 32},
                                                     {256, 64},
                                                     {384, 96},
                                                     {512, 32},
                                                     {512, 64},
                                                     // LDS budget only:
                                                     {128, 128},
                                                     {160, 160},
                                                     {192, 192},
                                                     {224, 224},
                                                     {256, 128},
                                                     {256, 256},
                                                     {288, 288},
                                                     {320, 160},
                                                     {320, 320},
                                                     {352, 352},
                                                     {384, 192},
                                                     {384, 384},
                                                     {416, 416},
                                                     {448, 224},
                                                     {448, 448},
                                                     {480, 480},
                                                     {512, 128},
                                                     {512, 256},
                                                     {512, 512}};

    for(const int64_t headSize : {int64_t{64}, int64_t{128}})
    {
        const auto& expected = headSize == 64 ? d64 : d128;
        std::set<std::pair<int64_t, int64_t>> admitted;
        for(int64_t blockM = -64; blockM <= 1088; blockM += 16)
        {
            for(int64_t blockN = -64; blockN <= 1088; blockN += 16)
            {
                if(isSupportedGfx942AttentionDenseTile(headSize, blockM, blockN))
                {
                    admitted.emplace(blockM, blockN);
                }
            }
        }
        EXPECT_EQ(admitted, expected) << "head_size " << headSize;
    }
}

TEST(TestGfx942AttentionDenseTile, RefusesNamedNeighboursOfTheLegalSet)
{
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(128, 64, 128)); // block_m % block_n
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(128, 1024, 64)); // 2048-thread CTA
    // Fails only under UBSan: a wrapped CTA product still fails block_n % num_waves.
    // UnbuildableBlockMThrowsBeforeDividingByIt's 2^62 is the case that pins the guard.
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(128, int64_t{1} << 62, 64));
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(128, 256, 0)); // block_n positive
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(128, 256, 48)); // block_n % 32
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(128, 0, 64)); // block_m positive
    EXPECT_FALSE(isSupportedGfx942AttentionDenseTile(96, 256, 64)); // head_size
    // And the positive neighbours those are closest to.
    EXPECT_TRUE(isSupportedGfx942AttentionDenseTile(64, 512, 32));
    EXPECT_TRUE(isSupportedGfx942AttentionDenseTile(128, 512, 64));
    EXPECT_TRUE(isSupportedGfx942AttentionDenseTile(128, 256, 64));
}

// =============================================================================
// The sliding-window zero-trip guard. The witnesses pin this pack's own decline policy at
// each tile and shape, for a causal bf16 D128 spec; the Python no longer declines these
// windows, and the row rule is not part of the guard.
// =============================================================================

TEST(TestGfx942AttentionDenseWindow, NoWindowAlwaysReachesKeys)
{
    EXPECT_TRUE(gfx942AttentionDenseWindowReachesKeys(256, 64, 4096, 64, 0));
}

TEST(TestGfx942AttentionDenseWindow, PinsThePackDeclinePolicyAtEachWitness)
{
    struct Witness
    {
        int64_t blockM, blockN, seqLenQ, seqLenKv, window;
        bool accepted;
    };
    // The guard alone. Some accepted witnesses (768/256 W512, 1024/256 W576, 128/64 W64)
    // leave their last rows no key, so the graph matcher declines them before this runs.
    const std::vector<Witness> witnesses{
        {256, 64, 512, 512, 128, true},
        {256, 64, 2048, 2048, 128, true},
        {256, 64, 768, 256, 512, true},
        {256, 64, 1024, 256, 576, true},
        {64, 64, 128, 64, 64, true},
        // The boundary: the last window starts exactly at the key-tile count.
        {256, 64, 1024, 256, 512, false},
        {64, 64, 192, 64, 64, false},
        {256, 64, 4096, 512, 128, false},
        {256, 64, 4096, 512, 2048, false},
        {256, 32, 1024, 256, 384, false},
        {512, 64, 2048, 512, 128, false},
    };
    for(const auto& w : witnesses)
    {
        EXPECT_EQ(gfx942AttentionDenseWindowReachesKeys(
                      w.blockM, w.blockN, w.seqLenQ, w.seqLenKv, w.window),
                  w.accepted)
            << "block " << w.blockM << "/" << w.blockN << ", Sq " << w.seqLenQ << ", Skv "
            << w.seqLenKv << ", window " << w.window;
    }
}

} // namespace gfx942

// =============================================================================
// gfx950
// =============================================================================

namespace gfx950
{
namespace
{

using hipdnn_plugin_sdk::HipdnnPluginException;

constexpr const char* KERNEL_NAME = "hipkernel:Gfx950AttentionDense/unit";

/// The two block_m values gfx950 builds (DENSE_TILE_GEOMETRIES).
constexpr int64_t BM256 = 256;
constexpr int64_t BM128 = 128;

/// attention_dense_block is `(spec.num_waves * 64, 1, 1)` and `num_waves = block_m // 32`:
/// (256 / 32) * 64 = 512 and (128 / 32) * 64 = 256.
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
// multiples of block_m, where the ceiling is the quotient; the non-multiple cases pin the
// behaviour outside that contract.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, AlignedSeqLenQIsOneBlockPerWholeTile)
{
    EXPECT_EQ(geometryFor(BM256, 256, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(BM256, 512, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM256, 4096, 1, 1).gridX, 16U);
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
    EXPECT_EQ(geometryFor(BM256, 513, 1, 1).gridX, 3U);
    EXPECT_EQ(geometryFor(BM256, 769, 1, 1).gridX, 4U);
    // ceil(384 / 256) = 2: a half-full final block, not a one-row one.
    EXPECT_EQ(geometryFor(BM256, 384, 1, 1).gridX, 2U);
    // Truncating gives 0 here, which is an empty grid: the kernel returns having written
    // nothing and reports success.
    EXPECT_EQ(geometryFor(BM256, 255, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(BM256, 1, 1, 1).gridX, 1U);
    EXPECT_EQ(geometryFor(BM128, 129, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM128, 197, 1, 1).gridX, 2U);
    EXPECT_EQ(geometryFor(BM128, 127, 1, 1).gridX, 1U);
}

// =============================================================================
// The full triple -- (nqb, num_query_heads, batch), in that order.
// =============================================================================

TEST(TestGfx950AttentionDenseGeometry, GridYIsQueryHeadsAndGridZIsBatch)
{
    // Heads and batch pass through untouched and are mutually distinct, so an exchanged
    // pair cannot read as correct.
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

/// Representative launches, read off attention_dense_grid/_block for each
/// block_m, at the qb_minor non-persistent block order. The same graph launches
/// differently per selected tile, so a prepare() that used one tile's block_m for
/// another's binary produces exactly one of these mismatches.
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
    for(const int64_t seqLenQ : {4096, 257, 1})
    {
        EXPECT_EQ(geometryFor(BM256, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM256) << seqLenQ;
        EXPECT_EQ(geometryFor(BM128, seqLenQ, 8, 4).blockX, PYTHON_BLOCK_X_BM128) << seqLenQ;
    }
}

TEST(TestGfx950AttentionDenseGeometry, BlockMIsTheTileTheCtaAndTheGridAgreeOn)
{
    // The same block_m divides the query grid and sizes the CTA; a shape of exactly one
    // tile makes both readings visible on one call.
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
    EXPECT_NO_THROW(geometryFor(BM256, 1, 1, 1));
    const Gfx950AttentionDenseGeometry expectedBm256{1U, 1U, 1U, PYTHON_BLOCK_X_BM256};
    EXPECT_TRUE(geometryFor(BM256, 1, 1, 1) == expectedBm256);
    const Gfx950AttentionDenseGeometry expectedBm128{1U, 1U, 1U, PYTHON_BLOCK_X_BM128};
    EXPECT_TRUE(geometryFor(BM128, 1, 1, 1) == expectedBm128);
}

TEST(TestGfx950AttentionDenseGeometry, RejectionNamesTheKernel)
{
    // The diagnostic exists so a failure identifies the descriptor that declared the
    // shape.
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

} // namespace gfx950

// =============================================================================
// Both arches: the extent guard reports the graph's values with the kernel name, since
// the extents come from the graph and the name from the descriptor.
// =============================================================================

namespace
{

struct GeometryArch
{
    const char* label;
    /// Calls the arch's geometry function at block_m 256, which both build.
    void (*launch)(int64_t seqLenQ, int64_t numQueryHeads, int64_t batch);
    const char* kernelName;
};

const GeometryArch GFX942_GEOMETRY{"Gfx942",
                                   [](int64_t seqLenQ, int64_t numQueryHeads, int64_t batch) {
                                       gfx942AttentionDenseGeometry(
                                           256, seqLenQ, numQueryHeads, batch, gfx942::KERNEL_NAME);
                                   },
                                   gfx942::KERNEL_NAME};

const GeometryArch GFX950_GEOMETRY{"Gfx950",
                                   [](int64_t seqLenQ, int64_t numQueryHeads, int64_t batch) {
                                       gfx950AttentionDenseGeometry(
                                           256, seqLenQ, numQueryHeads, batch, gfx950::KERNEL_NAME);
                                   },
                                   gfx950::KERNEL_NAME};

class TestAttentionDenseGeometry : public ::testing::TestWithParam<const GeometryArch*>
{
};

} // namespace

TEST_P(TestAttentionDenseGeometry, NonPositiveExtentRejectionPrintsTheGraphExtents)
{
    // Mutually distinct, so an exchanged pair cannot read as correct.
    const std::vector<std::vector<int64_t>> extents{{0, 8, 4}, {512, -3, 4}, {512, 8, 0}};
    for(const auto& extent : extents)
    {
        try
        {
            GetParam()->launch(extent[0], extent[1], extent[2]);
            ADD_FAILURE() << "expected seqlen_q " << extent[0] << ", num_query_heads " << extent[1]
                          << ", batch " << extent[2] << " to be rejected";
        }
        catch(const hipdnn_plugin_sdk::HipdnnPluginException& e)
        {
            const std::string message = e.what();
            EXPECT_NE(message.find(GetParam()->kernelName), std::string::npos) << message;
            const std::string values = "non-positive extent: seqlen_q " + std::to_string(extent[0])
                                       + ", num_query_heads " + std::to_string(extent[1])
                                       + ", batch " + std::to_string(extent[2]);
            EXPECT_NE(message.find(values), std::string::npos) << message;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(,
                         TestAttentionDenseGeometry,
                         ::testing::Values(&GFX942_GEOMETRY, &GFX950_GEOMETRY),
                         [](const ::testing::TestParamInfo<const GeometryArch*>& info) {
                             return std::string(info.param->label);
                         });

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
