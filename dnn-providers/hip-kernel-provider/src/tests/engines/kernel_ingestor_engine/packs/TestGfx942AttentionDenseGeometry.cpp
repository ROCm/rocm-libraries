// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <set>
#include <string>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"

/**
 * @file TestGfx942AttentionDenseGeometry.cpp
 * @brief Pins gfx942AttentionDenseGeometry(), the tile rules and the sliding-window guard
 *        against their Python originals.
 *
 * A geometry that disagrees with the Python does not fault, it computes something else,
 * and nothing else in the suite compares the two halves. Every expected number below is
 * read off `kernels/gfx942/attention_dense.py` (`attention_dense_grid`,
 * `attention_dense_block`, `supports_attention_dense`) and written as a literal; deriving
 * one from the expression under test would assert only that the expression equals itself.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
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
    // needs a 2048-thread CTA, past the workgroup limit.
    for(const int64_t blockM : {int64_t{0}, int64_t{-256}, int64_t{48}, int64_t{1024}})
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
    // supports_attention_dense's accepted tiles, plus the tiles it rejects ONLY on the LDS
    // budget, which this predicate does not restate (no such tile can be packed).
    const std::set<std::pair<int64_t, int64_t>> d64{{32, 32},
                                                    {64, 32},
                                                    {64, 64},
                                                    {128, 32},
                                                    {128, 64},
                                                    {128, 128},
                                                    {256, 32},
                                                    {256, 64},
                                                    {256, 128},
                                                    {512, 32},
                                                    {512, 64},
                                                    {512, 128},
                                                    // LDS budget only:
                                                    {256, 256},
                                                    {512, 256},
                                                    {512, 512}};
    const std::set<std::pair<int64_t, int64_t>> d128{{32, 32},
                                                     {64, 32},
                                                     {64, 64},
                                                     {128, 32},
                                                     {128, 64},
                                                     {256, 32},
                                                     {256, 64},
                                                     {512, 32},
                                                     {512, 64},
                                                     // LDS budget only:
                                                     {128, 128},
                                                     {256, 128},
                                                     {256, 256},
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
// The sliding-window zero-trip guard. Every verdict is supports_attention_dense's for a
// causal bf16 D128 spec at that tile and shape.
// =============================================================================

TEST(TestGfx942AttentionDenseWindow, NoWindowAlwaysReachesKeys)
{
    EXPECT_TRUE(gfx942AttentionDenseWindowReachesKeys(256, 64, 4096, 64, 0));
}

TEST(TestGfx942AttentionDenseWindow, MatchesThePythonVerdictAtEachWitness)
{
    struct Witness
    {
        int64_t blockM, blockN, seqLenQ, seqLenKv, window;
        bool accepted;
    };
    const Witness witnesses[] = {
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
