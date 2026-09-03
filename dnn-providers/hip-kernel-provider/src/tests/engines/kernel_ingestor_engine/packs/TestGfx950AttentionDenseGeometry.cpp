// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"

/**
 * @file TestGfx950AttentionDenseGeometry.cpp
 * @brief The launch contract this engine restates from Python, per shape family.
 *
 * The engine relaunches a rocKE binary from C++, recomputing by hand what the Python
 * launch path computes. Nothing in the build, the packer, the validator or the rest
 * of this suite compares the two, and a mismatch does not fail -- the kernel runs,
 * returns, and leaves part of the output untouched.
 *
 * These are the SECOND, independent statement of the rule. The assertions below spell
 * the expected grid out per shape family rather than calling a shared helper: a test
 * that recomputes with the same expression as the code under test agrees with it by
 * construction, including when both are wrong. Where the arithmetic is restated it is
 * restated from the PYTHON (`attention_dense_grid`,
 * kernels/gfx950/attention_dense.py:1874), which is the actual source of truth.
 *
 * TWO THINGS DIFFER FROM THE gfx942 TWIN, and both are why this file is not that one
 * with the numbers changed:
 *
 *  1. **The ceiling is live.** gfx950 serves RAGGED shapes, where `Sq % 256 != 0` is
 *     legal and the final query block is partial. On gfx942 every servable shape has
 *     `Sq % block_m == 0`, so floor and ceil agree and a truncating bug is invisible.
 *     Here it drops the tail rows of a real, shipped variant, so the ragged families
 *     below are the ones that matter most.
 *  2. **The block is constant.** `_BLOCK_M` is a module constant, not a spec field, so
 *     there is no block_m to pass and every variant launches 512 lanes. The test
 *     asserts that invariant directly rather than deriving it per family.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

/// Baked into the kernel as `_BLOCK_M` (kernels/gfx950/attention_dense.py:88).
/// Restated here as a literal rather than reusing the header's constant: this file is
/// the independent statement, and importing the value under test would make the
/// comparison circular.
constexpr int64_t BLOCK_M = 256;
/// MI355X CU count, the gfx950 dispatcher's default (dispatch/attention/gfx950.py).
/// NOT 304 -- that is the gfx942 value, and the kernel's own note records 304 as
/// measured WORSE here because it oversubscribes the CUs.
constexpr int64_t NUM_PERSISTENT = 256;
/// `attention_dense_block` is `(num_waves * 64, 1, 1)` with `num_waves = 256 // 32`.
constexpr unsigned EXPECTED_BLOCK_X = 512;
constexpr int64_t OFF = 0;
constexpr int64_t ON = 1;

/// One servable shape, named the way the descriptor names it.
struct Family
{
    const char* name;
    int64_t seqLenQ;
    int64_t numQueryHeads;
    int64_t batch;
};

Gfx950AttentionDenseGeometry geometryFor(const Family& family, int64_t persistent)
{
    return gfx950AttentionDenseGeometry(family.seqLenQ,
                                        family.numQueryHeads,
                                        family.batch,
                                        persistent,
                                        NUM_PERSISTENT,
                                        family.name);
}

/// Tile-ALIGNED families spanning the real corpus: short prefill through
/// long-context, MHA and GQA head counts, batched and unbatched. Every one of these
/// is a shape the mined gfx950 corpus actually contains.
const std::vector<Family>& alignedFamilies()
{
    static const std::vector<Family> s_kFamilies{
        {"sq512_hq64_b1", 512, 64, 1},
        {"sq1024_hq64_b1", 1024, 64, 1},
        {"sq2048_hq32_b1", 2048, 32, 1},
        {"sq4096_hq32_b1", 4096, 32, 1},
        {"sq8192_hq64_b2", 8192, 64, 2},
        {"sq256_hq8_b1", 256, 8, 1},
    };
    return s_kFamilies;
}

/// RAGGED families: `Sq % 256 != 0`, which gfx942 rejects outright and gfx950 serves
/// through its on-chip boundary-padding path. The partial final block is the whole
/// point -- these are the shapes a truncating grid silently under-covers.
const std::vector<Family>& raggedFamilies()
{
    static const std::vector<Family> s_kFamilies{
        {"sq4000_hq32_b1", 4000, 32, 1}, // one short of 16 blocks
        {"sq257_hq8_b1", 257, 8, 1}, // one row into the second block
        {"sq1_hq8_b1", 1, 8, 1}, // a single row: one whole block
        {"sq6143_hq16_b2", 6143, 16, 2}, // one short of 24 blocks
    };
    return s_kFamilies;
}

} // namespace

// ---------------------------------------------------------------------------
// The default arm: a 3-D grid over (query blocks, heads, batch)
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, DefaultArmTilesQueryBlocksByHeadsByBatch)
{
    for(const auto& family : alignedFamilies())
    {
        const auto geometry = geometryFor(family, OFF);
        // Restated from the Python, not from the engine: ceil(Sq / _BLOCK_M).
        const int64_t expectedBlocks = (family.seqLenQ + BLOCK_M - 1) / BLOCK_M;
        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(expectedBlocks)) << family.name;
        EXPECT_EQ(geometry.gridY, static_cast<unsigned>(family.numQueryHeads)) << family.name;
        EXPECT_EQ(geometry.gridZ, static_cast<unsigned>(family.batch)) << family.name;
    }
}

TEST(TestGfx950AttentionDenseGeometry, DefaultArmCoversEveryQueryRowExactly)
{
    // The property underneath the arithmetic, and the one whose violation is silent:
    // gridX * BLOCK_M must cover Sq with no row left over. Under-covering leaves rows
    // unwritten; over-covering writes some twice.
    for(const auto& family : alignedFamilies())
    {
        const auto geometry = geometryFor(family, OFF);
        const int64_t covered = static_cast<int64_t>(geometry.gridX) * BLOCK_M;
        EXPECT_GE(covered, family.seqLenQ) << family.name << ": query rows unwritten";
        EXPECT_LT(covered - BLOCK_M, family.seqLenQ)
            << family.name << ": an entire redundant query block";
    }
}

// ---------------------------------------------------------------------------
// Ragged shapes: the arm gfx942 cannot express at all
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, RaggedShapesGetAPartialFinalBlock)
{
    // THE regression this file exists for. A floor divide here under-covers by one
    // block: the tail query rows are never written, the kernel returns cleanly, and
    // nothing reports it. gfx942's suite cannot catch this because gfx942 declines
    // every shape that reaches it.
    for(const auto& family : raggedFamilies())
    {
        const auto geometry = geometryFor(family, OFF);
        const int64_t expectedBlocks = (family.seqLenQ + BLOCK_M - 1) / BLOCK_M;
        const int64_t truncating = family.seqLenQ / BLOCK_M;

        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(expectedBlocks)) << family.name;
        EXPECT_GT(expectedBlocks, truncating)
            << family.name
            << ": this family is not actually ragged, so it cannot defend the ceiling";
        EXPECT_GE(static_cast<int64_t>(geometry.gridX) * BLOCK_M, family.seqLenQ)
            << family.name << ": tail query rows would never be written";
    }
}

TEST(TestGfx950AttentionDenseGeometry, ASingleQueryRowStillLaunchesOneBlock)
{
    // The degenerate end of the ragged range: floor would launch ZERO blocks and
    // write nothing at all, exiting 0.
    const Family family{"sq1", 1, 8, 1};
    const auto geometry = geometryFor(family, OFF);
    EXPECT_EQ(geometry.gridX, 1U);
}

// ---------------------------------------------------------------------------
// The persistent arm: a 1-D grid of num_persistent CTAs
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, PersistentArmIsAOneDimensionalGridOfCtas)
{
    // A different binary with a different grid RANK. Launching it on the default 3-D
    // grid leaves output rows unwritten, with no error anywhere.
    for(const auto& family : alignedFamilies())
    {
        const auto geometry = geometryFor(family, ON);
        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(NUM_PERSISTENT)) << family.name;
        EXPECT_EQ(geometry.gridY, 1U) << family.name;
        EXPECT_EQ(geometry.gridZ, 1U) << family.name;
    }
}

TEST(TestGfx950AttentionDenseGeometry, PersistentArmIgnoresTheShapeEntirely)
{
    // The persistent grid is a function of num_persistent ALONE. If a shape term
    // leaked into it, two shapes would produce two grids and one of them would be
    // wrong for its binary.
    const auto small = geometryFor({"small", 512, 8, 1}, ON);
    const auto large = geometryFor({"large", 16384, 64, 4}, ON);
    EXPECT_EQ(small, large);
}

TEST(TestGfx950AttentionDenseGeometry, TheTwoArmsDisagreeOnGridRank)
{
    // Guards against a refactor that collapses the branch: if these ever matched,
    // one arm would be launching the other's grid.
    const Family family{"sq4096_hq32_b1", 4096, 32, 1};
    EXPECT_FALSE(geometryFor(family, OFF) == geometryFor(family, ON));
}

// ---------------------------------------------------------------------------
// The block, which is constant on this arch
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, BlockIsAlwaysTheBakedWaveCount)
{
    // gfx950 has no block_m knob: `_BLOCK_M` is a module constant, so every variant
    // launches the same 512 lanes regardless of shape or arm. Asserted directly
    // because a future block_m field would silently invalidate the geometry header's
    // constant-folding.
    for(const auto& family : alignedFamilies())
    {
        EXPECT_EQ(geometryFor(family, OFF).blockX, EXPECTED_BLOCK_X) << family.name;
        EXPECT_EQ(geometryFor(family, ON).blockX, EXPECTED_BLOCK_X) << family.name;
    }
    for(const auto& family : raggedFamilies())
    {
        EXPECT_EQ(geometryFor(family, OFF).blockX, EXPECTED_BLOCK_X) << family.name;
    }
}

// ---------------------------------------------------------------------------
// Degenerate metadata fails LOUDLY rather than launching an empty grid
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, APersistentVariantWithoutCtasThrows)
{
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, ON, 0, "bad_np"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, ON, -1, "negative_np"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, ADefaultVariantWithANonPositiveShapeThrows)
{
    // An empty launch returns cleanly having written nothing, which is the
    // silent-wrong-answer case; prepare() is the last place a named failure is cheap.
    //
    // The named locals are not decoration: `OFF` next to `NUM_PERSISTENT` reads as a
    // plausible argument swap to clang-tidy (readability-suspicious-call-argument),
    // and it is right that the two are easy to transpose at a call site -- which is
    // exactly the class of mistake this file exists to catch in the ENGINE.
    const int64_t notPersistent = OFF;
    const int64_t ctaCount = NUM_PERSISTENT;

    EXPECT_THROW(gfx950AttentionDenseGeometry(0, 32, 1, notPersistent, ctaCount, "zero_sq"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 0, 1, notPersistent, ctaCount, "zero_hq"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 0, notPersistent, ctaCount, "zero_b"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, APersistentVariantDoesNotCareAboutShapeTerms)
{
    // Control for the throw above: the persistent arm does not index gridY/gridZ, so
    // a zero head count is not a reason to refuse it. If this threw, the guard would
    // be over-firing and declining variants the kernel serves.
    const int64_t isPersistent = ON;
    const int64_t ctaCount = NUM_PERSISTENT;
    EXPECT_NO_THROW(gfx950AttentionDenseGeometry(0, 0, 0, isPersistent, ctaCount, "persistent"));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
