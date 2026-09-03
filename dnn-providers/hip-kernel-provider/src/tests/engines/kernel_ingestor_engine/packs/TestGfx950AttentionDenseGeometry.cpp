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
 *  1. **The ceiling is live.** gfx950 serves RAGGED shapes, where `Sq % block_m != 0`
 *     is legal and the final query block is partial. On gfx942 every servable shape
 *     has `Sq % block_m == 0`, so floor and ceil agree and a truncating bug is
 *     invisible. Here it drops the tail rows of a real, shipped variant, so the ragged
 *     families below are the ones that matter most.
 *  2. **The tile is a spec field, as of ROCm/rocm-libraries#11627, and the test used to
 *     assert the opposite.** This file previously carried a
 *     `BlockIsAlwaysTheBakedWaveCount` case asserting a constant 512 lanes for every
 *     variant, justified by `_BLOCK_M` being a module constant. #11627 deleted that
 *     constant and made `block_m` a real field with two legal values, so the CTA is
 *     `block_m * 2` lanes and the grid's ceil divides by the variant's own tile. The
 *     old assertion would still have PASSED -- every family here is bm256 -- which is
 *     precisely why it was worthless as a defence. The bm128 families below are what
 *     make the parameterisation load-bearing.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

/// The two tile geometries `supports_attention_dense` admits
/// (kernels/gfx950/attention_dense.py:591-598, from `DENSE_TILE_GEOMETRIES`).
/// Restated here as literals rather than reusing the header's constants: this file is
/// the independent statement, and importing the values under test would make the
/// comparison circular.
constexpr int64_t BLOCK_M_DEFAULT = 256;
constexpr int64_t BLOCK_M_BM128 = 128;
/// MI355X CU count, the gfx950 dispatcher's default (dispatch/attention/gfx950.py).
/// NOT 304 -- that is the gfx942 value, and the kernel's own note records 304 as
/// measured WORSE here because it oversubscribes the CUs.
constexpr int64_t NUM_PERSISTENT = 256;
/// `attention_dense_block` is `(num_waves * 64, 1, 1)` with `num_waves = block_m // 32`
/// (kernels/gfx950/attention_dense.py:483-484, 2313-2315), i.e. `block_m * 2` lanes.
constexpr unsigned expectedBlockX(int64_t blockM)
{
    return static_cast<unsigned>(blockM / 32 * 64);
}
constexpr int64_t OFF = 0;
constexpr int64_t ON = 1;

/// One servable shape, named the way the descriptor names it, plus the tile the
/// variant serving it was compiled for. `blockM` is part of the family rather than a
/// global because it is a per-variant field now: two variants can serve the same
/// logical shape with different tiles, and they need different grids.
struct Family
{
    const char* name;
    int64_t seqLenQ;
    int64_t numQueryHeads;
    int64_t batch;
    int64_t blockM;
};

Gfx950AttentionDenseGeometry geometryFor(const Family& family, int64_t persistent)
{
    return gfx950AttentionDenseGeometry(family.seqLenQ,
                                        family.numQueryHeads,
                                        family.batch,
                                        persistent,
                                        NUM_PERSISTENT,
                                        family.blockM,
                                        family.name);
}

/// Tile-ALIGNED families spanning the real corpus: short prefill through
/// long-context, MHA and GQA head counts, batched and unbatched. Every one of these
/// is a shape the mined gfx950 corpus actually contains, at the dispatcher's own
/// bm256 geometry -- which is the only one it resolves.
const std::vector<Family>& alignedFamilies()
{
    static const std::vector<Family> s_kFamilies{
        {"sq512_hq64_b1", 512, 64, 1, BLOCK_M_DEFAULT},
        {"sq1024_hq64_b1", 1024, 64, 1, BLOCK_M_DEFAULT},
        {"sq2048_hq32_b1", 2048, 32, 1, BLOCK_M_DEFAULT},
        {"sq4096_hq32_b1", 4096, 32, 1, BLOCK_M_DEFAULT},
        {"sq8192_hq64_b2", 8192, 64, 2, BLOCK_M_DEFAULT},
        {"sq256_hq8_b1", 256, 8, 1, BLOCK_M_DEFAULT},
    };
    return s_kFamilies;
}

/// RAGGED families: `Sq % block_m != 0`, which gfx942 rejects outright and gfx950
/// serves through its on-chip boundary-padding path. The partial final block is the
/// whole point -- these are the shapes a truncating grid silently under-covers.
const std::vector<Family>& raggedFamilies()
{
    static const std::vector<Family> s_kFamilies{
        {"sq4000_hq32_b1", 4000, 32, 1, BLOCK_M_DEFAULT}, // one short of 16 blocks
        {"sq257_hq8_b1", 257, 8, 1, BLOCK_M_DEFAULT}, // one row into the second block
        {"sq1_hq8_b1", 1, 8, 1, BLOCK_M_DEFAULT}, // a single row: one whole block
        {"sq6143_hq16_b2", 6143, 16, 2, BLOCK_M_DEFAULT}, // one short of 24 blocks
    };
    return s_kFamilies;
}

/// The bm128 geometry, which #11627 introduced and the dispatcher does not currently
/// resolve to. These families are what make block_m's parameterisation testable: they
/// are chosen so a header still hardcoding 256 gets a DIFFERENT answer, which is the
/// only way this file can defend the change.
///
/// `sq384` is the sharp one: 384 % 128 == 0 but 384 % 256 != 0, so the old code would
/// have called it ragged and given it 2 blocks where bm128 needs 3.
const std::vector<Family>& bm128Families()
{
    static const std::vector<Family> s_kFamilies{
        {"sq384_hq32_b1_bm128", 384, 32, 1, BLOCK_M_BM128}, // aligned at 128, not at 256
        {"sq1024_hq64_b1_bm128", 1024, 64, 1, BLOCK_M_BM128}, // aligned at both
        {"sq129_hq8_b1_bm128", 129, 8, 1, BLOCK_M_BM128}, // ragged at both
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
        // Restated from the Python, not from the engine: ceil(Sq / spec.block_m).
        const int64_t expectedBlocks = (family.seqLenQ + family.blockM - 1) / family.blockM;
        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(expectedBlocks)) << family.name;
        EXPECT_EQ(geometry.gridY, static_cast<unsigned>(family.numQueryHeads)) << family.name;
        EXPECT_EQ(geometry.gridZ, static_cast<unsigned>(family.batch)) << family.name;
    }
}

TEST(TestGfx950AttentionDenseGeometry, DefaultArmCoversEveryQueryRowExactly)
{
    // The property underneath the arithmetic, and the one whose violation is silent:
    // gridX * block_m must cover Sq with no row left over. Under-covering leaves rows
    // unwritten; over-covering writes some twice.
    for(const auto& family : alignedFamilies())
    {
        const auto geometry = geometryFor(family, OFF);
        const int64_t covered = static_cast<int64_t>(geometry.gridX) * family.blockM;
        EXPECT_GE(covered, family.seqLenQ) << family.name << ": query rows unwritten";
        EXPECT_LT(covered - family.blockM, family.seqLenQ)
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
        const int64_t expectedBlocks = (family.seqLenQ + family.blockM - 1) / family.blockM;
        const int64_t truncating = family.seqLenQ / family.blockM;

        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(expectedBlocks)) << family.name;
        EXPECT_GT(expectedBlocks, truncating)
            << family.name
            << ": this family is not actually ragged, so it cannot defend the ceiling";
        EXPECT_GE(static_cast<int64_t>(geometry.gridX) * family.blockM, family.seqLenQ)
            << family.name << ": tail query rows would never be written";
    }
}

TEST(TestGfx950AttentionDenseGeometry, ASingleQueryRowStillLaunchesOneBlock)
{
    // The degenerate end of the ragged range: floor would launch ZERO blocks and
    // write nothing at all, exiting 0.
    const Family family{"sq1", 1, 8, 1, BLOCK_M_DEFAULT};
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
    // wrong for its binary. block_m is included in "shape terms" deliberately: it
    // reaches the DEFAULT arm's grid and must not reach this one.
    const auto small = geometryFor({"small", 512, 8, 1, BLOCK_M_DEFAULT}, ON);
    const auto large = geometryFor({"large", 16384, 64, 4, BLOCK_M_DEFAULT}, ON);
    EXPECT_EQ(small.gridX, large.gridX);
    EXPECT_EQ(small.gridY, large.gridY);
    EXPECT_EQ(small.gridZ, large.gridZ);
}

TEST(TestGfx950AttentionDenseGeometry, TheTwoArmsDisagreeOnGridRank)
{
    // Guards against a refactor that collapses the branch: if these ever matched,
    // one arm would be launching the other's grid.
    const Family family{"sq4096_hq32_b1", 4096, 32, 1, BLOCK_M_DEFAULT};
    EXPECT_FALSE(geometryFor(family, OFF) == geometryFor(family, ON));
}

// ---------------------------------------------------------------------------
// The block and the grid both scale with block_m -- the #11627 change
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, BlockIsTwoLanesPerQueryRowOfTheTile)
{
    // `num_waves * 64` with `num_waves = block_m // 32`, i.e. block_m * 2 lanes.
    // This used to assert a flat 512 for every variant, justified by `_BLOCK_M` being
    // a module constant; #11627 made it a field and the flat assertion would still
    // have passed, because every family it looped over was bm256. The bm128 loop is
    // what makes this a defence rather than a restatement.
    for(const auto& family : alignedFamilies())
    {
        EXPECT_EQ(geometryFor(family, OFF).blockX, expectedBlockX(family.blockM)) << family.name;
        EXPECT_EQ(geometryFor(family, ON).blockX, expectedBlockX(family.blockM)) << family.name;
    }
    for(const auto& family : raggedFamilies())
    {
        EXPECT_EQ(geometryFor(family, OFF).blockX, expectedBlockX(family.blockM)) << family.name;
    }
    for(const auto& family : bm128Families())
    {
        EXPECT_EQ(geometryFor(family, OFF).blockX, 256U) << family.name;
        EXPECT_EQ(geometryFor(family, ON).blockX, 256U) << family.name;
    }
}

TEST(TestGfx950AttentionDenseGeometry, Bm128TilesTheGridWithItsOwnBlockM)
{
    for(const auto& family : bm128Families())
    {
        const auto geometry = geometryFor(family, OFF);
        const int64_t expectedBlocks = (family.seqLenQ + family.blockM - 1) / family.blockM;
        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(expectedBlocks)) << family.name;
        EXPECT_GE(static_cast<int64_t>(geometry.gridX) * family.blockM, family.seqLenQ)
            << family.name << ": query rows unwritten";
    }
}

TEST(TestGfx950AttentionDenseGeometry, ATileThatOnlyDividesAt128GetsThreeBlocksNotTwo)
{
    // The single assertion that a header still hardcoding 256 fails. Sq=384 is exactly
    // three bm128 blocks and one-and-a-half bm256 ones, so the old constant-folded
    // arithmetic returns 2 -- under-covering by 128 query rows, silently, on a binary
    // whose CTA is also half the size the old code assumed.
    const Family bm128{"sq384_bm128", 384, 32, 1, BLOCK_M_BM128};
    const Family bm256{"sq384_bm256", 384, 32, 1, BLOCK_M_DEFAULT};

    EXPECT_EQ(geometryFor(bm128, OFF).gridX, 3U);
    EXPECT_EQ(geometryFor(bm256, OFF).gridX, 2U);
    EXPECT_EQ(geometryFor(bm128, OFF).blockX, 256U);
    EXPECT_EQ(geometryFor(bm256, OFF).blockX, 512U);
}

// ---------------------------------------------------------------------------
// Degenerate metadata fails LOUDLY rather than launching an empty grid
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGeometry, APersistentVariantWithoutCtasThrows)
{
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, ON, 0, BLOCK_M_DEFAULT, "bad_np"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, ON, -1, BLOCK_M_DEFAULT, "negative_np"),
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
    const int64_t tile = BLOCK_M_DEFAULT;

    EXPECT_THROW(gfx950AttentionDenseGeometry(0, 32, 1, notPersistent, ctaCount, tile, "zero_sq"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 0, 1, notPersistent, ctaCount, tile, "zero_hq"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 0, notPersistent, ctaCount, tile, "zero_b"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, ADegenerateBlockMThrowsOnBothArms)
{
    // block_m divides in BOTH the CTA size and the default grid's ceil, so zero is a
    // divide-by-zero rather than a wrong answer, and a non-multiple of 32 is a CTA
    // size the kernel was never built with. The persistent arm is checked too: it does
    // not use block_m for its grid, but it DOES use it for the block.
    const int64_t ctaCount = NUM_PERSISTENT;
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, OFF, ctaCount, 0, "zero_bm"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, OFF, ctaCount, -256, "negative_bm"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, OFF, ctaCount, 100, "unaligned_bm"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(gfx950AttentionDenseGeometry(4096, 32, 1, ON, ctaCount, 0, "zero_bm_persistent"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx950AttentionDenseGeometry, APersistentVariantDoesNotCareAboutShapeTerms)
{
    // Control for the throw above: the persistent arm does not index gridY/gridZ, so
    // a zero head count is not a reason to refuse it. If this threw, the guard would
    // be over-firing and declining variants the kernel serves.
    const int64_t isPersistent = ON;
    const int64_t ctaCount = NUM_PERSISTENT;
    EXPECT_NO_THROW(gfx950AttentionDenseGeometry(
        0, 0, 0, isPersistent, ctaCount, BLOCK_M_DEFAULT, "persistent"));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
