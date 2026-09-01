// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseGeometry.hpp"

/**
 * @file TestGfx942AttentionDenseGeometry.cpp
 * @brief The launch contract this engine restates from Python, per shape family.
 *
 * The engine relaunches a rocKE binary from C++, recomputing by hand what the Python
 * launch path computes. Nothing in the build, the packer, the validator or the rest
 * of this suite compared the two, and a mismatch does not fail -- the kernel runs,
 * returns, and leaves part of the output untouched. Two defects shipped exactly this
 * way, and most shipped shape-keys never executed on GPU at all, so the coverage that
 * did exist was a minority of descriptors and excluded the risky branch almost
 * entirely.
 *
 * These are the SECOND, independent statement of the rule. The assertions below spell
 * the expected grid out per shape family rather than calling a shared helper: a test
 * that recomputes with the same expression as the code under test agrees with it by
 * construction, including when both are wrong. Where the arithmetic is restated it is
 * restated from the PYTHON (`attention_dense_grid`, attention_dense.py:1803), which is
 * the actual source of truth.
 *
 * Shape families, not sampled shapes. The dispatcher's `persistent` rule partitions
 * every servable shape into two arms with different grid RANKS, and the arm that was
 * never exercised is the one whose failure mode is silent. Each family below is a
 * region of that partition plus its boundary.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

constexpr int64_t BLOCK_M = 256; // baked; the kernel faults at other values
constexpr int64_t NUM_PERSISTENT = 304; // gfx942 CU count, from the dispatcher
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

/// `nqb * Hq * B`, the dispatcher's own work estimate
/// (dispatch/attention/gfx942.py::_dense_spec). Restated from the dispatcher, not
/// from the engine, so the two can disagree.
int64_t work(const Family& family)
{
    const int64_t queryBlocks = (family.seqLenQ + BLOCK_M - 1) / BLOCK_M;
    return queryBlocks * family.numQueryHeads * family.batch;
}

AttentionDenseGeometry geometryFor(const Family& family, int64_t persistent)
{
    return attentionDenseGeometry(BLOCK_M,
                                  family.seqLenQ,
                                  family.numQueryHeads,
                                  family.batch,
                                  persistent,
                                  NUM_PERSISTENT,
                                  family.name);
}

/// Families spanning the real corpus: short prefill through long-context, MHA and
/// GQA head counts, batched and unbatched.
const std::vector<Family>& families()
{
    static const std::vector<Family> s_kFamilies{
        {"sq512_hq32_b1", 512, 32, 1},
        {"sq2048_hq16_b1", 2048, 16, 1},
        {"sq4096_hq32_b1", 4096, 32, 1},
        {"sq8192_hq64_b2", 8192, 64, 2},
        {"sq16384_hq32_b1", 16384, 32, 1},
        {"sq256_hq8_b1", 256, 8, 1},
    };
    return s_kFamilies;
}

} // namespace

// ---------------------------------------------------------------------------
// The default arm: a 3-D grid over (query blocks, heads, batch)
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGeometry, DefaultArmTilesQueryBlocksByHeadsByBatch)
{
    for(const auto& family : families())
    {
        const auto geometry = geometryFor(family, OFF);
        // Restated from the Python, not from the engine: ceil(Sq / BLOCK_M).
        const int64_t expectedBlocks = (family.seqLenQ + BLOCK_M - 1) / BLOCK_M;
        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(expectedBlocks)) << family.name;
        EXPECT_EQ(geometry.gridY, static_cast<unsigned>(family.numQueryHeads)) << family.name;
        EXPECT_EQ(geometry.gridZ, static_cast<unsigned>(family.batch)) << family.name;
    }
}

TEST(TestGfx942AttentionDenseGeometry, DefaultArmCoversEveryQueryRowExactly)
{
    // The property underneath the arithmetic, and the one whose violation is silent:
    // gridX * BLOCK_M must cover Sq with no row left over. Under-covering leaves rows
    // unwritten; over-covering writes some twice.
    for(const auto& family : families())
    {
        const auto geometry = geometryFor(family, OFF);
        const int64_t covered = static_cast<int64_t>(geometry.gridX) * BLOCK_M;
        EXPECT_GE(covered, family.seqLenQ) << family.name << ": query rows unwritten";
        EXPECT_LT(covered - BLOCK_M, family.seqLenQ)
            << family.name << ": an entire redundant query block";
    }
}

TEST(TestGfx942AttentionDenseGeometry, DefaultArmRoundsQueryBlocksUpNotDown)
{
    // Every family above has Sq % BLOCK_M == 0, where floor and ceil AGREE -- so
    // none of them can tell the two apart, and a floor would pass the whole suite.
    // Found by mutation, and it is the same shape of blind spot that once made half
    // a tuning axis unreachable because every shipped shape divided evenly.
    //
    // `supports_attention_dense` enforces the divisibility, so a ragged Sq cannot
    // reach the engine TODAY. That is exactly why it is pinned here: the guard is
    // load-bearing on an invariant enforced in another language and another
    // repository, and if it is ever relaxed, rounding down silently drops the last
    // partial query block instead of failing.
    const int64_t ragged = BLOCK_M * 3 + 1;
    const int64_t persistent = OFF;
    const int64_t numPersistent = NUM_PERSISTENT;
    const auto geometry
        = attentionDenseGeometry(BLOCK_M, ragged, 8, 1, persistent, numPersistent, "ragged");
    EXPECT_EQ(geometry.gridX, 4U)
        << "a partial query block must get its own CTA; rounding down leaves those "
           "rows unwritten";
    EXPECT_GE(static_cast<int64_t>(geometry.gridX) * BLOCK_M, ragged);
}

// ---------------------------------------------------------------------------
// The persistent arm: a 1-D grid of num_persistent CTAs
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGeometry, PersistentArmLaunchesAFlatGridOfNumPersistent)
{
    // The branch that shipped missing. The engine implemented only the default arm,
    // so every persistent variant launched the 3-D grid and left output rows
    // unwritten -- no fault, no error, ~0.004% of elements wrong.
    for(const auto& family : families())
    {
        const auto geometry = geometryFor(family, ON);
        EXPECT_EQ(geometry.gridX, static_cast<unsigned>(NUM_PERSISTENT)) << family.name;
        EXPECT_EQ(geometry.gridY, 1U) << family.name;
        EXPECT_EQ(geometry.gridZ, 1U) << family.name;
    }
}

TEST(TestGfx942AttentionDenseGeometry, TheTwoArmsDisagreeOnEveryRealFamily)
{
    // If they ever agreed, every test above would pass with the branch deleted --
    // which is precisely the defect that shipped. This is the assertion that makes
    // the persistent case load-bearing rather than decorative.
    for(const auto& family : families())
    {
        EXPECT_FALSE(geometryFor(family, OFF) == geometryFor(family, ON))
            << family.name
            << ": the arms coincide, so this family cannot detect a "
               "missing persistent branch";
    }
}

// ---------------------------------------------------------------------------
// The block size, which both arms share
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGeometry, BlockIsWave64LanesPerQueryRowGroup)
{
    // attention_dense_block (:1822): (block_m // 32 * 64, 1, 1).
    for(const auto& family : families())
    {
        EXPECT_EQ(geometryFor(family, OFF).blockX, static_cast<unsigned>(BLOCK_M / 32 * 64))
            << family.name;
        EXPECT_EQ(geometryFor(family, ON).blockX, static_cast<unsigned>(BLOCK_M / 32 * 64))
            << family.name;
    }
}

// ---------------------------------------------------------------------------
// Agreement with the dispatcher's own rule
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGeometry, FamiliesStraddleThePersistentThreshold)
{
    // Guards the suite itself: if every family fell on one side, the arm tests would
    // still pass while covering one branch. The corpus must exercise both.
    bool anyBelow = false;
    bool anyAtOrAbove = false;
    for(const auto& family : families())
    {
        (work(family) >= NUM_PERSISTENT ? anyAtOrAbove : anyBelow) = true;
    }
    EXPECT_TRUE(anyBelow) << "no family below the persistent threshold";
    EXPECT_TRUE(anyAtOrAbove) << "no family at or above the persistent threshold";
}

// ---------------------------------------------------------------------------
// Refusals: a degenerate grid must be named, not launched
// ---------------------------------------------------------------------------

TEST(TestGfx942AttentionDenseGeometry, RefusesNonPositiveBlockM)
{
    // blockM reaches an integer division, so without the guard this is SIGFPE rather
    // than a wrong answer. Verified by mutation: deleting the guard takes the whole
    // test binary down with exit -8 instead of producing a clean failure. That is
    // still a detection -- nobody ships a suite that crashes -- but the guard is what
    // turns it into a NAMED refusal at prepare(), which is the difference between an
    // operator seeing which descriptor is malformed and seeing a dead process.
    // Named for the callee's parameters; OFF/NUM_PERSISTENT read as swapped otherwise.
    const int64_t persistent = OFF;
    const int64_t numPersistent = NUM_PERSISTENT;
    EXPECT_THROW(attentionDenseGeometry(0, 4096, 32, 1, persistent, numPersistent, "k"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx942AttentionDenseGeometry, RefusesPersistentWithoutACtaCount)
{
    // Would launch an empty or negative grid and return having written nothing.
    EXPECT_THROW(attentionDenseGeometry(BLOCK_M, 4096, 32, 1, ON, 0, "k"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx942AttentionDenseGeometry, RefusesADefaultArmThatWouldLaunchZeroCtas)
{
    // gridY/gridZ come straight from metadata, so a zero head count or batch is a
    // launch of nothing that reports success.
    // Named for the callee's parameters; OFF/NUM_PERSISTENT read as swapped otherwise.
    const int64_t persistent = OFF;
    const int64_t numPersistent = NUM_PERSISTENT;
    EXPECT_THROW(attentionDenseGeometry(BLOCK_M, 4096, 0, 1, persistent, numPersistent, "k"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(attentionDenseGeometry(BLOCK_M, 4096, 32, 0, persistent, numPersistent, "k"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
    EXPECT_THROW(attentionDenseGeometry(BLOCK_M, 0, 32, 1, persistent, numPersistent, "k"),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestGfx942AttentionDenseGeometry, PersistentArmIgnoresTheShapeItDoesNotIndex)
{
    // The persistent grid is (num_persistent, 1, 1) whatever the shape, so a zero
    // head count is NOT a refusal here -- the arm never indexes it. Pinned so the
    // guard above is not widened into the arm it does not apply to.
    EXPECT_NO_THROW(attentionDenseGeometry(BLOCK_M, 4096, 0, 0, ON, NUM_PERSISTENT, "k"));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
