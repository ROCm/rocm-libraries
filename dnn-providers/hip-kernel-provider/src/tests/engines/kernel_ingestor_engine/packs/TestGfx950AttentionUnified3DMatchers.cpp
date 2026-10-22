// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified3DMatchers.cpp
 * @brief hipkernel:Gfx950AttentionUnified3D's graph match, kernel match and score.
 *
 * The rules that are this engine's own, each against the neighbouring graph or record that
 * must be refused: decode only (a longer query leaves rows unwritten), a GQA group dividing
 * 16, D256 at bf16 only, the NO_MASK decode graph served by the bottom-right causal kernel,
 * and a record that is not a decode-grid kernel with exactly one companion. The SDPA-shape
 * rules it shares with Unified2D are spot-checked; Unified2D's suite covers them in full.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace unified3d;
using hipdnn_plugin_sdk::ingestor::tryGetBoundInt;

int64_t boundMask(const GraphSpec& spec)
{
    const auto bound = matchGraph(spec);
    EXPECT_TRUE(bound.has_value());
    return bound.has_value() ? tryGetBoundInt(*bound, MASK_TOKEN).value_or(-1) : -1;
}

} // namespace

// =============================================================================
// graph_match
// =============================================================================

TEST(TestGfx950AttentionUnified3DMatchers, AcceptsTheDefaultDecodeGraph)
{
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
}

/// The decode grid launches one CTA per sequence and reads one query row from it; the rows
/// after it would be left unwritten with a success status.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesAQueryLongerThanOneToken)
{
    for(const int64_t seqLenQ : {2, 16, 128})
    {
        SCOPED_TRACE("seqlen_q " + std::to_string(seqLenQ));
        GraphSpec spec;
        spec.seqLenQ = seqLenQ;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
}

/// The segment kernel fixes BLOCK_M at 16 rows, so the group must divide it: 8 over 2 (4)
/// and 16 over 1 are served; 12 over 2 (6) is a whole group that does not divide 16.
TEST(TestGfx950AttentionUnified3DMatchers, NeedsAGqaGroupDividingSixteen)
{
    GraphSpec dividing;
    dividing.numQueryHeads = 16;
    dividing.numKvHeads = 1;
    EXPECT_TRUE(matchGraph(dividing).has_value());

    GraphSpec six;
    six.numQueryHeads = 12;
    six.numKvHeads = 2;
    EXPECT_FALSE(matchGraph(six).has_value());

    GraphSpec thirtyTwo;
    thirtyTwo.numQueryHeads = 32;
    thirtyTwo.numKvHeads = 1;
    EXPECT_FALSE(matchGraph(thirtyTwo).has_value());
}

/// rocKE's d256_decode cohort is bf16 only.
TEST(TestGfx950AttentionUnified3DMatchers, ServesD256AtBf16Only)
{
    GraphSpec bf16;
    bf16.headSize = 256;
    EXPECT_TRUE(matchGraph(bf16).has_value());

    GraphSpec fp16 = bf16;
    fp16.dataType = data_objects::DataType::HALF;
    EXPECT_FALSE(matchGraph(fp16).has_value());

    GraphSpec fp16d128;
    fp16d128.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(matchGraph(fp16d128).has_value());
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesAHeadSizeOutsideTheCatalog)
{
    GraphSpec spec;
    spec.headSize = 96;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

/// One query row sees every key under full attention and under bottom-right causal alike,
/// so the catalog carries one kernel for both and the match binds that kernel's key. Top-left
/// causal (one row sees key 0 only) keeps its own.
TEST(TestGfx950AttentionUnified3DMatchers, BindsNoMaskDecodeAsBottomRightCausal)
{
    GraphSpec none;
    setMask(none, NO_MASK);
    EXPECT_EQ(boundMask(none), BOTTOM_RIGHT_CAUSAL);

    GraphSpec bottomRight;
    setMask(bottomRight, BOTTOM_RIGHT_CAUSAL);
    EXPECT_EQ(boundMask(bottomRight), BOTTOM_RIGHT_CAUSAL);

    GraphSpec topLeft;
    setMask(topLeft, TOP_LEFT_CAUSAL);
    EXPECT_EQ(boundMask(topLeft), TOP_LEFT_CAUSAL);
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesASlidingWindow)
{
    GraphSpec spec;
    spec.leftBound = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesANonBshdKeyCache)
{
    GraphSpec spec;
    spec.kLayout = Layout::BHSD;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesAKeyLengthNotAWholeNumberOfBlocks)
{
    GraphSpec spec;
    spec.seqLenKv = 100;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesAnAdditiveMask)
{
    GraphSpec spec;
    spec.extra = [](data_objects::SdpaAttributesBuilder& builder) {
        builder.add_attn_mask_tensor_uid(EXTRA_UID);
    };
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// =============================================================================
// kernel_match
// =============================================================================

TEST(TestGfx950AttentionUnified3DMatchers, MatchesTheShippedKernelForTheDefaultGraph)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx950AttentionUnified3DMatchers, MatchesTheBottomRightKernelForANoMaskGraph)
{
    GraphSpec graph;
    setMask(graph, NO_MASK);
    KernelSpec bottomRight;
    bottomRight.maskType = BOTTOM_RIGHT_CAUSAL;
    EXPECT_TRUE(matchesKernel(graph, bottomRight));
    EXPECT_FALSE(matchesKernel(graph, KernelSpec{}));
}

/// Each baked field against the graph, one at a time.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARecordThatDisagreesWithTheGraph)
{
    KernelSpec dtype;
    dtype.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, dtype));

    KernelSpec head;
    head.headSize = 64;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, head));

    KernelSpec queryHeads;
    queryHeads.numQueryHeads = 16;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, queryHeads));

    KernelSpec kvHeads;
    kvHeads.numKvHeads = 4;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kvHeads));

    KernelSpec mask;
    mask.maskType = BOTTOM_RIGHT_CAUSAL;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, mask));

    KernelSpec window;
    window.slidingWindow = 128;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, window));
}

/// Without its reduce kernel the segment kernel's partial results are never combined and
/// the output is never written; with two, which one reduces is a guess.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARecordWithoutExactlyOneCompanion)
{
    KernelSpec none;
    none.companions = 0;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, none));

    KernelSpec two;
    two.companions = 2;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, two));
}

/// This engine launches the decode grid only; a prefill-grid record would be launched with
/// the wrong grid.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARecordThatIsNotADecodeGridKernel)
{
    KernelSpec prefill;
    prefill.useDecodeGrid = 0;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, prefill));
}

/// Decode carries no NO_MASK kernel; a record claiming one is malformed.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARecordKeyedNoMask)
{
    GraphSpec graph;
    setMask(graph, NO_MASK);
    KernelSpec none;
    none.maskType = NO_MASK;
    EXPECT_FALSE(matchesKernel(graph, none));
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARecordWithAnUnbuildableGeometry)
{
    KernelSpec segments;
    segments.numSegments = 0;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, segments));

    KernelSpec block;
    block.blockSize = 24;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, block));
}

/// The identity cache is whole blocks of the candidate's own size: Skv 272 is seventeen
/// 16-token blocks, and not a whole number of 32-token ones.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesABlockSizeThatDoesNotDivideTheKeyLength)
{
    GraphSpec graph;
    graph.seqLenKv = 272;
    EXPECT_TRUE(matchesKernel(graph, KernelSpec{}));
    KernelSpec block32;
    block32.blockSize = 32;
    EXPECT_FALSE(matchesKernel(graph, block32));
}

TEST(TestGfx950AttentionUnified3DMatchers, MatchesTheD256KernelForABf16D256Graph)
{
    GraphSpec graph;
    graph.headSize = 256;
    KernelSpec d256;
    d256.headSize = 256;
    d256.numSegments = 128;
    EXPECT_TRUE(matchesKernel(graph, d256));
}

// =============================================================================
// score
// =============================================================================

TEST(TestGfx950AttentionUnified3DMatchers, RanksFewerSegmentsFirstWithDistinctScores)
{
    const KernelSpec sixteen;
    KernelSpec thirtyTwo;
    thirtyTwo.numSegments = 32;
    KernelSpec oneTwentyEight;
    oneTwentyEight.numSegments = 128;
    const double a = scoreOf(GraphSpec{}, sixteen);
    const double b = scoreOf(GraphSpec{}, thirtyTwo);
    const double c = scoreOf(GraphSpec{}, oneTwentyEight);
    EXPECT_GT(a, b);
    EXPECT_GT(b, c);
    EXPECT_GT(c, 0.0);
}

TEST(TestGfx950AttentionUnified3DMatchers, ScoresAnUnlaunchableRecordZero)
{
    KernelSpec none;
    none.companions = 0;
    EXPECT_EQ(scoreOf(GraphSpec{}, none), 0.0);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
