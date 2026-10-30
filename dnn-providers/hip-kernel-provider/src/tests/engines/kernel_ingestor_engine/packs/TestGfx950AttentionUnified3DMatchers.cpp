// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified3DMatchers.cpp
 * @brief hipkernel:Gfx950AttentionUnified3D's graph match, kernel match and score.
 *
 * The rules that are this engine's own, each against the neighbouring graph or record that
 * must be refused: decode only (a longer query leaves rows unwritten), a GQA group dividing
 * 16, D256 at bf16 only, the NO_MASK decode graph served by the bottom-right causal kernel,
 * and a record that is not a decode-grid kernel launching exactly its segment and reduce
 * kernels. graph_match is its own copy of Unified2D's (layout, shape/dtype agreement, the
 * 32-bit K/V addressing cap, and the alibi/padding/mma_core_mode/implementation declines), so
 * this suite repeats those rows against the decode shape rather than relying on Unified2D's
 * suite to cover its copy.
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

/// A named perturbation of the default graph that graph_match must decline.
struct DeclineCase
{
    std::string name;
    GraphSpec spec;
};

void expectEveryCaseDeclines(const std::vector<DeclineCase>& cases)
{
    ASSERT_FALSE(cases.empty());
    for(const auto& declineCase : cases)
    {
        EXPECT_FALSE(matchGraph(declineCase.spec).has_value())
            << declineCase.name << ": graph_match bound a graph the kernel cannot serve";
    }
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

/// The kernel reads K/V as a gap-free paged cache and takes no stride arguments, so a
/// differently-strided K or V is read as if it were this one: elements in bounds, wrong
/// values, no fault. Each operand on its own, in each wrong layout.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesAKeyOrValueThatIsNotDenseBshd)
{
    std::vector<DeclineCase> cases;
    for(const auto layout : {Layout::BHSD, Layout::PADDED_BSHD})
    {
        const std::string spelling = layout == Layout::BHSD ? "BHSD" : "padded BSHD";
        GraphSpec k;
        k.kLayout = layout;
        cases.push_back({"K " + spelling, k});
        GraphSpec v;
        v.vLayout = layout;
        cases.push_back({"V " + spelling, v});
    }
    expectEveryCaseDeclines(cases);
}

/// Q and O carry one query token at decode (seqlen_q 1), so BHSD and BSHD are byte-identical
/// for them -- the sequence axis's stride is never read when its extent is 1 -- and declining
/// BHSD would refuse a graph the kernel serves perfectly. Padded BSHD still introduces a real
/// batch-stride gap at any seqlen_q, so it is declined like K/V's.
TEST(TestGfx950AttentionUnified3DMatchers, AcceptsQAndOInBhsdButNotPaddedBshd)
{
    GraphSpec qBhsd;
    qBhsd.qLayout = Layout::BHSD;
    EXPECT_TRUE(matchGraph(qBhsd).has_value());
    GraphSpec oBhsd;
    oBhsd.oLayout = Layout::BHSD;
    EXPECT_TRUE(matchGraph(oBhsd).has_value());

    std::vector<DeclineCase> cases;
    GraphSpec qPadded;
    qPadded.qLayout = Layout::PADDED_BSHD;
    cases.push_back({"Q padded BSHD", qPadded});
    GraphSpec oPadded;
    oPadded.oLayout = Layout::PADDED_BSHD;
    cases.push_back({"O padded BSHD", oPadded});
    expectEveryCaseDeclines(cases);
}

TEST(TestGfx950AttentionUnified3DMatchers, DeclinesOperandsThatDisagreeInShapeOrType)
{
    std::vector<DeclineCase> cases;
    GraphSpec mixed;
    mixed.kDataType = data_objects::DataType::HALF;
    cases.push_back({"K dtype differs from Q", mixed});
    GraphSpec vShorter;
    vShorter.vSeqLen = 64;
    cases.push_back({"V sequence differs from K", vShorter});
    GraphSpec oHeads;
    oHeads.oHeads = 4;
    cases.push_back({"O heads differ from Q", oHeads});
    GraphSpec f32;
    f32.dataType = data_objects::DataType::FLOAT;
    cases.push_back({"float32", f32});
    expectEveryCaseDeclines(cases);
}

/// The kernels are built with 32-bit K/V byte addressing and bound their buffer loads at
/// 0x7FFF0000 bytes: past that, loads return zeros. The default decode graph's Sq is 1, so only
/// K/V (not Q/O) can reach the bound; batch 1, Hkv 8, D128 matches Unified2D's boundary case.
/// 1048592 keys is 2147516416 bytes, over the bound; 1048576 keys is 2^31 bytes, still over;
/// 1048000 is 2146304000, under.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesACacheBeyondThirtyTwoBitAddressing)
{
    GraphSpec over;
    over.batch = 1;
    over.numQueryHeads = 8;
    over.numKvHeads = 8;
    over.seqLenKv = 1048592;
    EXPECT_FALSE(matchGraph(over).has_value());

    GraphSpec under = over;
    under.seqLenKv = 1048576; // 2147483648 bytes
    EXPECT_FALSE(matchGraph(under).has_value());
    under.seqLenKv = 1048000; // 2146304000 bytes
    EXPECT_TRUE(matchGraph(under).has_value());
}

/// This copy of Unified2D's catch-all feature decline: no ALiBi, no padding mask, an
/// allow-listed mma_core_mode, and AUTO-only execution hints.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesAlibiPaddingMmaCoreModeAndImplementationHints)
{
    std::vector<DeclineCase> cases;
    GraphSpec alibi;
    alibi.alibiMask = true;
    cases.push_back({"alibi_mask", alibi});
    GraphSpec padding;
    padding.paddingMask = true;
    cases.push_back({"padding_mask", padding});
    GraphSpec mma;
    mma.mmaCoreMode = data_objects::DataType::FLOAT;
    cases.push_back({"mma_core_mode FLOAT", mma});
    GraphSpec composite;
    composite.implementation = data_objects::AttentionImplementation::COMPOSITE;
    cases.push_back({"implementation COMPOSITE", composite});
    expectEveryCaseDeclines(cases);
}

/// Explicit UNSET/HALF/BFLOAT16 mma_core_mode is inert, not a feature: declining it would
/// refuse decode graphs the cuDNN-compat shim writes by default.
TEST(TestGfx950AttentionUnified3DMatchers, AcceptsTheInertMmaCoreModeSpellings)
{
    for(const auto mode : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        spec.mmaCoreMode = mode;
        EXPECT_TRUE(matchGraph(spec).has_value());
    }
}

/// A ragged (RFC-0014) operand packs its tokens behind an offsets tensor, padded per
/// sequence; read as dense BSHD it gives wrong elements in bounds. Each operand on its own.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARaggedOperand)
{
    ASSERT_TRUE(matchGraph(GraphSpec{}).has_value());
    for(const int64_t uid : {Q_UID, K_UID, V_UID, O_UID})
    {
        SCOPED_TRACE("ragged operand uid " + std::to_string(uid));
        GraphSpec spec;
        spec.raggedOperand = uid;
        EXPECT_FALSE(matchGraph(spec).has_value());
    }
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

/// Without its reduce launch the segment kernel's partial results are never combined and
/// the output is never written; an extra or renamed launch is one this handler never runs.
/// The state manager drops these against launchNames() first; this is the native's own
/// refusal for a caller that hands it a definition directly.
TEST(TestGfx950AttentionUnified3DMatchers, DeclinesARecordWhoseLaunchesAreNotSegmentAndReduce)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));

    for(const auto& launches : std::vector<std::vector<std::string>>{
            {},
            {"segment"},
            {"segment", "merge"},
            {"segment", "reduce", "extra"},
        })
    {
        KernelSpec spec;
        spec.launches = launches;
        EXPECT_FALSE(matchesKernel(GraphSpec{}, spec)) << launches.size() << " launch(es)";
    }
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
    KernelSpec noReduce;
    noReduce.launches = {"segment"};
    EXPECT_EQ(scoreOf(GraphSpec{}, noReduce), 0.0);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
