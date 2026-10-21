// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "tests/engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DTestGraphs.hpp"

/**
 * @file TestGfx950AttentionUnified2DMatchers.cpp
 * @brief Applicability for hipkernel:Gfx950AttentionUnified2D: what the graph match declines,
 *        what the kernel match compares against a candidate's baked metadata, and how the
 *        survivors rank.
 *
 * One case per rule, in severity order: the silent-wrong-answer rules first (layout, shape,
 * mask, batch capacity), then faults (addressing limits), then declined features. A graph
 * match that returns nullopt empties this engine's WHOLE catalog, so every case that expects
 * a decline is paired with a positive control that shows the same graph one field away is
 * served.
 *
 * Matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace unified2d;

/// graph_match's answer for @p spec: true when it binds, false when it declines.
bool graphMatches(const GraphSpec& spec)
{
    return matchGraph(spec).has_value();
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
        EXPECT_FALSE(graphMatches(declineCase.spec))
            << declineCase.name << ": graph_match bound a graph the kernel cannot serve";
    }
}

} // namespace

// =============================================================================
// graph_match accepts
// =============================================================================

/// The positive control every decline below depends on: a matcher that refused everything
/// would pass them all and serve nothing.
TEST(TestGfx950AttentionUnified2DMatchers, AcceptsTheDefaultGraph)
{
    EXPECT_TRUE(graphMatches(GraphSpec{}));
}

TEST(TestGfx950AttentionUnified2DMatchers, AcceptsEveryMaskKindAtBothDataTypes)
{
    for(const auto dataType : {data_objects::DataType::BFLOAT16, data_objects::DataType::HALF})
    {
        for(const int64_t mask : {NO_MASK, TOP_LEFT_CAUSAL, BOTTOM_RIGHT_CAUSAL})
        {
            GraphSpec spec;
            spec.dataType = dataType;
            setMask(spec, mask);
            EXPECT_TRUE(graphMatches(spec)) << "mask " << mask;
        }
    }
}

/// Both head sizes, and the group sizes the kernel is built for: MHA, a group of 2 and the
/// widest group of 16. A shape the engine declines would silently fall to another provider.
TEST(TestGfx950AttentionUnified2DMatchers, AcceptsBothHeadSizesAndEveryGroupUpToSixteen)
{
    for(const int64_t headSize : {64, 128})
    {
        for(const auto& [hq, hkv] : std::vector<std::pair<int64_t, int64_t>>{
                {8, 8}, {8, 4}, {16, 1}, {32, 2}, {28, 4}, {128, 8}})
        {
            GraphSpec spec;
            spec.headSize = headSize;
            spec.numQueryHeads = hq;
            spec.numKvHeads = hkv;
            EXPECT_TRUE(graphMatches(spec)) << "D" << headSize << " Hq " << hq << " Hkv " << hkv;
        }
    }
}

/// q_len == kv_len and q_len < kv_len both satisfy the kernel's seq_lens contract.
TEST(TestGfx950AttentionUnified2DMatchers, AcceptsAQueryAsLongAsOrShorterThanTheKeys)
{
    GraphSpec equal;
    equal.seqLenQ = 256;
    equal.seqLenKv = 256;
    EXPECT_TRUE(graphMatches(equal));
    GraphSpec shorter;
    shorter.seqLenQ = 1;
    shorter.seqLenKv = 4096;
    EXPECT_TRUE(graphMatches(shorter));
}

// =============================================================================
// graph_match declines: silent wrong answers
// =============================================================================

/// The kernel reads Q and writes O as token-major rows and reads K/V as a gap-free cache, and
/// takes no stride arguments, so a differently-strided tensor is read as if it were this one:
/// elements in bounds, wrong values, no fault. Each operand on its own, in each wrong layout.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesAnyOperandThatIsNotDenseBshd)
{
    std::vector<DeclineCase> cases;
    for(const auto layout : {Layout::BHSD, Layout::PADDED_BSHD})
    {
        const std::string spelling = layout == Layout::BHSD ? "BHSD" : "padded BSHD";
        GraphSpec q;
        q.qLayout = layout;
        cases.push_back({"Q " + spelling, q});
        GraphSpec k;
        k.kLayout = layout;
        cases.push_back({"K " + spelling, k});
        GraphSpec v;
        v.vLayout = layout;
        cases.push_back({"V " + spelling, v});
        GraphSpec o;
        o.oLayout = layout;
        cases.push_back({"O " + spelling, o});
    }
    expectEveryCaseDeclines(cases);
}

/// A single head is byte-identically BSHD and BHSD while the two spellings disagree on the head
/// stride, so declining it would refuse a graph the kernel serves perfectly.
TEST(TestGfx950AttentionUnified2DMatchers, AcceptsASingleHeadInEitherSpelling)
{
    GraphSpec spec;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.qLayout = Layout::BHSD;
    spec.kLayout = Layout::BHSD;
    spec.vLayout = Layout::BHSD;
    spec.oLayout = Layout::BHSD;
    EXPECT_TRUE(graphMatches(spec));
}

TEST(TestGfx950AttentionUnified2DMatchers, DeclinesOperandsThatDisagreeInShapeOrType)
{
    std::vector<DeclineCase> cases;
    GraphSpec mixed;
    mixed.kDataType = data_objects::DataType::HALF;
    cases.push_back({"K dtype differs from Q", mixed});
    GraphSpec vShorter;
    vShorter.vSeqLen = 128;
    cases.push_back({"V sequence differs from K", vShorter});
    GraphSpec oHeads;
    oHeads.oHeads = 4;
    cases.push_back({"O heads differ from Q", oHeads});
    GraphSpec f32;
    f32.dataType = data_objects::DataType::FLOAT;
    cases.push_back({"float32", f32});
    expectEveryCaseDeclines(cases);
}

/// The kernel derives its group size by integer division, so a non-dividing pair silently
/// drops heads; a group above 16 is a shape it is not built for.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesAGroupTheKernelCannotServe)
{
    std::vector<DeclineCase> cases;
    GraphSpec nonDividing;
    nonDividing.numQueryHeads = 9;
    nonDividing.numKvHeads = 2;
    cases.push_back({"Hq 9 over Hkv 2", nonDividing});
    GraphSpec tooWide;
    tooWide.numQueryHeads = 34;
    tooWide.numKvHeads = 2;
    cases.push_back({"a group of 17", tooWide});
    GraphSpec moreKvThanQ;
    moreKvThanQ.numQueryHeads = 2;
    moreKvThanQ.numKvHeads = 8;
    cases.push_back({"Hkv above Hq", moreKvThanQ});
    expectEveryCaseDeclines(cases);
}

TEST(TestGfx950AttentionUnified2DMatchers, DeclinesAHeadSizeOtherThanSixtyFourOrOneTwentyEight)
{
    std::vector<DeclineCase> cases;
    for(const int64_t headSize : {32, 96, 192, 256})
    {
        GraphSpec spec;
        spec.headSize = headSize;
        cases.push_back({"head size " + std::to_string(headSize), spec});
    }
    expectEveryCaseDeclines(cases);
}

/// The kernel's seq_lens contract is q_len <= kv_len per sequence; its causal limit is
/// `seq_len - q_len`, which a longer query makes negative. Declined for every mask.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesAQueryLongerThanTheKeysForEveryMask)
{
    std::vector<DeclineCase> cases;
    for(const int64_t mask : {NO_MASK, TOP_LEFT_CAUSAL, BOTTOM_RIGHT_CAUSAL})
    {
        GraphSpec spec;
        spec.seqLenQ = 512;
        spec.seqLenKv = 256;
        setMask(spec, mask);
        cases.push_back({"Sq 512 over Skv 256, mask " + std::to_string(mask), spec});
    }
    expectEveryCaseDeclines(cases);
}

/// The identity cache is whole blocks, and the smallest block is 16.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesASeqLenKvThatSixteenDoesNotDivide)
{
    std::vector<DeclineCase> cases;
    for(const int64_t skv : {250, 255, 257, 1000})
    {
        GraphSpec spec;
        spec.seqLenQ = 128;
        spec.seqLenKv = skv;
        cases.push_back({"Skv " + std::to_string(skv), spec});
    }
    expectEveryCaseDeclines(cases);

    GraphSpec multiple;
    multiple.seqLenKv = 272; // 17 * 16
    EXPECT_TRUE(graphMatches(multiple));
}

// =============================================================================
// graph_match declines: the mask
// =============================================================================

/// A band other than unmasked or causal is a sliding window, and no shipped variant carries
/// one: serving it as plain causal would attend to keys the window excludes.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesABandThatIsNeitherUnmaskedNorCausal)
{
    std::vector<DeclineCase> cases;
    GraphSpec window;
    window.leftBound = 128;
    window.rightBound = 0;
    cases.push_back({"left bound 128", window});
    GraphSpec lookahead;
    lookahead.leftBound = std::nullopt;
    lookahead.rightBound = 5;
    cases.push_back({"right bound 5", lookahead});
    GraphSpec both;
    both.causalMaskDeprecated = true;
    both.causalMaskBottomRightDeprecated = true;
    both.leftBound = std::nullopt;
    both.rightBound = std::nullopt;
    cases.push_back({"both deprecated causal booleans", both});
    expectEveryCaseDeclines(cases);
}

/// The deprecated booleans still mean what they meant: they merge with the bounds and
/// alignment into the same three kinds.
TEST(TestGfx950AttentionUnified2DMatchers, ReadsTheDeprecatedCausalBooleansAsTheirMaskKinds)
{
    GraphSpec topLeft;
    topLeft.leftBound = std::nullopt;
    topLeft.rightBound = std::nullopt;
    topLeft.causalMaskDeprecated = true;
    const auto topLeftBound = matchGraph(topLeft);
    ASSERT_TRUE(topLeftBound.has_value());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*topLeftBound, MASK_TOKEN),
              TOP_LEFT_CAUSAL);

    GraphSpec bottomRight;
    bottomRight.leftBound = std::nullopt;
    bottomRight.rightBound = std::nullopt;
    bottomRight.causalMaskBottomRightDeprecated = true;
    const auto bottomRightBound = matchGraph(bottomRight);
    ASSERT_TRUE(bottomRightBound.has_value());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bottomRightBound, MASK_TOKEN),
              BOTTOM_RIGHT_CAUSAL);
}

// =============================================================================
// graph_match declines: addressing limits
// =============================================================================

/// The kernels are built with 32-bit K/V byte addressing and bound their buffer loads at
/// 0x7FFF0000 bytes: past that, loads return zeros. 1048592 keys of 8 heads x 128 x 2 bytes is
/// 2147516416 bytes, over the bound; 1048576 keys are 2^31 bytes, still over; 1048000 are
/// 2146304000, under.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesACacheBeyondThirtyTwoBitAddressing)
{
    GraphSpec over;
    over.batch = 1;
    over.numQueryHeads = 8;
    over.numKvHeads = 8;
    over.seqLenQ = 16;
    over.seqLenKv = 1048592;
    EXPECT_FALSE(graphMatches(over));

    GraphSpec under = over;
    under.seqLenKv = 1048576; // 2147483648 bytes
    EXPECT_FALSE(graphMatches(under));
    under.seqLenKv = 1048000; // 2146304000 bytes
    EXPECT_TRUE(graphMatches(under));
}

/// Extents whose product does not fit int64_t are declined, not wrapped into a passing value.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesExtentsWhoseProductOverflows)
{
    GraphSpec spec;
    spec.batch = std::numeric_limits<int64_t>::max() / 4;
    spec.seqLenQ = 16;
    spec.seqLenKv = 16;
    EXPECT_FALSE(graphMatches(spec));
}

// =============================================================================
// graph_match declines: features this catalog does not carry
// =============================================================================

TEST(TestGfx950AttentionUnified2DMatchers, DeclinesEveryOptionalFeatureTheKernelsAreNotBuiltWith)
{
    std::vector<DeclineCase> cases;
    const auto withExtra
        = [&cases](const std::string& name,
                   std::function<void(data_objects::SdpaAttributesBuilder&)> extra) {
              GraphSpec spec;
              spec.extra = std::move(extra);
              cases.push_back({name, spec});
          };

    withExtra("attn_mask tensor", [](auto& b) { b.add_attn_mask_tensor_uid(EXTRA_UID); });
    withExtra("scale tensor", [](auto& b) { b.add_scale_tensor_uid(EXTRA_UID); });
    withExtra("seq_len_q tensor", [](auto& b) { b.add_seq_len_q_tensor_uid(EXTRA_UID); });
    withExtra("seq_len_kv tensor", [](auto& b) { b.add_seq_len_kv_tensor_uid(EXTRA_UID); });
    withExtra("seed tensor", [](auto& b) { b.add_seed_tensor_uid(EXTRA_UID); });
    withExtra("offset tensor", [](auto& b) { b.add_offset_tensor_uid(EXTRA_UID); });
    withExtra("dropout mask", [](auto& b) { b.add_dropout_mask_tensor_uid(EXTRA_UID); });
    withExtra("dropout scale", [](auto& b) { b.add_dropout_scale_tensor_uid(EXTRA_UID); });
    withExtra("dropout probability", [](auto& b) { b.add_dropout_probability(0.1F); });
    withExtra("page table K", [](auto& b) { b.add_page_table_k_tensor_uid(EXTRA_UID); });
    withExtra("page table V", [](auto& b) { b.add_page_table_v_tensor_uid(EXTRA_UID); });
    withExtra("max_seq_len_kv", [](auto& b) { b.add_max_seq_len_kv(512); });
    withExtra("block mask", [](auto& b) { b.add_block_mask_tensor_uid(EXTRA_UID); });
    withExtra("sink token", [](auto& b) { b.add_sink_token_tensor_uid(EXTRA_UID); });
    withExtra("descale Q", [](auto& b) { b.add_descale_q_tensor_uid(EXTRA_UID); });
    withExtra("descale K", [](auto& b) { b.add_descale_k_tensor_uid(EXTRA_UID); });
    withExtra("descale V", [](auto& b) { b.add_descale_v_tensor_uid(EXTRA_UID); });
    withExtra("descale S", [](auto& b) { b.add_descale_s_tensor_uid(EXTRA_UID); });
    withExtra("scale S", [](auto& b) { b.add_scale_s_tensor_uid(EXTRA_UID); });
    withExtra("scale O", [](auto& b) { b.add_scale_o_tensor_uid(EXTRA_UID); });
    withExtra("amax S", [](auto& b) { b.add_amax_s_tensor_uid(EXTRA_UID); });
    withExtra("amax O", [](auto& b) { b.add_amax_o_tensor_uid(EXTRA_UID); });
    withExtra("stats tensor", [](auto& b) { b.add_stats_tensor_uid(EXTRA_UID); });
    withExtra("max tensor", [](auto& b) { b.add_max_tensor_uid(EXTRA_UID); });
    withExtra("sum_exp tensor", [](auto& b) { b.add_sum_exp_tensor_uid(EXTRA_UID); });
    withExtra("rng_dump tensor", [](auto& b) { b.add_rng_dump_tensor_uid(EXTRA_UID); });
    withExtra("generate_stats", [](auto& b) { b.add_generate_stats(true); });

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

/// Explicit false and an unset mma_core_mode / HALF are inert, not features: declining them
/// would refuse graphs the cuDNN-compat shim writes by default.
TEST(TestGfx950AttentionUnified2DMatchers, AcceptsTheInertSpellingsOfThoseFeatures)
{
    GraphSpec explicitFalse;
    explicitFalse.extra = [](auto& b) { b.add_generate_stats(false); };
    EXPECT_TRUE(graphMatches(explicitFalse));

    for(const auto mode : {data_objects::DataType::HALF, data_objects::DataType::BFLOAT16})
    {
        GraphSpec spec;
        spec.mmaCoreMode = mode;
        EXPECT_TRUE(graphMatches(spec));
    }
}

/// The scale is an f32 kernarg and the kernel folds it into an exp2-domain multiplier: zero,
/// negative, NaN and infinite scales are not softmax scales.
TEST(TestGfx950AttentionUnified2DMatchers, DeclinesAScaleThatIsNotAFinitePositiveNumber)
{
    std::vector<DeclineCase> cases;
    for(const float scale : {0.0F,
                             -0.125F,
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity()})
    {
        GraphSpec spec;
        spec.attnScaleValue = scale;
        cases.push_back({"scale " + std::to_string(scale), spec});
    }
    expectEveryCaseDeclines(cases);
}

TEST(TestGfx950AttentionUnified2DMatchers, DeclinesAGraphThatIsNotOneSdpaForwardNode)
{
    std::vector<DeclineCase> cases;
    GraphSpec two;
    two.twoNodes = true;
    cases.push_back({"two SDPA nodes", two});
    GraphSpec pointwise;
    pointwise.pointwiseOnly = true;
    cases.push_back({"a pointwise node", pointwise});
    expectEveryCaseDeclines(cases);
}

// =============================================================================
// kernel_match
// =============================================================================

/// The positive control: the default kernel is the shipped baseline for the default graph.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchAcceptsTheBaselineCandidate)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

/// One baked field at a time: the graph is the default, the candidate differs in exactly one
/// way, and the kernel computes something else if it is served.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchDeclinesACandidateBuiltForAnotherProblem)
{
    std::vector<std::pair<std::string, KernelSpec>> cases;
    const auto with = [&cases](const std::string& name, auto mutate) {
        KernelSpec spec;
        mutate(spec);
        cases.emplace_back(name, spec);
    };
    with("dtype FP16 for a bf16 graph", [](KernelSpec& s) { s.dtype = "FP16"; });
    with("head size 64 for a D128 graph", [](KernelSpec& s) { s.headSize = 64; });
    with("Hq 16 for an Hq 8 graph", [](KernelSpec& s) { s.numQueryHeads = 16; });
    with("Hkv 4 for an Hkv 2 graph", [](KernelSpec& s) { s.numKvHeads = 4; });
    with("a window of 128", [](KernelSpec& s) { s.slidingWindow = 128; });
    with("no mask for a causal graph", [](KernelSpec& s) { s.maskType = NO_MASK; });
    with("bottom-right for a top-left graph",
         [](KernelSpec& s) { s.maskType = BOTTOM_RIGHT_CAUSAL; });

    ASSERT_FALSE(cases.empty());
    for(const auto& [name, kernelSpec] : cases)
    {
        EXPECT_FALSE(matchesKernel(GraphSpec{}, kernelSpec)) << name;
    }
}

/// Top-left and bottom-right causal coincide only at Sq == Skv, but they are distinct kernel
/// bodies, and each graph must reach its own: the matrix of every candidate against every
/// graph mask matches on the diagonal and nowhere else.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchPairsEachMaskKindWithItsOwnKernelOnly)
{
    for(const int64_t graphMask : {NO_MASK, TOP_LEFT_CAUSAL, BOTTOM_RIGHT_CAUSAL})
    {
        for(const int64_t kernelMask : {NO_MASK, TOP_LEFT_CAUSAL, BOTTOM_RIGHT_CAUSAL})
        {
            GraphSpec graph;
            setMask(graph, graphMask);
            KernelSpec kernel;
            kernel.maskType = kernelMask;
            EXPECT_EQ(matchesKernel(graph, kernel), graphMask == kernelMask)
                << "graph mask " << graphMask << ", kernel mask " << kernelMask;
        }
    }
}

/// A block of 64 divides Skv 256 and one of 64 does not divide 272: the cache is whole blocks
/// of THIS candidate's size, so one graph can admit some block sizes of a cohort and not
/// others.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchRequiresTheCandidatesBlockToDivideSeqLenKv)
{
    KernelSpec block64;
    block64.blockSize = 64;
    block64.tileSize = 256;
    KernelSpec block32;
    block32.blockSize = 32;
    block32.tileSize = 128;

    const GraphSpec divisible; // Skv 256
    EXPECT_TRUE(matchesKernel(divisible, block64));
    EXPECT_TRUE(matchesKernel(divisible, block32));

    GraphSpec sixteenOnly;
    sixteenOnly.seqLenKv = 272; // 17 * 16: a multiple of 16, of neither 32 nor 64
    EXPECT_TRUE(matchesKernel(sixteenOnly, KernelSpec{}));
    EXPECT_FALSE(matchesKernel(sixteenOnly, block32));
    EXPECT_FALSE(matchesKernel(sixteenOnly, block64));
}

/// The kernel bisects the sequences in a fixed number of unrolled steps, which resolves at
/// most 2^iters - 1 of them. Past that the last blocks read another sequence's lengths and the
/// launch still succeeds, so the boundary is exact: capacity 3 (2 iterations) serves a batch
/// of 3 and not 4, and the shipped 511 (9 iterations) serves 511 and not 512.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchBoundsTheBatchByTheBakedCapacity)
{
    const auto graphOfBatch = [](int64_t batch) {
        GraphSpec spec;
        spec.batch = batch;
        spec.seqLenQ = 16;
        spec.seqLenKv = 16;
        return spec;
    };
    const auto kernelOfCapacity = [](int64_t numSeqs) {
        KernelSpec spec;
        spec.numSeqs = numSeqs;
        return spec;
    };

    EXPECT_TRUE(matchesKernel(graphOfBatch(3), kernelOfCapacity(3)));
    EXPECT_FALSE(matchesKernel(graphOfBatch(4), kernelOfCapacity(3)));
    EXPECT_TRUE(matchesKernel(graphOfBatch(511), kernelOfCapacity(511)));
    EXPECT_FALSE(matchesKernel(graphOfBatch(512), kernelOfCapacity(511)));
    // A capacity that is not 2^k - 1 still rounds UP to the iterations it needs: num_seqs 4
    // takes 3 iterations, so it serves 7.
    EXPECT_TRUE(matchesKernel(graphOfBatch(7), kernelOfCapacity(4)));
    EXPECT_FALSE(matchesKernel(graphOfBatch(8), kernelOfCapacity(4)));
}

/// The metadata carries no shape: one binary serves any Sq and Skv for its configuration.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchServesAnySequenceLengthsForOneConfiguration)
{
    for(const auto& [sq, skv] :
        std::vector<std::pair<int64_t, int64_t>>{{1, 16}, {16, 16}, {100, 4096}, {2048, 2048}})
    {
        GraphSpec spec;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        EXPECT_TRUE(matchesKernel(spec, KernelSpec{})) << "Sq " << sq << " Skv " << skv;
    }
}

/// A geometry the Python builder would refuse is a record this engine cannot launch, so it is
/// declined even where every semantic field agrees.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchDeclinesACandidateWithAnUnbuildableGeometry)
{
    std::vector<KernelSpec> unbuildable;
    const auto with = [&unbuildable](auto mutate) {
        KernelSpec spec;
        mutate(spec);
        unbuildable.push_back(spec);
    };
    with([](KernelSpec& s) { s.numWarps = 3; });
    with([](KernelSpec& s) { s.blockMPerWarp = 24; });
    with([](KernelSpec& s) {
        s.blockMPerWarp = 32;
        s.numWarps = 8;
    });
    with([](KernelSpec& s) { s.tileSize = 70; });
    with([](KernelSpec& s) { s.tileSize = 144; });
    with([](KernelSpec& s) { s.blockSize = 24; });
    with([](KernelSpec& s) { s.useMfma32x32 = 1; });
    with([](KernelSpec& s) { s.useTransposedQk32x32 = 1; });
    with([](KernelSpec& s) { s.numSeqs = 0; });
    with([](KernelSpec& s) { s.numSeqs = -3; });
    with([](KernelSpec& s) { s.maskType = 7; });

    for(std::size_t i = 0; i < unbuildable.size(); ++i)
    {
        EXPECT_FALSE(matchesKernel(GraphSpec{}, unbuildable.at(i))) << "unbuildable " << i;
    }
}

/// A completed record omits no field; one that does, or that spells one in another type, is
/// declined rather than given a default.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchDeclinesAnIncompleteOrMistypedRecord)
{
    const Graph graph;
    for(const auto& entry : metadataOf(KernelSpec{}))
    {
        SCOPED_TRACE(entry.first);
        auto kernel = makeKernel(KernelSpec{});
        kernel.metadata.erase(entry.first);
        EXPECT_FALSE(matchesKernel(graph, kernel)) << "without the field";
    }

    auto mistyped = makeKernel(KernelSpec{});
    mistyped.metadata[std::string("tile_size")] = std::string("64");
    EXPECT_FALSE(matchesKernel(graph, mistyped));
}

/// Geometry alternatives the catalog may grow into are served exactly like the baseline when
/// they are buildable and everything else agrees: nothing in kernel_match keys on the baseline.
TEST(TestGfx950AttentionUnified2DMatchers, KernelMatchServesEveryBuildableGeometry)
{
    for(const int64_t warps : {1, 2, 4, 8})
    {
        for(const int64_t multiplier : {2, 4, 8})
        {
            KernelSpec spec;
            spec.numWarps = warps;
            spec.tileSize = 16 * multiplier;
            EXPECT_TRUE(matchesKernel(GraphSpec{}, spec))
                << "warps " << warps << " tile " << spec.tileSize;
        }
    }
    KernelSpec stacked;
    stacked.numWarps = 2;
    stacked.blockMPerWarp = 32;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, stacked));
    KernelSpec transposed;
    transposed.blockMPerWarp = 32;
    transposed.useMfma32x32 = 1;
    transposed.useTransposedQk32x32 = 1;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, transposed));
}

// =============================================================================
// score
// =============================================================================

namespace
{

/// A candidate of the given geometry, with the other fields at the baseline.
KernelSpec geometry(int64_t tile, int64_t warps, int64_t rows, int64_t mfma32 = 0, int64_t qk32 = 0)
{
    KernelSpec spec;
    spec.tileSize = tile;
    spec.numWarps = warps;
    spec.blockMPerWarp = rows;
    spec.useMfma32x32 = mfma32;
    spec.useTransposedQk32x32 = qk32;
    return spec;
}

} // namespace

/// The ranking is a stable fallback order for cold selection, not a performance model: the
/// baseline geometry above every alternative, the alternatives in ascending lexicographic
/// (tile, warps, rows, mfma32, transposed) order, and every distinct geometry a distinct score
/// so the selector's descriptor-id tie-break never decides between two.
TEST(TestGfx950AttentionUnified2DMatchers, ScoreRanksTheBaselineFirstThenAscendingGeometry)
{
    const double baseline = scoreOf(GraphSpec{}, geometry(64, 4, 16));

    const std::vector<KernelSpec> ascending{geometry(32, 1, 16),
                                            geometry(32, 2, 16),
                                            geometry(32, 2, 32),
                                            geometry(32, 4, 16),
                                            geometry(32, 4, 32),
                                            geometry(32, 4, 32, 1, 0),
                                            geometry(32, 4, 32, 1, 1),
                                            geometry(64, 1, 16),
                                            geometry(64, 2, 16),
                                            geometry(64, 4, 32),
                                            geometry(64, 8, 16),
                                            geometry(128, 1, 16),
                                            geometry(128, 4, 16)};

    double previous = baseline;
    for(const auto& candidate : ascending)
    {
        const double score = scoreOf(GraphSpec{}, candidate);
        EXPECT_GT(score, 0.0) << "tile " << candidate.tileSize << " warps " << candidate.numWarps;
        EXPECT_LT(score, previous)
            << "tile " << candidate.tileSize << " warps " << candidate.numWarps << " rows "
            << candidate.blockMPerWarp << " should rank below the one before it";
        previous = score;
    }
    // Every alternative sits in the band below ALTERNATIVE_CEILING (1.0), under the baseline.
    EXPECT_LT(scoreOf(GraphSpec{}, ascending.front()), 1.0);
    EXPECT_EQ(baseline, 2.0);
}

/// Only a record kernel_match already declined scores below every legal candidate.
TEST(TestGfx950AttentionUnified2DMatchers, ScoreGivesAnUnbuildableRecordTheFloor)
{
    EXPECT_EQ(scoreOf(GraphSpec{}, geometry(64, 3, 16)), 0.0);

    // The widest legal candidate -- block 64 at the dispatcher's 8x tile, 8 warps -- still
    // scores above the floor.
    KernelSpec widest = geometry(512, 8, 16);
    widest.blockSize = 64;
    EXPECT_GT(scoreOf(GraphSpec{}, widest), 0.0);
}

/// The ranking does not read the mask, dtype or head configuration: two candidates of one
/// geometry for different problems score alike, so a catalog cannot be steered by them.
TEST(TestGfx950AttentionUnified2DMatchers, ScoreDependsOnlyOnTheGeometry)
{
    KernelSpec other = geometry(64, 4, 16);
    other.maskType = NO_MASK;
    other.dtype = "FP16";
    EXPECT_EQ(scoreOf(GraphSpec{}, other), scoreOf(GraphSpec{}, geometry(64, 4, 16)));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
