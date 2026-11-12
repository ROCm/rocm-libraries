// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified2DGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx942AttentionUnified2DTestGraphs.hpp"

/**
 * @file TestGfx942AttentionUnified2DMatchers.cpp
 * @brief Applicability for hipkernel:Gfx942AttentionUnified2D: what the graph match declines,
 *        what the kernel match compares against a candidate's baked metadata, and how the
 *        survivors rank.
 *
 * One case per rule, in severity order: the silent-wrong-answer rules first (layout, shape,
 * mask), then faults (addressing limits), then declined features. A graph match that returns
 * nullopt empties this engine's WHOLE catalog, so every case that expects a decline is paired
 * with a positive control that shows the same graph one field away is served.
 *
 * Matcher-only: no device, no compile, no launch.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

using namespace gfx942_unified2d;

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

/// Whether the catalog ships GQA group @p ratio, by the list graph_match reads.
bool shipsGqaRatio(int64_t ratio)
{
    return std::find(GFX942_UNIFIED_2D_SHIPPED_RATIOS.begin(),
                     GFX942_UNIFIED_2D_SHIPPED_RATIOS.end(),
                     ratio)
           != GFX942_UNIFIED_2D_SHIPPED_RATIOS.end();
}

/// The cause key of the decline graph_match logs at INFO for @p spec ("<engine> declined the
/// graph [<cause>]: ..."), or empty when it logged none.
std::string loggedDeclineCause(const GraphSpec& spec)
{
    const std::string marker
        = std::string(GFX942_ATTENTION_UNIFIED_2D_ENGINE_NAME) + " declined the graph [";
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    graphMatches(spec);
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        const auto causeEnd
            = at == std::string::npos ? at : log.message.find(']', at + marker.size());
        if(log.severity == HIPDNN_SEV_INFO && causeEnd != std::string::npos)
        {
            return log.message.substr(at + marker.size(), causeEnd - at - marker.size());
        }
    }
    return {};
}

} // namespace

// =============================================================================
// graph_match accepts
// =============================================================================

/// The positive control every decline below depends on: a matcher that refused everything
/// would pass them all and serve nothing.
TEST(TestGfx942AttentionUnified2DMatchers, AcceptsTheDefaultGraph)
{
    EXPECT_TRUE(graphMatches(GraphSpec{}));
}

TEST(TestGfx942AttentionUnified2DMatchers, AcceptsBottomRightCausalAtBothDataTypes)
{
    for(const auto dataType : {data_objects::DataType::BFLOAT16, data_objects::DataType::HALF})
    {
        GraphSpec spec;
        spec.dataType = dataType;
        setMask(spec, BOTTOM_RIGHT_CAUSAL);
        const auto bound = matchGraph(spec);
        ASSERT_TRUE(bound.has_value());
        EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bound, MASK_TOKEN),
                  BOTTOM_RIGHT_CAUSAL);
    }
}

/// Both head sizes, and every group size the catalog ships (1, 2, 4, 8 and 16) on a spread of
/// head pairs. graph_match declines any other group by name (below).
TEST(TestGfx942AttentionUnified2DMatchers, AcceptsBothHeadSizesAndEveryGroupTheCatalogShips)
{
    for(const int64_t headSize : {64, 128})
    {
        for(const int64_t ratio : GFX942_UNIFIED_2D_SHIPPED_RATIOS)
        {
            for(const int64_t hkv : {1, 4, 8})
            {
                GraphSpec spec;
                spec.headSize = headSize;
                spec.numKvHeads = hkv;
                spec.numQueryHeads = ratio * hkv;
                EXPECT_TRUE(graphMatches(spec))
                    << "D" << headSize << " Hq " << spec.numQueryHeads << " Hkv " << hkv;
            }
        }
    }
}

/// q_len == kv_len and q_len < kv_len both satisfy the kernel's seq_lens contract.
TEST(TestGfx942AttentionUnified2DMatchers, AcceptsAQueryAsLongAsOrShorterThanTheKeys)
{
    GraphSpec equal;
    equal.seqLenQ = 256;
    equal.seqLenKv = 256;
    EXPECT_TRUE(graphMatches(equal));
    GraphSpec shorter;
    shorter.seqLenQ = 2;
    shorter.seqLenKv = 4096;
    EXPECT_TRUE(graphMatches(shorter));
}

/// Decode (seqlen_q 1) is the split-KV Unified3d engine's at every GQA ratio its catalog
/// ships (1, 2, 4, 8, 16), and named so in the decline.
TEST(TestGfx942AttentionUnified2DMatchers, LeavesDecodeToTheSplitKvEngineAtTheRatiosItShips)
{
    GraphSpec decode;
    decode.seqLenQ = 1;
    decode.seqLenKv = 4096;
    for(const auto& [hq, hkv] :
        std::vector<std::pair<int64_t, int64_t>>{{8, 8}, {8, 4}, {8, 2}, {64, 8}, {16, 1}})
    {
        decode.numQueryHeads = hq;
        decode.numKvHeads = hkv;
        EXPECT_FALSE(graphMatches(decode)) << "Hq " << hq << " Hkv " << hkv;
        EXPECT_EQ(loggedDeclineCause(decode), "decode") << "Hq " << hq << " Hkv " << hkv;
    }
}

/// A group the catalog ships no kernel for is declined by name rather than left to empty the
/// catalog at kernel_match: at prefill and at decode alike, since the split-KV engine ships no
/// such group either.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAGroupTheCatalogDoesNotShipByName)
{
    int64_t checked = 0;
    for(int64_t ratio = 1; ratio <= GFX942_UNIFIED_2D_MAX_QUERIES_PER_KV; ++ratio)
    {
        if(shipsGqaRatio(ratio))
        {
            continue;
        }
        ++checked;
        for(const int64_t seqLenQ : {1, 96})
        {
            GraphSpec spec;
            spec.numKvHeads = 4;
            spec.numQueryHeads = ratio * spec.numKvHeads;
            spec.seqLenQ = seqLenQ;
            EXPECT_FALSE(graphMatches(spec)) << "group " << ratio << ", Sq " << seqLenQ;
            EXPECT_EQ(loggedDeclineCause(spec), "gqa") << "group " << ratio << ", Sq " << seqLenQ;
        }
    }
    // Every group up to the widest, less the shipped ones.
    EXPECT_EQ(checked,
              GFX942_UNIFIED_2D_MAX_QUERIES_PER_KV
                  - static_cast<int64_t>(GFX942_UNIFIED_2D_SHIPPED_RATIOS.size()));
}

// =============================================================================
// graph_match declines: silent wrong answers
// =============================================================================

/// The kernel reads Q and writes O as token-major rows and reads K/V as a gap-free cache, and
/// takes no stride arguments, so a differently-strided tensor is read as if it were this one:
/// elements in bounds, wrong values, no fault. Each operand on its own, in each wrong layout.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAnyOperandThatIsNotDenseBshd)
{
    std::vector<DeclineCase> cases;
    for(const auto layout : {SdpaLayout::BHSD, SdpaLayout::PADDED_BSHD})
    {
        const std::string spelling = layout == SdpaLayout::BHSD ? "BHSD" : "padded BSHD";
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

/// A ragged (RFC-0014) operand packs its tokens behind an offsets tensor, padded per
/// sequence; read as dense BSHD it gives wrong elements in bounds. Each operand on its own.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesARaggedOperand)
{
    ASSERT_TRUE(graphMatches(GraphSpec{}));
    std::vector<DeclineCase> cases;
    for(const auto& [role, uid] : std::vector<std::pair<std::string, int64_t>>{
            {"Q", Q_UID}, {"K", K_UID}, {"V", V_UID}, {"O", O_UID}})
    {
        GraphSpec spec;
        spec.raggedOperand = uid;
        cases.push_back({"ragged " + role, spec});
    }
    expectEveryCaseDeclines(cases);
}

/// A single head is byte-identically BSHD and BHSD while the two spellings disagree on the head
/// stride, so declining it would refuse a graph the kernel serves perfectly.
TEST(TestGfx942AttentionUnified2DMatchers, AcceptsASingleHeadInEitherSpelling)
{
    GraphSpec spec;
    spec.numQueryHeads = 1;
    spec.numKvHeads = 1;
    spec.qLayout = SdpaLayout::BHSD;
    spec.kLayout = SdpaLayout::BHSD;
    spec.vLayout = SdpaLayout::BHSD;
    spec.oLayout = SdpaLayout::BHSD;
    EXPECT_TRUE(graphMatches(spec));
}

TEST(TestGfx942AttentionUnified2DMatchers, DeclinesOperandsThatDisagreeInShapeOrType)
{
    std::vector<DeclineCase> cases;
    GraphSpec mixed;
    mixed.kDataType = data_objects::DataType::HALF;
    cases.push_back({"K dtype differs from Q", mixed});
    GraphSpec kBatch;
    kBatch.kBatch = 1;
    cases.push_back({"K batch differs from Q", kBatch});
    GraphSpec kHeadSize;
    kHeadSize.kHeadSize = 64;
    cases.push_back({"K head size differs from Q", kHeadSize});
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
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAGroupTheKernelCannotServe)
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

TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAHeadSizeOtherThanSixtyFourOrOneTwentyEight)
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
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAQueryLongerThanTheKeysForEveryMask)
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
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesASeqLenKvThatSixteenDoesNotDivide)
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

/// The gfx942 kernel computes the bottom-right causal band only, so no shipped kernel is
/// built for no mask or top-left causal and graph_match declines both by name. Top-left at
/// Sq == Skv is the same computation as bottom-right; declining it is a scope choice.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesEveryMaskButBottomRightCausal)
{
    std::vector<DeclineCase> cases;
    for(const int64_t mask : {NO_MASK, TOP_LEFT_CAUSAL})
    {
        GraphSpec prefill;
        setMask(prefill, mask);
        cases.push_back({"mask " + std::to_string(mask) + ", Sq 128 over Skv 256", prefill});
        GraphSpec square;
        square.seqLenQ = 256;
        square.seqLenKv = 256;
        setMask(square, mask);
        cases.push_back({"mask " + std::to_string(mask) + ", Sq == Skv", square});
    }
    expectEveryCaseDeclines(cases);
}

/// A band other than unmasked or causal is a sliding window, and no shipped variant carries
/// one: serving it as plain causal would attend to keys the window excludes.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesABandThatIsNeitherUnmaskedNorCausal)
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

/// The deprecated booleans still mean what they meant: the bottom-right one is served as
/// bottom-right causal, and the top-left one is declined like any top-left graph.
TEST(TestGfx942AttentionUnified2DMatchers, ReadsTheDeprecatedCausalBooleansAsTheirMaskKinds)
{
    // The default graph's bottom-right alignment is reset, so the boolean alone says which.
    GraphSpec bottomRight;
    bottomRight.leftBound = std::nullopt;
    bottomRight.rightBound = std::nullopt;
    bottomRight.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    bottomRight.causalMaskBottomRightDeprecated = true;
    const auto bottomRightBound = matchGraph(bottomRight);
    ASSERT_TRUE(bottomRightBound.has_value());
    EXPECT_EQ(hipdnn_plugin_sdk::ingestor::tryGetBoundInt(*bottomRightBound, MASK_TOKEN),
              BOTTOM_RIGHT_CAUSAL);

    GraphSpec topLeft;
    topLeft.leftBound = std::nullopt;
    topLeft.rightBound = std::nullopt;
    topLeft.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    topLeft.causalMaskDeprecated = true;
    EXPECT_FALSE(graphMatches(topLeft));
}

// =============================================================================
// graph_match declines: addressing limits
// =============================================================================

/// The kernels are built with 32-bit K/V byte addressing and bound their buffer loads at
/// 0x7FFF0000 bytes: past that, loads return zeros. 1048544 keys of 8 heads x 128 x 2 bytes is
/// exactly 0x7FFF0000 bytes, served; one more 16-key block, 1048560 keys (2147450880 bytes), is
/// over the bound.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesACacheBeyondThirtyTwoBitAddressing)
{
    GraphSpec atCap;
    atCap.batch = 1;
    atCap.numQueryHeads = 8;
    atCap.numKvHeads = 8;
    atCap.seqLenQ = 16;
    atCap.seqLenKv = 1048544; // 2147418112 bytes
    EXPECT_TRUE(graphMatches(atCap));

    GraphSpec over = atCap;
    over.seqLenKv = 1048560; // 2147450880 bytes
    EXPECT_FALSE(graphMatches(over));
    over.seqLenKv = 1048576; // 2^31 bytes
    EXPECT_FALSE(graphMatches(over));
}

/// Q and O are held to the same 0x7FFF0000-byte bound, and at GQA ratio 16 Q binds first: K/V
/// here are 16x smaller. 65520 query rows of 128 heads x 128 x 2 bytes is 2146959360 bytes,
/// served; 65536 rows is 2^31 bytes, declined, while K/V stay about 134 MB.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAQueryBeyondThirtyTwoBitAddressing)
{
    GraphSpec atCap;
    atCap.batch = 1;
    atCap.numQueryHeads = 128;
    atCap.numKvHeads = 8;
    atCap.headSize = 128;
    atCap.seqLenQ = 65520; // Q 2146959360 bytes
    atCap.seqLenKv = 65520; // K/V 134184960 bytes
    EXPECT_TRUE(graphMatches(atCap));

    GraphSpec over = atCap;
    over.seqLenQ = 65536; // Q 2^31 bytes
    over.seqLenKv = 65536; // K/V 134217728 bytes
    EXPECT_FALSE(graphMatches(over));
}

/// Extents whose product does not fit int64_t are declined, not wrapped into a passing value.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesExtentsWhoseProductOverflows)
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

TEST(TestGfx942AttentionUnified2DMatchers, DeclinesEveryOptionalFeatureTheKernelsAreNotBuiltWith)
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
TEST(TestGfx942AttentionUnified2DMatchers, AcceptsTheInertSpellingsOfThoseFeatures)
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

/// No LSE-on kernel ships on gfx942, so a stats output is declined by name even when it is the
/// head-major FP32 tensor gfx950's LSE-on kernels write: the shared stats match would accept
/// it, and the catalog would empty at kernel_match with no line to say why.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAWellFormedStatsOutputByName)
{
    GraphSpec spec;
    spec.stats = true;
    EXPECT_FALSE(graphMatches(spec));
    EXPECT_EQ(loggedDeclineCause(spec), "stats");

    const GraphSpec sameWithoutStats;
    EXPECT_TRUE(graphMatches(sameWithoutStats));
}

/// The scale is an f32 kernarg and the kernel folds it into an exp2-domain multiplier: zero,
/// negative, NaN and infinite scales are not softmax scales.
TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAScaleThatIsNotAFinitePositiveNumber)
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

TEST(TestGfx942AttentionUnified2DMatchers, DeclinesAGraphThatIsNotOneSdpaForwardNode)
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

/// The positive control: the default kernel is the D128 cells' default for the default graph.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchAcceptsTheDefaultCandidate)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

/// One baked field at a time: the graph is the default, the candidate differs in exactly one
/// way, and the kernel computes something else if it is served.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchDeclinesACandidateBuiltForAnotherProblem)
{
    std::vector<std::pair<std::string, KernelSpec>> cases;
    const auto with = [&cases](const std::string& name, auto mutate) {
        KernelSpec spec;
        mutate(spec);
        cases.emplace_back(name, spec);
    };
    with("dtype FP16 for a bf16 graph", [](KernelSpec& s) { s.dtype = "FP16"; });
    with("head size 64 for a D128 graph", [](KernelSpec& s) {
        s.headSize = 64;
        s.tileSize = 64;
        s.numWarps = 4;
    });
    with("ratio 8 for a ratio-4 graph", [](KernelSpec& s) { s.numQueriesPerKv = 8; });
    with("ratio 2 for a ratio-4 graph", [](KernelSpec& s) { s.numQueriesPerKv = 2; });
    with("a window of 128", [](KernelSpec& s) { s.slidingWindow = 128; });
    with("no mask for a causal graph", [](KernelSpec& s) { s.maskType = NO_MASK; });
    with("top-left for a bottom-right graph", [](KernelSpec& s) { s.maskType = TOP_LEFT_CAUSAL; });

    ASSERT_FALSE(cases.empty());
    for(const auto& [name, kernelSpec] : cases)
    {
        EXPECT_FALSE(matchesKernel(GraphSpec{}, kernelSpec)) << name;
    }
}

/// The kernel reads the head counts as kernargs and bakes only their ratio: one ratio-4
/// candidate serves every head pair of ratio 4, and none of another. Every pair is of a ratio
/// the catalog ships (1, 2, 4, 8, 16): graph_match declines any other, so a pair of one would
/// never reach kernel_match.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchServesEveryHeadPairOfItsRatio)
{
    for(const auto& [hq, hkv] :
        std::vector<std::pair<int64_t, int64_t>>{{4, 1}, {8, 2}, {16, 4}, {32, 8}, {64, 16}})
    {
        GraphSpec spec;
        spec.numQueryHeads = hq;
        spec.numKvHeads = hkv;
        EXPECT_TRUE(matchesKernel(spec, KernelSpec{})) << "Hq " << hq << " Hkv " << hkv;
    }
    for(const auto& [hq, hkv] :
        std::vector<std::pair<int64_t, int64_t>>{{8, 8}, {8, 4}, {32, 4}, {16, 1}})
    {
        GraphSpec spec;
        spec.numQueryHeads = hq;
        spec.numKvHeads = hkv;
        EXPECT_FALSE(matchesKernel(spec, KernelSpec{})) << "Hq " << hq << " Hkv " << hkv;
    }
}

/// A block of 64 divides Skv 256 and one of 64 does not divide 272: the cache is whole blocks
/// of THIS candidate's size, so one graph can admit some block sizes of a cohort and not
/// others.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchRequiresTheCandidatesBlockToDivideSeqLenKv)
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

/// The kernel derives its cu_q search trip count from the `num_seqs` kernarg and the KMD
/// declares no batch: one record serves any batch.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchServesAnyBatch)
{
    for(const int64_t batch : {1, 3, 511, 512, 4096})
    {
        GraphSpec spec;
        spec.batch = batch;
        spec.seqLenQ = 16;
        spec.seqLenKv = 16;
        EXPECT_TRUE(matchesKernel(spec, KernelSpec{})) << "batch " << batch;
    }
}

/// The metadata carries no shape: one binary serves any Sq and Skv for its configuration.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchServesAnySequenceLengthsForOneConfiguration)
{
    for(const auto& [sq, skv] :
        std::vector<std::pair<int64_t, int64_t>>{{2, 16}, {16, 16}, {100, 4096}, {2048, 2048}})
    {
        GraphSpec spec;
        spec.seqLenQ = sq;
        spec.seqLenKv = skv;
        EXPECT_TRUE(matchesKernel(spec, KernelSpec{})) << "Sq " << sq << " Skv " << skv;
    }
}

/// A geometry the Python builder would refuse is a record this engine cannot launch, so it is
/// declined even where every semantic field agrees.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchDeclinesACandidateWithAnUnbuildableGeometry)
{
    std::vector<std::pair<std::string, KernelSpec>> unbuildable;
    const auto with = [&unbuildable](const std::string& name, auto mutate) {
        KernelSpec spec;
        mutate(spec);
        unbuildable.emplace_back(name, spec);
    };
    with("3 warps", [](KernelSpec& s) { s.numWarps = 3; });
    with("24 rows per warp", [](KernelSpec& s) { s.blockMPerWarp = 24; });
    with("a tile that is not whole blocks", [](KernelSpec& s) { s.tileSize = 70; });
    with("a tile over 8 blocks", [](KernelSpec& s) { s.tileSize = 144; });
    with("block 24", [](KernelSpec& s) { s.blockSize = 24; });
    with("32x32x8 at 16 rows per warp", [](KernelSpec& s) {
        s.blockMPerWarp = 16;
        s.useTransposedQk32x32 = 0;
    });
    with("32x32x8 with a tile that is not a multiple of 32",
         [](KernelSpec& s) { s.tileSize = 48; });
    with("transposed QK without 32x32x8", [](KernelSpec& s) { s.useMfma32x32x8 = 0; });
    with("a flag of 2", [](KernelSpec& s) { s.useMfma32x32x8 = 2; });
    with("a K ring flag of 2", [](KernelSpec& s) { s.useKSlicedRing = 2; });
    with("a mask limit flag of -1", [](KernelSpec& s) { s.useTransposedMaskLimit = -1; });
    with("mask 7", [](KernelSpec& s) { s.maskType = 7; });
    with("ratio 0", [](KernelSpec& s) { s.numQueriesPerKv = 0; });
    with("ratio 17", [](KernelSpec& s) { s.numQueriesPerKv = 17; });

    for(const auto& [name, spec] : unbuildable)
    {
        EXPECT_FALSE(matchesKernel(GraphSpec{}, spec)) << name;
    }
}

/// A completed record omits no field; one that does, or that spells one in another type, is
/// declined rather than given a default.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchDeclinesAnIncompleteOrMistypedRecord)
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
    mistyped.metadata[std::string("tile_size")] = std::string("32");
    EXPECT_FALSE(matchesKernel(graph, mistyped));
}

/// Geometry alternatives the catalog may grow into are served exactly like the default when
/// they are buildable and everything else agrees: nothing in kernel_match keys on the default.
TEST(TestGfx942AttentionUnified2DMatchers, KernelMatchServesEveryBuildableGeometry)
{
    for(const int64_t warps : {1, 2, 4, 8})
    {
        for(const int64_t multiplier : {1, 2, 4, 8})
        {
            KernelSpec spec = narrowKernel();
            spec.numWarps = warps;
            spec.tileSize = 16 * multiplier;
            EXPECT_TRUE(matchesKernel(GraphSpec{}, spec))
                << "narrow, warps " << warps << " tile " << spec.tileSize;
        }
        for(const int64_t multiplier : {2, 4, 8})
        {
            KernelSpec spec;
            spec.numWarps = warps;
            spec.tileSize = 16 * multiplier;
            EXPECT_TRUE(matchesKernel(GraphSpec{}, spec))
                << "transposed 32x32x8, warps " << warps << " tile " << spec.tileSize;
        }
    }
    // 8 warps x 32 rows is a gfx942 geometry (the builder pairs it with direct-Q and
    // conflict-free V stores, which the spec carries).
    KernelSpec eightWide;
    eightWide.numWarps = 8;
    eightWide.tileSize = 64;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, eightWide));
    KernelSpec wide = narrowKernel();
    wide.blockMPerWarp = 32;
    wide.useMfma32x32x8 = 1;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, wide)) << "wide 32x32x8, not transposed";
}

// =============================================================================
// score
// =============================================================================

namespace
{

/// A candidate of the given geometry, with the other fields at the D128 default.
KernelSpec geometry(int64_t tile, int64_t warps, int64_t rows, int64_t x8 = 0, int64_t qk32 = 0)
{
    KernelSpec spec;
    spec.tileSize = tile;
    spec.numWarps = warps;
    spec.blockMPerWarp = rows;
    spec.useMfma32x32x8 = x8;
    spec.useTransposedQk32x32 = qk32;
    return spec;
}

} // namespace

/// The ranking is a stable fallback order for cold selection, not a performance model: the
/// head size's default geometry above every alternative, the alternatives in ascending
/// lexicographic (tile, warps, rows, 32x32x8, transposed) order, and every distinct geometry a
/// distinct score so the selector's descriptor-id tie-break never decides between two.
TEST(TestGfx942AttentionUnified2DMatchers, ScoreRanksTheDefaultFirstThenAscendingGeometry)
{
    const double defaultScore = scoreOf(GraphSpec{}, KernelSpec{});

    const std::vector<KernelSpec> ascending{geometry(16, 2, 16),
                                            geometry(32, 1, 16),
                                            geometry(32, 2, 16),
                                            geometry(32, 2, 32, 1, 0),
                                            geometry(32, 4, 16),
                                            geometry(32, 4, 32, 1, 1),
                                            geometry(64, 1, 16),
                                            geometry(64, 2, 32, 1, 1),
                                            geometry(64, 4, 32, 1, 1),
                                            geometry(64, 8, 16),
                                            geometry(128, 1, 16),
                                            geometry(128, 4, 16)};

    double previous = defaultScore;
    for(const auto& candidate : ascending)
    {
        const double score = scoreOf(GraphSpec{}, candidate);
        EXPECT_GT(score, 0.0) << "tile " << candidate.tileSize << " warps " << candidate.numWarps;
        EXPECT_LT(score, previous)
            << "tile " << candidate.tileSize << " warps " << candidate.numWarps << " rows "
            << candidate.blockMPerWarp << " should rank below the one before it";
        previous = score;
    }
    // Every alternative sits in the band below ALTERNATIVE_CEILING (1.0), under the default.
    EXPECT_LT(scoreOf(GraphSpec{}, ascending.front()), 1.0);
    EXPECT_EQ(defaultScore, 2.0);
}

/// Each head size has its own default: D64's (T64, 4 warps) on a D64 graph, D128's (T32,
/// 2 warps) on a D128 graph. The other head size's default geometry is just an alternative.
TEST(TestGfx942AttentionUnified2DMatchers, ScoreRanksEachHeadSizesOwnDefaultFirst)
{
    GraphSpec d64;
    d64.headSize = 64;
    EXPECT_EQ(scoreOf(d64, d64DefaultKernel()), 2.0);

    KernelSpec d128Geometry = d64DefaultKernel();
    d128Geometry.tileSize = 32;
    d128Geometry.numWarps = 2;
    EXPECT_LT(scoreOf(d64, d128Geometry), 1.0);

    KernelSpec d64Geometry;
    d64Geometry.tileSize = 64;
    d64Geometry.numWarps = 4;
    EXPECT_LT(scoreOf(GraphSpec{}, d64Geometry), 1.0);
}

/// Only a record kernel_match already declined scores below every legal candidate.
TEST(TestGfx942AttentionUnified2DMatchers, ScoreGivesAnUnbuildableRecordTheFloor)
{
    EXPECT_EQ(scoreOf(GraphSpec{}, geometry(64, 3, 16)), 0.0);

    // The widest legal candidate, block 64 at the dispatcher's 8x tile and 8 warps, still
    // scores above the floor.
    KernelSpec widest = geometry(512, 8, 16);
    widest.blockSize = 64;
    EXPECT_GT(scoreOf(GraphSpec{}, widest), 0.0);
}

/// The ranking does not read the mask, dtype or ratio: two candidates of one geometry for
/// different problems score alike, so a catalog cannot be steered by them.
TEST(TestGfx942AttentionUnified2DMatchers, ScoreDependsOnlyOnTheGeometryAndHeadSize)
{
    KernelSpec other = geometry(64, 4, 16);
    other.maskType = NO_MASK;
    other.dtype = "FP16";
    other.numQueriesPerKv = 8;
    EXPECT_EQ(scoreOf(GraphSpec{}, other), scoreOf(GraphSpec{}, geometry(64, 4, 16)));
}

/// rocKE's production stack (the mask limit and the invariant hoist on) ranks above the
/// head size's default geometry, whatever geometry carries it: bf16 D128's production kernel
/// is 2 warps x 32 rows at T64, not the T32 default.
TEST(TestGfx942AttentionUnified2DMatchers, ScoreRanksTheProductionStackAboveTheDefault)
{
    EXPECT_EQ(scoreOf(GraphSpec{}, KernelSpec{}), 2.0);

    KernelSpec d128Production;
    d128Production.tileSize = 64;
    d128Production.useTransposedMaskLimit = 1;
    d128Production.useTransposedInvariantHoist = 1;
    EXPECT_EQ(scoreOf(GraphSpec{}, d128Production), 3.0);

    GraphSpec d64;
    d64.headSize = 64;
    KernelSpec d64Production = d64DefaultKernel();
    d64Production.useKSlicedRing = 1;
    d64Production.useQDirectGlobal = 1;
    d64Production.useConflictFreeVStore = 1;
    d64Production.useTransposedMaskLimit = 1;
    d64Production.useTransposedInvariantHoist = 1;
    EXPECT_EQ(scoreOf(d64, d64Production), 3.0);
    EXPECT_LT(scoreOf(d64, d64DefaultKernel()), scoreOf(d64, d64Production));

    // One half of the pair is not the stack: it ranks no higher than the default geometry it rides.
    KernelSpec half = d64DefaultKernel();
    half.useTransposedMaskLimit = 1;
    EXPECT_LE(scoreOf(d64, half), 2.0);
}

/// Alternatives that differ only in a codegen flag still get distinct scores below the default,
/// so the descriptor-id tie-break never decides between them.
TEST(TestGfx942AttentionUnified2DMatchers,
     ScoreSeparatesAlternativesThatDifferOnlyInTheCodegenStack)
{
    std::set<double> scores;
    int variants = 0;
    for(int bits = 0; bits < 32; ++bits)
    {
        KernelSpec spec = geometry(64, 8, 32, 1, 1);
        spec.useKSlicedRing = bits & 1;
        spec.useQDirectGlobal = (bits >> 1) & 1;
        spec.useConflictFreeVStore = (bits >> 2) & 1;
        spec.useTransposedMaskLimit = (bits >> 3) & 1;
        spec.useTransposedInvariantHoist = (bits >> 4) & 1;
        if(spec.useTransposedMaskLimit != 0 && spec.useTransposedInvariantHoist != 0)
        {
            continue; // the production stack, ranked above
        }
        ++variants;
        const double score = scoreOf(GraphSpec{}, spec);
        EXPECT_GT(score, 0.0);
        EXPECT_LT(score, 1.0);
        scores.insert(score);
    }
    EXPECT_EQ(scores.size(), static_cast<std::size_t>(variants));
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
