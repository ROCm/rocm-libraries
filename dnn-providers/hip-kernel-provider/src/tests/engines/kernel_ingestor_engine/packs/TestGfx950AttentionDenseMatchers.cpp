// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>

#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file TestGfx950AttentionDenseMatchers.cpp
 * @brief Applicability negatives for hipkernel:Gfx950AttentionDense.
 *
 * One case per "must decline" row of the rejection checklist in `mining.md`, in the
 * same severity order: silent-wrong-answer rows first, then faults, then declined
 * features. Each of these is a graph the kernel would accept and compute something
 * plausible-but-wrong for if the matcher did not stop it.
 *
 * These are matcher-only: no device, no compile, no launch.
 *
 * gfx950-specific behaviors covered:
 *   - LAYOUT: every operand's stride spelling is set independently, so each tensor's clause
 *     of the layout gate is pinned by a case that flips that tensor and nothing else. All
 *     four operands are gated in graph_match; prepare() re-checks the output as defence in
 *     depth for a caller that reaches the handler without having matched.
 *   - SHAPE: per-operand dimension overrides sit beside the layout fields, so a single
 *     operand can disagree with the problem shape on one axis while staying dense BSHD for
 *     its own extents. That is what makes the cross-operand agreement clauses reachable.
 *   - RAGGED: gfx950 serves non-tile-multiple self-attention lengths through a separately
 *     compiled boundary-padding path; kernel_match's tile rule is conditional on `ragged`.
 *   - SLIDING WINDOW: declined. No variant in this catalog carries a non-zero
 *     sliding_window, so a windowed graph has nothing that could serve it.
 *   - NO block_m KMD field: the KMD declares only what varies, and the tile does not. The
 *     value the engine launches with is pinned to the shipped set, not baked into the
 *     binary -- see note 1 in Gfx950AttentionDenseGeometry.hpp for what must change if a
 *     variant ever moves it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::MatchContext;

constexpr std::string_view GRAPH_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "hipkernel.gfx950_attention_dense.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_dense.score";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t EXTRA_UID = 99;

/// Base shape: bf16, D128, B=2, Hq=Hkv=4, Sq=Skv=256, top-left causal.
constexpr int64_t BATCH = 2;
constexpr int64_t HEADS = 4;
constexpr int64_t SEQ = 256;
constexpr int64_t HEAD_SIZE = 128;
constexpr float SCALE = 0.08838834764831843F;

DeviceProperties testDeviceProperties()
{
    DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    return properties;
}

/// BSHD strides for (B, H, S, D) LOGICAL dims -- token-major, head varying fastest.
std::vector<int64_t> bshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {sequence * heads * headSize, headSize, heads * headSize, 1};
}

/// BHSD strides -- the layout a BHSD-strided graph carries.
std::vector<int64_t> bhsdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {heads * sequence * headSize, sequence * headSize, headSize, 1};
}

/// Elements of slack a real allocator leaves between token rows.
constexpr int64_t ROW_PAD = 8;

/// BSHD axis order with the per-token row stride padded past `heads * headSize`. The
/// tensor is still token-major and still dense in the frontend's sense, but the matcher
/// compares strides by exact equality, so the padding alone is disqualifying.
std::vector<int64_t> paddedBshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    const int64_t rowStride = heads * headSize + ROW_PAD;
    return {sequence * rowStride, headSize, rowStride, 1};
}

/// The stride spelling one operand carries. Chosen per tensor so a fixture can hand Q
/// one layout and K, V or O another -- the mixed-layout graph is the hazard the
/// per-operand clauses of the matcher exist to catch.
enum class StrideLayout
{
    BSHD,
    BHSD,
    PADDED_BSHD
};

std::vector<int64_t>
    stridesFor(StrideLayout layout, int64_t heads, int64_t sequence, int64_t headSize)
{
    switch(layout)
    {
    case StrideLayout::BHSD:
        return bhsdStrides(heads, sequence, headSize);
    case StrideLayout::PADDED_BSHD:
        return paddedBshdStrides(heads, sequence, headSize);
    case StrideLayout::BSHD:
    default:
        return bshdStrides(heads, sequence, headSize);
    }
}

struct GraphSpec
{
    int64_t batch = BATCH;
    int64_t numQueryHeads = HEADS;
    int64_t numKvHeads = HEADS;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t headSize = HEAD_SIZE;
    int64_t headSizeV = HEAD_SIZE;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    std::optional<data_objects::DataType> vDataType;
    StrideLayout qLayout = StrideLayout::BSHD;
    StrideLayout kLayout = StrideLayout::BSHD;
    StrideLayout vLayout = StrideLayout::BSHD;
    StrideLayout oLayout = StrideLayout::BSHD;
    bool omitStrides = false;

    // Per-operand dimension overrides, the dimension counterpart of the per-operand
    // layout fields above. Each falls back to the shared value, so a spec that leaves
    // them unset builds four operands that agree on every axis.
    //
    // They spell the one family of graphs the shared fields cannot: an operand whose
    // extents disagree with the problem shape the kernel derives from Q and K. Strides
    // follow the override, so a perturbed operand is still dense BSHD for its own
    // extents and the layout gate is not what rejects the graph.
    std::optional<int64_t> qBatch;
    std::optional<int64_t> kBatch;
    std::optional<int64_t> vBatch;
    std::optional<int64_t> oBatch;
    std::optional<int64_t> oNumHeads;
    std::optional<int64_t> oSeqLen;
    std::optional<int64_t> oHeadSize;

    // Mask. Defaults to top-left causal.
    std::optional<int64_t> leftBound = -1;
    std::optional<int64_t> rightBound = 0;
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    bool causalMaskDeprecated = false;
    bool causalMaskBottomRightDeprecated = false;

    std::optional<float> attnScaleValue = SCALE;

    // Optional features.
    std::optional<int64_t> attnMaskUid;
    std::optional<int64_t> scaleTensorUid;
    std::optional<int64_t> seqLenQUid;
    std::optional<int64_t> pageTableKUid;
    std::optional<int64_t> sinkTokenUid;
    std::optional<int64_t> blockMaskUid;
    std::optional<int64_t> statsUid;
    std::optional<int64_t> descaleQUid;
    std::optional<float> dropoutProbability;
    std::optional<bool> generateStats;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation implementation
        = data_objects::AttentionImplementation::AUTO;

    bool twoNodes = false;

    /// Gives Q, K, V and O the same stride spelling.
    void setEveryLayout(StrideLayout layout)
    {
        qLayout = layout;
        kLayout = layout;
        vLayout = layout;
        oLayout = layout;
    }
};

flatbuffers::FlatBufferBuilder buildSdpaGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    // The output's extents: heads and sequence follow Q, head size follows V, which is
    // what an SDPA output carries. An override replaces the inherited value.
    const int64_t outputHeads = spec.oNumHeads.value_or(spec.numQueryHeads);
    const int64_t outputSeqLen = spec.oSeqLen.value_or(spec.seqLenQ);
    const int64_t outputHeadSize = spec.oHeadSize.value_or(spec.headSizeV);

    const std::vector<int64_t> qDims{
        spec.qBatch.value_or(spec.batch), spec.numQueryHeads, spec.seqLenQ, spec.headSize};
    const std::vector<int64_t> kDims{
        spec.kBatch.value_or(spec.batch), spec.numKvHeads, spec.seqLenKv, spec.headSize};
    const std::vector<int64_t> vDims{
        spec.vBatch.value_or(spec.batch), spec.numKvHeads, spec.seqLenKv, spec.headSizeV};
    const std::vector<int64_t> oDims{
        spec.oBatch.value_or(spec.batch), outputHeads, outputSeqLen, outputHeadSize};

    const auto qStrides = stridesFor(spec.qLayout, spec.numQueryHeads, spec.seqLenQ, spec.headSize);
    const auto kStrides = stridesFor(spec.kLayout, spec.numKvHeads, spec.seqLenKv, spec.headSize);
    const auto vStrides = stridesFor(spec.vLayout, spec.numKvHeads, spec.seqLenKv, spec.headSizeV);
    const auto oStrides = stridesFor(spec.oLayout, outputHeads, outputSeqLen, outputHeadSize);

    const std::vector<int64_t>* const qStridesPtr = spec.omitStrides ? nullptr : &qStrides;
    const std::vector<int64_t>* const kStridesPtr = spec.omitStrides ? nullptr : &kStrides;
    const std::vector<int64_t>* const vStridesPtr = spec.omitStrides ? nullptr : &vStrides;
    const std::vector<int64_t>* const oStridesPtr = spec.omitStrides ? nullptr : &oStrides;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, Q_UID, nullptr, spec.dataType, qStridesPtr, &qDims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, K_UID, nullptr, spec.dataType, kStridesPtr, &kDims, false));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   V_UID,
                                                   nullptr,
                                                   spec.vDataType.value_or(spec.dataType),
                                                   vStridesPtr,
                                                   &vDims,
                                                   false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, O_UID, nullptr, spec.dataType, oStridesPtr, &oDims, false));

    const auto attributesFor = [&]() {
        data_objects::SdpaAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_q_tensor_uid(Q_UID);
        attributesBuilder.add_k_tensor_uid(K_UID);
        attributesBuilder.add_v_tensor_uid(V_UID);
        attributesBuilder.add_o_tensor_uid(O_UID);

        if(spec.leftBound.has_value())
        {
            attributesBuilder.add_left_bound(*spec.leftBound);
        }
        if(spec.rightBound.has_value())
        {
            attributesBuilder.add_right_bound(*spec.rightBound);
        }
        attributesBuilder.add_diagonal_alignment(spec.alignment);
        attributesBuilder.add_causal_mask(spec.causalMaskDeprecated);
        attributesBuilder.add_causal_mask_bottom_right(spec.causalMaskBottomRightDeprecated);
        if(spec.attnScaleValue.has_value())
        {
            attributesBuilder.add_attn_scale_value(*spec.attnScaleValue);
        }

        if(spec.attnMaskUid.has_value())
        {
            attributesBuilder.add_attn_mask_tensor_uid(*spec.attnMaskUid);
        }
        if(spec.scaleTensorUid.has_value())
        {
            attributesBuilder.add_scale_tensor_uid(*spec.scaleTensorUid);
        }
        if(spec.seqLenQUid.has_value())
        {
            attributesBuilder.add_seq_len_q_tensor_uid(*spec.seqLenQUid);
        }
        if(spec.pageTableKUid.has_value())
        {
            attributesBuilder.add_page_table_k_tensor_uid(*spec.pageTableKUid);
        }
        if(spec.sinkTokenUid.has_value())
        {
            attributesBuilder.add_sink_token_tensor_uid(*spec.sinkTokenUid);
        }
        if(spec.blockMaskUid.has_value())
        {
            attributesBuilder.add_block_mask_tensor_uid(*spec.blockMaskUid);
        }
        if(spec.statsUid.has_value())
        {
            attributesBuilder.add_stats_tensor_uid(*spec.statsUid);
        }
        if(spec.descaleQUid.has_value())
        {
            attributesBuilder.add_descale_q_tensor_uid(*spec.descaleQUid);
        }
        if(spec.dropoutProbability.has_value())
        {
            attributesBuilder.add_dropout_probability(*spec.dropoutProbability);
        }
        if(spec.generateStats.has_value())
        {
            attributesBuilder.add_generate_stats(*spec.generateStats);
        }
        attributesBuilder.add_alibi_mask(spec.alibiMask);
        attributesBuilder.add_padding_mask(spec.paddingMask);
        attributesBuilder.add_mma_core_mode(spec.mmaCoreMode);
        attributesBuilder.add_implementation(spec.implementation);
        return attributesBuilder.Finish();
    };

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                   "sdpa",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::NodeAttributes::SdpaAttributes,
                                                   attributesFor().Union()));
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa2",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }

    auto name = builder.CreateString("gfx950_attention_dense_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));

    auto builder = buildSdpaGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};
    return matcher(context);
}

/// KernelSpec spells the fields the engine's KMD declares: dtype, head_size,
/// num_query_heads, num_kv_heads, batch, seqlen_q, seqlen_kv, causal, sliding_window,
/// ragged.
///
/// The KMD carries only what VARIES between candidates, so a knob the catalog holds at a
/// single value -- block_m, block_n, waves_per_eu and the rest of the tuning surface --
/// has no field here. A knob that starts varying must be added to the KMD first: without
/// a field of its own, two candidates differing only in that knob complete to the same
/// catalog key, and the loader keeps one of them and drops the other.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = HEAD_SIZE;
    int64_t numQueryHeads = HEADS;
    int64_t numKvHeads = HEADS;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t batch = BATCH;
    int64_t causal = 1;
    int64_t slidingWindow = 0;
    int64_t ragged = 0;
};

hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea1");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea3");
    kernel.metadata = {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("num_query_heads"), spec.numQueryHeads},
        {std::string("num_kv_heads"), spec.numKvHeads},
        {std::string("seqlen_q"), spec.seqLenQ},
        {std::string("seqlen_kv"), spec.seqLenKv},
        {std::string("batch"), spec.batch},
        {std::string("causal"), spec.causal},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("ragged"), spec.ragged},
    };
    return kernel;
}

bool matchesKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));

    auto builder = buildSdpaGraph(graphSpec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto bound = graphMatcher(context);
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return kernelMatcher(context, *bound, makeKernel(kernelSpec));
}

double scoreOf(const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto graphMatcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));

    auto builder = buildSdpaGraph(GraphSpec{});
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    const auto properties = testDeviceProperties();
    const MatchContext context{graph, 0, properties};

    const auto bound = graphMatcher(context);
    EXPECT_TRUE(bound.has_value());
    return scorer(context, bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

// ---------------------------------------------------------------------------
// Positive controls. Every negative below is only meaningful because these pass.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsDenseBshdCausalGraph)
{
    EXPECT_TRUE(matchGraph(GraphSpec{}).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsNoMaskGraph)
{
    GraphSpec spec;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsGroupedQueryAttention)
{
    GraphSpec spec;
    spec.numQueryHeads = 8;
    spec.numKvHeads = 2;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsHeadSize64)
{
    GraphSpec spec;
    spec.headSize = 64;
    spec.headSizeV = 64;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Silent-wrong-answer rows first (Tier 1 in mining.md)
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdLayout)
{
    // The kernel bakes BSHD strides and takes no stride kernargs; BHSD is wrong elements.
    GraphSpec spec;
    spec.setEveryLayout(StrideLayout::BHSD);
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdQueryAlone)
{
    // Q is the only operand flipped, so only the Q clause of the layout gate can stop it.
    GraphSpec spec;
    spec.qLayout = StrideLayout::BHSD;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdKeyAlone)
{
    // The mixed-layout graph: a BSHD query beside a BHSD key. Q sails through the layout
    // gate, so the K clause is the only thing between this graph and a launch that reads
    // K as if it were packed -- wrong elements in bounds, no fault and no status code.
    GraphSpec spec;
    spec.kLayout = StrideLayout::BHSD;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdValueAlone)
{
    // Same hazard on the value operand. V shares K's base and stride in the kernel builder,
    // so a BHSD V is addressed with the packed stride the builder computed for K.
    GraphSpec spec;
    spec.vLayout = StrideLayout::BHSD;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsSingleHeadUnderEitherStrideSpelling)
{
    // A single-head tensor is byte-identically BSHD and BHSD; strict compare would
    // decline a graph the kernel serves perfectly and empty the whole catalog.
    GraphSpec bshd;
    bshd.numQueryHeads = 1;
    bshd.numKvHeads = 1;
    EXPECT_TRUE(matchGraph(bshd).has_value());

    GraphSpec bhsd = bshd;
    bhsd.setEveryLayout(StrideLayout::BHSD);
    EXPECT_TRUE(matchGraph(bhsd).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddedSequenceStride)
{
    // Far likelier in a real graph than a full transpose: BSHD axis ORDER, but token rows
    // padded past heads * headSize. The kernel bakes heads * headSize as the row stride,
    // so every row after the first is read off by the accumulated padding.
    //
    // batch = 1 makes the batch axis unit-extent and therefore exempt from the stride
    // compare, which leaves the sequence-stride clause as the only thing that can catch
    // this. Heads and sequence both stay above 1 so neither of those is exempted away.
    GraphSpec packed;
    packed.batch = 1;
    EXPECT_TRUE(matchGraph(packed).has_value());

    GraphSpec padded = packed;
    padded.qLayout = StrideLayout::PADDED_BSHD;
    EXPECT_FALSE(matchGraph(padded).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBhsdOutput)
{
    // The layout gate covers all four operands. The kernel bakes BSHD for the epilogue
    // exactly as it does for the inputs, so a differently-strided output is outside the
    // capability set and declines here -- leaving the graph free for another engine
    // rather than being claimed and then faulted on in prepare().
    GraphSpec spec;
    spec.oLayout = StrideLayout::BHSD;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddedOutputSequenceStride)
{
    // Exact-equality rule, same as the inputs: a dense-but-padded row stride is not the
    // layout the epilogue bakes. Batch is 1 so the batch axis is unit-extent and exempt,
    // leaving the sequence stride as the only clause that can reject this graph.
    GraphSpec spec;
    spec.batch = 1;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec padded = spec;
    padded.oLayout = StrideLayout::PADDED_BSHD;
    EXPECT_FALSE(matchGraph(padded).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBottomRightCausalWhenSeqLensDiffer)
{
    // Top-left causal clamp != bottom-right when Sq != Skv: serving it is a wrong answer.
    GraphSpec spec;
    spec.seqLenKv = SEQ * 2;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsBottomRightCausalWhenSeqLensMatch)
{
    // The complement: every shipped quick/SdpaFwd causal bundle sets BOTTOM_RIGHT at
    // Sq == Skv; declining it outright declines all of them.
    GraphSpec spec;
    spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Faults and malformed input
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesGraphWithNoStrides)
{
    GraphSpec spec;
    spec.omitStrides = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMultiNodeGraph)
{
    GraphSpec spec;
    spec.twoNodes = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedHeadSize)
{
    GraphSpec spec;
    spec.headSize = 256;
    spec.headSizeV = 256;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMismatchedHeadSizes)
{
    // hipDNN permits D_qk != D_v; the kernel has ONE head_size.
    GraphSpec spec;
    spec.headSizeV = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Cross-operand shape agreement.
//
// The kernel derives ONE problem shape from Q and K and addresses every operand from
// it -- there are no per-tensor extent kernargs. An operand whose dims disagree is
// therefore walked with the wrong bounds: in-bounds wrong elements where the operand
// is larger, an out-of-bounds write where the output is smaller.
//
// Each case perturbs exactly one axis of one operand and leaves that operand's strides
// dense BSHD for its own extents, so the layout gate passes and the cross-tensor clause
// named in the comment is the only thing that can decline the graph. Each is paired
// with the unperturbed spec as its positive control.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesOutputBatchMismatch)
{
    // Kills the O-vs-batch clause. The epilogue reuses the query base and stride, so an
    // output allocated for a different batch count is written past its own end.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.oBatch = BATCH + 1;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesOutputHeadCountMismatch)
{
    // Kills the O-vs-numQueryHeads clause. The grid is sized from Q's head count, so a
    // narrower or wider output is indexed by head ids it has no storage for. Q, K and V
    // are untouched, so the GQA divisibility check still sees Hq == Hkv.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.oNumHeads = HEADS * 2;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesOutputSequenceLengthMismatch)
{
    // Kills the O-vs-seqLenQ clause. seqlen_q is a launch argument taken from Q, and the
    // query block id indexes the output with it; an output of a different length is the
    // same walk over the wrong extent.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.oSeqLen = SEQ * 2;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesOutputHeadSizeMismatch)
{
    // Kills the O-vs-headSize clause, which compares O against Q's head size. headSizeV
    // is left at the default so V still agrees with Q and the V clause -- the one
    // DeclinesMismatchedHeadSizes pins -- cannot be what stops this graph.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.oHeadSize = 64;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesKeyBatchMismatch)
{
    // Kills the K-vs-batch clause. batch comes from Q; V and O are left agreeing with it
    // so their own batch clauses pass and K's is the only one left to fire.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.kBatch = BATCH + 1;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesValueBatchMismatch)
{
    // Kills the V-vs-batch clause. V shares K's base and stride in the kernel builder,
    // so a V holding a different number of batches is read as if it held K's.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.vBatch = BATCH + 1;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesQueryBatchDisagreement)
{
    // The complementary direction: Q is the operand that disagrees, so the problem's
    // batch moves and K, V and O are all left behind. V is compared first, so the V
    // clause is what declines this one -- the case exists to show that the batch
    // agreement is judged against Q, not against a majority of the operands.
    const GraphSpec spec;
    EXPECT_TRUE(matchGraph(spec).has_value());

    GraphSpec mismatched = spec;
    mismatched.qBatch = BATCH + 1;
    EXPECT_FALSE(matchGraph(mismatched).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesNonDivisibleGqaGrouping)
{
    // Integer division drops the remainder heads silently.
    GraphSpec spec;
    spec.numQueryHeads = 6;
    spec.numKvHeads = 4;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMixedOperandDataTypes)
{
    GraphSpec spec;
    spec.vDataType = data_objects::DataType::HALF;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedDataType)
{
    GraphSpec spec;
    spec.dataType = data_objects::DataType::FLOAT;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesMissingAttentionScale)
{
    GraphSpec spec;
    spec.attnScaleValue = std::nullopt;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBothDeprecatedCausalBooleans)
{
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.causalMaskBottomRightDeprecated = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesGraphsPastThe32BitExtentLimit)
{
    GraphSpec spec;
    spec.batch = 1;
    spec.numQueryHeads = 128;
    spec.numKvHeads = 128;
    spec.seqLenQ = 131072;
    spec.seqLenKv = 131072;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Declined optional features
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAdditiveAttentionMask)
{
    GraphSpec spec;
    spec.attnMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDeviceResidentScaleTensor)
{
    GraphSpec spec;
    spec.scaleTensorUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesVarlen)
{
    GraphSpec spec;
    spec.seqLenQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPagedKv)
{
    GraphSpec spec;
    spec.pageTableKUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAttentionSinks)
{
    GraphSpec spec;
    spec.sinkTokenUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBlockSparseMask)
{
    GraphSpec spec;
    spec.blockMaskUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSoftmaxStatsBothSpellings)
{
    GraphSpec byUid;
    byUid.statsUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(byUid).has_value());

    GraphSpec byFlag;
    byFlag.generateStats = true;
    EXPECT_FALSE(matchGraph(byFlag).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsExplicitlyDisabledStats)
{
    // generate_stats is optional<bool>; explicit false is not a request for stats.
    GraphSpec spec;
    spec.generateStats = false;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesDropout)
{
    GraphSpec spec;
    spec.dropoutProbability = 0.1F;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesFp8Descale)
{
    GraphSpec spec;
    spec.descaleQUid = EXTRA_UID;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesAlibiMask)
{
    GraphSpec spec;
    spec.alibiMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesPaddingMask)
{
    GraphSpec spec;
    spec.paddingMask = true;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesNonAutoImplementationHint)
{
    GraphSpec spec;
    spec.implementation = data_objects::AttentionImplementation::COMPOSITE;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, AcceptsMmaCoreModeFloat)
{
    // Allow-list, not `!= UNSET`: shipped SdpaFwd bundles set FLOAT.
    GraphSpec spec;
    spec.mmaCoreMode = data_objects::DataType::FLOAT;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesUnsupportedMmaCoreMode)
{
    GraphSpec spec;
    spec.mmaCoreMode = data_objects::DataType::BFLOAT16;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// Sliding-window: declined outright. No variant in this catalog carries a
// non-zero sliding_window, so a windowed graph has nothing that could serve it.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSlidingWindowForSelfAttention)
{
    // Every shipped variant is sliding_window = 0. Serving a windowed graph on a
    // full-length causal binary would attend the whole lower triangle instead of the
    // requested band: wrong numerics, no error.
    GraphSpec spec;
    spec.leftBound = 127;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesSlidingWindowForCrossAttention)
{
    // Declined for the same reason as the self-attention case above; the unequal
    // sequence lengths are incidental, not the cause.
    GraphSpec spec;
    spec.seqLenKv = SEQ * 2;
    spec.leftBound = 127;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesWhenDeprecatedBoolIsSetAlongsideBound)
{
    // A REAL BOUND WINS OVER THE DEPRECATED BOOLEAN, and this case is the reason the
    // ordering is load-bearing. causal_mask=true alone is served as plain causal; add
    // left_bound and the graph is asking for a band the catalog cannot supply, so it
    // must decline. If the boolean won instead, the window would be silently widened
    // to the full triangle -- accepted, dispatched, and wrong.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = 127;
    spec.rightBound = 0;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, StillServesPlainDeprecatedCausalWithNoBound)
{
    // The control: bound-wins must not over-fire and decline ordinary deprecated-causal.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = std::nullopt;
    EXPECT_TRUE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesBidirectionalSlidingWindow)
{
    // A graph with both left_bound and a non-zero right_bound is a bidirectional
    // window. The gfx950 kernel is hard-causal (upper mask only) and has no
    // right-bound field, so serving it would produce silent wrong numerics.
    // The review's exact scenario: left=127, right=64, same shape as a shipped SWA variant.
    GraphSpec spec;
    spec.leftBound = 127;
    spec.rightBound = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

TEST(TestGfx950AttentionDenseGraphMatch, DeclinesCausalWithNonZeroRightBound)
{
    // A graph with causal_mask=true and right_bound > 0 also describes a shape the kernel
    // cannot serve correctly. The right_bound wins over the deprecated causal boolean.
    GraphSpec spec;
    spec.causalMaskDeprecated = true;
    spec.leftBound = std::nullopt;
    spec.rightBound = 64;
    EXPECT_FALSE(matchGraph(spec).has_value());
}

// ---------------------------------------------------------------------------
// kernel_match
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseKernelMatch, AcceptsTheCandidateBakedForThisGraph)
{
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherDtype)
{
    KernelSpec kernel;
    kernel.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, AlignedCandidateAcceptsDifferentBatch)
{
    // Aligned (ragged=0) kernels are shape-generic: batch/seqlen equality is not
    // enforced. A KD compiled with batch=BATCH+1 serves a graph with batch=BATCH.
    KernelSpec kernel;
    kernel.batch = BATCH + 1;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, AlignedCandidateAcceptsDifferentSeqLen)
{
    // Same shape-generic rule: seqlen mismatch is not a rejection for aligned kernels.
    // The graph has seqLenKv=SEQ (default); kernel was compiled with SEQ*2.
    KernelSpec kernel;
    kernel.seqLenKv = SEQ * 2;
    EXPECT_TRUE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForAnotherHeadCount)
{
    KernelSpec kernel;
    kernel.numQueryHeads = HEADS * 2;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesACandidateBakedForTheOtherMask)
{
    KernelSpec kernel;
    kernel.causal = 0; // default GraphSpec is causal
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesARaggedCandidateForAnAlignedGraph)
{
    // The ragged path pads boundary tiles on-chip; the binaries are different.
    KernelSpec kernel;
    kernel.ragged = 1;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAnAlignedCandidateForARaggedGraph)
{
    // An aligned binary's grid does not cover the partial final query block.
    GraphSpec graph;
    graph.seqLenQ = 4000;
    graph.seqLenKv = 4000;
    KernelSpec aligned;
    aligned.seqLenQ = 4000;
    aligned.seqLenKv = 4000;
    EXPECT_FALSE(matchesKernel(graph, aligned));
}

TEST(TestGfx950AttentionDenseKernelMatch, AcceptsARaggedCandidateForARaggedGraph)
{
    // Positive control for the ragged pair.
    GraphSpec graph;
    graph.seqLenQ = 4000;
    graph.seqLenKv = 4000;
    KernelSpec ragged;
    ragged.ragged = 1;
    ragged.seqLenQ = 4000;
    ragged.seqLenKv = 4000;
    EXPECT_TRUE(matchesKernel(graph, ragged));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesAnAlignedCandidateWhoseTileDoesNotDivideSeqLenKv)
{
    // Tile alignment is checked against GFX950_ATTENTION_DENSE_BLOCK_N, a constant in the
    // pack rather than a KMD field, because the tile does not vary across the catalog.
    GraphSpec graph;
    graph.seqLenKv = 288; // not a multiple of 64
    KernelSpec kernel;
    kernel.seqLenKv = 288;
    EXPECT_FALSE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RaggedKernelRefusesWrongBatch)
{
    // Tail (ragged=1) KDs bake the exact shape; a request with a different batch
    // must not reuse the baked binary (silent wrong bounds). shape_generic=false
    // for ragged==1, so batch equality is enforced. ViT-B/16 scenario: KD has
    // B=16 but caller requests B=32.
    GraphSpec graph;
    graph.seqLenQ = 197;
    graph.seqLenKv = 197;
    graph.batch = 32;
    graph.numQueryHeads = 12;
    graph.numKvHeads = 12;
    graph.headSize = 64;
    graph.headSizeV = 64;
    graph.leftBound = std::nullopt; // noncausal
    graph.rightBound = std::nullopt;

    KernelSpec kernel;
    kernel.ragged = 1;
    kernel.seqLenQ = 197;
    kernel.seqLenKv = 197;
    kernel.batch = 16;
    kernel.numQueryHeads = 12;
    kernel.numKvHeads = 12;
    kernel.headSize = 64;
    kernel.causal = 0;

    EXPECT_FALSE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RaggedKernelRefusesWrongSeqLen)
{
    // A ragged KD baked for S=197 must not serve S=394 (different tail length,
    // different on-chip boundary-padding bounds).
    GraphSpec graph;
    graph.seqLenQ = 394;
    graph.seqLenKv = 394;
    graph.batch = 16;
    graph.numQueryHeads = 12;
    graph.numKvHeads = 12;
    graph.headSize = 64;
    graph.headSizeV = 64;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;

    KernelSpec kernel;
    kernel.ragged = 1;
    kernel.seqLenQ = 197;
    kernel.seqLenKv = 197;
    kernel.batch = 16;
    kernel.numQueryHeads = 12;
    kernel.numKvHeads = 12;
    kernel.headSize = 64;
    kernel.causal = 0;

    EXPECT_FALSE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RaggedKernelAcceptsExactShape)
{
    // Positive control: the exact baked shape matches.
    GraphSpec graph;
    graph.seqLenQ = 197;
    graph.seqLenKv = 197;
    graph.batch = 16;
    graph.numQueryHeads = 12;
    graph.numKvHeads = 12;
    graph.headSize = 64;
    graph.headSizeV = 64;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;

    KernelSpec kernel;
    kernel.ragged = 1;
    kernel.seqLenQ = 197;
    kernel.seqLenKv = 197;
    kernel.batch = 16;
    kernel.numQueryHeads = 12;
    kernel.numKvHeads = 12;
    kernel.headSize = 64;
    kernel.causal = 0;

    EXPECT_TRUE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, AlignedKernelAcceptsDifferentBatch)
{
    // Aligned (ragged=0) KDs are shape-generic: a KD compiled with B=1 must
    // serve a graph with B=4. shape_generic=true, so batch equality is skipped.
    GraphSpec graph;
    graph.batch = 4;

    KernelSpec kernel;
    kernel.ragged = 0;
    kernel.batch = 1; // canonical build input, not a runtime constraint

    EXPECT_TRUE(matchesKernel(graph, kernel));
}

TEST(TestGfx950AttentionDenseKernelMatch, RefusesWindowedCandidateForPlainGraph)
{
    // The KV-loop bound is baked at compile time; a windowed variant must not serve
    // a plain graph.
    KernelSpec kernel;
    kernel.slidingWindow = 128;
    EXPECT_FALSE(matchesKernel(GraphSpec{}, kernel));
}

// ---------------------------------------------------------------------------
// score
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseScore, ScoresACandidateDeterministicallyAsAPositiveFiniteWeight)
{
    // The three properties the selector relies on, each true of a neutral placeholder and
    // of a real tuning model alike:
    //
    //   DETERMINISM -- one candidate scored twice under the same context yields the same
    //   number. A scorer that drifts between calls makes plan selection unreproducible,
    //   and the same graph would pick different binaries on successive builds.
    //
    //   POSITIVITY -- the number is usable as a ranking weight rather than read as a
    //   refusal. Applicability is kernel_match's job; score ranks what already matched.
    //
    //   FINITENESS -- NaN compares false against everything, so a single NaN score turns
    //   the ranking into whatever order the sort happened to visit the candidates in.
    //
    // Deliberately NOT asserted: any ordering BETWEEN candidates. The scorer claims no
    // ranking over them, so pinning one would pin an accident of the implementation.
    const KernelSpec bf16;
    KernelSpec fp16;
    fp16.dtype = "FP16";

    for(const auto& candidate : {bf16, fp16})
    {
        const double score = scoreOf(candidate);
        EXPECT_EQ(score, scoreOf(candidate))
            << "candidate '" << candidate.dtype << "' scored differently on a repeat call";
        EXPECT_GT(score, 0.0) << "candidate '" << candidate.dtype << "' scored non-positive";
        EXPECT_TRUE(std::isfinite(score))
            << "candidate '" << candidate.dtype << "' scored a non-finite value";
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
