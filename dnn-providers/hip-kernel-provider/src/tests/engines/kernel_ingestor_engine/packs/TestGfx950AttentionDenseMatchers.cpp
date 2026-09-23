// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Catalog.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelHeuristic.hpp>
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
 *   - TILE: every candidate carries its own completed (block_m, block_n). Applicability is
 *     candidate-relative -- one graph admits some tiles of a cohort and not others -- and
 *     a missing, mistyped or unbuildable tile declines before anything divides by it.
 *     Cold ranking puts the 256/64 baseline first, then ascending (block_m, block_n).
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
/// num_query_heads, num_kv_heads, causal, ragged, sliding_window, batch, seqlen_q,
/// seqlen_kv, block_m, block_n -- as a completed record carries them, so a spec that
/// leaves the tile alone is a legacy record completed to 256/64.
///
/// The KMD carries only what VARIES between candidates. A knob that starts varying must
/// be added to the KMD first: without a field of its own, two candidates differing only
/// in that knob complete to the same catalog key, and the loader keeps one of them and
/// drops the other.
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
    int64_t blockM = 256;
    int64_t blockN = 64;
    /// The descriptor id's final byte. Only the ranking cases vary it.
    unsigned idByte = 0xa1;
};

/// A descriptor id ending in @p lastByte. Distinct bytes give distinct ids, ordered by
/// the byte, so a case can choose which candidate the selector's id tie-break favours.
hipdnn_plugin_sdk::ingestor::DescriptorId idEndingIn(unsigned lastByte)
{
    constexpr const char* HEX = "0123456789abcdef";
    std::string text = "00000000-0000-4000-8000-0000000000";
    text.push_back(HEX[(lastByte >> 4U) & 0xFU]);
    text.push_back(HEX[lastByte & 0xFU]);
    return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
}

hipdnn_plugin_sdk::ingestor::KernelDefinition makeKernel(const KernelSpec& spec)
{
    hipdnn_plugin_sdk::ingestor::KernelDefinition kernel;
    kernel.kernelId = idEndingIn(spec.idByte);
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
        {std::string("block_m"), spec.blockM},
        {std::string("block_n"), spec.blockN},
    };
    return kernel;
}

/// Runs graph_match then kernel_match for @p kernel, exactly as a catalog build does.
bool matchesKernelDefinition(const GraphSpec& graphSpec,
                             const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel)
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
    return kernelMatcher(context, *bound, kernel);
}

bool matchesKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    return matchesKernelDefinition(graphSpec, makeKernel(kernelSpec));
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

/// A (block_m, block_n) pair.
using Tile = std::pair<int64_t, int64_t>;
using TileSet = std::set<Tile>;

/// The tiles the catalog authors per aligned cohort at each head size.
const std::vector<Tile>& d64Tiles()
{
    static const std::vector<Tile> s_tiles{
        {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}, {256, 256}};
    return s_tiles;
}

const std::vector<Tile>& d128Tiles()
{
    static const std::vector<Tile> s_tiles{
        {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}};
    return s_tiles;
}

TileSet tileSetOf(const std::vector<Tile>& tiles)
{
    return TileSet(tiles.begin(), tiles.end());
}

KernelSpec withTile(KernelSpec spec, int64_t blockM, int64_t blockN)
{
    spec.blockM = blockM;
    spec.blockN = blockN;
    return spec;
}

/// An aligned record's canonical build inputs: B1, Sq=Skv=512. Not runtime constraints.
KernelSpec canonicalAligned()
{
    KernelSpec spec;
    spec.batch = 1;
    spec.seqLenQ = 512;
    spec.seqLenKv = 512;
    spec.ragged = 0;
    return spec;
}

/// One candidate per tile, sharing @p semantic's other fields, with distinct ids.
std::vector<KernelSpec> cohortOf(const KernelSpec& semantic, const std::vector<Tile>& tiles)
{
    std::vector<KernelSpec> cohort;
    cohort.reserve(tiles.size());
    unsigned idByte = 0x10;
    for(const auto& [blockM, blockN] : tiles)
    {
        auto candidate = withTile(semantic, blockM, blockN);
        candidate.idByte = idByte++;
        cohort.push_back(candidate);
    }
    return cohort;
}

/// BF16/D128/H9/9 noncausal: the cross-attention cohort the long-KV witnesses use.
KernelSpec canonicalD128H9()
{
    KernelSpec spec = canonicalAligned();
    spec.headSize = 128;
    spec.numQueryHeads = 9;
    spec.numKvHeads = 9;
    spec.causal = 0;
    return spec;
}

std::vector<KernelSpec> d128H9Cohort()
{
    return cohortOf(canonicalD128H9(), d128Tiles());
}

GraphSpec d128H9Noncausal(int64_t batch, int64_t seqLenQ, int64_t seqLenKv)
{
    GraphSpec graph;
    graph.batch = batch;
    graph.numQueryHeads = 9;
    graph.numKvHeads = 9;
    graph.seqLenQ = seqLenQ;
    graph.seqLenKv = seqLenKv;
    graph.headSize = 128;
    graph.headSizeV = 128;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;
    return graph;
}

/// BF16/D64/H32/32 noncausal: the cohort that carries the D64-only 256/256 tile.
std::vector<KernelSpec> d64H32Cohort()
{
    KernelSpec spec = canonicalAligned();
    spec.headSize = 64;
    spec.numQueryHeads = 32;
    spec.numKvHeads = 32;
    spec.causal = 0;
    return cohortOf(spec, d64Tiles());
}

GraphSpec d64H32Noncausal(int64_t seqLenQ, int64_t seqLenKv)
{
    GraphSpec graph;
    graph.batch = 1;
    graph.numQueryHeads = 32;
    graph.numKvHeads = 32;
    graph.seqLenQ = seqLenQ;
    graph.seqLenKv = seqLenKv;
    graph.headSize = 64;
    graph.headSizeV = 64;
    graph.leftBound = std::nullopt;
    graph.rightBound = std::nullopt;
    return graph;
}

/// BF16/D64/H64/8 top-left causal -- the semantic cohort of the shipped B1/S2016 tail.
GraphSpec tailGraph(int64_t batch, int64_t seqLenQ, int64_t seqLenKv)
{
    GraphSpec graph;
    graph.batch = batch;
    graph.numQueryHeads = 64;
    graph.numKvHeads = 8;
    graph.seqLenQ = seqLenQ;
    graph.seqLenKv = seqLenKv;
    graph.headSize = 64;
    graph.headSizeV = 64;
    return graph;
}

KernelSpec tail2016()
{
    KernelSpec spec;
    spec.headSize = 64;
    spec.numQueryHeads = 64;
    spec.numKvHeads = 8;
    spec.causal = 1;
    spec.ragged = 1;
    spec.batch = 1;
    spec.seqLenQ = 2016;
    spec.seqLenKv = 2016;
    spec.idByte = 0xe0;
    return spec;
}

/// The aligned candidates sharing the tail's semantic fields.
std::vector<KernelSpec> tailCohort()
{
    KernelSpec spec = canonicalAligned();
    spec.headSize = 64;
    spec.numQueryHeads = 64;
    spec.numKvHeads = 8;
    spec.causal = 1;
    return cohortOf(spec, d64Tiles());
}

/// Runs graph_match, then kernel_match over @p candidates, and returns the survivors'
/// tiles -- ranked through the engine's score symbol and the SDK's NativeKernelHeuristic
/// when @p rank, in authoring order otherwise.
std::vector<Tile> matchCandidates(const GraphSpec& graphSpec,
                                  const std::vector<KernelSpec>& candidates,
                                  bool rank)
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
        return {};
    }

    hipdnn_plugin_sdk::ingestor::Catalog catalog;
    catalog.bound = *bound;
    for(const auto& spec : candidates)
    {
        auto kernel = makeKernel(spec);
        if(kernelMatcher(context, *bound, kernel))
        {
            catalog.entries.push_back(std::move(kernel));
        }
    }

    if(rank)
    {
        const hipdnn_plugin_sdk::ingestor::NativeKernelHeuristic heuristic{
            std::string(SCORE_SYMBOL)};
        catalog.entries = heuristic.rank(catalog, context);
    }

    std::vector<Tile> tiles;
    tiles.reserve(catalog.entries.size());
    for(const auto& entry : catalog.entries)
    {
        tiles.emplace_back(entry.getIntMetadata("block_m"), entry.getIntMetadata("block_n"));
    }
    return tiles;
}

TileSet admittedTiles(const GraphSpec& graph, const std::vector<KernelSpec>& candidates)
{
    return tileSetOf(matchCandidates(graph, candidates, /*rank=*/false));
}

/// The order a cold plan build tries the admitted candidates in.
std::vector<Tile> coldOrder(const GraphSpec& graph, const std::vector<KernelSpec>& candidates)
{
    return matchCandidates(graph, candidates, /*rank=*/true);
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
    // The candidate's own block_n must divide Skv. 288 is not a multiple of the baseline's
    // 64, so the baseline declines; the BN32 neighbour that does serve it is pinned below.
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
// Candidate-relative tiles. Each cohort is authored as the catalog authors it --
// canonical B1/Sq512/Skv512 build inputs, one candidate per tile -- and every graph
// below differs from those inputs, so an admitted candidate is runtime-shape reuse.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseTileMatch, D128CrossAttentionQ384AdmitsExactlyTheBm128Tiles)
{
    // 384 is a multiple of 128 and not of 256; 512 is a multiple of every block_n.
    EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 384, 512), d128H9Cohort()),
              (TileSet{{128, 32}, {128, 64}, {128, 128}}));
}

TEST(TestGfx950AttentionDenseTileMatch, D128CrossAttentionKv192AdmitsBothBlockMsAtBn32And64)
{
    // 192 is a multiple of 32 and 64 but not of 128.
    EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 1024, 192), d128H9Cohort()),
              (TileSet{{128, 32}, {128, 64}, {256, 32}, {256, 64}}));
}

TEST(TestGfx950AttentionDenseTileMatch, D128CrossAttentionKv288And416AdmitOnlyBn32)
{
    // Both are odd multiples of 32: the baseline declines, and only the BN32 tiles serve.
    for(const int64_t seqLenKv : {int64_t{288}, int64_t{416}})
    {
        EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 1024, seqLenKv), d128H9Cohort()),
                  (TileSet{{128, 32}, {256, 32}}))
            << "seqlen_kv " << seqLenKv;
    }
}

TEST(TestGfx950AttentionDenseTileMatch, D128LongKvAdmitsEveryAuthoredTile)
{
    // 62208 = 486 * 128, so every block_n divides it.
    EXPECT_EQ(admittedTiles(d128H9Noncausal(1, 1024, 62208), d128H9Cohort()),
              tileSetOf(d128Tiles()));
}

TEST(TestGfx950AttentionDenseTileMatch, D128GqaCausalRuntimeBatchAdmitsAllSixTiles)
{
    // BF16/D128/H32/8 top-left causal at B3, Sq=Skv=1536: 1536 = 6 * 256 = 12 * 128.
    GraphSpec graph;
    graph.batch = 3;
    graph.numQueryHeads = 32;
    graph.numKvHeads = 8;
    graph.seqLenQ = 1536;
    graph.seqLenKv = 1536;

    KernelSpec semantic = canonicalAligned();
    semantic.numQueryHeads = 32;
    semantic.numKvHeads = 8;
    semantic.causal = 1;

    EXPECT_EQ(admittedTiles(graph, cohortOf(semantic, d128Tiles())), tileSetOf(d128Tiles()));
}

TEST(TestGfx950AttentionDenseTileMatch, D64SelfAttentionAdmitsAllSevenTilesIncludingBn256)
{
    EXPECT_EQ(admittedTiles(d64H32Noncausal(1024, 1024), d64H32Cohort()), tileSetOf(d64Tiles()));
}

TEST(TestGfx950AttentionDenseTileMatch, D64Kv128ExcludesOnlyTheBn256Tile)
{
    EXPECT_EQ(admittedTiles(d64H32Noncausal(1024, 128), d64H32Cohort()),
              (TileSet{{128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 64}, {256, 128}}));
}

TEST(TestGfx950AttentionDenseTileMatch, RefusesTilesGfx950DoesNotBuildEvenWhereTheyWouldDivide)
{
    // Every length below is a multiple of every block_n, so only the tile rules decide.
    // 128/256 fails block_m % block_n at both head sizes; D128 256/256 passes the Python
    // rules and fails the LDS budget. D64 256/256 is the positive neighbour.
    const GraphSpec d64 = d64H32Noncausal(1024, 1024);
    const GraphSpec d128 = d128H9Noncausal(1, 1024, 1024);

    EXPECT_FALSE(matchesKernel(d64, withTile(d64H32Cohort().front(), 128, 256)));
    EXPECT_FALSE(matchesKernel(d128, withTile(d128H9Cohort().front(), 128, 256)));
    EXPECT_FALSE(matchesKernel(d128, withTile(d128H9Cohort().front(), 256, 256)));
    EXPECT_TRUE(matchesKernel(d64, withTile(d64H32Cohort().front(), 256, 256)));
}

TEST(TestGfx950AttentionDenseTileMatch, BothLengthsAreCheckedAgainstTheCandidateTile)
{
    // One graph per failing length, each beside a tile that the same graph admits.
    // Sq 128 x Skv 512: block_m 256 declines, 128 admits.
    const auto q128 = d128H9Noncausal(1, 128, 512);
    EXPECT_FALSE(matchesKernel(q128, withTile(canonicalD128H9(), 256, 64)));
    EXPECT_TRUE(matchesKernel(q128, withTile(canonicalD128H9(), 128, 64)));
    // Sq 512 x Skv 96: block_n 64 declines, 32 admits.
    const auto kv96 = d128H9Noncausal(1, 512, 96);
    EXPECT_FALSE(matchesKernel(kv96, withTile(canonicalD128H9(), 256, 64)));
    EXPECT_TRUE(matchesKernel(kv96, withTile(canonicalD128H9(), 256, 32)));
}

TEST(TestGfx950AttentionDenseTileMatch, AlternativeTilesKeepTheMaskRules)
{
    // TOP_LEFT_CAUSAL at unequal lengths is served; BOTTOM_RIGHT_CAUSAL at unequal
    // lengths is declined by graph_match for every tile alike, and served at equal ones.
    GraphSpec topLeft;
    topLeft.seqLenQ = 384;
    topLeft.seqLenKv = 512;
    EXPECT_TRUE(matchesKernel(topLeft, withTile(KernelSpec{}, 128, 32)));

    GraphSpec bottomRightUnequal = topLeft;
    bottomRightUnequal.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    EXPECT_FALSE(matchGraph(bottomRightUnequal).has_value());

    GraphSpec bottomRightEqual = bottomRightUnequal;
    bottomRightEqual.seqLenKv = 384;
    EXPECT_TRUE(matchesKernel(bottomRightEqual, withTile(KernelSpec{}, 128, 128)));
}

TEST(TestGfx950AttentionDenseTileMatch, AlternativeTilesKeepTheSemanticFieldComparisons)
{
    // A BM128 candidate differing in one semantic field from a graph it would otherwise
    // serve. The unperturbed candidate is the positive neighbour.
    const GraphSpec graph = d128H9Noncausal(1, 384, 512);
    const KernelSpec match = withTile(canonicalD128H9(), 128, 32);
    EXPECT_TRUE(matchesKernel(graph, match));

    KernelSpec otherDtype = match;
    otherDtype.dtype = "FP16";
    EXPECT_FALSE(matchesKernel(graph, otherDtype));

    KernelSpec otherHeads = match;
    otherHeads.numKvHeads = 3;
    EXPECT_FALSE(matchesKernel(graph, otherHeads));

    KernelSpec otherMask = match;
    otherMask.causal = 1;
    EXPECT_FALSE(matchesKernel(graph, otherMask));

    // FP16 graph against the FP16 candidate, and D64 against D64: both dtypes and both
    // head sizes are served at the alternative tile.
    GraphSpec fp16Graph = graph;
    fp16Graph.dataType = data_objects::DataType::HALF;
    EXPECT_TRUE(matchesKernel(fp16Graph, otherDtype));

    GraphSpec d64Graph = graph;
    d64Graph.headSize = 64;
    d64Graph.headSizeV = 64;
    KernelSpec d64Candidate = match;
    d64Candidate.headSize = 64;
    EXPECT_TRUE(matchesKernel(d64Graph, d64Candidate));
}

// ---------------------------------------------------------------------------
// Malformed tile metadata. Each is declined outright -- no default substituted, and no
// division by the value: a zero block_n that reached `Skv % block_n` would fault the
// process rather than fail the expectation.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseTileMatch, AcceptsARecordCompletedToTheBaselineTile)
{
    // A legacy record's raw form omits the tile; completion supplies 256/64 and the
    // candidate is served. This is the positive neighbour of every case below.
    EXPECT_TRUE(matchesKernel(GraphSpec{}, KernelSpec{}));
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesARecordWhoseTileWasNeverCompleted)
{
    for(const char* field : {"block_m", "block_n"})
    {
        auto kernel = makeKernel(KernelSpec{});
        kernel.metadata.erase(field);
        EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, kernel)) << "missing " << field;
    }
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesZeroNegativeAndUnbuiltTileValues)
{
    // Sq = Skv = 256 is a multiple of every legal tile, so only validation can decline.
    const std::vector<std::pair<int64_t, int64_t>> malformed{
        {0, 64}, {256, 0}, {0, 0}, {-256, 64}, {256, -64}, {64, 64}, {512, 64}, {256, 48}};
    for(const auto& [blockM, blockN] : malformed)
    {
        EXPECT_FALSE(matchesKernel(GraphSpec{}, withTile(KernelSpec{}, blockM, blockN)))
            << blockM << "/" << blockN;
    }
}

TEST(TestGfx950AttentionDenseTileMatch, DeclinesATileFieldOfTheWrongType)
{
    // Each value names 256 or 64 in some other type; none may be read as the integer.
    using hipdnn_plugin_sdk::ingestor::MetadataValue;
    const std::vector<MetadataValue> blockMSpellings{MetadataValue{std::string("256")},
                                                     MetadataValue{256.0},
                                                     MetadataValue{true},
                                                     MetadataValue{std::vector<int64_t>{256}}};
    for(const auto& value : blockMSpellings)
    {
        auto kernel = makeKernel(KernelSpec{});
        kernel.metadata[std::string("block_m")] = value;
        EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, kernel))
            << "block_m held alternative " << value.index();
    }

    auto kernel = makeKernel(KernelSpec{});
    kernel.metadata[std::string("block_n")] = MetadataValue{std::string("64")};
    EXPECT_FALSE(matchesKernelDefinition(GraphSpec{}, kernel));
}

// ---------------------------------------------------------------------------
// Exact tails beside aligned alternatives. A real shipped tail --
// BF16/D64/H64/8 causal, B1, Sq=Skv=2016 -- with its aligned cohort alongside.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseTileMatch, TailServesOnlyItsExactAuthoredShape)
{
    // 2016 is a multiple of 32 but of no block_m, so no aligned candidate applies and
    // the tail is the only one served.
    EXPECT_TRUE(matchesKernel(tailGraph(1, 2016, 2016), tail2016()));
    EXPECT_EQ(admittedTiles(tailGraph(1, 2016, 2016), tailCohort()), TileSet{});
}

TEST(TestGfx950AttentionDenseTileMatch, TailRefusesEachIndependentShapeChange)
{
    // Batch, seqlen_q and seqlen_kv changed one at a time, then both lengths together.
    // Each drops the tail; the aligned candidates each change admits are exactly those
    // whose own tile divides the new lengths, so the tail's refusal is not global.
    struct Case
    {
        const char* what;
        GraphSpec graph;
        TileSet aligned;
    };
    const std::vector<Case> cases{
        {"batch 2", tailGraph(2, 2016, 2016), TileSet{}},
        // TOP_LEFT_CAUSAL with Sq != Skv is served: 2048 is a multiple of both block_m,
        // 2016 of block_n 32 only.
        {"seqlen_q 2048", tailGraph(1, 2048, 2016), TileSet{{128, 32}, {256, 32}}},
        {"seqlen_kv 2048", tailGraph(1, 2016, 2048), TileSet{}},
        {"both 2048", tailGraph(1, 2048, 2048), tileSetOf(d64Tiles())},
    };
    for(const auto& c : cases)
    {
        SCOPED_TRACE(c.what);
        EXPECT_FALSE(matchesKernel(c.graph, tail2016()));
        EXPECT_EQ(admittedTiles(c.graph, tailCohort()), c.aligned);
    }
}

TEST(TestGfx950AttentionDenseTileMatch, TailRefusesAShapeItsOwnTileDivides)
{
    // A tail record at a length its own 256/64 tile divides is not a tail's shape: the
    // aligned binary serves it, and the tail declines even at exact metadata equality.
    KernelSpec tail = tail2016();
    tail.seqLenQ = 2048;
    tail.seqLenKv = 2048;
    EXPECT_FALSE(matchesKernel(tailGraph(1, 2048, 2048), tail));
}

TEST(TestGfx950AttentionDenseTileMatch, TailWithAMalformedTileIsDeclined)
{
    // The tail's exact shape match does not excuse its tile: the geometry divides by it.
    auto kernel = makeKernel(tail2016());
    kernel.metadata.erase("block_m");
    EXPECT_FALSE(matchesKernelDefinition(tailGraph(1, 2016, 2016), kernel));
}

// ---------------------------------------------------------------------------
// Cold ranking, through the engine's own score symbol and the SDK heuristic that
// consumes it: the order a cold (unbenchmarked) plan build tries candidates in.
// ---------------------------------------------------------------------------

TEST(TestGfx950AttentionDenseScore, BaselineFirstThenAscendingTilesWhateverTheIdOrder)
{
    // D64 S1024 admits all seven tiles. Ids are assigned twice -- ascending with the
    // expected order, then descending -- so an order the id tie-break decided would
    // differ between the two runs.
    const std::vector<Tile> expected{
        {256, 64}, {128, 32}, {128, 64}, {128, 128}, {256, 32}, {256, 128}, {256, 256}};
    const GraphSpec graph = d64H32Noncausal(1024, 1024);

    for(const bool descendingIds : {false, true})
    {
        SCOPED_TRACE(descendingIds ? "descending ids" : "ascending ids");
        auto cohort = d64H32Cohort();
        for(std::size_t i = 0; i < cohort.size(); ++i)
        {
            const auto& tile = expected.at(i);
            auto& candidate = cohort.at(i);
            candidate.blockM = tile.first;
            candidate.blockN = tile.second;
            candidate.idByte = static_cast<unsigned>(descendingIds ? std::size_t{0x70} - i
                                                                   : std::size_t{0x10} + i);
        }
        EXPECT_EQ(coldOrder(graph, cohort), expected);
    }
}

TEST(TestGfx950AttentionDenseScore, BestApplicableAlternativeLeadsWhenTheBaselineCannotServe)
{
    // Sq 384 rules out every block_m 256 tile, the baseline included. Ids descend against
    // the authoring order, so the selector's ascending-id tie-break alone would put
    // 128/128 first.
    auto cohort = d128H9Cohort();
    for(std::size_t i = 0; i < cohort.size(); ++i)
    {
        cohort.at(i).idByte = static_cast<unsigned>(std::size_t{0x80} - i);
    }
    EXPECT_EQ(coldOrder(d128H9Noncausal(1, 384, 512), cohort),
              (std::vector<Tile>{{128, 32}, {128, 64}, {128, 128}}));
}

TEST(TestGfx950AttentionDenseScore, ScoresEveryTileDeterministicallyAsAPositiveFiniteWeight)
{
    // The properties the selector relies on beyond the order itself:
    //
    //   DETERMINISM -- one candidate scored twice under the same context yields the same
    //   number. A scorer that drifts between calls makes plan selection unreproducible.
    //
    //   POSITIVITY -- the number is usable as a ranking weight rather than read as a
    //   refusal. Applicability is kernel_match's job; score ranks what already matched.
    //
    //   FINITENESS -- NaN compares false against everything, so a single NaN score turns
    //   the ranking into whatever order the sort happened to visit the candidates in.
    for(const auto& candidate : d64H32Cohort())
    {
        SCOPED_TRACE(std::to_string(candidate.blockM) + "/" + std::to_string(candidate.blockN));
        const double score = scoreOf(candidate);
        EXPECT_EQ(score, scoreOf(candidate));
        EXPECT_GT(score, 0.0);
        EXPECT_TRUE(std::isfinite(score));
    }
}

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
