// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"

/**
 * @file Gfx950AttentionUnified3DTestGraphs.hpp
 * @brief Graphs and candidate records the hipkernel:Gfx950AttentionUnified3D suites share.
 *
 * The symbol and token spellings are written out here rather than shared with the pack:
 * they are the contract between the descriptors, the matcher that writes a token and the
 * dispatch that reads it, and a test sharing the constant would keep passing through a rename.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified3d
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::KernelSourceKind;
using hipdnn_plugin_sdk::ingestor::MatchContext;

inline constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_3d.graph_match";
inline constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_3d.kernel_match";
inline constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_unified_3d.score";
inline constexpr std::string_view DISPATCH_SYMBOL
    = "hipkernel.gfx950_attention_unified_3d.dispatch";

inline constexpr std::string_view Q_TOKEN = "gfx950_attention_unified_3d.q.uid";
inline constexpr std::string_view K_TOKEN = "gfx950_attention_unified_3d.k.uid";
inline constexpr std::string_view V_TOKEN = "gfx950_attention_unified_3d.v.uid";
inline constexpr std::string_view O_TOKEN = "gfx950_attention_unified_3d.o.uid";
inline constexpr std::string_view MASK_TOKEN = "gfx950_attention_unified_3d.mask_type";
inline constexpr std::string_view SLIDING_WINDOW_TOKEN
    = "gfx950_attention_unified_3d.sliding_window";
inline constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_unified_3d.scale_bits";

/// Mutually distinct, none a small ordinal: a uid that equalled an axis index could read as
/// correct after a transposition.
inline constexpr int64_t Q_UID = 11;
inline constexpr int64_t K_UID = 22;
inline constexpr int64_t V_UID = 33;
inline constexpr int64_t O_UID = 44;
inline constexpr int64_t EXTRA_UID = 99;
/// The RFC-0014 offsets tensor a ragged operand names.
inline constexpr int64_t RAGGED_OFFSET_UID = 55;

/// The mask kinds in the catalog's `mask_type` numbering.
inline constexpr int64_t NO_MASK = 0;
inline constexpr int64_t TOP_LEFT_CAUSAL = 1;
inline constexpr int64_t BOTTOM_RIGHT_CAUSAL = 2;

inline constexpr float SCALE = 0.08838834764831843F;

/// BSHD strides for (B, H, S, D) logical dims -- token-major, head varying fastest.
inline std::vector<int64_t> bshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {sequence * heads * headSize, headSize, heads * headSize, 1};
}

/// BHSD strides for the same dims: head-major.
inline std::vector<int64_t> bhsdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    return {heads * sequence * headSize, sequence * headSize, headSize, 1};
}

/// BSHD axis order with the token row padded past `heads * headSize`: still token-major,
/// but not gap-free, so it is not a paged cache's layout.
inline std::vector<int64_t> paddedBshdStrides(int64_t heads, int64_t sequence, int64_t headSize)
{
    constexpr int64_t ROW_PAD = 8;
    const int64_t rowStride = heads * headSize + ROW_PAD;
    return {sequence * rowStride, headSize, rowStride, 1};
}

enum class Layout
{
    BSHD,
    BHSD,
    PADDED_BSHD
};

inline std::vector<int64_t>
    stridesFor(Layout layout, int64_t heads, int64_t sequence, int64_t headSize)
{
    switch(layout)
    {
    case Layout::BHSD:
        return bhsdStrides(heads, sequence, headSize);
    case Layout::PADDED_BSHD:
        return paddedBshdStrides(heads, sequence, headSize);
    case Layout::BSHD:
    default:
        return bshdStrides(heads, sequence, headSize);
    }
}

/// An SDPA graph, defaulting to one the engine serves: bf16, D128, B2, Hq 8 over Hkv 2,
/// one query token over Skv 256 (decode), top-left causal. Fields perturb one rule at a time.
struct GraphSpec
{
    int64_t batch = 2;
    int64_t numQueryHeads = 8;
    int64_t numKvHeads = 2;
    int64_t seqLenQ = 1;
    int64_t seqLenKv = 256;
    int64_t headSize = 128;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    std::optional<data_objects::DataType> kDataType;

    Layout qLayout = Layout::BSHD;
    Layout kLayout = Layout::BSHD;
    Layout vLayout = Layout::BSHD;
    Layout oLayout = Layout::BSHD;

    /// Dims of V and O, when they should disagree with the shared values.
    std::optional<int64_t> vSeqLen;
    std::optional<int64_t> oHeads;

    std::optional<int64_t> leftBound = -1;
    std::optional<int64_t> rightBound = 0;
    data_objects::DiagonalAlignment alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    bool causalMaskDeprecated = false;
    bool causalMaskBottomRightDeprecated = false;
    std::optional<float> attnScaleValue = SCALE;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation implementation
        = data_objects::AttentionImplementation::AUTO;

    /// Sets any further attribute on the SDPA node (an optional tensor uid, a probability).
    std::function<void(data_objects::SdpaAttributesBuilder&)> extra;

    /// Operand uid (Q/K/V/O) that carries a ragged offsets tensor, if any.
    std::optional<int64_t> raggedOperand;

    bool twoNodes = false;
    /// One RELU node from Q to O in place of the SDPA node.
    bool pointwiseOnly = false;
};

/// A mask spelled the way hipDNN's frontend spells it.
inline void setMask(GraphSpec& spec, int64_t maskType)
{
    spec.leftBound = std::nullopt;
    spec.causalMaskDeprecated = false;
    spec.causalMaskBottomRightDeprecated = false;
    if(maskType == NO_MASK)
    {
        spec.rightBound = std::nullopt;
        spec.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    }
    else if(maskType == TOP_LEFT_CAUSAL)
    {
        spec.rightBound = 0;
        spec.alignment = data_objects::DiagonalAlignment::TOP_LEFT;
    }
    else
    {
        spec.rightBound = 0;
        spec.alignment = data_objects::DiagonalAlignment::BOTTOM_RIGHT;
    }
}

inline flatbuffers::FlatBufferBuilder buildSdpaGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> qDims{spec.batch, spec.numQueryHeads, spec.seqLenQ, spec.headSize};
    const std::vector<int64_t> kDims{spec.batch, spec.numKvHeads, spec.seqLenKv, spec.headSize};
    const std::vector<int64_t> vDims{
        spec.batch, spec.numKvHeads, spec.vSeqLen.value_or(spec.seqLenKv), spec.headSize};
    const int64_t oHeads = spec.oHeads.value_or(spec.numQueryHeads);
    const std::vector<int64_t> oDims{spec.batch, oHeads, spec.seqLenQ, spec.headSize};

    const auto qStrides = stridesFor(spec.qLayout, spec.numQueryHeads, spec.seqLenQ, spec.headSize);
    const auto kStrides = stridesFor(spec.kLayout, spec.numKvHeads, spec.seqLenKv, spec.headSize);
    const auto vStrides = stridesFor(
        spec.vLayout, spec.numKvHeads, spec.vSeqLen.value_or(spec.seqLenKv), spec.headSize);
    const auto oStrides = stridesFor(spec.oLayout, oHeads, spec.seqLenQ, spec.headSize);

    const auto tensorFor = [&](int64_t uid,
                               data_objects::DataType dataType,
                               const std::vector<int64_t>& strides,
                               const std::vector<int64_t>& dims) {
        const ::flatbuffers::Optional<int64_t> ragged
            = spec.raggedOperand == uid ? ::flatbuffers::Optional<int64_t>(RAGGED_OFFSET_UID)
                                        : ::flatbuffers::nullopt;
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          nullptr,
                                                          dataType,
                                                          &strides,
                                                          &dims,
                                                          false,
                                                          data_objects::TensorValue::NONE,
                                                          0,
                                                          false,
                                                          ragged);
    };

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(tensorFor(Q_UID, spec.dataType, qStrides, qDims));
    tensors.push_back(tensorFor(K_UID, spec.kDataType.value_or(spec.dataType), kStrides, kDims));
    tensors.push_back(tensorFor(V_UID, spec.dataType, vStrides, vDims));
    tensors.push_back(tensorFor(O_UID, spec.dataType, oStrides, oDims));
    if(spec.raggedOperand.has_value())
    {
        const std::vector<int64_t> offsetDims{spec.batch + 1, 1, 1, 1};
        const std::vector<int64_t> offsetStrides{1, 1, 1, 1};
        tensors.push_back(
            tensorFor(RAGGED_OFFSET_UID, data_objects::DataType::INT32, offsetStrides, offsetDims));
    }

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
        attributesBuilder.add_alibi_mask(spec.alibiMask);
        attributesBuilder.add_padding_mask(spec.paddingMask);
        attributesBuilder.add_mma_core_mode(spec.mmaCoreMode);
        attributesBuilder.add_implementation(spec.implementation);
        if(spec.extra)
        {
            spec.extra(attributesBuilder);
        }
        return attributesBuilder.Finish();
    };

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    if(spec.pointwiseOnly)
    {
        const auto relu
            = data_objects::CreatePointwiseAttributes(builder,
                                                      data_objects::PointwiseMode::RELU_FWD,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      Q_UID,
                                                      flatbuffers::nullopt,
                                                      flatbuffers::nullopt,
                                                      O_UID);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "relu",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           relu.Union()));
    }
    else
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa2",
                                                       data_objects::DataType::FLOAT,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }

    auto name = builder.CreateString("gfx950_attention_unified_3d_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// Keeps a built graph alive for as long as a case reads it, and hands out the MatchContext an
/// engine sees. The context holds references into this object, so it must outlive every use.
class Graph
{
public:
    explicit Graph(const GraphSpec& spec = GraphSpec{})
        : _builder(buildSdpaGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
        _properties.gcnArchName = "gfx950";
        _properties.warpSize = 64;
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

private:
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

/// Runs the pack's graph match over @p graph.
inline std::optional<BoundTokens> matchGraph(const Graph& graph)
{
    registerNativeIngestorSymbols();
    const auto matcher = hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
        std::string(GRAPH_MATCHER_SYMBOL));
    return matcher(graph.context());
}

inline std::optional<BoundTokens> matchGraph(const GraphSpec& spec)
{
    const Graph graph(spec);
    return matchGraph(graph);
}

/// The fields the engine's KMD declares, as a completed catalog record carries them. The
/// defaults are the shipped kernel for the default graph (bf16, D128, Hq 8 / Hkv 2, top-left
/// causal, block 16, 16 segments, decode grid), launching its segment and reduce kernels.
struct KernelSpec
{
    std::string dtype = "BF16";
    int64_t headSize = 128;
    int64_t numQueryHeads = 8;
    int64_t numKvHeads = 2;
    int64_t blockSize = 16;
    int64_t maskType = TOP_LEFT_CAUSAL;
    int64_t slidingWindow = 0;
    int64_t numSegments = 16;
    int64_t useDecodeGrid = 1;
    /// The launch names the record's `sources` carries; the engine needs exactly `segment`
    /// and `reduce`. Empty is a single-launch record.
    std::vector<std::string> launches = {"segment", "reduce"};
    /// The descriptor id's final byte. Only the ranking cases vary it.
    unsigned idByte = 0xa1;
};

/// A descriptor id ending in @p lastByte, ordered by the byte.
inline hipdnn_plugin_sdk::ingestor::DescriptorId idEndingIn(unsigned lastByte)
{
    constexpr const char* HEX = "0123456789abcdef";
    std::string text = "00000000-0000-4000-8000-0000000000";
    text.push_back(HEX[(lastByte >> 4U) & 0xFU]);
    text.push_back(HEX[lastByte & 0xFU]);
    return hipdnn_flatbuffers_sdk::utilities::parseUuid(text);
}

/// The metadata of a completed record, as the descriptor loader hands it over.
inline hipdnn_plugin_sdk::ingestor::MetadataValues metadataOf(const KernelSpec& spec)
{
    return {
        {std::string("dtype"), spec.dtype},
        {std::string("head_size"), spec.headSize},
        {std::string("num_query_heads"), spec.numQueryHeads},
        {std::string("num_kv_heads"), spec.numKvHeads},
        {std::string("block_size"), spec.blockSize},
        {std::string("mask_type"), spec.maskType},
        {std::string("sliding_window"), spec.slidingWindow},
        {std::string("num_segments"), spec.numSegments},
        {std::string("use_decode_grid"), spec.useDecodeGrid},
    };
}

inline KernelDefinition makeKernel(const KernelSpec& spec)
{
    KernelDefinition kernel;
    kernel.kernelId = idEndingIn(spec.idByte);
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-00000000dea3");
    kernel.name = "attention_unified_3d.bf16_d128_hq8_kv2_m1_bs16_seg16.gfx950";
    kernel.metadata = metadataOf(spec);
    for(const auto& launch : spec.launches)
    {
        kernel.sources[launch] = {};
    }
    return kernel;
}

/// Runs graph_match then kernel_match for @p kernel, exactly as a catalog build does.
inline bool matchesKernel(const Graph& graph, const KernelDefinition& kernel)
{
    registerNativeIngestorSymbols();
    const auto kernelMatcher = hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(KERNEL_MATCHER_SYMBOL));
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return kernelMatcher(graph.context(), *bound, kernel);
}

inline bool matchesKernel(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    const Graph graph(graphSpec);
    return matchesKernel(graph, makeKernel(kernelSpec));
}

inline double scoreOf(const GraphSpec& graphSpec, const KernelSpec& kernelSpec)
{
    registerNativeIngestorSymbols();
    const auto scorer
        = hipdnn_plugin_sdk::ingestor::ScoreRegistry::resolve(std::string(SCORE_SYMBOL));
    const Graph graph(graphSpec);
    const auto bound = matchGraph(graph);
    EXPECT_TRUE(bound.has_value());
    return scorer(graph.context(), bound.value_or(BoundTokens{}), makeKernel(kernelSpec));
}

/// The 19-argument segment-kernel ABI the rocKE builder declares, under its own names.
inline std::vector<KernelArgument> pythonSegmentAbi()
{
    constexpr uint32_t POINTER_BYTES = 8;
    constexpr uint32_t SCALAR_BYTES = 4;
    const auto buffer
        = [](const char* name) { return KernelArgument{"global_buffer", POINTER_BYTES, 0, name}; };
    const auto scalar
        = [](const char* name) { return KernelArgument{"by_value", SCALAR_BYTES, 0, name}; };
    return {buffer("segm_output_ptr"),
            buffer("segm_max_ptr"),
            buffer("segm_expsum_ptr"),
            buffer("query_ptr"),
            buffer("key_cache_ptr"),
            buffer("value_cache_ptr"),
            buffer("sink_ptr"),
            buffer("block_tables_ptr"),
            buffer("seq_lens_ptr"),
            buffer("alibi_slopes_ptr"),
            buffer("qq_bias_ptr"),
            buffer("query_start_len_ptr"),
            scalar("scale"),
            scalar("k_scale"),
            scalar("v_scale"),
            scalar("softcap"),
            scalar("num_seqs"),
            scalar("block_table_stride"),
            scalar("qq_bias_stride_0")};
}

/// The 5-pointer reduce-kernel ABI.
inline std::vector<KernelArgument> pythonReduceAbi()
{
    constexpr uint32_t POINTER_BYTES = 8;
    const auto buffer
        = [](const char* name) { return KernelArgument{"global_buffer", POINTER_BYTES, 0, name}; };
    return {buffer("output_ptr"),
            buffer("segm_output_ptr"),
            buffer("segm_max_ptr"),
            buffer("segm_expsum_ptr"),
            buffer("seq_lens_ptr")};
}

/// A KPACK descriptor for this engine naming an archive that is not there, its `segment`
/// launch recording @p segment and its `reduce` launch recording @p reduce (any other
/// launch the spec names records @p segment). That absence is the
/// point: every check prepare() performs BEFORE the loader runs is reached, the signature
/// comparisons run next, and when they pass the loader reports the missing file. No device,
/// no compile, no launch.
inline KernelDefinition makeKpackKernel(const KernelSpec& spec,
                                        const std::vector<KernelArgument>& segment,
                                        const std::vector<KernelArgument>& reduce
                                        = pythonReduceAbi())
{
    auto kernel = makeKernel(spec);
    for(auto& [launch, source] : kernel.sources)
    {
        const bool isReduce = launch == "reduce";
        source.kind = KernelSourceKind::KPACK;
        // One module: both launches resolve out of the same library and toc key.
        source.library = "there-is-no-archive-here.kpack";
        source.tocKey = "toc#0";
        source.symbol = isReduce ? "rocke_uattn_reduce_tiled" : "rocke_uattn3d_tiled";
        source.signature = isReduce ? reduce : segment;
    }
    kernel.originDirectory = "/tmp/hkp-unified-3d-no-such-tree";
    kernel.treeRoot = "/tmp/hkp-unified-3d-no-such-tree";
    return kernel;
}

/// The message prepare() throws for @p bound and @p kernel over @p graph, or empty if it
/// returned a prepared dispatch -- which it cannot, since the archive is absent.
inline std::string
    prepareFailure(const Graph& graph, const BoundTokens& bound, const KernelDefinition& kernel)
{
    registerNativeIngestorSymbols();
    const auto* handler = hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(DISPATCH_SYMBOL));
    EXPECT_NE(handler, nullptr);
    if(handler == nullptr)
    {
        return {};
    }
    try
    {
        handler->prepare(graph.context(), bound, kernel);
    }
    catch(const std::exception& error)
    {
        return error.what();
    }
    return {};
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::unified3d

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
