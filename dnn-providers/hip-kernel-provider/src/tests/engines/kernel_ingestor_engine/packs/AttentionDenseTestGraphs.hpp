// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/KernelIngestorEngine.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/AttentionDenseTestArch.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/SdpaLayoutTestSupport.hpp"

/**
 * @file AttentionDenseTestGraphs.hpp
 * @brief The SDPA graphs, the decline-log reader and the prepare() seam every gfx942 and
 *        gfx950 dense-attention unit suite builds on.
 *
 * The defaults are the gfx950 suite's base graph; each arch's suite and each shared suite
 * starts from them and moves only what it is about, so a graph's defaults live in one
 * place. Symbols, tokens and candidate records stay with the suite that pins them.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::attention_dense_graphs
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;
using hipdnn_plugin_sdk::ingestor::DeviceProperties;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The decline lines name operands by these uids.
inline constexpr int64_t Q_UID = 1;
inline constexpr int64_t K_UID = 2;
inline constexpr int64_t V_UID = 3;
inline constexpr int64_t O_UID = 4;
inline constexpr int64_t STATS_UID = 5;
/// A uid no tensor carries.
inline constexpr int64_t EXTRA_UID = 99;
/// What tensor_attributes.fbs gives a tensor that declares no alignment, and what the dense
/// packs require. Spelled out here so a graph that under-declares is visibly the exception.
inline constexpr int64_t DEFAULT_ALIGNMENT_BYTES = 16;

/// The default shape: bf16, D128, B2, Hq = Hkv = 4, Sq = Skv = 256.
inline constexpr int64_t BATCH = 2;
inline constexpr int64_t HEADS = 4;
inline constexpr int64_t SEQ = 256;
inline constexpr int64_t HEAD_SIZE = 128;
/// 1 / sqrt(128).
inline constexpr float SCALE = 0.08838834764831843F;

/// Packed row-major strides over @p dims, the frontend's default at any rank.
inline std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(std::size_t axis = dims.size(); axis > 1; --axis)
    {
        strides[axis - 2] = strides[axis - 1] * dims[axis - 1];
    }
    return strides;
}

/// One operand's departures from the dense BSHD tensor the shape gives it.
struct OperandSpec
{
    /// Dims written out whole: an extent that disagrees with the shape, or another rank.
    std::optional<std::vector<int64_t>> dims;
    /// Strides written out verbatim: one perturbed axis, or extents whose product would
    /// not fit. Unset, they are `layout` over the operand's own dims, so an operand whose
    /// dims disagree is still dense for its own extents.
    std::optional<std::vector<int64_t>> strides;
    /// Chosen per operand so a graph can mix layouts: the mixed-layout graph is the hazard
    /// the matcher's per-operand clauses exist to catch.
    SdpaLayout layout = SdpaLayout::BSHD;
    std::optional<data_objects::DataType> dataType;
};

/// A tensor beside Q, K, V and O: the stats output, or one the node does not name.
struct TensorSpec
{
    int64_t uid = STATS_UID;
    /// Empty: [B, Hq, Sq, 1], the stats tensor the kernel writes.
    std::vector<int64_t> dims;
    /// Empty: packed row-major over the dims, the frontend's default.
    std::vector<int64_t> strides;
    data_objects::DataType dataType = data_objects::DataType::FLOAT;
    bool isVirtual = false;
};

/// An SDPA-forward graph. The defaults are a top-left causal graph both arches serve; each
/// field perturbs one rule.
struct GraphSpec
{
    int64_t batch = BATCH;
    int64_t numQueryHeads = HEADS;
    int64_t numKvHeads = HEADS;
    int64_t seqLenQ = SEQ;
    int64_t seqLenKv = SEQ;
    int64_t headSize = HEAD_SIZE;
    data_objects::DataType dataType = data_objects::DataType::BFLOAT16;
    /// The SDPA node's compute data type.
    data_objects::DataType computeType = data_objects::DataType::FLOAT;

    /// The uids of Q, K, V and O, in that order.
    std::array<int64_t, 4> uids{Q_UID, K_UID, V_UID, O_UID};
    /// Q, K, V and O, in the order of `uids`.
    std::array<OperandSpec, 4> operands;
    bool omitStrides = false;
    // The one operand whose dims are left out, and the one operand tensor the graph does
    // not carry at all (the node still names its uid).
    std::optional<int64_t> omitDimsUid;
    std::optional<int64_t> omitTensorUid;
    std::optional<int64_t> virtualUid;
    // The one operand that names a ragged offset tensor (packed sequences).
    std::optional<int64_t> raggedUid;
    // The one operand that declares a base alignment of its own, and the value it declares.
    // Every other operand keeps DEFAULT_ALIGNMENT_BYTES.
    std::optional<int64_t> alignmentUid;
    int64_t alignmentBytes = 8;
    std::optional<int64_t> passByValueUid;

    /// Tensors the node does not name, after the operands.
    std::vector<TensorSpec> unnamedTensors;
    /// The stats output's tensor, last; `statsUid` is what the node names.
    std::optional<TensorSpec> statsTensor;

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
    std::optional<int64_t> seqLenKvUid;
    std::optional<int64_t> seedUid;
    std::optional<int64_t> offsetUid;
    std::optional<int64_t> dropoutMaskUid;
    std::optional<int64_t> dropoutScaleUid;
    std::optional<int64_t> pageTableKUid;
    std::optional<int64_t> pageTableVUid;
    std::optional<int32_t> maxSeqLenKv;
    std::optional<int64_t> sinkTokenUid;
    std::optional<int64_t> blockMaskUid;
    std::optional<int64_t> statsUid;
    std::optional<int64_t> maxUid;
    std::optional<int64_t> sumExpUid;
    std::optional<int64_t> rngDumpUid;
    std::optional<int64_t> descaleQUid;
    std::optional<int64_t> descaleKUid;
    std::optional<int64_t> descaleVUid;
    std::optional<int64_t> descaleSUid;
    std::optional<int64_t> scaleSUid;
    std::optional<int64_t> scaleOUid;
    std::optional<int64_t> amaxSUid;
    std::optional<int64_t> amaxOUid;
    std::optional<float> dropoutProbability;
    std::optional<bool> generateStats;
    bool alibiMask = false;
    bool paddingMask = false;
    data_objects::DataType mmaCoreMode = data_objects::DataType::UNSET;
    data_objects::AttentionImplementation implementation
        = data_objects::AttentionImplementation::AUTO;

    bool twoNodes = false;
    /// One RELU node from Q to O in place of the SDPA node: a graph with no SDPA in it.
    bool pointwiseOnly = false;

    /// The position of @p uid in `uids`.
    std::size_t slotOf(int64_t uid) const
    {
        const auto it = std::find(uids.begin(), uids.end(), uid);
        if(it == uids.end())
        {
            throw std::out_of_range("uid " + std::to_string(uid) + " names no operand");
        }
        return static_cast<std::size_t>(std::distance(uids.begin(), it));
    }

    OperandSpec& operand(int64_t uid)
    {
        return operands.at(slotOf(uid));
    }

    const OperandSpec& operand(int64_t uid) const
    {
        return operands.at(slotOf(uid));
    }

    /// @p uid's dims as the graph carries them.
    std::vector<int64_t> dimsOf(int64_t uid) const
    {
        const auto& written = operand(uid).dims;
        if(written.has_value())
        {
            return *written;
        }
        // K and V take the key/value heads and length; Q and O the query's.
        const std::size_t slot = slotOf(uid);
        if(slot == 1 || slot == 2)
        {
            return {batch, numKvHeads, seqLenKv, headSize};
        }
        return {batch, numQueryHeads, seqLenQ, headSize};
    }

    /// @p uid's strides as the graph carries them. Derived only when not written out, since
    /// a written-out operand's extents may be chosen so their product does not fit.
    std::vector<int64_t> stridesOf(int64_t uid) const
    {
        const auto& spec = operand(uid);
        if(spec.strides.has_value())
        {
            return *spec.strides;
        }
        const auto dims = dimsOf(uid);
        if(dims.size() != 4)
        {
            return packedStrides(dims);
        }
        return sdpaStrides(spec.layout, dims[1], dims[2], dims[3]);
    }

    void setEveryLayout(SdpaLayout layout)
    {
        for(auto& spec : operands)
        {
            spec.layout = layout;
        }
    }
};

/// @p spec with the stats output the frontend builds: a stats uid, generate_stats true,
/// and a packed [B, Hq, Sq, 1] FLOAT tensor.
inline GraphSpec withStats(GraphSpec spec, int64_t uid = STATS_UID)
{
    spec.statsUid = uid;
    spec.generateStats = true;
    TensorSpec stats;
    stats.uid = uid;
    spec.statsTensor = stats;
    return spec;
}

inline flatbuffers::FlatBufferBuilder buildGraph(const GraphSpec& spec)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(std::size_t slot = 0; slot < spec.uids.size(); ++slot)
    {
        const int64_t uid = spec.uids.at(slot);
        if(spec.omitTensorUid == uid)
        {
            continue;
        }
        const auto dims = spec.dimsOf(uid);
        const auto strides = spec.stridesOf(uid);
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            uid,
            nullptr,
            spec.operands.at(slot).dataType.value_or(spec.dataType),
            spec.omitStrides ? nullptr : &strides,
            spec.omitDimsUid == uid ? nullptr : &dims,
            spec.virtualUid == uid,
            data_objects::TensorValue::NONE,
            0,
            spec.passByValueUid == uid,
            spec.raggedUid == uid ? ::flatbuffers::Optional<int64_t>(EXTRA_UID)
                                  : ::flatbuffers::Optional<int64_t>(),
            spec.alignmentUid == uid ? spec.alignmentBytes : DEFAULT_ALIGNMENT_BYTES));
    }

    const auto addTensor = [&](const TensorSpec& tensor) {
        const auto dims
            = tensor.dims.empty()
                  ? std::vector<int64_t>{spec.batch, spec.numQueryHeads, spec.seqLenQ, 1}
                  : tensor.dims;
        const auto strides = tensor.strides.empty() ? packedStrides(dims) : tensor.strides;
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, tensor.uid, nullptr, tensor.dataType, &strides, &dims, tensor.isVirtual));
    };
    for(const auto& tensor : spec.unnamedTensors)
    {
        addTensor(tensor);
    }
    if(spec.statsTensor.has_value())
    {
        addTensor(*spec.statsTensor);
    }

    const auto attributesFor = [&]() {
        data_objects::SdpaAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_q_tensor_uid(spec.uids[0]);
        attributesBuilder.add_k_tensor_uid(spec.uids[1]);
        attributesBuilder.add_v_tensor_uid(spec.uids[2]);
        attributesBuilder.add_o_tensor_uid(spec.uids[3]);

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
        if(spec.seqLenKvUid.has_value())
        {
            attributesBuilder.add_seq_len_kv_tensor_uid(*spec.seqLenKvUid);
        }
        if(spec.seedUid.has_value())
        {
            attributesBuilder.add_seed_tensor_uid(*spec.seedUid);
        }
        if(spec.offsetUid.has_value())
        {
            attributesBuilder.add_offset_tensor_uid(*spec.offsetUid);
        }
        if(spec.dropoutMaskUid.has_value())
        {
            attributesBuilder.add_dropout_mask_tensor_uid(*spec.dropoutMaskUid);
        }
        if(spec.dropoutScaleUid.has_value())
        {
            attributesBuilder.add_dropout_scale_tensor_uid(*spec.dropoutScaleUid);
        }
        if(spec.pageTableKUid.has_value())
        {
            attributesBuilder.add_page_table_k_tensor_uid(*spec.pageTableKUid);
        }
        if(spec.pageTableVUid.has_value())
        {
            attributesBuilder.add_page_table_v_tensor_uid(*spec.pageTableVUid);
        }
        if(spec.maxSeqLenKv.has_value())
        {
            attributesBuilder.add_max_seq_len_kv(*spec.maxSeqLenKv);
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
        if(spec.maxUid.has_value())
        {
            attributesBuilder.add_max_tensor_uid(*spec.maxUid);
        }
        if(spec.sumExpUid.has_value())
        {
            attributesBuilder.add_sum_exp_tensor_uid(*spec.sumExpUid);
        }
        if(spec.rngDumpUid.has_value())
        {
            attributesBuilder.add_rng_dump_tensor_uid(*spec.rngDumpUid);
        }
        if(spec.descaleQUid.has_value())
        {
            attributesBuilder.add_descale_q_tensor_uid(*spec.descaleQUid);
        }
        if(spec.descaleKUid.has_value())
        {
            attributesBuilder.add_descale_k_tensor_uid(*spec.descaleKUid);
        }
        if(spec.descaleVUid.has_value())
        {
            attributesBuilder.add_descale_v_tensor_uid(*spec.descaleVUid);
        }
        if(spec.descaleSUid.has_value())
        {
            attributesBuilder.add_descale_s_tensor_uid(*spec.descaleSUid);
        }
        if(spec.scaleSUid.has_value())
        {
            attributesBuilder.add_scale_s_tensor_uid(*spec.scaleSUid);
        }
        if(spec.scaleOUid.has_value())
        {
            attributesBuilder.add_scale_o_tensor_uid(*spec.scaleOUid);
        }
        if(spec.amaxSUid.has_value())
        {
            attributesBuilder.add_amax_s_tensor_uid(*spec.amaxSUid);
        }
        if(spec.amaxOUid.has_value())
        {
            attributesBuilder.add_amax_o_tensor_uid(*spec.amaxOUid);
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
    if(spec.pointwiseOnly)
    {
        const auto relu
            = data_objects::CreatePointwiseAttributes(builder,
                                                      data_objects::PointwiseMode::RELU_FWD,
                                                      flatbuffers::nullopt, // relu_lower_clip
                                                      flatbuffers::nullopt, // relu_upper_clip
                                                      flatbuffers::nullopt, // relu_lower_clip_slope
                                                      flatbuffers::nullopt, // axis_tensor_uid
                                                      spec.uids[0], // in_0_tensor_uid
                                                      flatbuffers::nullopt, // in_1_tensor_uid
                                                      flatbuffers::nullopt, // in_2_tensor_uid
                                                      spec.uids[3]); // out_0_tensor_uid
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
                                                       spec.computeType,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }
    if(spec.twoNodes)
    {
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "sdpa2",
                                                       spec.computeType,
                                                       data_objects::NodeAttributes::SdpaAttributes,
                                                       attributesFor().Union()));
    }

    auto name = builder.CreateString("attention_dense_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

/// A built graph and the MatchContext @p arch's engine sees over it. The context holds
/// references into this object, so it must outlive every use.
class Graph
{
public:
    /// @p multiProcessorCount is the device's CU count the gfx942 score reads (0:
    /// unresolved).
    Graph(const AttentionDenseTestArch& arch, const GraphSpec& spec, int multiProcessorCount = 0)
        : _arch(&arch)
        , _builder(buildGraph(spec))
        , _graph(_builder.GetBufferPointer(), _builder.GetSize())
    {
        _properties.gcnArchName = std::string(arch.gcnArchName);
        _properties.warpSize = 64;
        _properties.multiProcessorCount = multiProcessorCount;
        registerNativeIngestorSymbols();
    }

    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    const AttentionDenseTestArch& arch() const
    {
        return *_arch;
    }

    MatchContext context() const
    {
        return MatchContext{_graph, 0, _properties};
    }

    /// The arch's graph_match over this graph.
    std::optional<BoundTokens> match() const
    {
        return hipdnn_plugin_sdk::ingestor::GraphMatchRegistry::resolve(
            std::string(_arch->graphMatchSymbol))(context());
    }

private:
    const AttentionDenseTestArch* _arch;
    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper _graph;
    DeviceProperties _properties;
};

inline std::optional<BoundTokens> matchGraph(const AttentionDenseTestArch& arch,
                                             const GraphSpec& spec)
{
    return Graph(arch, spec).match();
}

/// The cause key and the text after it, of the decline an engine logged at INFO.
struct LoggedDecline
{
    std::string cause;
    std::string detail;
};

/// The decline @p arch's graph match logged at INFO for @p spec, or an empty cause when it
/// logged none or served the graph.
inline LoggedDecline loggedDecline(const AttentionDenseTestArch& arch, const GraphSpec& spec)
{
    const std::string marker = std::string(arch.engineName) + " declined the graph [";
    const auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
    EXPECT_FALSE(matchGraph(arch, spec).has_value());
    for(const auto& log : recorder.getRecordedLogs())
    {
        const auto at = log.message.find(marker);
        const auto causeEnd
            = at == std::string::npos ? at : log.message.find(']', at + marker.size());
        if(log.severity == HIPDNN_SEV_INFO && causeEnd != std::string::npos)
        {
            return {log.message.substr(at + marker.size(), causeEnd - at - marker.size()),
                    log.message.substr(causeEnd + 1)};
        }
    }
    return {};
}

/// graph_match then kernel_match, as a catalog build runs them.
inline bool matchesKernel(const AttentionDenseTestArch& arch,
                          const GraphSpec& spec,
                          const KernelDefinition& kernel)
{
    const Graph graph(arch, spec);
    const auto bound = graph.match();
    EXPECT_TRUE(bound.has_value()) << "graph_match declined the graph before kernel_match ran";
    if(!bound.has_value())
    {
        return false;
    }
    return hipdnn_plugin_sdk::ingestor::KernelMatcherRegistry::resolve(
        std::string(arch.kernelMatchSymbol))(graph.context(), *bound, kernel);
}

/// @p value's IEEE-754 storage, the way BoundTokens carries a float.
inline int64_t ieee754Bits(float value)
{
    int32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "float must be 32-bit to round-trip");
    std::memcpy(&bits, &value, sizeof(value));
    return static_cast<int64_t>(bits);
}

// ---------------------------------------------------------------------------
// The prepare() seam
// ---------------------------------------------------------------------------

/// `.value_kind` a device pointer parameter carries in the AMDGPU metadata note.
inline constexpr const char* BUFFER_KIND = "global_buffer";

/// `.value_kind` a scalar parameter carries.
inline constexpr const char* BY_VALUE_KIND = "by_value";

/// Bytes one `ptr<dtype, global>` occupies in the kernarg segment: a 64-bit address,
/// whatever the element type is.
inline constexpr uint32_t POINTER_BYTES = 8;

inline constexpr uint32_t SCALAR_BYTES = 4;

/// The argument list attention_dense_signature declares, under the names it gives them.
inline std::vector<KernelArgument> pythonAbi()
{
    return {KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "q_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "k_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "v_ptr"},
            KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "o_ptr"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, "scale"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, "batch"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, "seqlen_q"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, "seqlen_kv"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, "num_query_heads"},
            KernelArgument{BY_VALUE_KIND, SCALAR_BYTES, 0, "num_kv_heads"}};
}

/// The emit_lse form: the same ten arguments, then lse_ptr, the FP32 stats output.
inline std::vector<KernelArgument> pythonLseAbi()
{
    auto signature = pythonAbi();
    signature.push_back(KernelArgument{BUFFER_KIND, POINTER_BYTES, 0, "lse_ptr"});
    return signature;
}

/// A KPACK descriptor for @p arch's engine recording @p signature, carrying the completed
/// D128 metadata prepare() reads: head_size and the (@p blockM, @p blockN) tile.
///
/// The archive it names is not there, which is the point: every check prepare() makes
/// before the loader is reached, and when they all pass the loader reports the missing
/// file. No device, no compile, no launch.
inline KernelDefinition kpackKernel(const AttentionDenseTestArch& arch,
                                    const std::vector<KernelArgument>& signature,
                                    int64_t blockM = 256,
                                    int64_t blockN = 64)
{
    KernelDefinition kernel;
    kernel.kernelId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a1");
    kernel.packId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a2");
    kernel.dispatchId
        = hipdnn_flatbuffers_sdk::utilities::parseUuid("00000000-0000-4000-8000-0000000051a3");
    kernel.name = std::string(arch.kernelName);
    kernel.source.kind = hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK;
    kernel.source.library = "there-is-no-archive-here.kpack";
    kernel.source.tocKey = "toc#0";
    kernel.source.symbol = "attention_dense";
    kernel.source.signature = signature;
    kernel.originDirectory = "/nonexistent";
    kernel.treeRoot = "/nonexistent";
    kernel.metadata = completedAttentionDenseMetadata(arch, HEAD_SIZE, blockM, blockN);
    return kernel;
}

inline const hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>*
    dispatchHandler(const AttentionDenseTestArch& arch)
{
    registerNativeIngestorSymbols();
    return hipdnn_plugin_sdk::ingestor::DispatchRegistry<Handle>::resolve(
        std::string(arch.dispatchSymbol));
}

/// The message prepare() throws for @p bound and @p kernel over @p graph, or empty if it
/// returned a prepared dispatch -- which it cannot while the archive is absent.
inline std::string
    prepareFailure(const Graph& graph, const BoundTokens& bound, const KernelDefinition& kernel)
{
    const auto* handler = dispatchHandler(graph.arch());
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

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::attention_dense_graphs

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
