// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp"

/**
 * @file FlydslSdpaBwdTestGraphs.hpp
 * @brief Raw flatbuffer SdpaBackwardAttributes graphs for the shared backward matcher's suite.
 *
 * The default graph is the one the hipDNN test SDK's createValidSdpaBwdGraph builds: f16,
 * B=2, H=8, S=16, D=64, head-major dense strides, uids 1..9, no mask, no bias, no dropout.
 * A test states what it changes with SdpaBwdSpec::withEdit, which edits the graph's tensor
 * attributes, node attributes and graph-level flags before they are serialized, so a decline
 * case and its accepting control differ in exactly the lines the edit names.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The uids the default graph and the add* helpers assign.
namespace sdpa_bwd_uid
{
inline constexpr int64_t Q = 1;
inline constexpr int64_t K = 2;
inline constexpr int64_t V = 3;
inline constexpr int64_t O = 4;
inline constexpr int64_t DO = 5;
inline constexpr int64_t STATS = 6;
inline constexpr int64_t DQ = 7;
inline constexpr int64_t DK = 8;
inline constexpr int64_t DV = 9;
inline constexpr int64_t SCALE = 10;
inline constexpr int64_t BIAS = 11;
inline constexpr int64_t DBIAS = 12;
inline constexpr int64_t SEQ_LEN_Q = 13;
inline constexpr int64_t SEQ_LEN_KV = 14;
inline constexpr int64_t SEED = 15;
inline constexpr int64_t OFFSET = 16;
inline constexpr int64_t DROPOUT_MASK = 17;
inline constexpr int64_t DROPOUT_SCALE = 18;
inline constexpr int64_t DROPOUT_SCALE_INV = 19;
} // namespace sdpa_bwd_uid

/// A pass-by-value tensor's baked value; monostate for a tensor that has none.
using SdpaBwdTensorValue = std::variant<std::monostate, float, double, int32_t, int64_t>;

struct SdpaBwdTensor
{
    int64_t uid = 0;
    std::string name;
    hipdnn_flatbuffers_sdk::data_objects::DataType dtype
        = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    std::vector<int64_t> dims;
    std::vector<int64_t> strides;
    bool isVirtual = false;
    bool runtimePassByValue = false;
    SdpaBwdTensorValue value;
    std::optional<int64_t> raggedOffsetUid;
    int64_t alignment = 16;
};

/// Every SdpaBackwardAttributes field.
struct SdpaBwdAttributes
{
    int64_t q = sdpa_bwd_uid::Q;
    int64_t k = sdpa_bwd_uid::K;
    int64_t v = sdpa_bwd_uid::V;
    int64_t o = sdpa_bwd_uid::O;
    int64_t dO = sdpa_bwd_uid::DO;
    int64_t stats = sdpa_bwd_uid::STATS;
    int64_t dQ = sdpa_bwd_uid::DQ;
    int64_t dK = sdpa_bwd_uid::DK;
    int64_t dV = sdpa_bwd_uid::DV;
    std::optional<int64_t> scaleUid;
    std::optional<int64_t> biasUid;
    std::optional<int64_t> seqLenQUid;
    std::optional<int64_t> seqLenKvUid;
    std::optional<int64_t> seedUid;
    std::optional<int64_t> offsetUid;
    std::optional<int64_t> dropoutMaskUid;
    std::optional<int64_t> dropoutScaleUid;
    std::optional<int64_t> dropoutScaleInvUid;
    std::optional<int64_t> dbiasUid;
    bool alibiMask = false;
    bool paddingMask = false;
    bool causalMask = false;
    bool causalMaskBottomRight = false;
    std::optional<float> dropoutProbability;
    std::optional<float> attnScaleValue;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment diagonalAlignment
        = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
};

/// The node's operation: the one the matcher serves, or a forward SDPA node.
enum class SdpaBwdNodeKind
{
    BACKWARD,
    FORWARD
};

/// The graph as a test edits it, before serialization.
struct SdpaBwdGraph
{
    int64_t batch = 0;
    int64_t heads = 0;
    int64_t kvHeads = 0;
    int64_t seqQ = 0;
    int64_t seqKv = 0;
    int64_t headDimQk = 0;
    int64_t headDimV = 0;
    std::vector<SdpaBwdTensor> tensors;
    SdpaBwdAttributes attributes;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    SdpaBwdNodeKind nodeKind = SdpaBwdNodeKind::BACKWARD;
    uint32_t nodeCount = 1;
    bool overrideShape = false;

    /// The tensor named "q", "k", "v", "o", "do", "stats", "dq", "dk", "dv", or one an add*
    /// helper named.
    SdpaBwdTensor& tensor(std::string_view name)
    {
        for(auto& candidate : tensors)
        {
            if(candidate.name == name)
            {
                return candidate;
            }
        }
        throw std::out_of_range("no tensor named " + std::string(name));
    }
};

/// A graph the matcher is asked about: dimensions, then the edits applied in order. The row
/// strides of the default graph are the head dim rounded up to a 16-byte vector of its dtype.
struct SdpaBwdSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dtype
        = hipdnn_flatbuffers_sdk::data_objects::DataType::HALF;
    int64_t batch = 2;
    int64_t heads = 8;
    int64_t kvHeads = 8;
    int64_t seqQ = 16;
    int64_t seqKv = 16;
    int64_t headDimQk = 64;
    int64_t headDimV = 64;
    std::vector<std::function<void(SdpaBwdGraph&)>> edits;

    /// A copy with @p edit applied after the edits already present.
    SdpaBwdSpec withEdit(std::function<void(SdpaBwdGraph&)> edit) const
    {
        auto copy = *this;
        copy.edits.push_back(std::move(edit));
        return copy;
    }
};

/// Element strides of a dense row-major tensor of @p dims.
inline std::vector<int64_t> denseStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t axis = dims.size() - 1; axis > 0; --axis)
    {
        strides[axis - 1] = strides[axis] * dims[axis];
    }
    return strides;
}

/// Head-major (B, H, S, D) strides whose rows are @p rowStride elements apart.
inline std::vector<int64_t> headMajorStrides(int64_t heads, int64_t seq, int64_t rowStride)
{
    return {heads * seq * rowStride, seq * rowStride, rowStride, 1};
}

inline int64_t ceilToEight(int64_t value)
{
    return (value + 7) / 8 * 8;
}

/// Elements of one 16-byte vector of @p dtype: 8 for f16 and bf16, 4 for FLOAT.
inline int64_t vectorElementsOf(hipdnn_flatbuffers_sdk::data_objects::DataType dtype)
{
    return dtype == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT ? 4 : 8;
}

/// @p value rounded up to a whole vector of @p dtype.
inline int64_t ceilToVectorOf(hipdnn_flatbuffers_sdk::data_objects::DataType dtype, int64_t value)
{
    const auto vector = vectorElementsOf(dtype);
    return (value + vector - 1) / vector * vector;
}

/// A spec for an fp32 graph: FLOAT on every data tensor, bias and dBias.
inline SdpaBwdSpec fp32Spec()
{
    SdpaBwdSpec spec;
    spec.dtype = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    return spec;
}

/// The default graph for @p spec, with the spec's edits not yet applied.
inline SdpaBwdGraph makeSdpaBwdGraph(const SdpaBwdSpec& spec)
{
    namespace uid = sdpa_bwd_uid;
    SdpaBwdGraph graph;
    graph.batch = spec.batch;
    graph.heads = spec.heads;
    graph.kvHeads = spec.kvHeads;
    graph.seqQ = spec.seqQ;
    graph.seqKv = spec.seqKv;
    graph.headDimQk = spec.headDimQk;
    graph.headDimV = spec.headDimV;

    const auto add = [&](int64_t id,
                         const char* name,
                         hipdnn_flatbuffers_sdk::data_objects::DataType dtype,
                         std::vector<int64_t> dims,
                         std::vector<int64_t> strides) {
        SdpaBwdTensor tensor;
        tensor.uid = id;
        tensor.name = name;
        tensor.dtype = dtype;
        tensor.dims = std::move(dims);
        tensor.strides = std::move(strides);
        graph.tensors.push_back(std::move(tensor));
    };
    const std::vector<int64_t> qDims{spec.batch, spec.heads, spec.seqQ, spec.headDimQk};
    const std::vector<int64_t> kDims{spec.batch, spec.kvHeads, spec.seqKv, spec.headDimQk};
    const std::vector<int64_t> vDims{spec.batch, spec.kvHeads, spec.seqKv, spec.headDimV};
    const std::vector<int64_t> oDims{spec.batch, spec.heads, spec.seqQ, spec.headDimV};
    const auto rowQk = ceilToVectorOf(spec.dtype, spec.headDimQk);
    const auto rowV = ceilToVectorOf(spec.dtype, spec.headDimV);
    const auto qStrides = headMajorStrides(spec.heads, spec.seqQ, rowQk);
    const auto kStrides = headMajorStrides(spec.kvHeads, spec.seqKv, rowQk);
    const auto vStrides = headMajorStrides(spec.kvHeads, spec.seqKv, rowV);
    const auto oStrides = headMajorStrides(spec.heads, spec.seqQ, rowV);
    const std::vector<int64_t> statsDims{spec.batch, spec.heads, spec.seqQ, 1};
    const std::vector<int64_t> statsStrides{spec.heads * spec.seqQ, spec.seqQ, 1, 1};

    add(uid::Q, "q", spec.dtype, qDims, qStrides);
    add(uid::K, "k", spec.dtype, kDims, kStrides);
    add(uid::V, "v", spec.dtype, vDims, vStrides);
    add(uid::O, "o", spec.dtype, oDims, oStrides);
    add(uid::DO, "do", spec.dtype, oDims, oStrides);
    add(uid::STATS,
        "stats",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        statsDims,
        statsStrides);
    add(uid::DQ, "dq", spec.dtype, qDims, qStrides);
    add(uid::DK, "dk", spec.dtype, kDims, kStrides);
    add(uid::DV, "dv", spec.dtype, vDims, vStrides);
    return graph;
}

/// Adds a bias of @p dims (default dense [B, Hq, Sq, Skv]) in Q's dtype and names it.
inline SdpaBwdTensor& addBias(SdpaBwdGraph& graph, std::vector<int64_t> dims = {})
{
    if(dims.empty())
    {
        dims = {graph.batch, graph.heads, graph.seqQ, graph.seqKv};
    }
    SdpaBwdTensor tensor;
    tensor.uid = sdpa_bwd_uid::BIAS;
    tensor.name = "bias";
    tensor.dtype = graph.tensor("q").dtype;
    tensor.strides = denseStrides(dims);
    tensor.dims = std::move(dims);
    graph.attributes.biasUid = tensor.uid;
    graph.tensors.push_back(std::move(tensor));
    return graph.tensors.back();
}

/// Adds a dBias with the bias's dims, strides and dtype, and names it.
inline SdpaBwdTensor& addDbias(SdpaBwdGraph& graph)
{
    SdpaBwdTensor tensor = graph.tensor("bias");
    tensor.uid = sdpa_bwd_uid::DBIAS;
    tensor.name = "dbias";
    graph.attributes.dbiasUid = tensor.uid;
    graph.tensors.push_back(std::move(tensor));
    return graph.tensors.back();
}

/// Adds the padded-varlen length tensors (INT32 [B, 1, 1, 1], device) and sets padding_mask.
inline void addSeqLens(SdpaBwdGraph& graph)
{
    const auto add = [&](int64_t id, const char* name) {
        SdpaBwdTensor tensor;
        tensor.uid = id;
        tensor.name = name;
        tensor.dtype = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
        tensor.dims = {graph.batch, 1, 1, 1};
        tensor.strides = {1, 1, 1, 1};
        graph.tensors.push_back(std::move(tensor));
    };
    add(sdpa_bwd_uid::SEQ_LEN_Q, "seq_len_q");
    add(sdpa_bwd_uid::SEQ_LEN_KV, "seq_len_kv");
    graph.attributes.seqLenQUid = sdpa_bwd_uid::SEQ_LEN_Q;
    graph.attributes.seqLenKvUid = sdpa_bwd_uid::SEQ_LEN_KV;
    graph.attributes.paddingMask = true;
}

/// Sets dropout probability @p probability with INT32 or INT64 seed and offset tensors
/// (1, 1, 1, 1), device or, with @p passByValue, carrying a baked value.
inline void addDropout(SdpaBwdGraph& graph,
                       float probability = 0.2F,
                       hipdnn_flatbuffers_sdk::data_objects::DataType seedType
                       = hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                       bool passByValue = false)
{
    const auto add = [&](int64_t id, const char* name) {
        SdpaBwdTensor tensor;
        tensor.uid = id;
        tensor.name = name;
        tensor.dtype = seedType;
        tensor.dims = {1, 1, 1, 1};
        tensor.strides = {1, 1, 1, 1};
        if(passByValue)
        {
            if(seedType == hipdnn_flatbuffers_sdk::data_objects::DataType::INT32)
            {
                tensor.value = int32_t{7};
            }
            else
            {
                tensor.value = int64_t{7};
            }
        }
        graph.tensors.push_back(std::move(tensor));
    };
    add(sdpa_bwd_uid::SEED, "seed");
    add(sdpa_bwd_uid::OFFSET, "offset");
    graph.attributes.seedUid = sdpa_bwd_uid::SEED;
    graph.attributes.offsetUid = sdpa_bwd_uid::OFFSET;
    graph.attributes.dropoutProbability = probability;
}

/// Adds an f32 scalar scale tensor, pass-by-value: carrying @p baked, or, without one,
/// supplied at execute.
inline SdpaBwdTensor& addScaleTensor(SdpaBwdGraph& graph, std::optional<float> baked)
{
    SdpaBwdTensor tensor;
    tensor.uid = sdpa_bwd_uid::SCALE;
    tensor.name = "scale";
    tensor.dtype = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    tensor.dims = {1};
    tensor.strides = {1};
    if(baked.has_value())
    {
        tensor.value = *baked;
    }
    else
    {
        tensor.runtimePassByValue = true;
    }
    graph.attributes.scaleUid = tensor.uid;
    graph.tensors.push_back(std::move(tensor));
    return graph.tensors.back();
}

namespace detail
{

inline flatbuffers::Optional<int64_t> toFlatOptional(const std::optional<int64_t>& value)
{
    return value.has_value() ? flatbuffers::Optional<int64_t>(*value) : flatbuffers::nullopt;
}

inline flatbuffers::Optional<float> toFlatOptional(const std::optional<float>& value)
{
    return value.has_value() ? flatbuffers::Optional<float>(*value) : flatbuffers::nullopt;
}

} // namespace detail

/// Serializes @p spec's graph with its edits applied.
inline flatbuffers::FlatBufferBuilder buildSdpaBwdGraph(const SdpaBwdSpec& spec)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    auto graph = makeSdpaBwdGraph(spec);
    for(const auto& edit : spec.edits)
    {
        edit(graph);
    }

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto& tensor : graph.tensors)
    {
        auto valueType = data_objects::TensorValue::NONE;
        flatbuffers::Offset<void> value = 0;
        if(const auto* baked = std::get_if<float>(&tensor.value))
        {
            valueType = data_objects::TensorValue::Float32Value;
            value = builder.CreateStruct(data_objects::Float32Value(*baked)).Union();
        }
        else if(const auto* bakedDouble = std::get_if<double>(&tensor.value))
        {
            valueType = data_objects::TensorValue::Float64Value;
            value = builder.CreateStruct(data_objects::Float64Value(*bakedDouble)).Union();
        }
        else if(const auto* bakedInt = std::get_if<int32_t>(&tensor.value))
        {
            valueType = data_objects::TensorValue::Int32Value;
            value = builder.CreateStruct(data_objects::Int32Value(*bakedInt)).Union();
        }
        else if(const auto* bakedLong = std::get_if<int64_t>(&tensor.value))
        {
            valueType = data_objects::TensorValue::Int64Value;
            value = builder.CreateStruct(data_objects::Int64Value(*bakedLong)).Union();
        }
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            tensor.uid,
            tensor.name.c_str(),
            tensor.dtype,
            &tensor.strides,
            &tensor.dims,
            tensor.isVirtual,
            valueType,
            value,
            tensor.runtimePassByValue,
            detail::toFlatOptional(tensor.raggedOffsetUid),
            tensor.alignment));
    }

    const auto& a = graph.attributes;
    flatbuffers::Offset<void> attributes;
    data_objects::NodeAttributes attributesType;
    if(graph.nodeKind == SdpaBwdNodeKind::BACKWARD)
    {
        attributesType = data_objects::NodeAttributes::SdpaBackwardAttributes;
        attributes = data_objects::CreateSdpaBackwardAttributes(
                         builder,
                         a.q,
                         a.k,
                         a.v,
                         a.o,
                         a.dO,
                         a.stats,
                         a.dQ,
                         a.dK,
                         a.dV,
                         detail::toFlatOptional(a.scaleUid),
                         detail::toFlatOptional(a.biasUid),
                         detail::toFlatOptional(a.seqLenQUid),
                         detail::toFlatOptional(a.seqLenKvUid),
                         detail::toFlatOptional(a.seedUid),
                         detail::toFlatOptional(a.offsetUid),
                         detail::toFlatOptional(a.dropoutMaskUid),
                         detail::toFlatOptional(a.dropoutScaleUid),
                         detail::toFlatOptional(a.dropoutScaleInvUid),
                         detail::toFlatOptional(a.dbiasUid),
                         a.alibiMask,
                         a.paddingMask,
                         a.causalMask,
                         a.causalMaskBottomRight,
                         detail::toFlatOptional(a.dropoutProbability),
                         detail::toFlatOptional(a.attnScaleValue),
                         detail::toFlatOptional(a.leftBound),
                         detail::toFlatOptional(a.rightBound),
                         a.diagonalAlignment)
                         .Union();
    }
    else
    {
        attributesType = data_objects::NodeAttributes::SdpaAttributes;
        data_objects::SdpaAttributesBuilder forward(builder);
        forward.add_q_tensor_uid(a.q);
        forward.add_k_tensor_uid(a.k);
        forward.add_v_tensor_uid(a.v);
        forward.add_o_tensor_uid(a.o);
        attributes = forward.Finish().Union();
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(uint32_t index = 0; index < graph.nodeCount; ++index)
    {
        nodes.push_back(data_objects::CreateNodeDirect(
            builder, "sdpa_bwd", graph.computeType, attributesType, attributes));
    }

    const auto graphOffset = data_objects::CreateGraphDirect(builder,
                                                             "sdpa_bwd_test",
                                                             data_objects::DataType::FLOAT,
                                                             data_objects::DataType::FLOAT,
                                                             spec.dtype,
                                                             &tensors,
                                                             &nodes,
                                                             flatbuffers::nullopt,
                                                             graph.overrideShape);
    builder.Finish(graphOffset);
    return builder;
}

/// What the shared backward matcher says about @p spec's graph under @p options.
inline std::variant<flydsl_sdpa_bwd::SdpaBwdProblem, flydsl_sdpa_bwd::Decline>
    matchSdpaBwdSpec(const SdpaBwdSpec& spec, const flydsl_sdpa_bwd::MatchOptions& options = {})
{
    const auto builder = buildSdpaBwdGraph(spec);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    return flydsl_sdpa_bwd::matchSdpaBackward(
        hipdnn_plugin_sdk::ingestor::MatchContext{graph, 0, properties}, options);
}

/// What the matcher says about an already-built graph buffer under @p options, for graphs that
/// do not come from a spec (the hipDNN test SDK's createValidSdpaBwdGraph).
inline std::variant<flydsl_sdpa_bwd::SdpaBwdProblem, flydsl_sdpa_bwd::Decline>
    matchSdpaBwdBuffer(const flatbuffers::FlatBufferBuilder& builder,
                       const flydsl_sdpa_bwd::MatchOptions& options = {})
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());
    hipdnn_plugin_sdk::ingestor::DeviceProperties properties;
    properties.gcnArchName = "gfx950";
    properties.warpSize = 64;
    return flydsl_sdpa_bwd::matchSdpaBackward(
        hipdnn_plugin_sdk::ingestor::MatchContext{graph, 0, properties}, options);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
