// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_rope's contract. One pack, so `operationMatcher` is empty:
/// `RopeAttributes` is its own union arm. The three operand tokens are x/freqs/y.
inline constexpr PackSymbols FLYDSL_ROPE{"hipkernel:flydsl_rope",
                                         "hipkernel.flydsl_rope.graph_match",
                                         "",
                                         "hipkernel.flydsl_rope.kernel_match",
                                         "hipkernel.flydsl_rope.score",
                                         "hipkernel.flydsl_rope.dispatch",
                                         "flydsl_rope.x.uid",
                                         "flydsl_rope.freqs.uid",
                                         "flydsl_rope.y.uid"};

/// Tensor uids buildFlydslRopeGraph() uses.
constexpr int64_t FLYDSL_ROPE_X_UID = 1;
constexpr int64_t FLYDSL_ROPE_FREQS_UID = 2;
constexpr int64_t FLYDSL_ROPE_Y_UID = 3;

/// Row-major packed strides for @p dims.
inline std::vector<int64_t> flydslRopePackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// The strides of a logical [B, H, S, D] tensor stored B, S, H, D: how Qwen (and every
/// framework that projects q/k as [B, S, H*D]) hands RoPE its operands.
inline std::vector<int64_t> flydslRopeBshdStrides(const std::vector<int64_t>& bhsdDims)
{
    const int64_t heads = bhsdDims[1];
    const int64_t seq = bhsdDims[2];
    const int64_t headDim = bhsdDims[3];
    return {seq * heads * headDim, headDim, heads * headDim, 1};
}

/// Everything a RoPE graph varies, defaulting to Qwen3's partial rotary: bf16 x/y of
/// [1, 16, 7, 256] in BSHD strides, freqs [1, 1, 7, 64], r = 64 and o = 0.
struct RopeGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    std::vector<int64_t> xDims{1, 16, 7, 256};
    /// Unset means BSHD strides for rank 4, packed otherwise.
    std::optional<std::vector<int64_t>> xStrides;
    std::vector<int64_t> freqsDims{1, 1, 7, 64};
    /// Unset means packed.
    std::optional<std::vector<int64_t>> freqsStrides;
    hipdnn_flatbuffers_sdk::data_objects::DataType freqsDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    /// Unset means x's dtype, dims and strides.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> yDataType;
    std::optional<std::vector<int64_t>> yDims;
    std::optional<std::vector<int64_t>> yStrides;
    float outputScale = 1.0F;
    int64_t ropeDim = 64;
    /// Unset means the cuDNN default, D - r.
    std::optional<int64_t> ropeOffset = 0;
    /// Unset means x's dtype.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> computeDataType;
    std::set<int64_t> virtualTensors;
    std::set<int64_t> passByValueTensors;
    bool secondNode = false;
    bool overrideShapeEnabled = false;
};

inline std::vector<int64_t> flydslRopeXStrides(const RopeGraphSpec& spec)
{
    if(spec.xStrides.has_value())
    {
        return *spec.xStrides;
    }
    return spec.xDims.size() == 4 ? flydslRopeBshdStrides(spec.xDims)
                                  : flydslRopePackedStrides(spec.xDims);
}

/// A single-node (or, with `secondNode`, two-node) RoPE graph as @p spec describes it.
inline flatbuffers::FlatBufferBuilder buildFlydslRopeGraph(const RopeGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;

    const auto tensor = [&](int64_t uid,
                            const char* name,
                            data_objects::DataType type,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dims) {
        data_objects::TensorValue valueType = data_objects::TensorValue::NONE;
        flatbuffers::Offset<void> value = 0;
        if(spec.passByValueTensors.count(uid) != 0)
        {
            const data_objects::Float32Value hostValue(1.0F);
            valueType = data_objects::TensorValue::Float32Value;
            value = builder.CreateStruct(hostValue).Union();
        }
        return data_objects::CreateTensorAttributesDirect(builder,
                                                          uid,
                                                          name,
                                                          type,
                                                          &strides,
                                                          &dims,
                                                          spec.virtualTensors.count(uid) != 0,
                                                          valueType,
                                                          value);
    };

    const auto xStrides = flydslRopeXStrides(spec);
    const auto freqsStrides = spec.freqsStrides.value_or(flydslRopePackedStrides(spec.freqsDims));
    const auto yDims = spec.yDims.value_or(spec.xDims);
    const auto yStrides = spec.yStrides.value_or(
        spec.yDims.has_value() ? flydslRopePackedStrides(yDims) : xStrides);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(tensor(FLYDSL_ROPE_X_UID, "x", spec.dataType, xStrides, spec.xDims));
    tensors.push_back(
        tensor(FLYDSL_ROPE_FREQS_UID, "freqs", spec.freqsDataType, freqsStrides, spec.freqsDims));
    tensors.push_back(
        tensor(FLYDSL_ROPE_Y_UID, "y", spec.yDataType.value_or(spec.dataType), yStrides, yDims));

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        const auto attributes = data_objects::CreateRopeAttributes(builder,
                                                                   FLYDSL_ROPE_X_UID,
                                                                   FLYDSL_ROPE_FREQS_UID,
                                                                   FLYDSL_ROPE_Y_UID,
                                                                   spec.outputScale,
                                                                   spec.ropeDim,
                                                                   spec.ropeOffset);
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "rope",
                                                       spec.computeDataType.value_or(spec.dataType),
                                                       data_objects::NodeAttributes::RopeAttributes,
                                                       attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_rope_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapeEnabled);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
