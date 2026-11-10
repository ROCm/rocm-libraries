// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_causal_conv1d's contract. One pack, so `operationMatcher` is empty:
/// `CausalConv1dAttributes` is its own union arm. The operand tokens named here are
/// x/weight/y; the others are FLYDSL_CAUSAL_CONV1D_*_TOKEN below.
inline constexpr PackSymbols FLYDSL_CAUSAL_CONV1D{"hipkernel:flydsl_causal_conv1d",
                                                  "hipkernel.flydsl_causal_conv1d.graph_match",
                                                  "",
                                                  "hipkernel.flydsl_causal_conv1d.kernel_match",
                                                  "hipkernel.flydsl_causal_conv1d.score",
                                                  "hipkernel.flydsl_causal_conv1d.dispatch",
                                                  "flydsl_causal_conv1d.x.uid",
                                                  "flydsl_causal_conv1d.weight.uid",
                                                  "flydsl_causal_conv1d.y.uid"};
inline constexpr const char* FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_TOKEN
    = "flydsl_causal_conv1d.cu_seqlens.uid";
inline constexpr const char* FLYDSL_CAUSAL_CONV1D_BIAS_TOKEN = "flydsl_causal_conv1d.bias.uid";
inline constexpr const char* FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_TOKEN
    = "flydsl_causal_conv1d.initial_state.uid";
inline constexpr const char* FLYDSL_CAUSAL_CONV1D_STATE_INDICES_TOKEN
    = "flydsl_causal_conv1d.state_indices.uid";
inline constexpr const char* FLYDSL_CAUSAL_CONV1D_FINAL_STATE_TOKEN
    = "flydsl_causal_conv1d.final_state.uid";

/// Tensor uids buildFlydslCausalConv1dGraph() uses.
constexpr int64_t FLYDSL_CAUSAL_CONV1D_X_UID = 1;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_WEIGHT_UID = 2;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_UID = 3;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_Y_UID = 4;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_BIAS_UID = 5;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_UID = 6;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_STATE_INDICES_UID = 7;
constexpr int64_t FLYDSL_CAUSAL_CONV1D_FINAL_STATE_UID = 8;

/// Row-major packed strides for @p dims.
inline std::vector<int64_t> flydslCausalConv1dPackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// Everything a causal-conv1d graph varies, defaulting to a Qwen3.5 linear-attention decode
/// step: bf16, N = 4 sequences of one token, C = 2048, W = 4, SiLU, no bias, an initial and
/// a final state. Every tensor is packed unless its strides are given.
struct CausalConv1dGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    int64_t tokens = 4;
    int64_t channels = 2048;
    int64_t width = 4;
    int64_t sequences = 4;
    hipdnn_flatbuffers_sdk::data_objects::CausalConv1dActivation activation
        = hipdnn_flatbuffers_sdk::data_objects::CausalConv1dActivation::SILU;
    hipdnn_flatbuffers_sdk::data_objects::DataType cuSeqlensDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    bool bias = false;
    bool initialState = true;
    bool finalState = true;
    /// Unset means no pool; set, the states have this many rows and state_indices is bound.
    std::optional<int64_t> poolRows;
    hipdnn_flatbuffers_sdk::data_objects::DataType stateIndicesDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    std::optional<std::vector<int64_t>> xStrides;
    std::optional<std::vector<int64_t>> yStrides;
    std::optional<std::vector<int64_t>> weightStrides;
    std::optional<std::vector<int64_t>> initialStrides;
    std::optional<std::vector<int64_t>> finalStrides;
    std::optional<std::vector<int64_t>> cuSeqlensStrides;
    /// Per-tensor dtype and dims overrides, for the refusal cases.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> weightDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> stateDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> yDataType;
    std::optional<std::vector<int64_t>> yDims;
    std::optional<std::vector<int64_t>> finalDims;
    std::optional<int64_t> cuSeqlensAlignment;
    /// Unset means x's dtype.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> computeDataType;
    std::set<int64_t> virtualTensors;
    bool secondNode = false;
    bool overrideShapeEnabled = false;

    int64_t stateRows() const
    {
        return poolRows.value_or(sequences);
    }
};

/// A single-node (or, with `secondNode`, two-node) causal-conv1d graph as @p spec describes.
inline flatbuffers::FlatBufferBuilder buildFlydslCausalConv1dGraph(const CausalConv1dGraphSpec& spec
                                                                   = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto tensor = [&](int64_t uid,
                            const char* name,
                            data_objects::DataType type,
                            const std::vector<int64_t>& dims,
                            const std::optional<std::vector<int64_t>>& strides,
                            int64_t alignment = 16) {
        const auto packed = flydslCausalConv1dPackedStrides(dims);
        const auto& used = strides.has_value() ? *strides : packed;
        tensors.push_back(
            data_objects::CreateTensorAttributesDirect(builder,
                                                       uid,
                                                       name,
                                                       type,
                                                       &used,
                                                       &dims,
                                                       spec.virtualTensors.count(uid) != 0,
                                                       data_objects::TensorValue::NONE,
                                                       0,
                                                       false,
                                                       flatbuffers::nullopt,
                                                       alignment));
    };

    const std::vector<int64_t> rowDims{spec.tokens, spec.channels};
    const std::vector<int64_t> stateDims{spec.stateRows(), spec.channels, spec.width - 1};
    tensor(FLYDSL_CAUSAL_CONV1D_X_UID, "x", spec.dataType, rowDims, spec.xStrides);
    tensor(FLYDSL_CAUSAL_CONV1D_WEIGHT_UID,
           "weight",
           spec.weightDataType.value_or(spec.dataType),
           {spec.channels, spec.width},
           spec.weightStrides);
    tensor(FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_UID,
           "cu_seqlens",
           spec.cuSeqlensDataType,
           {spec.sequences + 1},
           spec.cuSeqlensStrides,
           spec.cuSeqlensAlignment.value_or(16));
    tensor(FLYDSL_CAUSAL_CONV1D_Y_UID,
           "y",
           spec.yDataType.value_or(spec.dataType),
           spec.yDims.value_or(rowDims),
           spec.yStrides);
    if(spec.bias)
    {
        tensor(FLYDSL_CAUSAL_CONV1D_BIAS_UID, "bias", spec.dataType, {spec.channels}, std::nullopt);
    }
    if(spec.initialState)
    {
        tensor(FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_UID,
               "initial_state",
               spec.stateDataType.value_or(spec.dataType),
               stateDims,
               spec.initialStrides);
    }
    if(spec.poolRows.has_value())
    {
        tensor(FLYDSL_CAUSAL_CONV1D_STATE_INDICES_UID,
               "state_indices",
               spec.stateIndicesDataType,
               {spec.sequences},
               std::nullopt);
    }
    if(spec.finalState)
    {
        tensor(FLYDSL_CAUSAL_CONV1D_FINAL_STATE_UID,
               "final_state",
               spec.stateDataType.value_or(spec.dataType),
               spec.finalDims.value_or(stateDims),
               spec.finalStrides);
    }

    const auto optionalUid = [](bool bound, int64_t uid) {
        return bound ? flatbuffers::Optional<int64_t>(uid) : flatbuffers::nullopt;
    };
    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        const auto attributes = data_objects::CreateCausalConv1dAttributes(
            builder,
            FLYDSL_CAUSAL_CONV1D_X_UID,
            FLYDSL_CAUSAL_CONV1D_WEIGHT_UID,
            FLYDSL_CAUSAL_CONV1D_CU_SEQLENS_UID,
            FLYDSL_CAUSAL_CONV1D_Y_UID,
            optionalUid(spec.bias, FLYDSL_CAUSAL_CONV1D_BIAS_UID),
            optionalUid(spec.initialState, FLYDSL_CAUSAL_CONV1D_INITIAL_STATE_UID),
            optionalUid(spec.poolRows.has_value(), FLYDSL_CAUSAL_CONV1D_STATE_INDICES_UID),
            optionalUid(spec.finalState, FLYDSL_CAUSAL_CONV1D_FINAL_STATE_UID),
            spec.activation);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "causal_conv1d",
                                           spec.computeDataType.value_or(spec.dataType),
                                           data_objects::NodeAttributes::CausalConv1dAttributes,
                                           attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_causal_conv1d_test");
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
