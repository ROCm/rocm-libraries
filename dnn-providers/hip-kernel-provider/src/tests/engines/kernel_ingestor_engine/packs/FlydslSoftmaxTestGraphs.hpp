// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_softmax's contract. One pack, so `operationMatcher` is empty:
/// `SoftmaxAttributes` is its own union arm. The operand tokens are p/stats/s.
inline constexpr PackSymbols FLYDSL_SOFTMAX{"hipkernel:flydsl_softmax",
                                            "hipkernel.flydsl_softmax.graph_match",
                                            "",
                                            "hipkernel.flydsl_softmax.kernel_match",
                                            "hipkernel.flydsl_softmax.score",
                                            "hipkernel.flydsl_softmax.dispatch",
                                            "flydsl_softmax.p.uid",
                                            "flydsl_softmax.stats.uid",
                                            "flydsl_softmax.s.uid"};

/// The tokens the matcher binds besides the operand uids.
constexpr const char* FLYDSL_SOFTMAX_COLUMNS_TOKEN = "flydsl_softmax.columns";
constexpr const char* FLYDSL_SOFTMAX_P_ROW_STRIDE_TOKEN = "flydsl_softmax.p_row_stride";
constexpr const char* FLYDSL_SOFTMAX_S_ROW_STRIDE_TOKEN = "flydsl_softmax.s_row_stride";

/// Tensor uids buildFlydslSoftmaxGraph() uses.
constexpr int64_t FLYDSL_SOFTMAX_P_UID = 1;
constexpr int64_t FLYDSL_SOFTMAX_S_UID = 2;
constexpr int64_t FLYDSL_SOFTMAX_STATS_UID = 3;

/// Row-major packed strides for @p dims.
inline std::vector<int64_t> flydslSoftmaxPackedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

/// @p dims with the last axis 1: the stats shape.
inline std::vector<int64_t> flydslSoftmaxStatsDims(std::vector<int64_t> dims)
{
    dims.back() = 1;
    return dims;
}

/// Everything a softmax graph varies, defaulting to Qwen3.8's router: bf16 logits of
/// [1, 7, 512], packed, s in p's dtype, no stats.
struct SoftmaxGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType pDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    /// Unset means p's dtype.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> sDataType;
    std::vector<int64_t> pDims{1, 7, 512};
    /// Unset means packed.
    std::optional<std::vector<int64_t>> pStrides;
    /// Unset means p's dims; strides unset mean p's strides when s has p's dims, else packed.
    std::optional<std::vector<int64_t>> sDims;
    std::optional<std::vector<int64_t>> sStrides;
    bool stats = false;
    bool statsLog2 = false;
    hipdnn_flatbuffers_sdk::data_objects::DataType statsDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    /// Unset means p's dims with the last 1, packed.
    std::optional<std::vector<int64_t>> statsDims;
    std::optional<std::vector<int64_t>> statsStrides;
    int64_t pAlignment = 16;
    int64_t sAlignment = 16;
    int64_t statsAlignment = 16;
    /// Unset means p's dtype.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> computeDataType;
    std::set<int64_t> virtualTensors;
    std::set<int64_t> passByValueTensors;
    bool secondNode = false;
    bool overrideShapeEnabled = false;
};

inline std::vector<int64_t> flydslSoftmaxPStrides(const SoftmaxGraphSpec& spec)
{
    return spec.pStrides.value_or(flydslSoftmaxPackedStrides(spec.pDims));
}

inline std::vector<int64_t> flydslSoftmaxSStrides(const SoftmaxGraphSpec& spec)
{
    if(spec.sStrides.has_value())
    {
        return *spec.sStrides;
    }
    return spec.sDims.has_value() ? flydslSoftmaxPackedStrides(*spec.sDims)
                                  : flydslSoftmaxPStrides(spec);
}

/// A single-node (or, with `secondNode`, two-node) softmax graph as @p spec describes it.
inline flatbuffers::FlatBufferBuilder buildFlydslSoftmaxGraph(const SoftmaxGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;

    const auto tensor = [&](int64_t uid,
                            const char* name,
                            data_objects::DataType type,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dims,
                            int64_t alignment) {
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
                                                          value,
                                                          false,
                                                          flatbuffers::nullopt,
                                                          alignment);
    };

    const auto sDims = spec.sDims.value_or(spec.pDims);
    const auto statsDims = spec.statsDims.value_or(flydslSoftmaxStatsDims(spec.pDims));
    const auto statsStrides = spec.statsStrides.value_or(flydslSoftmaxPackedStrides(statsDims));

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(tensor(FLYDSL_SOFTMAX_P_UID,
                             "p",
                             spec.pDataType,
                             flydslSoftmaxPStrides(spec),
                             spec.pDims,
                             spec.pAlignment));
    tensors.push_back(tensor(FLYDSL_SOFTMAX_S_UID,
                             "s",
                             spec.sDataType.value_or(spec.pDataType),
                             flydslSoftmaxSStrides(spec),
                             sDims,
                             spec.sAlignment));
    if(spec.stats)
    {
        tensors.push_back(tensor(FLYDSL_SOFTMAX_STATS_UID,
                                 "stats",
                                 spec.statsDataType,
                                 statsStrides,
                                 statsDims,
                                 spec.statsAlignment));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        const auto attributes = data_objects::CreateSoftmaxAttributes(
            builder,
            FLYDSL_SOFTMAX_P_UID,
            FLYDSL_SOFTMAX_S_UID,
            spec.stats ? flatbuffers::Optional<int64_t>(FLYDSL_SOFTMAX_STATS_UID)
                       : flatbuffers::nullopt,
            spec.statsLog2);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "softmax",
                                           spec.computeDataType.value_or(spec.pDataType),
                                           data_objects::NodeAttributes::SoftmaxAttributes,
                                           attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_softmax_test");
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
