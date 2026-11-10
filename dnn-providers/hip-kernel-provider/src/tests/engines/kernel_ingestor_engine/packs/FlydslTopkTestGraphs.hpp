// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_topk. One pack, so `operationMatcher` is empty: `TopkAttributes` is its
/// own union arm, and the graph matcher admits the node type itself.
inline constexpr PackSymbols FLYDSL_TOPK{"hipkernel:flydsl_topk",
                                         "hipkernel.flydsl_topk.graph_match",
                                         "",
                                         "hipkernel.flydsl_topk.kernel_match",
                                         "hipkernel.flydsl_topk.score",
                                         "hipkernel.flydsl_topk.dispatch",
                                         "flydsl_topk.x.uid",
                                         "flydsl_topk.values.uid",
                                         "flydsl_topk.indices.uid"};

/// What the graph match binds beside the operand uids.
constexpr const char* FLYDSL_TOPK_ROWS_TOKEN = "flydsl_topk.rows";
constexpr const char* FLYDSL_TOPK_COLUMNS_TOKEN = "flydsl_topk.columns";
constexpr const char* FLYDSL_TOPK_K_TOKEN = "flydsl_topk.k";
constexpr const char* FLYDSL_TOPK_LARGEST_TOKEN = "flydsl_topk.largest";

/// The fields the top-k KMD declares.
constexpr const char* FLYDSL_TOPK_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_TOPK_INDEX_DTYPE_FIELD = "index_dtype";
constexpr const char* FLYDSL_TOPK_K_MAX_FIELD = "k_max";
constexpr const char* FLYDSL_TOPK_BLOCK_THREADS_FIELD = "block_threads";

/// The general objects' k bound (`TOPK_GENERAL_K_MAX` in flydsl/generators/_instances.py).
constexpr int64_t FLYDSL_TOPK_GENERAL_K_MAX = 1024;

/// Tensor uids buildFlydslTopkGraph() uses.
constexpr int64_t FLYDSL_TOPK_X_UID = 1;
constexpr int64_t FLYDSL_TOPK_INDICES_UID = 2;
constexpr int64_t FLYDSL_TOPK_VALUES_UID = 3;

/// Everything buildFlydslTopkGraph() varies. The defaults are the MoE router: bf16 logits
/// `[16, 512]`, k 8, largest, INT32 indices and values.
struct TopkGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    std::vector<int64_t> xDims = {16, 512};
    int64_t k = 8;
    bool largest = true;
    bool withValues = true;

    /// The outputs' dtype and shape and every operand's strides; unset means x's dtype for
    /// values, x's dims with the last k, and packed strides.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> valuesDataType;
    std::optional<std::vector<int64_t>> indicesDims;
    std::optional<std::vector<int64_t>> valuesDims;
    std::optional<std::vector<int64_t>> xStrides;
    std::optional<std::vector<int64_t>> indicesStrides;
    std::optional<std::vector<int64_t>> valuesStrides;

    /// Declared base alignment by uid; unset means the schema's 16.
    std::map<int64_t, int64_t> alignment;
    std::set<int64_t> virtualTensors;
    bool secondNode = false;
    bool overrideShapeEnabled = false;
};

/// x's dims with the last one @p k.
inline std::vector<int64_t> flydslTopkOutputDims(std::vector<int64_t> xDims, int64_t k)
{
    if(!xDims.empty())
    {
        xDims.back() = k;
    }
    return xDims;
}

/// A single-node top-k graph built from @p spec.
inline flatbuffers::FlatBufferBuilder buildFlydslTopkGraph(const TopkGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    const auto outDims = flydslTopkOutputDims(spec.xDims, spec.k);
    const auto indicesDims = spec.indicesDims.value_or(outDims);
    const auto valuesDims = spec.valuesDims.value_or(outDims);
    const auto xStrides = spec.xStrides.value_or(flydslPackedStrides(spec.xDims));
    const auto indicesStrides = spec.indicesStrides.value_or(flydslPackedStrides(indicesDims));
    const auto valuesStrides = spec.valuesStrides.value_or(flydslPackedStrides(valuesDims));

    flatbuffers::FlatBufferBuilder builder;
    const auto tensor = [&](int64_t uid,
                            const char* name,
                            data_objects::DataType type,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dims) {
        const auto declared = spec.alignment.find(uid);
        return data_objects::CreateTensorAttributesDirect(
            builder,
            uid,
            name,
            type,
            &strides,
            &dims,
            spec.virtualTensors.count(uid) != 0,
            data_objects::TensorValue::NONE,
            0,
            false,
            flatbuffers::nullopt,
            declared == spec.alignment.end() ? 16 : declared->second);
    };

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(tensor(FLYDSL_TOPK_X_UID, "x", spec.dataType, xStrides, spec.xDims));
    tensors.push_back(tensor(
        FLYDSL_TOPK_INDICES_UID, "indices", spec.indexDataType, indicesStrides, indicesDims));
    if(spec.withValues)
    {
        tensors.push_back(tensor(FLYDSL_TOPK_VALUES_UID,
                                 "values",
                                 spec.valuesDataType.value_or(spec.dataType),
                                 valuesStrides,
                                 valuesDims));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        auto attributes = data_objects::CreateTopkAttributes(
            builder,
            FLYDSL_TOPK_X_UID,
            FLYDSL_TOPK_INDICES_UID,
            spec.withValues ? flatbuffers::Optional<int64_t>(FLYDSL_TOPK_VALUES_UID)
                            : flatbuffers::nullopt,
            spec.k,
            spec.largest);
        nodes.push_back(data_objects::CreateNodeDirect(builder,
                                                       "topk",
                                                       spec.computeDataType,
                                                       data_objects::NodeAttributes::TopkAttributes,
                                                       attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_topk_test");
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
