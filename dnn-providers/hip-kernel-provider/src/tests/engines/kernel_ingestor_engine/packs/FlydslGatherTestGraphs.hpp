// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/gather_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_gather. One pack, so `operationMatcher` is empty: `GatherAttributes` is
/// its own union arm, and the graph matcher admits the node type itself.
inline constexpr PackSymbols FLYDSL_GATHER{"hipkernel:flydsl_gather",
                                           "hipkernel.flydsl_gather.graph_match",
                                           "",
                                           "hipkernel.flydsl_gather.kernel_match",
                                           "hipkernel.flydsl_gather.score",
                                           "hipkernel.flydsl_gather.dispatch",
                                           "flydsl_gather.x.uid",
                                           "flydsl_gather.indices.uid",
                                           "flydsl_gather.y.uid"};

/// The extents the graph match binds beside the three operand uids.
constexpr const char* FLYDSL_GATHER_NUM_INDICES_TOKEN = "flydsl_gather.num_indices";
constexpr const char* FLYDSL_GATHER_AXIS_EXTENT_TOKEN = "flydsl_gather.axis_extent";
constexpr const char* FLYDSL_GATHER_INNER_TOKEN = "flydsl_gather.inner";
constexpr const char* FLYDSL_GATHER_OUTER_TOKEN = "flydsl_gather.outer";

/// The fields the gather KMD declares.
constexpr const char* FLYDSL_GATHER_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_GATHER_INDEX_DTYPE_FIELD = "index_dtype";
constexpr const char* FLYDSL_GATHER_VEC_BYTES_FIELD = "vec_bytes";
constexpr const char* FLYDSL_GATHER_BLOCK_THREADS_FIELD = "block_threads";

/// Tensor uids buildFlydslGatherGraph() uses, in kernel argument order.
constexpr int64_t FLYDSL_GATHER_X_UID = 1;
constexpr int64_t FLYDSL_GATHER_INDICES_UID = 2;
constexpr int64_t FLYDSL_GATHER_Y_UID = 3;

/// Everything buildFlydslGatherGraph() varies. The defaults are the Qwen embedding lookup
/// at a small vocabulary: a bf16 `[64, 256]` table, INT32 token ids `[1, 8]`, axis 0.
struct GatherGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    hipdnn_flatbuffers_sdk::data_objects::DataType indexDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    std::vector<int64_t> xDims = {64, 256};
    std::vector<int64_t> indexDims = {1, 8};
    int64_t axis = 0;

    /// y's dtype, shape and every operand's strides; unset means x's dtype, the op's shape
    /// (x.dims[:axis] ++ indices.dims ++ x.dims[axis + 1:]) and packed strides.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> yDataType;
    std::optional<std::vector<int64_t>> yDims;
    std::optional<std::vector<int64_t>> xStrides;
    std::optional<std::vector<int64_t>> indexStrides;
    std::optional<std::vector<int64_t>> yStrides;

    /// Declared base alignment by uid; unset means the schema's 16.
    std::map<int64_t, int64_t> alignment;
    std::set<int64_t> virtualTensors;
    bool secondNode = false;
    bool overrideShapeEnabled = false;
};

/// y's shape as the op defines it.
inline std::vector<int64_t> flydslGatherOutputDims(const std::vector<int64_t>& xDims,
                                                   const std::vector<int64_t>& indexDims,
                                                   int64_t axis)
{
    std::vector<int64_t> dims(xDims.begin(), xDims.begin() + axis);
    dims.insert(dims.end(), indexDims.begin(), indexDims.end());
    dims.insert(dims.end(), xDims.begin() + axis + 1, xDims.end());
    return dims;
}

/// A single-node gather graph built from @p spec.
inline flatbuffers::FlatBufferBuilder buildFlydslGatherGraph(const GatherGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    const auto yDims
        = spec.yDims.value_or(spec.axis >= 0 && spec.axis < static_cast<int64_t>(spec.xDims.size())
                                  ? flydslGatherOutputDims(spec.xDims, spec.indexDims, spec.axis)
                                  : spec.xDims);
    const auto xStrides = spec.xStrides.value_or(flydslPackedStrides(spec.xDims));
    const auto indexStrides = spec.indexStrides.value_or(flydslPackedStrides(spec.indexDims));
    const auto yStrides = spec.yStrides.value_or(flydslPackedStrides(yDims));

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
    tensors.push_back(tensor(FLYDSL_GATHER_X_UID, "x", spec.dataType, xStrides, spec.xDims));
    tensors.push_back(tensor(
        FLYDSL_GATHER_INDICES_UID, "indices", spec.indexDataType, indexStrides, spec.indexDims));
    tensors.push_back(
        tensor(FLYDSL_GATHER_Y_UID, "y", spec.yDataType.value_or(spec.dataType), yStrides, yDims));

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        auto attributes = data_objects::CreateGatherAttributes(builder,
                                                               FLYDSL_GATHER_X_UID,
                                                               FLYDSL_GATHER_INDICES_UID,
                                                               FLYDSL_GATHER_Y_UID,
                                                               spec.axis);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "gather",
                                           spec.dataType,
                                           data_objects::NodeAttributes::GatherAttributes,
                                           attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_gather_test");
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
