// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The backward engine. The three named tokens are x, gamma and dx; the rest are bound
/// too and read by name below.
inline constexpr PackSymbols FLYDSL_RMSNORM_BWD{"hipkernel:flydsl_rmsnorm_bwd",
                                                "hipkernel.flydsl_rmsnorm_bwd.graph_match",
                                                "",
                                                "hipkernel.flydsl_rmsnorm_bwd.kernel_match",
                                                "hipkernel.flydsl_rmsnorm_bwd.score",
                                                "hipkernel.flydsl_rmsnorm_bwd.dispatch",
                                                "flydsl_rmsnorm_bwd.x.uid",
                                                "flydsl_rmsnorm_bwd.scale.uid",
                                                "flydsl_rmsnorm_bwd.dx.uid"};

constexpr const char* FLYDSL_BWD_DY_TOKEN = "flydsl_rmsnorm_bwd.dy.uid";
constexpr const char* FLYDSL_BWD_INV_RMS_TOKEN = "flydsl_rmsnorm_bwd.inv_rms.uid";
constexpr const char* FLYDSL_BWD_DSCALE_TOKEN = "flydsl_rmsnorm_bwd.dscale.uid";
constexpr const char* FLYDSL_BWD_ROWS_TOKEN = "flydsl_rmsnorm_bwd.rows";
constexpr const char* FLYDSL_BWD_COLUMNS_TOKEN = "flydsl_rmsnorm_bwd.columns";

/// KMD fields beside dtype, N and scale_f32 (shared with the forward).
constexpr const char* FLYDSL_N_MAX_FIELD = "n_max";
constexpr const char* FLYDSL_REDUCE_SYMBOL_FIELD = "reduce_symbol";
/// The reduce kernel's symbol every backward object carries; mirrored from
/// generators/_instances.py RMSNORM_BWD_REDUCE_SYMBOL.
constexpr const char* FLYDSL_BWD_REDUCE_SYMBOL = "rmsnorm_bwd_dweight_reduce_kernel_1";

/// Tensor uids buildFlydslRmsNormBwdGraph() uses.
constexpr int64_t FLYDSL_BWD_DY_UID = 11;
constexpr int64_t FLYDSL_BWD_X_UID = 12;
constexpr int64_t FLYDSL_BWD_SCALE_UID = 13;
constexpr int64_t FLYDSL_BWD_INV_RMS_UID = 14;
constexpr int64_t FLYDSL_BWD_DX_UID = 15;
constexpr int64_t FLYDSL_BWD_DSCALE_UID = 16;
constexpr int64_t FLYDSL_BWD_DBIAS_UID = 17;

/// Everything buildFlydslRmsNormBwdGraph() can vary, each for the case that pins the
/// matcher's reading of it. Unset means the shape the shipped objects serve.
struct RmsNormBwdGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    std::vector<int64_t> xDims{8, 4096};
    /// Gamma's shape; unset means x's innermost extent, rank-1.
    std::optional<std::vector<int64_t>> scaleDims;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> scaleDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dyDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dxDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> dscaleDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> invRmsDataType;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> computeDataType;
    std::optional<std::vector<int64_t>> invRmsDims;
    /// dy's and dx's shapes; unset means x's.
    std::optional<std::vector<int64_t>> dyDims;
    std::optional<std::vector<int64_t>> dxDims;
    /// dgamma's shape; unset means gamma's.
    std::optional<std::vector<int64_t>> dscaleDims;
    /// Strides by uid; unset means packed for the tensor's shape.
    std::map<int64_t, std::vector<int64_t>> strides;
    std::map<int64_t, int64_t> alignment;
    std::set<int64_t> virtualTensors;
    bool withDbias = false;
    bool overrideShapeEnabled = false;
    /// A second RMS-norm backward node after the first, over the same tensors.
    bool secondNode = false;
};

inline flatbuffers::FlatBufferBuilder buildFlydslRmsNormBwdGraph(const RmsNormBwdGraphSpec& spec
                                                                 = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    const auto& xDims = spec.xDims;
    const auto scaleDims = spec.scaleDims.value_or(std::vector<int64_t>{xDims.back()});
    const auto scaleType = spec.scaleDataType.value_or(spec.dataType);

    // inv_rms as the frontend shapes it: x's extent where gamma (aligned at the innermost
    // axis) is 1, and 1 where gamma spans the axis.
    std::vector<int64_t> defaultInvRmsDims(xDims.size(), 1);
    const auto offset = xDims.size() - std::min(xDims.size(), scaleDims.size());
    for(size_t axis = 0; axis < xDims.size(); ++axis)
    {
        const int64_t scaleExtent = axis < offset ? 1 : scaleDims[axis - offset];
        defaultInvRmsDims[axis] = scaleExtent == 1 ? xDims[axis] : 1;
    }
    const auto invRmsDims = spec.invRmsDims.value_or(defaultInvRmsDims);
    const auto dxDims = spec.dxDims.value_or(xDims);

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    const auto add = [&](int64_t uid,
                         const char* name,
                         data_objects::DataType type,
                         const std::vector<int64_t>& dims) {
        const auto found = spec.strides.find(uid);
        const auto strides
            = found == spec.strides.end() ? flydslPackedStrides(dims) : found->second;
        const auto declared = spec.alignment.find(uid);
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
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
            declared == spec.alignment.end() ? 16 : declared->second));
    };
    add(FLYDSL_BWD_DY_UID,
        "dy",
        spec.dyDataType.value_or(spec.dataType),
        spec.dyDims.value_or(xDims));
    add(FLYDSL_BWD_X_UID, "x", spec.dataType, xDims);
    add(FLYDSL_BWD_SCALE_UID, "scale", scaleType, scaleDims);
    add(FLYDSL_BWD_INV_RMS_UID,
        "inv_rms",
        spec.invRmsDataType.value_or(data_objects::DataType::FLOAT),
        invRmsDims);
    add(FLYDSL_BWD_DX_UID, "dx", spec.dxDataType.value_or(spec.dataType), dxDims);
    add(FLYDSL_BWD_DSCALE_UID,
        "dscale",
        spec.dscaleDataType.value_or(scaleType),
        spec.dscaleDims.value_or(scaleDims));
    if(spec.withDbias)
    {
        add(FLYDSL_BWD_DBIAS_UID, "dbias", scaleType, scaleDims);
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        auto attributes = data_objects::CreateRMSNormBackwardAttributes(
            builder,
            FLYDSL_BWD_DY_UID,
            FLYDSL_BWD_X_UID,
            FLYDSL_BWD_SCALE_UID,
            FLYDSL_BWD_INV_RMS_UID,
            FLYDSL_BWD_DX_UID,
            FLYDSL_BWD_DSCALE_UID,
            spec.withDbias ? flatbuffers::Optional<int64_t>(FLYDSL_BWD_DBIAS_UID)
                           : flatbuffers::nullopt);
        nodes.push_back(data_objects::CreateNodeDirect(
            builder,
            "rmsnorm_backward",
            spec.computeDataType.value_or(data_objects::DataType::FLOAT),
            data_objects::NodeAttributes::RMSNormBackwardAttributes,
            attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_rmsnorm_bwd_test");
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
