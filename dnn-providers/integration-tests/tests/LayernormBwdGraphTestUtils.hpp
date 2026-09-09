// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

inline flatbuffers::FlatBufferBuilder
    createLayernormBwdGraph(const int64_t dyUid,
                            const int64_t xUid,
                            const int64_t scaleUid,
                            const int64_t dxUid,
                            const int64_t dscaleUid,
                            const int64_t dbiasUid,
                            const std::optional<int64_t> epsilonUid,
                            const std::optional<int64_t> meanUid,
                            const std::optional<int64_t> invVarianceUid,
                            const std::vector<int64_t>& dyDims,
                            const std::vector<int64_t>& xDims,
                            const std::vector<int64_t>& scaleDims,
                            const std::vector<int64_t>& dxDims,
                            const std::vector<int64_t>& dscaleDims,
                            const std::vector<int64_t>& dbiasDims,
                            const std::optional<std::vector<int64_t>>& meanDims,
                            const std::optional<std::vector<int64_t>>& invVarianceDims,
                            const std::vector<int64_t>& dyStrides,
                            const std::vector<int64_t>& xStrides,
                            const std::vector<int64_t>& scaleStrides,
                            const std::vector<int64_t>& dxStrides,
                            const std::vector<int64_t>& dscaleStrides,
                            const std::vector<int64_t>& dbiasStrides,
                            const std::optional<std::vector<int64_t>>& meanStrides,
                            const std::optional<std::vector<int64_t>>& invVarianceStrides,
                            const std::optional<float> epsilon,
                            const int64_t normalizedDimCount,
                            const DataType dyDataType,
                            const DataType dxDataType,
                            const DataType scaleBiasDataType,
                            const std::optional<DataType> meanInvVarianceDataType,
                            const DataType computeDataType)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, dyUid, "dy", dyDataType, &dyStrides, &dyDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, xUid, "x", dxDataType, &xStrides, &xDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, scaleUid, "scale", scaleBiasDataType, &scaleStrides, &scaleDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, dxUid, "dx", dxDataType, &dxStrides, &dxDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, dscaleUid, "dscale", scaleBiasDataType, &dscaleStrides, &dscaleDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, dbiasUid, "dbias", scaleBiasDataType, &dbiasStrides, &dbiasDims));
    if(epsilonUid.has_value() && epsilon.has_value())
    {
        const std::vector<int64_t> epsilonDimsStrides = {1};
        tensors.push_back(CreateTensorAttributesDirect(
            builder,
            epsilonUid.value(),
            "epsilon",
            DataType::FLOAT,
            &epsilonDimsStrides,
            &epsilonDimsStrides,
            false,
            TensorValue::Float32Value,
            builder.CreateStruct(Float32Value(epsilon.value())).Union()));
    }
    if(meanUid.has_value() && meanDims.has_value() && meanStrides.has_value()
       && meanInvVarianceDataType.has_value())
    {
        tensors.push_back(CreateTensorAttributesDirect(builder,
                                                       meanUid.value(),
                                                       "mean",
                                                       meanInvVarianceDataType.value(),
                                                       &meanStrides.value(),
                                                       &meanDims.value()));
    }
    if(invVarianceUid.has_value() && invVarianceDims.has_value() && invVarianceStrides.has_value()
       && meanInvVarianceDataType.has_value())
    {
        tensors.push_back(CreateTensorAttributesDirect(builder,
                                                       invVarianceUid.value(),
                                                       "rstd",
                                                       meanInvVarianceDataType.value(),
                                                       &invVarianceStrides.value(),
                                                       &invVarianceDims.value()));
    }

    auto layernormBackwardAttrs = CreateLayernormBackwardAttributes(builder,
                                                                    dyUid,
                                                                    xUid,
                                                                    scaleUid,
                                                                    meanUid,
                                                                    invVarianceUid,
                                                                    epsilonUid,
                                                                    dxUid,
                                                                    dscaleUid,
                                                                    dbiasUid,
                                                                    normalizedDimCount);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "layernorm_bwd_node",
                                     computeDataType,
                                     NodeAttributes::LayernormBackwardAttributes,
                                     layernormBackwardAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "LayernormBwdTestGraph",
                                   computeDataType,
                                   scaleBiasDataType,
                                   dxDataType,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createLayernormBwdGraph(const int64_t dyUid,
                            const int64_t xUid,
                            const int64_t scaleUid,
                            const int64_t dxUid,
                            const int64_t dscaleUid,
                            const int64_t dbiasUid,
                            const std::optional<int64_t> epsilonUid,
                            const std::optional<int64_t> meanUid,
                            const std::optional<int64_t> invVarianceUid,
                            const std::vector<int64_t>& ioDims,
                            const TensorLayout& layout,
                            const float epsilon,
                            const int64_t normalizedDimCount,
                            const DataType dyDataType,
                            const DataType dxDataType,
                            const DataType scaleBiasDataType,
                            const std::optional<DataType> meanInvVarianceDataType,
                            const DataType computeDataType)
{
    const auto normalizedDim = static_cast<int64_t>(ioDims.size()) - normalizedDimCount;

    auto normDims = std::vector<int64_t>(ioDims.size(), 1);
    auto batchDims = std::vector<int64_t>(ioDims.size(), 1);
    for(size_t i = 0; i < ioDims.size(); ++i)
    {
        if(static_cast<int64_t>(i) < normalizedDim)
        {
            batchDims[i] = ioDims[i];
        }
        else
        {
            normDims[i] = ioDims[i];
        }
    }

    const auto ioStrides = generateStrides(ioDims, layout.strideOrder);
    const auto normStrides = generateStrides(normDims, layout.strideOrder);
    const auto batchStrides = generateStrides(batchDims, layout.strideOrder);

    return createLayernormBwdGraph(dyUid,
                                   xUid,
                                   scaleUid,
                                   dxUid,
                                   dscaleUid,
                                   dbiasUid,
                                   epsilonUid,
                                   meanUid,
                                   invVarianceUid,
                                   ioDims,
                                   ioDims,
                                   normDims,
                                   ioDims,
                                   normDims,
                                   normDims,
                                   batchDims,
                                   batchDims,
                                   ioStrides,
                                   ioStrides,
                                   normStrides,
                                   ioStrides,
                                   normStrides,
                                   normStrides,
                                   batchStrides,
                                   batchStrides,
                                   epsilon,
                                   normalizedDimCount,
                                   dyDataType,
                                   dxDataType,
                                   scaleBiasDataType,
                                   meanInvVarianceDataType,
                                   computeDataType);
}

} // namespace hipdnn_integration_tests::test_utils
