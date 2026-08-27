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
    createLayernormFwdGraph(const int64_t xUid,
                            const int64_t yUid,
                            const int64_t scaleUid,
                            const int64_t biasUid,
                            const int64_t epsilonUid,
                            const std::optional<int64_t> meanUid,
                            const std::optional<int64_t> invVarianceUid,
                            const std::vector<int64_t>& xDims,
                            const std::vector<int64_t>& yDims,
                            const std::vector<int64_t>& scaleDims,
                            const std::vector<int64_t>& biasDims,
                            const std::optional<std::vector<int64_t>>& meanDims,
                            const std::optional<std::vector<int64_t>>& invVarianceDims,
                            const std::vector<int64_t>& xStrides,
                            const std::vector<int64_t>& yStrides,
                            const std::vector<int64_t>& scaleStrides,
                            const std::vector<int64_t>& biasStrides,
                            const std::optional<std::vector<int64_t>>& meanStrides,
                            const std::optional<std::vector<int64_t>>& invVarianceStrides,
                            const double epsilon,
                            const int64_t normalizedDimCount,
                            const DataType xDataType,
                            const DataType yDataType,
                            const DataType scaleBiasDataType,
                            const std::optional<DataType> meanInvVarianceDataType,
                            const DataType computeDataType)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, xUid, "x", xDataType, &xStrides, &xDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, yUid, "y", yDataType, &yStrides, &yDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, scaleUid, "scale", scaleBiasDataType, &scaleStrides, &scaleDims));
    tensors.push_back(CreateTensorAttributesDirect(
        builder, biasUid, "bias", scaleBiasDataType, &biasStrides, &biasDims));
    const std::vector<int64_t> epsilonDimsStrides = {1};
    tensors.push_back(
        CreateTensorAttributesDirect(builder,
                                     epsilonUid,
                                     "epsilon",
                                     DataType::DOUBLE,
                                     &epsilonDimsStrides,
                                     &epsilonDimsStrides,
                                     false,
                                     TensorValue::Float64Value,
                                     builder.CreateStruct(Float64Value(epsilon)).Union()));
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

    auto layernormAttrs = CreateLayernormAttributes(
        builder,
        xUid,
        scaleUid,
        biasUid,
        epsilonUid,
        yUid,
        normalizedDimCount,
        meanUid,
        invVarianceUid,
        meanUid.has_value() && invVarianceUid.has_value() ? NormFwdPhase::TRAINING
                                                          : NormFwdPhase::INFERENCE);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "layernorm_fwd_node",
                                     computeDataType,
                                     NodeAttributes::LayernormAttributes,
                                     layernormAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "LayernormFwdTestGraph",
                                   computeDataType,
                                   scaleBiasDataType,
                                   xDataType,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createLayernormFwdGraph(const int64_t xUid,
                            const int64_t yUid,
                            const int64_t scaleUid,
                            const int64_t biasUid,
                            const int64_t epsilonUid,
                            const std::optional<int64_t> meanUid,
                            const std::optional<int64_t> invVarianceUid,
                            const std::vector<int64_t>& ioDims,
                            const TensorLayout& layout,
                            const double epsilon,
                            const int64_t normalizedDimCount,
                            const DataType xDataType,
                            const DataType yDataType,
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

    return createLayernormFwdGraph(xUid,
                                   yUid,
                                   scaleUid,
                                   biasUid,
                                   epsilonUid,
                                   meanUid,
                                   invVarianceUid,
                                   ioDims,
                                   ioDims,
                                   normDims,
                                   normDims,
                                   batchDims,
                                   batchDims,
                                   ioStrides,
                                   ioStrides,
                                   normStrides,
                                   normStrides,
                                   batchStrides,
                                   batchStrides,
                                   epsilon,
                                   normalizedDimCount,
                                   xDataType,
                                   yDataType,
                                   scaleBiasDataType,
                                   meanInvVarianceDataType,
                                   computeDataType);
}

} // namespace hipdnn_integration_tests::test_utils
