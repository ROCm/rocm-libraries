// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

// Use the canonical implementation from data_sdk.
using hipdnn_data_sdk::utilities::generateStrides;

// Creates a minimal graph with a ConvolutionFwd node and separate input/output data types.
// Uses symmetric padding (prePadding == postPadding == padding).
// inputDataType applies to x and w tensors; outputDataType applies to y tensor.
inline flatbuffers::FlatBufferBuilder
    createConvFwdGraph(int64_t xUid,
                       int64_t wUid,
                       int64_t yUid,
                       const std::vector<int64_t>& xDims,
                       const std::vector<int64_t>& wDims,
                       const std::vector<int64_t>& yDims,
                       const std::vector<int64_t>& xStrides,
                       const std::vector<int64_t>& wStrides,
                       const std::vector<int64_t>& yStrides,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& convStride,
                       const std::vector<int64_t>& dilation,
                       hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                       hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.push_back(
        CreateTensorAttributesDirect(builder, xUid, "x", inputDataType, &xStrides, &xDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, wUid, "w", inputDataType, &wStrides, &wDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, yUid, "y", outputDataType, &yStrides, &yDims));

    auto convAttrs = CreateConvolutionFwdAttributesDirect(builder,
                                                          xUid,
                                                          wUid,
                                                          yUid,
                                                          &padding,
                                                          &padding,
                                                          &convStride,
                                                          &dilation,
                                                          ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "conv_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::ConvolutionFwdAttributes,
                                     convAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "ConvFwdTestGraph",
                                   inputDataType,
                                   outputDataType,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

// Convenience overload: single dataType used for all tensors (x, w, and y).
inline flatbuffers::FlatBufferBuilder
    createConvFwdGraph(int64_t xUid,
                       int64_t wUid,
                       int64_t yUid,
                       const std::vector<int64_t>& xDims,
                       const std::vector<int64_t>& wDims,
                       const std::vector<int64_t>& yDims,
                       const std::vector<int64_t>& xStrides,
                       const std::vector<int64_t>& wStrides,
                       const std::vector<int64_t>& yStrides,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& convStride,
                       const std::vector<int64_t>& dilation,
                       hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    return createConvFwdGraph(xUid,
                              wUid,
                              yUid,
                              xDims,
                              wDims,
                              yDims,
                              xStrides,
                              wStrides,
                              yStrides,
                              padding,
                              convStride,
                              dilation,
                              dataType,
                              dataType);
}

} // namespace hipdnn_integration_tests::test_utils
