#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

// Creates a minimal graph with a ConvolutionBwd node and separate input/output data types.
// Uses symmetric padding (prePadding == postPadding == padding).
// inputDataType applies to dy and w tensors; outputDataType applies to dx tensor.
inline flatbuffers::FlatBufferBuilder
    createConvBwdGraph(int64_t dyUid,
                       int64_t wUid,
                       int64_t dxUid,
                       const std::vector<int64_t>& dyDims,
                       const std::vector<int64_t>& wDims,
                       const std::vector<int64_t>& dxDims,
                       const std::vector<int64_t>& dyStrides,
                       const std::vector<int64_t>& wStrides,
                       const std::vector<int64_t>& dxStrides,
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
        CreateTensorAttributesDirect(builder, dyUid, "dy", inputDataType, &dyStrides, &dyDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, wUid, "w", inputDataType, &wStrides, &wDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, dxUid, "dx", outputDataType, &dxStrides, &dxDims));

    auto convBwdAttrs = CreateConvolutionBwdAttributesDirect(builder,
                                                             dyUid,
                                                             wUid,
                                                             dxUid,
                                                             &padding,
                                                             &padding,
                                                             &convStride,
                                                             &dilation,
                                                             ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "conv_bwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::ConvolutionBwdAttributes,
                                     convBwdAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "ConvBwdTestGraph",
                                   inputDataType,
                                   outputDataType,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

// Convenience overload: single dataType used for all tensors (dy, w, and dx).
inline flatbuffers::FlatBufferBuilder
    createConvBwdGraph(int64_t dyUid,
                       int64_t wUid,
                       int64_t dxUid,
                       const std::vector<int64_t>& dyDims,
                       const std::vector<int64_t>& wDims,
                       const std::vector<int64_t>& dxDims,
                       const std::vector<int64_t>& dyStrides,
                       const std::vector<int64_t>& wStrides,
                       const std::vector<int64_t>& dxStrides,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& convStride,
                       const std::vector<int64_t>& dilation,
                       hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    return createConvBwdGraph(dyUid,
                              wUid,
                              dxUid,
                              dyDims,
                              wDims,
                              dxDims,
                              dyStrides,
                              wStrides,
                              dxStrides,
                              padding,
                              convStride,
                              dilation,
                              dataType,
                              dataType);
}

} // namespace hipdnn_integration_tests::test_utils
