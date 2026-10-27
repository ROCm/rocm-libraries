#pragma once

#include <cstdint>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_integration_tests::test_utils
{

// Creates a minimal graph with a ConvolutionWrw node and separate input/output data types.
// Uses symmetric padding (prePadding == postPadding == padding).
// inputDataType applies to x and dy tensors; outputDataType applies to dw tensor.
inline flatbuffers::FlatBufferBuilder
    createConvWrwGraph(int64_t xUid,
                       int64_t dyUid,
                       int64_t dwUid,
                       const std::vector<int64_t>& xDims,
                       const std::vector<int64_t>& dyDims,
                       const std::vector<int64_t>& dwDims,
                       const std::vector<int64_t>& xStrides,
                       const std::vector<int64_t>& dyStrides,
                       const std::vector<int64_t>& dwStrides,
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
        CreateTensorAttributesDirect(builder, dyUid, "dy", inputDataType, &dyStrides, &dyDims));
    tensors.push_back(
        CreateTensorAttributesDirect(builder, dwUid, "dw", outputDataType, &dwStrides, &dwDims));

    auto convWrwAttrs = CreateConvolutionWrwAttributesDirect(builder,
                                                             xUid,
                                                             dyUid,
                                                             dwUid,
                                                             &padding,
                                                             &padding,
                                                             &convStride,
                                                             &dilation,
                                                             ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "conv_wrw_node",
                                     DataType::FLOAT,
                                     NodeAttributes::ConvolutionWrwAttributes,
                                     convWrwAttrs.Union()));

    auto graph = CreateGraphDirect(builder,
                                   "ConvWrwTestGraph",
                                   inputDataType,
                                   outputDataType,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);
    return builder;
}

// Convenience overload: single dataType used for all tensors (x, dy, and dw).
inline flatbuffers::FlatBufferBuilder
    createConvWrwGraph(int64_t xUid,
                       int64_t dyUid,
                       int64_t dwUid,
                       const std::vector<int64_t>& xDims,
                       const std::vector<int64_t>& dyDims,
                       const std::vector<int64_t>& dwDims,
                       const std::vector<int64_t>& xStrides,
                       const std::vector<int64_t>& dyStrides,
                       const std::vector<int64_t>& dwStrides,
                       const std::vector<int64_t>& padding,
                       const std::vector<int64_t>& convStride,
                       const std::vector<int64_t>& dilation,
                       hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
{
    return createConvWrwGraph(xUid,
                              dyUid,
                              dwUid,
                              xDims,
                              dyDims,
                              dwDims,
                              xStrides,
                              dyStrides,
                              dwStrides,
                              padding,
                              convStride,
                              dilation,
                              dataType,
                              dataType);
}

} // namespace hipdnn_integration_tests::test_utils
