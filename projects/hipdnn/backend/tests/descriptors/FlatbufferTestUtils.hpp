// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <string>
#include <vector>

namespace hipdnn_backend::test_utilities
{

/// Graph with metadata but no tensors or nodes, for testing empty-graph error paths.
inline flatbuffers::FlatBufferBuilder createEmptyGraph()
{
    const std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    const std::vector<::flatbuffers::Offset<hipdnn_data_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_data_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_data_sdk::data_objects::DataType::FLOAT,
        hipdnn_data_sdk::data_objects::DataType::HALF,
        hipdnn_data_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidGraph()
{
    using namespace hipdnn_data_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;

    // Build tensors
    TensorAttributesT xTensor;
    xTensor.uid = 1;
    xTensor.data_type = DataType::FLOAT;
    xTensor.dims = {1, 3, 4, 4};
    xTensor.strides = {48, 16, 4, 1};

    TensorAttributesT wTensor;
    wTensor.uid = 2;
    wTensor.data_type = DataType::FLOAT;
    wTensor.dims = {3, 3, 3, 3};
    wTensor.strides = {27, 9, 3, 1};

    TensorAttributesT yTensor;
    yTensor.uid = 3;
    yTensor.data_type = DataType::FLOAT;
    yTensor.dims = {1, 3, 2, 2};
    yTensor.strides = {12, 4, 2, 1};

    std::vector<flatbuffers::Offset<TensorAttributes>> tensorOffsets;
    tensorOffsets.push_back(TensorAttributes::Pack(builder, &xTensor));
    tensorOffsets.push_back(TensorAttributes::Pack(builder, &wTensor));
    tensorOffsets.push_back(TensorAttributes::Pack(builder, &yTensor));

    // Build node with conv attributes
    ConvolutionFwdAttributesT convAttrs;
    convAttrs.x_tensor_uid = 1;
    convAttrs.w_tensor_uid = 2;
    convAttrs.y_tensor_uid = 3;
    convAttrs.pre_padding = {0, 0};
    convAttrs.post_padding = {0, 0};
    convAttrs.stride = {1, 1};
    convAttrs.dilation = {1, 1};
    convAttrs.conv_mode = ConvMode::CROSS_CORRELATION;

    NodeT nodeT;
    nodeT.compute_data_type = DataType::FLOAT;
    nodeT.attributes.Set(ConvolutionFwdAttributesT(convAttrs));

    std::vector<flatbuffers::Offset<Node>> nodeOffsets;
    nodeOffsets.push_back(Node::Pack(builder, &nodeT));

    // Build graph
    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::HALF,
                                         DataType::BFLOAT16,
                                         &tensorOffsets,
                                         &nodeOffsets);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidEngineDetails(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetailsOffset
        = hipdnn_data_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

} // namespace hipdnn_backend::test_utilities
