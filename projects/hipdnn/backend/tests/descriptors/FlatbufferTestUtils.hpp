// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>
#include <string>
#include <vector>

namespace hipdnn_backend::test_utilities
{

/// Graph with metadata but no tensors or nodes, for testing empty-graph error paths.
inline flatbuffers::FlatBufferBuilder createEmptyGraph()
{
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;
    const std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidGraph()
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;

    using namespace hipdnn_tests::constants;

    // Build tensors from shared test constants
    TensorAttributesT xTensor;
    xTensor.uid = K_FPROP_TENSOR_X_UID;
    xTensor.data_type = DataType::FLOAT;
    xTensor.dims = hipdnn_tests::toVec(K_FPROP_TENSOR_X_DIMS);
    xTensor.strides = hipdnn_tests::toVec(K_FPROP_TENSOR_X_STRIDES);

    TensorAttributesT wTensor;
    wTensor.uid = K_FPROP_TENSOR_W_UID;
    wTensor.data_type = DataType::FLOAT;
    wTensor.dims = hipdnn_tests::toVec(K_FPROP_TENSOR_W_DIMS);
    wTensor.strides = hipdnn_tests::toVec(K_FPROP_TENSOR_W_STRIDES);

    TensorAttributesT yTensor;
    yTensor.uid = K_FPROP_TENSOR_Y_UID;
    yTensor.data_type = DataType::FLOAT;
    yTensor.dims = hipdnn_tests::toVec(K_FPROP_TENSOR_Y_DIMS);
    yTensor.strides = hipdnn_tests::toVec(K_FPROP_TENSOR_Y_STRIDES);

    std::vector<flatbuffers::Offset<TensorAttributes>> tensorOffsets;
    tensorOffsets.push_back(TensorAttributes::Pack(builder, &xTensor));
    tensorOffsets.push_back(TensorAttributes::Pack(builder, &wTensor));
    tensorOffsets.push_back(TensorAttributes::Pack(builder, &yTensor));

    // Build node with conv attributes
    ConvolutionFwdAttributesT convAttrs;
    convAttrs.x_tensor_uid = K_FPROP_TENSOR_X_UID;
    convAttrs.w_tensor_uid = K_FPROP_TENSOR_W_UID;
    convAttrs.y_tensor_uid = K_FPROP_TENSOR_Y_UID;
    convAttrs.pre_padding = hipdnn_tests::toVec(K_FPROP_CONV_PADDING);
    convAttrs.post_padding = hipdnn_tests::toVec(K_FPROP_CONV_PADDING);
    convAttrs.stride = hipdnn_tests::toVec(K_FPROP_CONV_STRIDE);
    convAttrs.dilation = hipdnn_tests::toVec(K_FPROP_CONV_DILATION);
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
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

} // namespace hipdnn_backend::test_utilities
