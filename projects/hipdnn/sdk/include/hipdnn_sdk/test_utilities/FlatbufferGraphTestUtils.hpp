// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>

namespace hipdnn_sdk::test_utilities
{

using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

inline flatbuffers::FlatBufferBuilder createEmptyValidGraph()
{
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    flatbuffers::FlatBufferBuilder builder;
    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType::FLOAT,
                                                                   DataType::HALF,
                                                                   DataType::BFLOAT16,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidBatchnormInferenceGraph(const std::vector<int64_t>& strides = {1, 3, 224, 224},
                                       const std::vector<int64_t>& dims = {1, 3, 224, 224},
                                       hipdnn_sdk::data_objects::DataType inputDataType
                                       = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedStrides = getDerivedShape(strides);
    std::vector<int64_t> derivedDims = getDerivedShape(dims);

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "y", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "est_mean",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "est_variance",
        hipdnn_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    auto bnormAttributes
        = hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                       1, // x uid
                                                                       5, // mean uid
                                                                       6, // inv_variance uid
                                                                       3, // scale uid
                                                                       4, // bias uid
                                                                       2 // y uid
        );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType::FLOAT,
                                                                   DataType::HALF,
                                                                   DataType::BFLOAT16,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder createValidBatchnormBwdGraph(
    const std::vector<int64_t>& strides = {1, 3, 224, 224},
    const std::vector<int64_t>& dims = {1, 3, 224, 224},
    bool hasOptionalAttributes = true,
    hipdnn_sdk::data_objects::DataType inputDataType = DataType::FLOAT,
    hipdnn_sdk::data_objects::DataType scaleBiasDataType = DataType::FLOAT,
    hipdnn_sdk::data_objects::DataType meanVarianceDataType = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::TensorAttributes>> tensorAttributes;

    std::vector<int64_t> derivedDims = getDerivedShape(dims);
    std::vector<int64_t> derivedStrides = generateStrides(derivedDims);

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "dy", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "dx", inputDataType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "scale", scaleBiasDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "dscale", scaleBiasDataType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "dbias", scaleBiasDataType, &derivedStrides, &derivedDims));

    if(hasOptionalAttributes)
    {
        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 7, "mean", meanVarianceDataType, &derivedStrides, &derivedDims));

        tensorAttributes.push_back(hipdnn_sdk::data_objects::CreateTensorAttributesDirect(
            builder, 8, "inv_variance", meanVarianceDataType, &derivedStrides, &derivedDims));
    }

    auto bnormAttributes = hipdnn_sdk::data_objects::CreateBatchnormBackwardAttributes(
        builder,
        2, // dy_tensor_uid
        1, // x_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(7)
                              : flatbuffers::nullopt, // mean_tensor_uid
        hasOptionalAttributes ? flatbuffers::Optional<int64_t>(8)
                              : flatbuffers::nullopt, // inv_variance_tensor_uid
        4, // scale_tensor_uid
        flatbuffers::Offset<flatbuffers::Vector<int64_t>>(), // peer_stats_tensor_uid
        3, // dx_tensor_uid
        5, // dscale_tensor_uid
        6 // dbias_tensor_uid
    );

    std::vector<::flatbuffers::Offset<hipdnn_sdk::data_objects::Node>> nodes;
    auto node = hipdnn_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_bwd",
        hipdnn_sdk::data_objects::NodeAttributes::BatchnormBackwardAttributes,
        bnormAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = hipdnn_sdk::data_objects::CreateGraphDirect(builder,
                                                                   "test",
                                                                   DataType::FLOAT,
                                                                   DataType::HALF,
                                                                   DataType::BFLOAT16,
                                                                   &tensorAttributes,
                                                                   &nodes);
    builder.Finish(graphOffset);
    return builder;
}

// TODO: Replace with a createValidBatchnormGraph function once one is made and tested
// This may be useful to keep in general though, as it has distinct and non-null values for all fields
inline flatbuffers::FlatBufferBuilder createBatchnormGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<Node>> nodes;
    std::vector<int64_t> peerStats = {-1, -2, -3, -4};
    auto batchnormNode = CreateBatchnormAttributesDirect(
        builder, 0, 1, 2, 3, &peerStats, 4, 5, 6, 7, 8, 9, 10, 11);
    nodes.push_back(CreateNodeDirect(
        builder, "Node", NodeAttributes::BatchnormAttributes, batchnormNode.Union()));

    std::array tensorNames = {"x",
                              "scale",
                              "bias",
                              "epsilon",
                              "peer_stats",
                              "prev_running_mean",
                              "momentum",
                              "y",
                              "mean",
                              "inv_variance",
                              "next_running_mean",
                              "next_running_variance"};
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.reserve(tensorNames.size());
    int64_t tensorUid = 0;
    std::vector<int64_t> dims = {1, 2, 3, 4};
    std::vector<int64_t> strides = {5, 6, 7, 8};
    for(auto name : tensorNames)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            builder, tensorUid++, name, DataType::UINT8, &strides, &dims, false));
    }

    auto graph = CreateGraphDirect(builder,
                                   "BatchnormGraph",
                                   DataType::FLOAT,
                                   DataType::HALF,
                                   DataType::BFLOAT16,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);

    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidConvFwdGraph(const std::vector<int64_t>& xDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& wDims = {4, 4, 1, 1},
                            const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
                            const std::vector<int64_t>& yDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& yStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& convPrePadding = {0, 0},
                            const std::vector<int64_t>& convPostPadding = {0, 0},
                            const std::vector<int64_t>& convStrides = {1, 1},
                            const std::vector<int64_t>& convDilation = {1, 1},
                            DataType dataType = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 3, "y", dataType, &yStrides, &yDims));

    auto convAttributes = CreateConvolutionFwdAttributesDirect(builder,
                                                               1, // x tensor uid
                                                               2, // w tensor uid
                                                               3, // y tensor uid
                                                               &convPrePadding,
                                                               &convPostPadding,
                                                               &convStrides,
                                                               &convDilation,
                                                               ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<Node>> nodes;
    auto node = CreateNodeDirect(
        builder, "conv_fwd", NodeAttributes::ConvolutionFwdAttributes, convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidConvBwdGraph(const std::vector<int64_t>& dxDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& dxStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& wDims = {4, 4, 1, 1},
                            const std::vector<int64_t>& wStrides = {4, 1, 1, 1},
                            const std::vector<int64_t>& dyDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& dyStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& convPrePadding = {0, 0},
                            const std::vector<int64_t>& convPostPadding = {0, 0},
                            const std::vector<int64_t>& convStrides = {1, 1},
                            const std::vector<int64_t>& convDilation = {1, 1},
                            DataType dataType = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 1, "dy", dataType, &dyStrides, &dyDims));

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 2, "w", dataType, &wStrides, &wDims));

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 3, "dx", dataType, &dxStrides, &dxDims));

    auto convAttributes = CreateConvolutionBwdAttributesDirect(builder,
                                                               1, // dy tensor uid
                                                               2, // w tensor uid
                                                               3, // dx tensor uid
                                                               &convPrePadding,
                                                               &convPostPadding,
                                                               &convStrides,
                                                               &convDilation,
                                                               ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<Node>> nodes;
    auto node = CreateNodeDirect(
        builder, "conv_bwd", NodeAttributes::ConvolutionBwdAttributes, convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

inline flatbuffers::FlatBufferBuilder
    createValidConvWrwGraph(const std::vector<int64_t>& xDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& xStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& dwDims = {4, 4, 1, 1},
                            const std::vector<int64_t>& dwStrides = {4, 1, 1, 1},
                            const std::vector<int64_t>& dyDims = {4, 4, 4, 4},
                            const std::vector<int64_t>& dyStrides = {64, 16, 4, 1},
                            const std::vector<int64_t>& convPrePadding = {0, 0},
                            const std::vector<int64_t>& convPostPadding = {0, 0},
                            const std::vector<int64_t>& convStrides = {1, 1},
                            const std::vector<int64_t>& convDilation = {1, 1},
                            DataType dataType = DataType::FLOAT)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 1, "x", dataType, &xStrides, &xDims));

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 2, "dy", dataType, &dyStrides, &dyDims));

    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, 3, "dw", dataType, &dwStrides, &dwDims));

    auto convAttributes = CreateConvolutionWrwAttributesDirect(builder,
                                                               1, // x tensor uid
                                                               2, // dy tensor uid
                                                               3, // w tensor uid
                                                               &convPrePadding,
                                                               &convPostPadding,
                                                               &convStrides,
                                                               &convDilation,
                                                               ConvMode::CROSS_CORRELATION);

    std::vector<::flatbuffers::Offset<Node>> nodes;
    auto node = CreateNodeDirect(
        builder, "conv_wrw", NodeAttributes::ConvolutionWrwAttributes, convAttributes.Union());
    nodes.push_back(node);

    auto graphOffset = CreateGraphDirect(builder,
                                         "test",
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         &tensorAttributes,
                                         &nodes);
    builder.Finish(graphOffset);
    return builder;
}

// TODO: Replace with a createValidPointwiseGraph function once one is made and tested
// This may be useful to keep in general though, as it has distinct and non-null values for all fields
inline flatbuffers::FlatBufferBuilder createPointwiseGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<flatbuffers::Offset<Node>> nodes;
    auto pointwiseNode = CreatePointwiseAttributes(builder,
                                                   PointwiseMode::DIV, // operation
                                                   1.f, // relu_lower_clip
                                                   2.f, // relu_upper_clip
                                                   3.f, // relu_lower_clip_slope
                                                   0, // axis_tensor_uid
                                                   1, // in_0_tensor_uid
                                                   2, // in_1_tensor_uid
                                                   3, // in_2_tensor_uid
                                                   4, // out_0_tensor_uid
                                                   4.f, // swish_beta
                                                   5.f, // elu_alpha
                                                   6.f); // softplus_beta

    nodes.push_back(CreateNodeDirect(
        builder, "Node", NodeAttributes::PointwiseAttributes, pointwiseNode.Union()));

    std::array tensorNames = {"axis", "in_0", "in_1", "in_2", "out_0"};
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    tensors.reserve(tensorNames.size());
    int64_t tensorUid = 0;
    std::vector<int64_t> dims = {1, 2, 3, 4};
    std::vector<int64_t> strides = {5, 6, 7, 8};
    for(auto name : tensorNames)
    {
        tensors.push_back(CreateTensorAttributesDirect(
            builder, tensorUid++, name, DataType::UINT8, &strides, &dims, false));
    }

    auto graph = CreateGraphDirect(builder,
                                   "PointwiseGraph",
                                   DataType::FLOAT,
                                   DataType::HALF,
                                   DataType::BFLOAT16,
                                   &tensors,
                                   &nodes);

    builder.Finish(graph);

    return builder;
}

inline hipdnnPluginConstData_t
    createValidConstDataGraph(flatbuffers::DetachedBuffer& serializedGraph)
{
    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = serializedGraph.data();
    opGraph.size = serializedGraph.size();
    return opGraph;
}

inline flatbuffers::FlatBufferBuilder createValidEngineDetails(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineDetailsOffset = CreateEngineDetails(builder, engineId);
    builder.Finish(engineDetailsOffset);
    return builder;
}

inline hipdnnPluginConstData_t
    createValidConstDataEngineDetails(flatbuffers::DetachedBuffer& serializedEngineDetails)
{
    hipdnnPluginConstData_t engineDetails;
    engineDetails.ptr = serializedEngineDetails.data();
    engineDetails.size = serializedEngineDetails.size();
    return engineDetails;
}

inline flatbuffers::FlatBufferBuilder createValidEngineConfig(int64_t configId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineConfigOffset = CreateEngineConfig(builder, configId);
    builder.Finish(engineConfigOffset);
    return builder;
}

inline hipdnnPluginConstData_t
    createValidConstDataEngineConfig(flatbuffers::DetachedBuffer& serializedEngineConfig)
{
    hipdnnPluginConstData_t engineConfig;
    engineConfig.ptr = serializedEngineConfig.data();
    engineConfig.size = serializedEngineConfig.size();
    return engineConfig;
}

}
