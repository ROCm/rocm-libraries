// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TEMPLATE ADAPTATION: As needed, to assist with testing, replace createReluFwdGraph()
// and createConvFwdGraph() with helpers that build your operation's FlatBuffer graphs.
// You can use createEngineConfig() as-is as it creates a generic engine config usable by
// any PlanBuilder test. Other functions can be used or modified as the situation dictates.
// The helper functions construct in-memory FlatBuffer graphs that simulate what the
// hipDNN frontend produces.

#pragma once

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

namespace example_provider::test_helpers
{

/// Build a single-node pointwise RELU_FWD graph with input and output tensors.
/// @param dims Tensor dimensions (e.g., {1, 1, 4} for 4 elements)
/// @param inputUid UID for the input tensor
/// @param outputUid UID for the output tensor
inline flatbuffers::FlatBufferBuilder createReluFwdGraph(const std::vector<int64_t>& dims
                                                         = {1, 1, 4},
                                                         int64_t inputUid = 1,
                                                         int64_t outputUid = 2)
{
    flatbuffers::FlatBufferBuilder builder;

    std::vector<int64_t> strides = {1};
    int64_t stride = 1;
    strides.resize(dims.size());
    for(int i = static_cast<int>(dims.size()) - 1; i >= 0; --i)
    {
        strides[static_cast<size_t>(i)] = stride;
        stride *= dims[static_cast<size_t>(i)];
    }

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        inputUid,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        outputUid,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    auto pointwiseAttrs = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt, // relu_lower_clip
        flatbuffers::nullopt, // relu_upper_clip
        flatbuffers::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        inputUid, // in_0_tensor_uid
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        outputUid); // out_0_tensor_uid

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttrs.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_relu",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Build a single-node graph with a non-ReLU pointwise operation (e.g., ADD).
inline flatbuffers::FlatBufferBuilder
    createNonReluPointwiseGraph(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode
                                = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD)
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> dims = {1, 1, 4};
    const std::vector<int64_t> strides = {4, 4, 1};

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "in0", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "in1", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "out", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));

    auto pointwiseAttrs = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        mode,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1, // in_0
        flatbuffers::Optional<int64_t>(2), // in_1
        flatbuffers::nullopt,
        3); // out_0

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "pointwise",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttrs.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_non_relu",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Build a two-node graph (not single-node) to test multi-node rejection.
inline flatbuffers::FlatBufferBuilder createMultiNodeReluGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    const std::vector<int64_t> dims = {1, 1, 4};
    const std::vector<int64_t> strides = {4, 4, 1};

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "intermediate",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true)); // virtual
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    auto relu1 = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        1,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2);

    auto relu2 = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        3);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu1",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        relu1.Union()));
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu2",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        relu2.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_multi_node",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Build a single-node ConvolutionFwd graph for testing.
/// Uses NCHW layout for input (N,C,H,W), KCRS for weight (K,C,R,S),
/// and NKHW for output (N,K,outH,outW).
/// @param inputUid  UID for the input tensor (default: 1)
/// @param weightUid UID for the weight tensor (default: 2)
/// @param outputUid UID for the output tensor (default: 3)
/// @param n Batch size (default: 1)
/// @param c Input channels (default: 1)
/// @param h Input height (default: 4)
/// @param w Input width (default: 4)
/// @param k Output channels (default: 1)
/// @param r Filter height (default: 3)
/// @param s Filter width (default: 3)
/// @param padH Padding height (default: 0)
/// @param padW Padding width (default: 0)
/// @param strideH Stride height (default: 1)
/// @param strideW Stride width (default: 1)
/// @param dilationH Dilation height (default: 1)
/// @param dilationW Dilation width (default: 1)
inline flatbuffers::FlatBufferBuilder createConvFwdGraph(int64_t inputUid = 1,
                                                         int64_t weightUid = 2,
                                                         int64_t outputUid = 3,
                                                         int64_t n = 1,
                                                         int64_t c = 1,
                                                         int64_t h = 4,
                                                         int64_t w = 4,
                                                         int64_t k = 1,
                                                         int64_t r = 3,
                                                         int64_t s = 3,
                                                         int64_t padH = 0,
                                                         int64_t padW = 0,
                                                         int64_t strideH = 1,
                                                         int64_t strideW = 1,
                                                         int64_t dilationH = 1,
                                                         int64_t dilationW = 1)
{
    flatbuffers::FlatBufferBuilder builder;

    // Compute output dimensions (accounting for dilation)
    const int64_t outH = (h + 2 * padH - (dilationH * (r - 1) + 1)) / strideH + 1;
    const int64_t outW = (w + 2 * padW - (dilationW * (s - 1) + 1)) / strideW + 1;

    // Input tensor: NCHW
    const std::vector<int64_t> inputDims = {n, c, h, w};
    const std::vector<int64_t> inputStrides = {c * h * w, h * w, w, 1};

    // Weight tensor: KCRS
    const std::vector<int64_t> weightDims = {k, c, r, s};
    const std::vector<int64_t> weightStrides = {c * r * s, r * s, s, 1};

    // Output tensor: NKHW (with output spatial dims)
    const std::vector<int64_t> outputDims = {n, k, outH, outW};
    const std::vector<int64_t> outputStrides = {k * outH * outW, outH * outW, outW, 1};

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        inputUid,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &inputStrides,
        &inputDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        weightUid,
        "weight",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &weightStrides,
        &weightDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        outputUid,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &outputStrides,
        &outputDims));

    // Padding and stride vectors
    const std::vector<int64_t> prePadding = {padH, padW};
    const std::vector<int64_t> postPadding = {padH, padW};
    const std::vector<int64_t> stride = {strideH, strideW};
    const std::vector<int64_t> dilation = {dilationH, dilationW};

    auto convAttrs = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        inputUid,
        weightUid,
        outputUid,
        &prePadding,
        &postPadding,
        &stride,
        &dilation,
        hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        convAttrs.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_conv_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Build a two-node convolution graph (not single-node) to test multi-node rejection.
inline flatbuffers::FlatBufferBuilder createMultiNodeConvGraph()
{
    flatbuffers::FlatBufferBuilder builder;

    // Use small 4x4 input with 3x3 filter -> 2x2 output for first conv
    // Second conv: 2x2 input with 1x1 filter -> 2x2 output
    const std::vector<int64_t> inputDims = {1, 1, 4, 4};
    const std::vector<int64_t> inputStrides = {16, 16, 4, 1};

    const std::vector<int64_t> weight1Dims = {1, 1, 3, 3};
    const std::vector<int64_t> weight1Strides = {9, 9, 3, 1};

    const std::vector<int64_t> interDims = {1, 1, 2, 2};
    const std::vector<int64_t> interStrides = {4, 4, 2, 1};

    const std::vector<int64_t> weight2Dims = {1, 1, 1, 1};
    const std::vector<int64_t> weight2Strides = {1, 1, 1, 1};

    const std::vector<int64_t> outputDims = {1, 1, 2, 2};
    const std::vector<int64_t> outputStrides = {4, 4, 2, 1};

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        1,
        "input",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &inputStrides,
        &inputDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "weight1",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &weight1Strides,
        &weight1Dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "intermediate",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &interStrides,
        &interDims,
        true)); // virtual
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "weight2",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &weight2Strides,
        &weight2Dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "output",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &outputStrides,
        &outputDims));

    const std::vector<int64_t> pad1 = {0, 0};
    const std::vector<int64_t> stride1 = {1, 1};
    const std::vector<int64_t> dilation1 = {1, 1};

    auto conv1 = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        1,
        2,
        3,
        &pad1,
        &pad1,
        &stride1,
        &dilation1,
        hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    const std::vector<int64_t> pad2 = {0, 0};
    const std::vector<int64_t> stride2 = {1, 1};
    const std::vector<int64_t> dilation2 = {1, 1};

    auto conv2 = hipdnn_flatbuffers_sdk::data_objects::CreateConvolutionFwdAttributesDirect(
        builder,
        3,
        4,
        5,
        &pad2,
        &pad2,
        &stride2,
        &dilation2,
        hipdnn_flatbuffers_sdk::data_objects::ConvMode::CROSS_CORRELATION);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv1",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        conv1.Union()));
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "conv2",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes,
        conv2.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test_multi_node_conv",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}

/// Create an EngineConfig FlatBuffer for testing.
inline flatbuffers::FlatBufferBuilder createEngineConfig(int64_t engineId)
{
    flatbuffers::FlatBufferBuilder builder;
    auto engineConfigOffset
        = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfig(builder, engineId);
    builder.Finish(engineConfigOffset);
    return builder;
}

/// Create an EngineConfig FlatBuffer with a single float knob setting for testing.
/// @param engineId The engine ID
/// @param knobId The knob identifier string
/// @param knobValue The float value for the knob
inline flatbuffers::FlatBufferBuilder
    createEngineConfigWithFloatKnob(int64_t engineId, const char* knobId, double knobValue)
{
    flatbuffers::FlatBufferBuilder builder;

    auto floatVal = hipdnn_flatbuffers_sdk::data_objects::CreateFloatValue(builder, knobValue);

    std::vector<flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::KnobSetting>>
        knobSettings;
    knobSettings.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateKnobSettingDirect(
        builder,
        knobId,
        hipdnn_flatbuffers_sdk::data_objects::KnobValue::FloatValue,
        floatVal.Union()));

    auto engineConfigOffset = hipdnn_flatbuffers_sdk::data_objects::CreateEngineConfigDirect(
        builder, engineId, &knobSettings);
    builder.Finish(engineConfigOffset);
    return builder;
}

} // namespace example_provider::test_helpers
