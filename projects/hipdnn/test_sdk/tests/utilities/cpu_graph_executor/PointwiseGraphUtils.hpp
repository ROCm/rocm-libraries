// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "PointwiseTensorBundles.hpp"

namespace hipdnn_sdk_test_utils
{

inline std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  PointwiseUnaryTensorBundle,
                  std::unordered_map<int64_t, void*>>
    buildPointwiseUnaryGraph(const std::vector<int64_t>& inputDims,
                             const std::vector<int64_t>& outputDims,
                             hipdnn_flatbuffers_sdk::data_objects::DataType input0DataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType,
                             hipdnn_frontend::PointwiseMode operation,
                             unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed(),
                             const hipdnn_data_sdk::utilities::TensorLayout& layout
                             = hipdnn_data_sdk::utilities::TensorLayout::NCHW,
                             std::optional<float> reluLowerClip = std::nullopt,
                             std::optional<float> reluUpperClip = std::nullopt,
                             std::optional<float> reluLowerClipSlope = std::nullopt,
                             std::optional<float> swishBeta = std::nullopt,
                             std::optional<float> eluAlpha = std::nullopt,
                             std::optional<float> softplusBeta = std::nullopt)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PointwiseUnaryTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(input0DataType))
        .set_compute_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType));

    int64_t uid = 1;

    // Create input tensor attribute
    auto inputStrides = hipdnn_data_sdk::utilities::generateStrides(inputDims, layout.strideOrder);
    const auto& inputDimsCopy = inputDims;
    auto inputAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(input0DataType),
        inputDimsCopy,
        inputStrides);
    inputAttr.set_uid(uid++);
    auto inputTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(inputAttr));

    hipdnn_frontend::graph::PointwiseAttributes pointwiseAttrs;
    pointwiseAttrs.set_name("PointwiseUnary");
    pointwiseAttrs.set_mode(operation);
    if(reluLowerClip.has_value())
    {
        pointwiseAttrs.set_relu_lower_clip(reluLowerClip.value());
    }
    if(reluUpperClip.has_value())
    {
        pointwiseAttrs.set_relu_upper_clip(reluUpperClip.value());
    }
    if(reluLowerClipSlope.has_value())
    {
        pointwiseAttrs.set_relu_lower_clip_slope(reluLowerClipSlope.value());
    }
    if(swishBeta.has_value())
    {
        pointwiseAttrs.set_swish_beta(swishBeta.value());
    }
    if(eluAlpha.has_value())
    {
        pointwiseAttrs.set_elu_alpha(eluAlpha.value());
    }
    if(softplusBeta.has_value())
    {
        pointwiseAttrs.set_softplus_beta(softplusBeta.value());
    }

    auto outputTensorAttr = graph->pointwise(inputTensorAttr, pointwiseAttrs);

    if(!outputTensorAttr->has_uid())
    {
        outputTensorAttr->set_uid(uid++);
    }
    outputTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(outputDataType));
    outputTensorAttr->set_dim(outputDims);
    outputTensorAttr->set_stride(
        hipdnn_data_sdk::utilities::generateStrides(outputDims, layout.strideOrder));
    outputTensorAttr->set_output(true);

    // Ensure properties are inferred
    auto validateResult = graph->validate();
    if(validateResult.is_bad())
    {
        throw std::runtime_error("Graph validation failed: " + validateResult.get_message());
    }

    // Serialize graph and create tensor bundle
    auto [serializedGraph, serErr] = graph->to_binary();
    if(serErr.is_bad())
    {
        throw std::runtime_error("Graph serialization failed: " + serErr.get_message());
    }
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());
    auto nodeWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::NodeWrapper(&graphWrap.getNode(0));

    PointwiseUnaryTensorBundle tensorBundle(nodeWrap, graphWrap.getTensorMap(), seed);
    auto variantPack = tensorBundle.toHostVariantPack();

    return std::make_tuple(graph, std::move(tensorBundle), variantPack);
}

inline std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  PointwiseBinaryTensorBundle,
                  std::unordered_map<int64_t, void*>>
    buildPointwiseBinaryGraph(const std::vector<int64_t>& input1Dims,
                              const std::vector<int64_t>& input2Dims,
                              const std::vector<int64_t>& outputDims,
                              hipdnn_flatbuffers_sdk::data_objects::DataType input0DataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType input1DataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType,
                              hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType,
                              hipdnn_frontend::PointwiseMode operation,
                              unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed(),
                              const hipdnn_data_sdk::utilities::TensorLayout& layout
                              = hipdnn_data_sdk::utilities::TensorLayout::NCHW,
                              std::optional<float> reluLowerClip = std::nullopt,
                              std::optional<float> reluUpperClip = std::nullopt,
                              std::optional<float> reluLowerClipSlope = std::nullopt,
                              std::optional<float> swishBeta = std::nullopt,
                              std::optional<float> eluAlpha = std::nullopt,
                              std::optional<float> softplusBeta = std::nullopt)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PointwiseBinaryTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(input0DataType))
        .set_compute_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType));

    int64_t uid = 1;

    // Create input tensor attributes
    auto input1Strides
        = hipdnn_data_sdk::utilities::generateStrides(input1Dims, layout.strideOrder);
    const auto& input1DimsCopy = input1Dims;
    auto input1Attr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input1",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(input0DataType),
        input1DimsCopy,
        input1Strides);
    input1Attr.set_uid(uid++);
    auto input1TensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(input1Attr));

    auto input2Strides
        = hipdnn_data_sdk::utilities::generateStrides(input2Dims, layout.strideOrder);
    const auto& input2DimsCopy = input2Dims;
    auto input2Attr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input2",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(input1DataType),
        input2DimsCopy,
        input2Strides);
    input2Attr.set_uid(uid++);
    auto input2TensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(input2Attr));

    hipdnn_frontend::graph::PointwiseAttributes pointwiseAttrs;
    pointwiseAttrs.set_name("PointwiseBinary");
    pointwiseAttrs.set_mode(operation);
    if(reluLowerClip.has_value())
    {
        pointwiseAttrs.set_relu_lower_clip(reluLowerClip.value());
    }
    if(reluUpperClip.has_value())
    {
        pointwiseAttrs.set_relu_upper_clip(reluUpperClip.value());
    }
    if(reluLowerClipSlope.has_value())
    {
        pointwiseAttrs.set_relu_lower_clip_slope(reluLowerClipSlope.value());
    }
    if(swishBeta.has_value())
    {
        pointwiseAttrs.set_swish_beta(swishBeta.value());
    }
    if(eluAlpha.has_value())
    {
        pointwiseAttrs.set_elu_alpha(eluAlpha.value());
    }
    if(softplusBeta.has_value())
    {
        pointwiseAttrs.set_softplus_beta(softplusBeta.value());
    }

    auto outputTensorAttr = graph->pointwise(input1TensorAttr, input2TensorAttr, pointwiseAttrs);

    if(!outputTensorAttr->has_uid())
    {
        outputTensorAttr->set_uid(uid++);
    }
    outputTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(outputDataType));
    outputTensorAttr->set_dim(outputDims);
    outputTensorAttr->set_stride(
        hipdnn_data_sdk::utilities::generateStrides(outputDims, layout.strideOrder));
    outputTensorAttr->set_output(true);

    // Ensure properties are inferred
    auto validateResult = graph->validate();
    if(validateResult.is_bad())
    {
        throw std::runtime_error("Graph validation failed: " + validateResult.get_message());
    }

    // Serialize graph and create tensor bundle
    auto [serializedGraph, serErr] = graph->to_binary();
    if(serErr.is_bad())
    {
        throw std::runtime_error("Graph serialization failed: " + serErr.get_message());
    }
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());
    auto nodeWrap
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::NodeWrapper(&graphWrap.getNode(0));

    PointwiseBinaryTensorBundle tensorBundle(nodeWrap, graphWrap.getTensorMap(), seed);
    auto variantPack = tensorBundle.toHostVariantPack();

    return std::make_tuple(graph, std::move(tensorBundle), variantPack);
}

}
