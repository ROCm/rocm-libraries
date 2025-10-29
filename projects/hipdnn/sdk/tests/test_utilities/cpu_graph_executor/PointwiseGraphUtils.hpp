// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "PointwiseTensorBundles.hpp"

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

namespace hipdnn_sdk_test_utils
{

static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  PointwiseUnaryTensorBundle,
                  std::unordered_map<int64_t, void*>>
    buildPointwiseUnaryGraph(const std::vector<int64_t>& inputDims,
                             const std::vector<int64_t>& outputDims,
                             hipdnn_sdk::data_objects::DataType input0DataType,
                             hipdnn_sdk::data_objects::DataType accumulatorDataType,
                             hipdnn_sdk::data_objects::DataType outputDataType,
                             hipdnn_frontend::PointwiseMode operation,
                             unsigned int seed = getGlobalTestSeed(),
                             const TensorLayout& layout = TensorLayout::NCHW)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PointwiseUnaryTest");
    graph->set_compute_data_type(hipdnn_frontend::fromSdkType(accumulatorDataType));

    int64_t uid = 1;

    // Create input tensor attribute
    auto inputStrides = generateStrides(inputDims, layout.strideOrder);
    const auto& inputDimsCopy = inputDims;
    auto inputAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input", hipdnn_frontend::fromSdkType(input0DataType), inputDimsCopy, inputStrides);
    inputAttr.set_uid(uid++);
    auto inputTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(inputAttr));

    hipdnn_frontend::graph::PointwiseAttributes pointwiseAttrs;
    pointwiseAttrs.set_name("PointwiseUnary");
    pointwiseAttrs.set_mode(operation);

    auto outputTensorAttr = graph->pointwise(inputTensorAttr, pointwiseAttrs);

    if(!outputTensorAttr->has_uid())
    {
        outputTensorAttr->set_uid(uid++);
    }
    outputTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(outputDataType));
    outputTensorAttr->set_dim(outputDims);
    outputTensorAttr->set_stride(generateStrides(outputDims, layout.strideOrder));

    // Serialize graph and create tensor bundle
    auto serializedGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_plugin::GraphWrapper(serializedGraph.data(), serializedGraph.size());
    auto nodeWrap = hipdnn_plugin::NodeWrapper(&graphWrap.getNode(0));

    PointwiseUnaryTensorBundle tensorBundle(nodeWrap, graphWrap.getTensorMap(), seed);
    auto variantPack = tensorBundle.toHostVariantPack();

    return std::make_tuple(graph, std::move(tensorBundle), variantPack);
}

static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  PointwiseBinaryTensorBundle,
                  std::unordered_map<int64_t, void*>>
    buildPointwiseBinaryGraph(const std::vector<int64_t>& input1Dims,
                              const std::vector<int64_t>& input2Dims,
                              const std::vector<int64_t>& outputDims,
                              hipdnn_sdk::data_objects::DataType input0DataType,
                              hipdnn_sdk::data_objects::DataType input1DataType,
                              hipdnn_sdk::data_objects::DataType accumulatorDataType,
                              hipdnn_sdk::data_objects::DataType outputDataType,
                              hipdnn_frontend::PointwiseMode operation,
                              unsigned int seed = getGlobalTestSeed(),
                              const TensorLayout& layout = TensorLayout::NCHW)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PointwiseBinaryTest");
    graph->set_compute_data_type(hipdnn_frontend::fromSdkType(accumulatorDataType));

    int64_t uid = 1;

    // Create input tensor attributes
    auto input1Strides = generateStrides(input1Dims, layout.strideOrder);
    const auto& input1DimsCopy = input1Dims;
    auto input1Attr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input1", hipdnn_frontend::fromSdkType(input0DataType), input1DimsCopy, input1Strides);
    input1Attr.set_uid(uid++);
    auto input1TensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(input1Attr));

    auto input2Strides = generateStrides(input2Dims, layout.strideOrder);
    const auto& input2DimsCopy = input2Dims;
    auto input2Attr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input2", hipdnn_frontend::fromSdkType(input1DataType), input2DimsCopy, input2Strides);
    input2Attr.set_uid(uid++);
    auto input2TensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(input2Attr));

    hipdnn_frontend::graph::PointwiseAttributes pointwiseAttrs;
    pointwiseAttrs.set_name("PointwiseBinary");
    pointwiseAttrs.set_mode(operation);

    auto outputTensorAttr = graph->pointwise(input1TensorAttr, input2TensorAttr, pointwiseAttrs);

    if(!outputTensorAttr->has_uid())
    {
        outputTensorAttr->set_uid(uid++);
    }
    outputTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(outputDataType));
    outputTensorAttr->set_dim(outputDims);
    outputTensorAttr->set_stride(generateStrides(outputDims, layout.strideOrder));

    // Serialize graph and create tensor bundle
    auto serializedGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_plugin::GraphWrapper(serializedGraph.data(), serializedGraph.size());
    auto nodeWrap = hipdnn_plugin::NodeWrapper(&graphWrap.getNode(0));

    PointwiseBinaryTensorBundle tensorBundle(nodeWrap, graphWrap.getTensorMap(), seed);
    auto variantPack = tensorBundle.toHostVariantPack();

    return std::make_tuple(graph, std::move(tensorBundle), variantPack);
}

}
