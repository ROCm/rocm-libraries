// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "PointwiseTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildPointwiseUnaryGraph(PointwiseUnaryTensorBundle<InputType>& tensorBundle,
                             hipdnn_sdk::data_objects::DataType input0DataType,
                             hipdnn_sdk::data_objects::DataType accumulatorDataType,
                             hipdnn_sdk::data_objects::DataType outputDataType,
                             hipdnn_frontend::PointwiseMode operation)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PointwiseUnaryTest");
    graph->set_compute_data_type(hipdnn_frontend::fromSdkType(accumulatorDataType));

    int64_t uid = 1;
    auto inputAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input", hipdnn_frontend::fromSdkType(input0DataType), tensorBundle.inputTensor);
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

    auto variantPack = tensorBundle.createVariantPack(*inputTensorAttr, *outputTensorAttr);

    return std::make_tuple(graph, variantPack);
}

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildPointwiseBinaryGraph(PointwiseBinaryTensorBundle<InputType>& tensorBundle,
                              hipdnn_sdk::data_objects::DataType input0DataType,
                              hipdnn_sdk::data_objects::DataType input1DataType,
                              hipdnn_sdk::data_objects::DataType accumulatorDataType,
                              hipdnn_sdk::data_objects::DataType outputDataType,
                              hipdnn_frontend::PointwiseMode operation)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("PointwiseBinaryTest");
    graph->set_compute_data_type(hipdnn_frontend::fromSdkType(accumulatorDataType));

    int64_t uid = 1;
    auto input1Attr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input1", hipdnn_frontend::fromSdkType(input0DataType), tensorBundle.input1Tensor);
    input1Attr.set_uid(uid++);
    auto input1TensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(input1Attr));

    auto input2Attr = hipdnn_frontend::graph::makeTensorAttributes(
        "Input2", hipdnn_frontend::fromSdkType(input1DataType), tensorBundle.input2Tensor);
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

    auto variantPack
        = tensorBundle.createVariantPack(*input1TensorAttr, *input2TensorAttr, *outputTensorAttr);

    return std::make_tuple(graph, variantPack);
}

}
