// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildConvolutionFwdGraph(ConvolutionFwdTensorBundle<InputType>& tensorBundle,
                             hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType)
{
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("ConvolutionFwdTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType));

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto wAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "W",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.wTensor);
    wAttr.set_uid(uid++);
    auto wTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(wAttr));

    hipdnn_frontend::graph::ConvFpropAttributes convAttrs;
    convAttrs.set_name("Convolution_fwd_inference");
    convAttrs.set_stride(strides);
    convAttrs.set_dilation(dilation);
    convAttrs.set_pre_padding(padding);
    convAttrs.set_post_padding(padding);
    convAttrs.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    auto yTensorAttr = graph->conv_fprop(xTensorAttr, wTensorAttr, convAttrs);

    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }

    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));

    auto variantPack = tensorBundle.createVariantPack(*xTensorAttr, *wTensorAttr, *yTensorAttr);

    return std::make_tuple(graph, variantPack);
}

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildConvolutionBwdGraph(ConvolutionBwdTensorBundle<InputType>& tensorBundle,
                             hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType)
{
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("ConvolutionBwdTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType));

    int64_t uid = 1;

    auto dyAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dY",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.dyTensor);
    dyAttr.set_uid(uid++);
    auto dyTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(dyAttr));

    auto wAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "W",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.wTensor);
    wAttr.set_uid(uid++);
    auto wTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(wAttr));

    hipdnn_frontend::graph::ConvDgradAttributes convBwdAttrs;
    convBwdAttrs.set_name("Convolution_bwd");
    convBwdAttrs.set_stride(strides);
    convBwdAttrs.set_dilation(dilation);
    convBwdAttrs.set_pre_padding(padding);
    convBwdAttrs.set_post_padding(padding);
    convBwdAttrs.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    auto dxTensorAttr = graph->conv_dgrad(dyTensorAttr, wTensorAttr, convBwdAttrs);

    if(!dxTensorAttr->has_uid())
    {
        dxTensorAttr->set_uid(uid++);
    }

    dxTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));

    auto variantPack = tensorBundle.createVariantPack(*dyTensorAttr, *wTensorAttr, *dxTensorAttr);

    return std::make_tuple(graph, variantPack);
}

template <typename InputType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildConvolutionWrwGraph(ConvolutionWrwTensorBundle<InputType>& tensorBundle,
                             hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType accumulatorDataType)
{
    const std::vector<int64_t> strides = {1, 1};
    const std::vector<int64_t> dilation = {1, 1};
    const std::vector<int64_t> padding = {0, 0};

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("ConvolutionWrwTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(accumulatorDataType));

    int64_t uid = 1;

    auto dyAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dY",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.dyTensor);
    dyAttr.set_uid(uid++);
    auto dyTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(dyAttr));

    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    hipdnn_frontend::graph::ConvWgradAttributes convWrwAttrs;
    convWrwAttrs.set_name("Convolution_wrw");
    convWrwAttrs.set_stride(strides);
    convWrwAttrs.set_dilation(dilation);
    convWrwAttrs.set_pre_padding(padding);
    convWrwAttrs.set_post_padding(padding);
    convWrwAttrs.set_convolution_mode(hipdnn_frontend::ConvolutionMode::CROSS_CORRELATION);

    auto dwTensorAttr = graph->conv_wgrad(dyTensorAttr, xTensorAttr, convWrwAttrs);

    if(!dwTensorAttr->has_uid())
    {
        dwTensorAttr->set_uid(uid++);
    }

    dwTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));

    auto variantPack = tensorBundle.createVariantPack(*xTensorAttr, *dwTensorAttr, *dyTensorAttr);

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
