// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildLayernormFpropGraph(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                             const std::vector<int64_t>& dims,
                             const int64_t normalizedDimCount,
                             const hipdnn_data_sdk::utilities::TensorLayout& layout,
                             bool useTrainingPhase = false,
                             bool onePadded = false)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("LayernormFpropTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    // Scale/bias shape = normalized dims (last normalizedDimCount dims for NCHW)
    std::vector<int64_t> normalizedDims;
    if(onePadded)
    {
        normalizedDims = std::vector<int64_t>(dims.size(), 1);
        for(size_t i = static_cast<size_t>(
                std::max(static_cast<int64_t>(dims.size()) - normalizedDimCount, int64_t{0}));
            i < dims.size();
            ++i)
        {
            normalizedDims[i] = dims[i];
        }
    }
    else
    {
        normalizedDims = std::vector<int64_t>(
            dims.begin()
                + std::max(static_cast<int64_t>(dims.size()) - normalizedDimCount, int64_t{0}),
            dims.end());
    }
    auto normalizedStrides = hipdnn_data_sdk::utilities::generateStrides(normalizedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        normalizedDims,
        normalizedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        normalizedDims,
        normalizedStrides);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim({1})
        .set_stride({1})
        .set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);

    hipdnn_frontend::graph::LayernormAttributes lnAttrs;
    lnAttrs.set_name("layernorm_fprop");
    lnAttrs.set_epsilon(epsilonTensor);
    lnAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    if(useTrainingPhase)
    {
        lnAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::TRAINING);
    }
    else
    {
        lnAttrs.set_forward_phase(hipdnn_frontend::NormFwdPhase::INFERENCE);
    }

    auto outputTensorsAttr
        = graph->layernorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, lnAttrs);

    auto& yTensorAttr = outputTensorsAttr[0];
    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_name("Y");
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(false);

    if(useTrainingPhase)
    {
        // Mean shape = batch dims (e.g., [N, 1, 1, 1] for input [N, C, H, W] and normalizedDimCount 3)
        std::vector<int64_t> statsDims(dims.size(), 1);
        for(size_t i = 0; i < static_cast<size_t>(std::max(
                              static_cast<int64_t>(dims.size()) - normalizedDimCount, int64_t{0}));
            ++i)
        {
            statsDims[i] = dims[i];
        }
        auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(statsDims);

        auto& meanTensorAttr = outputTensorsAttr[1];
        if(meanTensorAttr && !meanTensorAttr->has_uid())
        {
            meanTensorAttr->set_uid(uid++);
        }
        if(meanTensorAttr)
        {
            meanTensorAttr->set_data_type(
                hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanInvVarianceDataType));
            meanTensorAttr->set_dim(statsDims);
            meanTensorAttr->set_stride(statsStrides);
            meanTensorAttr->set_is_virtual(false);
        }

        auto& invVarianceTensorAttr = outputTensorsAttr[2];
        if(invVarianceTensorAttr && !invVarianceTensorAttr->has_uid())
        {
            invVarianceTensorAttr->set_uid(uid++);
        }
        if(invVarianceTensorAttr)
        {
            invVarianceTensorAttr->set_data_type(
                hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanInvVarianceDataType));
            invVarianceTensorAttr->set_dim(statsDims);
            invVarianceTensorAttr->set_stride(statsStrides);
            invVarianceTensorAttr->set_is_virtual(false);
        }
    }

    return graph;
}

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildLayernormBpropGraph(hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType meanInvVarianceDataType,
                             hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
                             const std::vector<int64_t>& dims,
                             const int64_t normalizedDimCount,
                             const hipdnn_data_sdk::utilities::TensorLayout& layout,
                             bool meanInvVar = false,
                             bool onePadded = false)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("LayernormBpropTest");

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    // Scale/bias shape = normalized dims (last normalizedDimCount dims for NCHW)
    std::vector<int64_t> normalizedDims;
    std::vector<int64_t> statDims;
    if(onePadded)
    {
        normalizedDims = std::vector<int64_t>(dims.size(), 1);
        statDims = std::vector<int64_t>(dims.size(), 1);
        for(size_t i = 0; i < dims.size(); ++i)
        {
            if(static_cast<int64_t>(i) < static_cast<int64_t>(dims.size()) - normalizedDimCount)
            {
                statDims[i] = dims[i];
            }
            else
            {
                normalizedDims[i] = dims[i];
            }
        }
    }
    else
    {
        normalizedDims = std::vector<int64_t>(
            dims.begin()
                + std::max(static_cast<int64_t>(dims.size()) - normalizedDimCount, int64_t{0}),
            dims.end());
        statDims = std::vector<int64_t>(
            dims.begin(),
            dims.begin()
                + std::max(static_cast<int64_t>(dims.size()) - normalizedDimCount, int64_t{0}));
    }
    auto normalizedStrides = hipdnn_data_sdk::utilities::generateStrides(normalizedDims);
    auto statStrides = hipdnn_data_sdk::utilities::generateStrides(statDims);

    int64_t uid = 1;
    auto dyAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dy", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    dyAttr.set_uid(uid++);
    auto dyTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(dyAttr));

    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        normalizedDims,
        normalizedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim({1})
        .set_stride({1})
        .set_value(hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON);

    hipdnn_frontend::graph::LayernormBackwardAttributes lnAttrs;
    lnAttrs.set_name("layernorm_bprop");
    lnAttrs.set_epsilon(epsilonTensor);
    lnAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    if(meanInvVar)
    {
        auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "mean",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanInvVarianceDataType),
            statDims,
            statStrides);
        meanAttr.set_uid(uid++);
        auto meanTensorAttr
            = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));
        lnAttrs.set_mean(std::move(meanTensorAttr));

        auto invVarianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "inv_variance",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanInvVarianceDataType),
            statDims,
            statStrides);
        invVarianceAttr.set_uid(uid++);
        auto invVarianceTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
            std::move(invVarianceAttr));
        lnAttrs.set_inv_variance(std::move(invVarianceTensorAttr));
    }

    auto outputTensorsAttr
        = graph->layernorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, lnAttrs);

    auto& dxTensorAttr = outputTensorsAttr[0];
    if(!dxTensorAttr->has_uid())
    {
        dxTensorAttr->set_uid(uid++);
    }
    dxTensorAttr->set_name("DX");
    dxTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    dxTensorAttr->set_dim(dims);
    dxTensorAttr->set_stride(strides);
    dxTensorAttr->set_is_virtual(false);

    auto& dscaleTensorAttr = outputTensorsAttr[1];
    if(!dscaleTensorAttr->has_uid())
    {
        dscaleTensorAttr->set_uid(uid++);
    }
    dscaleTensorAttr->set_name("DSCALE");
    dscaleTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    dscaleTensorAttr->set_dim(normalizedDims);
    dscaleTensorAttr->set_stride(normalizedStrides);
    dscaleTensorAttr->set_is_virtual(false);

    auto& dbiasTensorAttr = outputTensorsAttr[2];
    if(!dbiasTensorAttr->has_uid())
    {
        dbiasTensorAttr->set_uid(uid++);
    }
    dbiasTensorAttr->set_name("DBIAS");
    dbiasTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    dbiasTensorAttr->set_dim(normalizedDims);
    dbiasTensorAttr->set_stride(normalizedStrides);
    dbiasTensorAttr->set_is_virtual(false);

    return graph;
}

}
