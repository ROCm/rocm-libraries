// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BatchnormTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;

namespace hipdnn_sdk_test_utils
{

template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildBatchnormTrainGraph(
        BatchnormTrainTensorBundle<InputType, ScaleBiasType, MeanVarianceType>& tensorBundle,
        hipdnn_sdk::data_objects::DataType inputDataType,
        hipdnn_sdk::data_objects::DataType scaleBiasDataType,
        hipdnn_sdk::data_objects::DataType meanVarianceDataType,
        bool useOptionalTensors)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormTrainTest");

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X", hipdnn_frontend::fromSdkType(inputDataType), tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale", hipdnn_frontend::fromSdkType(scaleBiasDataType), tensorBundle.scaleTensor);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias", hipdnn_frontend::fromSdkType(scaleBiasDataType), tensorBundle.biasTensor);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::fromSdkType(meanVarianceDataType))
        .set_dim({1})
        .set_stride({1});

    hipdnn_frontend::graph::BatchnormAttributes bnAttrs;
    bnAttrs.set_name("batchnorm_fwd_train");
    bnAttrs.set_epsilon(epsilonTensor);

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> momentumTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prevRunningMeanTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prevRunningVarianceTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> nextRunningMeanTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> nextRunningVarianceTensorAttr;

    if(useOptionalTensors)
    {
        auto momentumAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "momentum",
            hipdnn_frontend::fromSdkType(meanVarianceDataType),
            tensorBundle.momentumTensor.value());
        momentumAttr.set_uid(uid++);
        momentumTensorAttr
            = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(momentumAttr));

        auto prevRunningMeanAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "prev_running_mean",
            hipdnn_frontend::fromSdkType(meanVarianceDataType),
            tensorBundle.prevRunningMeanTensor.value());
        prevRunningMeanAttr.set_uid(uid++);
        prevRunningMeanTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
            std::move(prevRunningMeanAttr));

        auto prevRunningVarianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "prev_running_variance",
            hipdnn_frontend::fromSdkType(meanVarianceDataType),
            tensorBundle.prevRunningVarianceTensor.value());
        prevRunningVarianceAttr.set_uid(uid++);
        prevRunningVarianceTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
            std::move(prevRunningVarianceAttr));

        bnAttrs.set_momentum(momentumTensorAttr);
        bnAttrs.set_prev_running_mean(prevRunningMeanTensorAttr);
        bnAttrs.set_prev_running_variance(prevRunningVarianceTensorAttr);
    }

    auto outputTensorsAttr
        = graph->batchnorm(xTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

    auto& yTensorAttr = outputTensorsAttr[0];
    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(inputDataType));

    auto& meanTensorAttr = outputTensorsAttr[1];
    if(!meanTensorAttr->has_uid())
    {
        meanTensorAttr->set_uid(uid++);
    }
    meanTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(meanVarianceDataType));

    auto& invVarianceTensorAttr = outputTensorsAttr[2];
    if(!invVarianceTensorAttr->has_uid())
    {
        invVarianceTensorAttr->set_uid(uid++);
    }
    invVarianceTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(meanVarianceDataType));

    if(useOptionalTensors)
    {
        nextRunningMeanTensorAttr = outputTensorsAttr[3];
        if(!nextRunningMeanTensorAttr->has_uid())
        {
            nextRunningMeanTensorAttr->set_uid(uid++);
        }
        nextRunningMeanTensorAttr->set_data_type(
            hipdnn_frontend::fromSdkType(meanVarianceDataType));

        nextRunningVarianceTensorAttr = outputTensorsAttr[4];
        if(!nextRunningVarianceTensorAttr->has_uid())
        {
            nextRunningVarianceTensorAttr->set_uid(uid++);
        }
        nextRunningVarianceTensorAttr->set_data_type(
            hipdnn_frontend::fromSdkType(meanVarianceDataType));
    }

    auto variantPack = tensorBundle.createVariantPack(*xTensorAttr,
                                                      *scaleTensorAttr,
                                                      *biasTensorAttr,
                                                      *meanTensorAttr,
                                                      *invVarianceTensorAttr,
                                                      *epsilonTensor,
                                                      *yTensorAttr,
                                                      momentumTensorAttr,
                                                      prevRunningMeanTensorAttr,
                                                      prevRunningVarianceTensorAttr,
                                                      nextRunningMeanTensorAttr,
                                                      nextRunningVarianceTensorAttr);

    return std::make_tuple(graph, variantPack);
}

inline std::shared_ptr<hipdnn_frontend::graph::Graph>
    buildBatchnormFwdInferenceGraph(hipdnn_sdk::data_objects::DataType inputDataType,
                                    hipdnn_sdk::data_objects::DataType scaleBiasDataType,
                                    hipdnn_sdk::data_objects::DataType meanVarianceDataType,
                                    const std::vector<int64_t>& dims,
                                    const TensorLayout& layout,
                                    bool isOutputVirtual = false)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormFwdInferenceTest");

    auto strides = generateStrides(dims, layout.strideOrder);

    auto derivedDims = getDerivedShape(dims);
    auto derivedStrides = generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_frontend::fromSdkType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale", hipdnn_frontend::fromSdkType(scaleBiasDataType), derivedDims, derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias", hipdnn_frontend::fromSdkType(scaleBiasDataType), derivedDims, derivedStrides);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mean", hipdnn_frontend::fromSdkType(meanVarianceDataType), derivedDims, derivedStrides);
    meanAttr.set_uid(uid++);
    auto meanTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));

    auto varianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "invVariance",
        hipdnn_frontend::fromSdkType(meanVarianceDataType),
        derivedDims,
        derivedStrides);
    varianceAttr.set_uid(uid++);
    auto varianceTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(varianceAttr));

    hipdnn_frontend::graph::BatchnormInferenceAttributes bnAttrs;
    bnAttrs.set_name("batchnorm_fwd_inference");

    auto yTensorAttr = graph->batchnorm_inference(
        xTensorAttr, meanTensorAttr, varianceTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_name("Y");
    yTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(isOutputVirtual);

    return graph;
}

template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildBatchnormBwdGraph(
        BatchnormBwdTensorBundle<InputType, ScaleBiasType, MeanVarianceType>& tensorBundle,
        hipdnn_sdk::data_objects::DataType inputDataType,
        hipdnn_sdk::data_objects::DataType scaleBiasDataType,
        hipdnn_sdk::data_objects::DataType meanVarianceDataType)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormBwdTest");

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X", hipdnn_frontend::fromSdkType(inputDataType), tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto dyAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dY", hipdnn_frontend::fromSdkType(inputDataType), tensorBundle.dyTensor);
    dyAttr.set_uid(uid++);
    auto dyTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(dyAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale", hipdnn_frontend::fromSdkType(scaleBiasDataType), tensorBundle.scaleTensor);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mean", hipdnn_frontend::fromSdkType(meanVarianceDataType), tensorBundle.meanTensor);
    meanAttr.set_uid(uid++);
    auto meanTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));

    auto invVarianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "inv_variance",
        hipdnn_frontend::fromSdkType(meanVarianceDataType),
        tensorBundle.invVarianceTensor);
    invVarianceAttr.set_uid(uid++);
    auto invVarianceTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(invVarianceAttr));

    hipdnn_frontend::graph::BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_name("batchnorm_bwd");
    bnBwdAttrs.set_mean(meanTensorAttr);
    bnBwdAttrs.set_inv_variance(invVarianceTensorAttr);

    auto outputTensorsAttr
        = graph->batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

    auto& dxTensorAttr = outputTensorsAttr[0];
    if(!dxTensorAttr->has_uid())
    {
        dxTensorAttr->set_uid(uid++);
    }
    dxTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(inputDataType));

    auto& dScaleTensorAttr = outputTensorsAttr[1];
    if(!dScaleTensorAttr->has_uid())
    {
        dScaleTensorAttr->set_uid(uid++);
    }
    dScaleTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(scaleBiasDataType));

    auto& dBiasTensorAttr = outputTensorsAttr[2];
    if(!dBiasTensorAttr->has_uid())
    {
        dBiasTensorAttr->set_uid(uid++);
    }
    dBiasTensorAttr->set_data_type(hipdnn_frontend::fromSdkType(scaleBiasDataType));

    auto variantPack = tensorBundle.createVariantPack(*xTensorAttr,
                                                      *dyTensorAttr,
                                                      *scaleTensorAttr,
                                                      *meanTensorAttr,
                                                      *invVarianceTensorAttr,
                                                      *dxTensorAttr,
                                                      *dScaleTensorAttr,
                                                      *dBiasTensorAttr);

    return std::make_tuple(graph, variantPack);
}
}
