// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "BatchnormTensorBundles.hpp"
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
static std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>,
                  std::unordered_map<int64_t, void*>>
    buildBatchnormTrainGraph(
        BatchnormTrainTensorBundle<InputType, ScaleBiasType, MeanVarianceType>& tensorBundle,
        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
        bool useOptionalTensors)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormTrainTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        tensorBundle.scaleTensor);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        tensorBundle.biasTensor);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim({1})
        .set_stride({1})
        .set_value(hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON);

    hipdnn_frontend::graph::BatchnormAttributes bnAttrs;
    bnAttrs.set_name("batchnorm_fwd_train");
    bnAttrs.set_epsilon(epsilonTensor);
    bnAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> momentumTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prevRunningMeanTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> prevRunningVarianceTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> nextRunningMeanTensorAttr;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> nextRunningVarianceTensorAttr;

    if(useOptionalTensors)
    {
        auto momentumTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
        momentumTensor->set_uid(uid++)
            .set_name("MomentumTensor")
            .set_data_type(hipdnn_frontend::DataType::DOUBLE)
            .set_dim({1})
            .set_stride({1})
            .set_value(0.1);
        momentumTensorAttr = momentumTensor;

        auto prevRunningMeanAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "prev_running_mean",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
            tensorBundle.prevRunningMeanTensor.value());
        prevRunningMeanAttr.set_uid(uid++);
        prevRunningMeanTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
            std::move(prevRunningMeanAttr));

        auto prevRunningVarianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
            "prev_running_variance",
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
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
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));

    auto& meanTensorAttr = outputTensorsAttr[1];
    if(!meanTensorAttr->has_uid())
    {
        meanTensorAttr->set_uid(uid++);
    }
    meanTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType));

    auto& invVarianceTensorAttr = outputTensorsAttr[2];
    if(!invVarianceTensorAttr->has_uid())
    {
        invVarianceTensorAttr->set_uid(uid++);
    }
    invVarianceTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType));

    if(useOptionalTensors)
    {
        nextRunningMeanTensorAttr = outputTensorsAttr[3];
        if(!nextRunningMeanTensorAttr->has_uid())
        {
            nextRunningMeanTensorAttr->set_uid(uid++);
        }
        nextRunningMeanTensorAttr->set_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType));

        nextRunningVarianceTensorAttr = outputTensorsAttr[4];
        if(!nextRunningVarianceTensorAttr->has_uid())
        {
            nextRunningVarianceTensorAttr->set_uid(uid++);
        }
        nextRunningVarianceTensorAttr->set_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType));
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

inline std::shared_ptr<hipdnn_frontend::graph::Graph> buildBatchnormFwdInferenceGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
    const std::vector<int64_t>& dims,
    const hipdnn_data_sdk::utilities::TensorLayout& layout,
    bool isOutputVirtual = false)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormFwdInferenceTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        derivedDims,
        derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        derivedDims,
        derivedStrides);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mean",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
        derivedDims,
        derivedStrides);
    meanAttr.set_uid(uid++);
    auto meanTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));

    auto varianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "invVariance",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
        derivedDims,
        derivedStrides);
    varianceAttr.set_uid(uid++);
    auto varianceTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(varianceAttr));

    hipdnn_frontend::graph::BatchnormInferenceAttributes bnAttrs;
    bnAttrs.set_name("batchnorm_fwd_inference");
    bnAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto yTensorAttr = graph->batchnorm_inference(
        xTensorAttr, meanTensorAttr, varianceTensorAttr, scaleTensorAttr, biasTensorAttr, bnAttrs);

    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_name("Y");
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
    yTensorAttr->set_dim(dims);
    yTensorAttr->set_stride(strides);
    yTensorAttr->set_is_virtual(isOutputVirtual);

    return graph;
}

inline std::shared_ptr<hipdnn_frontend::graph::Graph> buildBatchnormFwdInferenceWithVarianceGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
    const std::vector<int64_t>& dims,
    const hipdnn_data_sdk::utilities::TensorLayout& layout,
    bool isOutputVirtual = false)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormFwdInferenceWithVarianceTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto strides = hipdnn_data_sdk::utilities::generateStrides(dims, layout.strideOrder);

    auto derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    auto derivedStrides = hipdnn_data_sdk::utilities::generateStrides(derivedDims);

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "x", hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType), dims, strides);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        derivedDims,
        derivedStrides);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto biasAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "bias",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        derivedDims,
        derivedStrides);
    biasAttr.set_uid(uid++);
    auto biasTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(biasAttr));

    auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mean",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
        derivedDims,
        derivedStrides);
    meanAttr.set_uid(uid++);
    auto meanTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));

    auto varianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "variance",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
        derivedDims,
        derivedStrides);
    varianceAttr.set_uid(uid++);
    auto varianceTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(varianceAttr));

    auto epsilonTensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    epsilonTensor->set_uid(uid++)
        .set_name("EpsilonTensor")
        .set_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_dim({1})
        .set_stride({1})
        .set_value(hipdnn_data_sdk::utilities::BATCHNORM_DEFAULT_EPSILON);

    hipdnn_frontend::graph::BatchnormInferenceAttributesVarianceExt bnAttrs;
    bnAttrs.set_name("batchnorm_fwd_inference_with_variance");
    bnAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto yTensorAttr = graph->batchnorm_inference_variance_ext(xTensorAttr,
                                                               meanTensorAttr,
                                                               varianceTensorAttr,
                                                               scaleTensorAttr,
                                                               biasTensorAttr,
                                                               epsilonTensor,
                                                               bnAttrs);

    if(!yTensorAttr->has_uid())
    {
        yTensorAttr->set_uid(uid++);
    }
    yTensorAttr->set_name("Y");
    yTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));
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
        hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType scaleBiasDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType meanVarianceDataType,
        hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType)
{
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("BatchnormBwdTest");
    graph->set_io_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType))
        .set_compute_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType))
        .set_intermediate_data_type(
            hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    int64_t uid = 1;
    auto xAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "X",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(xAttr));

    auto dyAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dY",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType),
        tensorBundle.dyTensor);
    dyAttr.set_uid(uid++);
    auto dyTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(dyAttr));

    auto scaleAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "scale",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType),
        tensorBundle.scaleTensor);
    scaleAttr.set_uid(uid++);
    auto scaleTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(scaleAttr));

    auto meanAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "mean",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
        tensorBundle.meanTensor);
    meanAttr.set_uid(uid++);
    auto meanTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(meanAttr));

    auto invVarianceAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "inv_variance",
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(meanVarianceDataType),
        tensorBundle.invVarianceTensor);
    invVarianceAttr.set_uid(uid++);
    auto invVarianceTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(invVarianceAttr));

    hipdnn_frontend::graph::BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_name("batchnorm_bwd");
    bnBwdAttrs.set_mean(meanTensorAttr);
    bnBwdAttrs.set_inv_variance(invVarianceTensorAttr);
    bnBwdAttrs.set_compute_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(computeDataType));

    auto outputTensorsAttr
        = graph->batchnorm_backward(dyTensorAttr, xTensorAttr, scaleTensorAttr, bnBwdAttrs);

    auto& dxTensorAttr = outputTensorsAttr[0];
    if(!dxTensorAttr->has_uid())
    {
        dxTensorAttr->set_uid(uid++);
    }
    dxTensorAttr->set_data_type(hipdnn_test_sdk::utilities::sdkToFrontendDataType(inputDataType));

    auto& dScaleTensorAttr = outputTensorsAttr[1];
    if(!dScaleTensorAttr->has_uid())
    {
        dScaleTensorAttr->set_uid(uid++);
    }
    dScaleTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType));

    auto& dBiasTensorAttr = outputTensorsAttr[2];
    if(!dBiasTensorAttr->has_uid())
    {
        dBiasTensorAttr->set_uid(uid++);
    }
    dBiasTensorAttr->set_data_type(
        hipdnn_test_sdk::utilities::sdkToFrontendDataType(scaleBiasDataType));

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
