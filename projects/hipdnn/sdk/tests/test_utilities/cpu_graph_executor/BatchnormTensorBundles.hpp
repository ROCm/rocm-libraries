// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/NodeWrapper.hpp>
#include <hipdnn_sdk/test_utilities/Seeds.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/GraphTensorBundle.hpp>
#include <hipdnn_sdk/utilities/Constants.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

namespace hipdnn_sdk_test_utils
{

struct BatchnormFwdTensorBundle : public GraphTensorBundle
{
    BatchnormFwdTensorBundle(
        const hipdnn_plugin::INodeWrapper& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        unsigned int seed)
        : GraphTensorBundle(tensorMap)
    {
        const auto& attributes
            = node.attributesAs<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>();

        randomizeTensor(attributes.x_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.scale_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.bias_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.mean_tensor_uid(), 0.0f, 1.0f, seed);
        randomizeTensor(attributes.inv_variance_tensor_uid(), 0.1f, 1.0f, seed);
    }
};

template <typename InputDataType, typename ScaleBiasDataType, typename MeanVarianceDataType>
struct BatchnormTrainTensorBundle
{
    BatchnormTrainTensorBundle(const std::vector<int64_t>& dims,
                               unsigned int seed = getGlobalTestSeed(),
                               const TensorLayout& layout = TensorLayout::NCHW,
                               bool useOptionalTensors = false)
        : derivedDims(getDerivedShape(dims))
        , xTensor(dims, layout)
        , scaleTensor(derivedDims)
        , biasTensor(derivedDims)
        , meanTensor(derivedDims)
        , invVarianceTensor(derivedDims)
        , epsilonTensor({1})
        , yTensor(dims, layout)
    {
        xTensor.fillWithRandomValues(
            static_cast<InputDataType>(-1.0f), static_cast<InputDataType>(1.0f), seed);

        scaleTensor.fillWithRandomValues(
            static_cast<ScaleBiasDataType>(-0.1f), static_cast<ScaleBiasDataType>(0.1f), seed);
        biasTensor.fillWithRandomValues(
            static_cast<ScaleBiasDataType>(-0.1f), static_cast<ScaleBiasDataType>(0.1f), seed);

        meanTensor.fillWithRandomValues(static_cast<MeanVarianceDataType>(-0.1f),
                                        static_cast<MeanVarianceDataType>(0.1f),
                                        seed);

        invVarianceTensor.fillWithRandomValues(
            static_cast<MeanVarianceDataType>(1.9f), static_cast<MeanVarianceDataType>(2.0f), seed);

        epsilonTensor.fillWithValue(
            static_cast<MeanVarianceDataType>(static_cast<float>(BATCHNORM_DEFAULT_EPSILON)));

        if(useOptionalTensors)
        {
            momentumTensor = Tensor<MeanVarianceDataType>({1});
            momentumTensor->fillWithValue(static_cast<MeanVarianceDataType>(0.1f));

            prevRunningMeanTensor = Tensor<MeanVarianceDataType>(derivedDims);
            prevRunningMeanTensor->fillWithRandomValues(static_cast<MeanVarianceDataType>(-0.1f),
                                                        static_cast<MeanVarianceDataType>(0.1f),
                                                        seed);

            prevRunningVarianceTensor = Tensor<MeanVarianceDataType>(derivedDims);
            prevRunningVarianceTensor->fillWithRandomValues(static_cast<MeanVarianceDataType>(1.9f),
                                                            static_cast<MeanVarianceDataType>(2.0f),
                                                            seed);

            nextRunningMeanTensor = Tensor<MeanVarianceDataType>(derivedDims);
            nextRunningVarianceTensor = Tensor<MeanVarianceDataType>(derivedDims);
        }
    }

    std::unordered_map<int64_t, void*> createVariantPack(
        const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& scaleTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& biasTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& meanTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& invVarianceTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& epsilonTensorAttr,
        const hipdnn_frontend::graph::TensorAttributes& yTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& momentumTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& prevRunningMeanTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>&
            prevRunningVarianceTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>& nextRunningMeanTensorAttr,
        const std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>&
            nextRunningVarianceTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;

        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[scaleTensorAttr.get_uid()] = scaleTensor.memory().hostData();
        variantPack[biasTensorAttr.get_uid()] = biasTensor.memory().hostData();
        variantPack[meanTensorAttr.get_uid()] = meanTensor.memory().hostData();
        variantPack[invVarianceTensorAttr.get_uid()] = invVarianceTensor.memory().hostData();
        variantPack[epsilonTensorAttr.get_uid()] = epsilonTensor.memory().hostData();
        variantPack[yTensorAttr.get_uid()] = yTensor.memory().hostData();

        //optionals
        if(momentumTensorAttr != nullptr)
        {
            variantPack[momentumTensorAttr->get_uid()] = momentumTensor.value().memory().hostData();
        }

        if(prevRunningMeanTensorAttr != nullptr)
        {
            variantPack[prevRunningMeanTensorAttr->get_uid()]
                = prevRunningMeanTensor.value().memory().hostData();
        }

        if(prevRunningVarianceTensorAttr != nullptr)
        {
            variantPack[prevRunningVarianceTensorAttr->get_uid()]
                = prevRunningVarianceTensor.value().memory().hostData();
        }

        if(nextRunningMeanTensorAttr != nullptr)
        {
            variantPack[nextRunningMeanTensorAttr->get_uid()]
                = nextRunningMeanTensor.value().memory().hostData();
        }

        if(nextRunningVarianceTensorAttr != nullptr)
        {
            variantPack[nextRunningVarianceTensorAttr->get_uid()]
                = nextRunningVarianceTensor.value().memory().hostData();
        }

        return variantPack;
    }

    std::vector<int64_t> derivedDims;
    Tensor<InputDataType> xTensor;
    Tensor<ScaleBiasDataType> scaleTensor;
    Tensor<ScaleBiasDataType> biasTensor;
    Tensor<MeanVarianceDataType> meanTensor;
    Tensor<MeanVarianceDataType> invVarianceTensor;
    Tensor<MeanVarianceDataType> epsilonTensor;
    Tensor<InputDataType> yTensor;

    std::optional<Tensor<MeanVarianceDataType>> momentumTensor;
    std::optional<Tensor<MeanVarianceDataType>> prevRunningMeanTensor;
    std::optional<Tensor<MeanVarianceDataType>> prevRunningVarianceTensor;
    std::optional<Tensor<MeanVarianceDataType>> nextRunningMeanTensor;
    std::optional<Tensor<MeanVarianceDataType>> nextRunningVarianceTensor;
};

template <typename InputType, typename ScaleBiasType, typename MeanVarianceType>
struct BatchnormBwdTensorBundle
{
    BatchnormBwdTensorBundle(const std::vector<int64_t>& dims,
                             unsigned int seed = getGlobalTestSeed(),
                             const TensorLayout& layout = TensorLayout::NCHW)
        : derivedDims(getDerivedShape(dims))
        , xTensor(dims, layout)
        , dyTensor(dims, layout)
        , dxTensor(dims, layout)
        , scaleTensor(derivedDims)
        , dscaleTensor(derivedDims)
        , dbiasTensor(derivedDims)
        , meanTensor(derivedDims)
        , invVarianceTensor(derivedDims)
    {
        xTensor.fillWithRandomValues(
            static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);
        dyTensor.fillWithRandomValues(
            static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed);
        scaleTensor.fillWithRandomValues(
            static_cast<ScaleBiasType>(-0.1f), static_cast<ScaleBiasType>(0.1f), seed);
        meanTensor.fillWithRandomValues(
            static_cast<MeanVarianceType>(-0.1f), static_cast<MeanVarianceType>(0.1f), seed);
        invVarianceTensor.fillWithRandomValues(
            static_cast<MeanVarianceType>(1.9f), static_cast<MeanVarianceType>(2.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& xTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dyTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& scaleTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& meanTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& invVarianceAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dxTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dScaleTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dBiasTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xTensorAttr.get_uid()] = xTensor.memory().hostData();
        variantPack[dyTensorAttr.get_uid()] = dyTensor.memory().hostData();
        variantPack[scaleTensorAttr.get_uid()] = scaleTensor.memory().hostData();
        variantPack[meanTensorAttr.get_uid()] = meanTensor.memory().hostData();
        variantPack[invVarianceAttr.get_uid()] = invVarianceTensor.memory().hostData();
        variantPack[dxTensorAttr.get_uid()] = dxTensor.memory().hostData();
        variantPack[dScaleTensorAttr.get_uid()] = dscaleTensor.memory().hostData();
        variantPack[dBiasTensorAttr.get_uid()] = dbiasTensor.memory().hostData();
        return variantPack;
    }

    std::vector<int64_t> derivedDims;
    Tensor<InputType> xTensor;
    Tensor<InputType> dyTensor;
    Tensor<InputType> dxTensor;
    Tensor<ScaleBiasType> scaleTensor;
    Tensor<ScaleBiasType> dscaleTensor;
    Tensor<ScaleBiasType> dbiasTensor;
    Tensor<MeanVarianceType> meanTensor;
    Tensor<MeanVarianceType> invVarianceTensor;
};
}
