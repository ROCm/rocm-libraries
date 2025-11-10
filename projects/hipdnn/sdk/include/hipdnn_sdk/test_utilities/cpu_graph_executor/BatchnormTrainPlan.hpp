// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanUtils.hpp>
#include <hipdnn_sdk/utilities/FlatbufferUtils.hpp>
#include <optional>
#include <variant>

namespace hipdnn_sdk::test_utilities
{

template <typename MeanVarianceDataType>
struct BatchnormTrainParams
{
    BatchnormTrainParams() = default;
    BatchnormTrainParams(
        const hipdnn_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_sdk::data_objects::TensorAttributes& epsilonAttributes,
        // Optional mean/variance tensors
        const hipdnn_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_sdk::data_objects::TensorAttributes* invVarianceAttributes = nullptr,
        // Optional running mean/variance tensors
        const hipdnn_sdk::data_objects::TensorAttributes* momentumAttributes = nullptr,
        const hipdnn_sdk::data_objects::TensorAttributes* prevRunningMeanAttributes = nullptr,
        const hipdnn_sdk::data_objects::TensorAttributes* prevRunningVarianceAttributes = nullptr,
        const hipdnn_sdk::data_objects::TensorAttributes* nextRunningMeanAttributes = nullptr,
        const hipdnn_sdk::data_objects::TensorAttributes* nextRunningVarianceAttributes = nullptr)
        : xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , biasTensor(unpackTensorAttributes(biasAttributes))
        , epsilonTensor(unpackTensorAttributes(epsilonAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , meanTensor(meanAttributes != nullptr
                         ? std::make_optional(unpackTensorAttributes(*meanAttributes))
                         : std::nullopt)
        , invVarianceTensor(invVarianceAttributes != nullptr
                                ? std::make_optional(unpackTensorAttributes(*invVarianceAttributes))
                                : std::nullopt)
        , momentumTensor(momentumAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*momentumAttributes))
                             : std::nullopt)
        , prevRunningMeanTensor(
              prevRunningMeanAttributes != nullptr
                  ? std::make_optional(unpackTensorAttributes(*prevRunningMeanAttributes))
                  : std::nullopt)
        , prevRunningVarianceTensor(
              prevRunningVarianceAttributes != nullptr
                  ? std::make_optional(unpackTensorAttributes(*prevRunningVarianceAttributes))
                  : std::nullopt)
        , nextRunningMeanTensor(
              nextRunningMeanAttributes != nullptr
                  ? std::make_optional(unpackTensorAttributes(*nextRunningMeanAttributes))
                  : std::nullopt)
        , nextRunningVarianceTensor(
              nextRunningVarianceAttributes != nullptr
                  ? std::make_optional(unpackTensorAttributes(*nextRunningVarianceAttributes))
                  : std::nullopt)
    {
    }

    hipdnn_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_sdk::data_objects::TensorAttributesT epsilonTensor;
    hipdnn_sdk::data_objects::TensorAttributesT yTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> invVarianceTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> momentumTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> prevRunningMeanTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> prevRunningVarianceTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> nextRunningMeanTensor;
    std::optional<hipdnn_sdk::data_objects::TensorAttributesT> nextRunningVarianceTensor;
};

template <typename InputDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename ComputeDataType>
class BatchnormTrainPlan : public IGraphNodePlanExecutor
{
public:
    BatchnormTrainPlan(BatchnormTrainParams<MeanVarianceDataType>&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor = createShallowTensor<InputDataType>(
            _params.xTensor, variantPack.at(_params.xTensor.uid));
        auto shallowScaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));
        auto shallowBiasTensor = createShallowTensor<ScaleBiasDataType>(
            _params.biasTensor, variantPack.at(_params.biasTensor.uid));
        auto shallowYTensor = createShallowTensor<InputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        // Extract epsilon from pass-by-value tensor (cast to double)
        double epsilon
            = hipdnn_sdk::utilities::extractDoubleFromTensorValue(_params.epsilonTensor, "Epsilon");

        // Optional batch statistics tensors
        std::unique_ptr<TensorBase<MeanVarianceDataType>> mean;
        std::unique_ptr<TensorBase<MeanVarianceDataType>> invVariance;
        TensorBase<MeanVarianceDataType>* meanPtr = nullptr;
        TensorBase<MeanVarianceDataType>* invVariancePtr = nullptr;

        if(_params.meanTensor.has_value())
        {
            mean = createShallowTensor<MeanVarianceDataType>(
                _params.meanTensor.value(), variantPack.at(_params.meanTensor.value().uid));
            meanPtr = mean.get();
        }

        if(_params.invVarianceTensor.has_value())
        {
            invVariance = createShallowTensor<MeanVarianceDataType>(
                _params.invVarianceTensor.value(),
                variantPack.at(_params.invVarianceTensor.value().uid));
            invVariancePtr = invVariance.get();
        }

        // Optional momentum and running statistics tensors
        std::unique_ptr<TensorBase<MeanVarianceDataType>> prevRunningMean;
        std::unique_ptr<TensorBase<MeanVarianceDataType>> prevRunningVariance;
        std::unique_ptr<TensorBase<MeanVarianceDataType>> nextRunningMean;
        std::unique_ptr<TensorBase<MeanVarianceDataType>> nextRunningVariance;
        TensorBase<MeanVarianceDataType>* prevRunningMeanPtr = nullptr;
        TensorBase<MeanVarianceDataType>* prevRunningVariancePtr = nullptr;
        TensorBase<MeanVarianceDataType>* nextRunningMeanPtr = nullptr;
        TensorBase<MeanVarianceDataType>* nextRunningVariancePtr = nullptr;

        // Extract momentum from pass-by-value tensor if present (cast to double)
        double momentumValue = 0.1;
        if(_params.momentumTensor.has_value())
        {
            momentumValue = hipdnn_sdk::utilities::extractDoubleFromTensorValue(
                _params.momentumTensor.value(), "Momentum");
        }

        if(_params.prevRunningMeanTensor.has_value())
        {
            prevRunningMean = createShallowTensor<MeanVarianceDataType>(
                _params.prevRunningMeanTensor.value(),
                variantPack.at(_params.prevRunningMeanTensor.value().uid));
            prevRunningMeanPtr = prevRunningMean.get();
        }
        if(_params.prevRunningVarianceTensor.has_value())
        {
            prevRunningVariance = createShallowTensor<MeanVarianceDataType>(
                _params.prevRunningVarianceTensor.value(),
                variantPack.at(_params.prevRunningVarianceTensor.value().uid));
            prevRunningVariancePtr = prevRunningVariance.get();
        }
        if(_params.nextRunningMeanTensor.has_value())
        {
            nextRunningMean = createShallowTensor<MeanVarianceDataType>(
                _params.nextRunningMeanTensor.value(),
                variantPack.at(_params.nextRunningMeanTensor.value().uid));
            nextRunningMeanPtr = nextRunningMean.get();
        }
        if(_params.nextRunningVarianceTensor.has_value())
        {
            nextRunningVariance = createShallowTensor<MeanVarianceDataType>(
                _params.nextRunningVarianceTensor.value(),
                variantPack.at(_params.nextRunningVarianceTensor.value().uid));
            nextRunningVariancePtr = nextRunningVariance.get();
        }

        CpuFpReferenceBatchnormImpl<InputDataType,
                                    ScaleBiasDataType,
                                    MeanVarianceDataType,
                                    ComputeDataType>::batchnormFwdTraining(*shallowXTensor,
                                                                           *shallowScaleTensor,
                                                                           *shallowBiasTensor,
                                                                           *shallowYTensor,
                                                                           epsilon,
                                                                           momentumValue,
                                                                           meanPtr,
                                                                           invVariancePtr,
                                                                           prevRunningMeanPtr,
                                                                           prevRunningVariancePtr,
                                                                           nextRunningMeanPtr,
                                                                           nextRunningVariancePtr);
    }

private:
    BatchnormTrainParams<MeanVarianceDataType> _params;
};

template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum>
class BatchnormTrainPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using InputDataType = DataTypeToNative<InputDataTypeEnum>;
    using ScaleBiasDataType = DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType = DataTypeToNative<MeanVarianceDataTypeEnum>;
    using ComputeDataType = DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_BatchnormAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        // Check required tensors
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), InputDataTypeEnum);

        // Optional batch statistics tensors
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid(), MeanVarianceDataTypeEnum);
        }

        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanVarianceDataTypeEnum);
        }

        // Momentum can be any type - will be cast to double during extraction
        if(nodeAttributes->momentum_tensor_uid())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->momentum_tensor_uid());
        }

        // Optional running mean/variance tensors
        if(nodeAttributes->prev_running_mean_tensor_uid())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->prev_running_mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap,
                                       nodeAttributes->prev_running_mean_tensor_uid(),
                                       MeanVarianceDataTypeEnum);
        }

        if(nodeAttributes->prev_running_variance_tensor_uid())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap,
                                         nodeAttributes->prev_running_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap,
                                       nodeAttributes->prev_running_variance_tensor_uid(),
                                       MeanVarianceDataTypeEnum);
        }

        if(nodeAttributes->next_running_mean_tensor_uid())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->next_running_mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap,
                                       nodeAttributes->next_running_mean_tensor_uid(),
                                       MeanVarianceDataTypeEnum);
        }

        if(nodeAttributes->next_running_variance_tensor_uid())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap,
                                         nodeAttributes->next_running_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap,
                                       nodeAttributes->next_running_variance_tensor_uid(),
                                       MeanVarianceDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type BatchnormTrainingAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const hipdnn_sdk::data_objects::TensorAttributes* mean = nullptr;
        const hipdnn_sdk::data_objects::TensorAttributes* invVariance = nullptr;
        const hipdnn_sdk::data_objects::TensorAttributes* momentum = nullptr;
        const hipdnn_sdk::data_objects::TensorAttributes* prevRunningMean = nullptr;
        const hipdnn_sdk::data_objects::TensorAttributes* prevRunningVariance = nullptr;
        const hipdnn_sdk::data_objects::TensorAttributes* nextRunningMean = nullptr;
        const hipdnn_sdk::data_objects::TensorAttributes* nextRunningVariance = nullptr;

        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            mean = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
        }

        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            invVariance = tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value());
        }

        if(nodeAttributes->momentum_tensor_uid())
        {
            momentum = tensorMap.at(nodeAttributes->momentum_tensor_uid().value());
        }

        if(nodeAttributes->prev_running_mean_tensor_uid()
           && nodeAttributes->prev_running_variance_tensor_uid()
           && nodeAttributes->next_running_mean_tensor_uid()
           && nodeAttributes->next_running_variance_tensor_uid())
        {
            prevRunningMean = tensorMap.at(nodeAttributes->prev_running_mean_tensor_uid().value());
            prevRunningVariance
                = tensorMap.at(nodeAttributes->prev_running_variance_tensor_uid().value());
            nextRunningMean = tensorMap.at(nodeAttributes->next_running_mean_tensor_uid().value());
            nextRunningVariance
                = tensorMap.at(nodeAttributes->next_running_variance_tensor_uid().value());
        }

        BatchnormTrainParams<MeanVarianceDataType> params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
            mean,
            invVariance,
            momentum,
            prevRunningMean,
            prevRunningVariance,
            nextRunningMean,
            nextRunningVariance);

        return std::make_unique<BatchnormTrainPlan<InputDataType,
                                                   ScaleBiasDataType,
                                                   MeanVarianceDataType,
                                                   ComputeDataType>>(std::move(params));
    }
};

}
