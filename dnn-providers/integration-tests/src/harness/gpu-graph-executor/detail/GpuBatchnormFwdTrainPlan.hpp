// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBatchnormFwdTrainParams
{
    GpuBatchnormFwdTrainParams() = default;
    GpuBatchnormFwdTrainParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& inputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& epsilonAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& momentumAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* prevRunningMeanAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* prevRunningVarianceAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* nextRunningMeanAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* nextRunningVarianceAttributes
        = nullptr)
        : inputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(inputAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , biasTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(biasAttributes))
        , outputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(outputAttributes))
        , epsilonTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(epsilonAttributes))
        , momentumTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(momentumAttributes))
        , meanTensor(meanAttributes != nullptr
                         ? std::make_optional(
                               hipdnn_test_sdk::detail::unpackTensorAttributes(*meanAttributes))
                         : std::nullopt)
        , invVarianceTensor(
              invVarianceAttributes != nullptr
                  ? std::make_optional(
                        hipdnn_test_sdk::detail::unpackTensorAttributes(*invVarianceAttributes))
                  : std::nullopt)
        , prevRunningMeanTensor(
              prevRunningMeanAttributes != nullptr
                  ? std::make_optional(
                        hipdnn_test_sdk::detail::unpackTensorAttributes(*prevRunningMeanAttributes))
                  : std::nullopt)
        , prevRunningVarianceTensor(
              prevRunningVarianceAttributes != nullptr
                  ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                        *prevRunningVarianceAttributes))
                  : std::nullopt)
        , nextRunningMeanTensor(
              nextRunningMeanAttributes != nullptr
                  ? std::make_optional(
                        hipdnn_test_sdk::detail::unpackTensorAttributes(*nextRunningMeanAttributes))
                  : std::nullopt)
        , nextRunningVarianceTensor(
              nextRunningVarianceAttributes != nullptr
                  ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                        *nextRunningVarianceAttributes))
                  : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT inputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT momentumTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invVarianceTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> prevRunningMeanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>
        prevRunningVarianceTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> nextRunningMeanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>
        nextRunningVarianceTensor;
};

template <typename InputDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename OutputDataType,
          typename ComputeDataType>
class GpuBatchnormFwdTrainPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuBatchnormFwdTrainPlan(GpuBatchnormFwdTrainParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<InputDataType> inputTensor(
            variantPack.at(_params.inputTensor.uid),
            _params.inputTensor.dims,
            _params.inputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> scaleTensor(
            variantPack.at(_params.scaleTensor.uid),
            _params.scaleTensor.dims,
            _params.scaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> biasTensor(
            variantPack.at(_params.biasTensor.uid),
            _params.biasTensor.dims,
            _params.biasTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> outputTensor(
            variantPack.at(_params.outputTensor.uid),
            _params.outputTensor.dims,
            _params.outputTensor.strides);

        const auto epsilonValue
            = hipdnn_flatbuffers_sdk::utilities::resolveDoubleScalarFromVariantPack(
                _params.epsilonTensor, variantPack, "Epsilon");
        const auto momentumValue
            = hipdnn_flatbuffers_sdk::utilities::resolveDoubleScalarFromVariantPack(
                _params.momentumTensor, variantPack, "Momentum");

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>> meanTensor;
        if(_params.meanTensor.has_value())
        {
            meanTensor.emplace(variantPack.at(_params.meanTensor->uid),
                               _params.meanTensor->dims,
                               _params.meanTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>> invVarianceTensor;
        if(_params.invVarianceTensor.has_value())
        {
            invVarianceTensor.emplace(variantPack.at(_params.invVarianceTensor->uid),
                                      _params.invVarianceTensor->dims,
                                      _params.invVarianceTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>> prevRunningMeanTensor;
        if(_params.prevRunningMeanTensor.has_value())
        {
            prevRunningMeanTensor.emplace(variantPack.at(_params.prevRunningMeanTensor->uid),
                                          _params.prevRunningMeanTensor->dims,
                                          _params.prevRunningMeanTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>>
            prevRunningVarianceTensor;
        if(_params.prevRunningVarianceTensor.has_value())
        {
            prevRunningVarianceTensor.emplace(
                variantPack.at(_params.prevRunningVarianceTensor->uid),
                _params.prevRunningVarianceTensor->dims,
                _params.prevRunningVarianceTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>> nextRunningMeanTensor;
        if(_params.nextRunningMeanTensor.has_value())
        {
            nextRunningMeanTensor.emplace(variantPack.at(_params.nextRunningMeanTensor->uid),
                                          _params.nextRunningMeanTensor->dims,
                                          _params.nextRunningMeanTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>>
            nextRunningVarianceTensor;
        if(_params.nextRunningVarianceTensor.has_value())
        {
            nextRunningVarianceTensor.emplace(
                variantPack.at(_params.nextRunningVarianceTensor->uid),
                _params.nextRunningVarianceTensor->dims,
                _params.nextRunningVarianceTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceBatchnorm::fwdTraining<InputDataType,
                                                             ScaleBiasDataType,
                                                             MeanVarianceDataType,
                                                             OutputDataType,
                                                             ComputeDataType>(
            inputTensor,
            scaleTensor,
            biasTensor,
            outputTensor,
            epsilonValue,
            momentumValue,
            meanTensor.has_value() ? &meanTensor.value() : nullptr,
            invVarianceTensor.has_value() ? &invVarianceTensor.value() : nullptr,
            prevRunningMeanTensor.has_value() ? &prevRunningMeanTensor.value() : nullptr,
            prevRunningVarianceTensor.has_value() ? &prevRunningVarianceTensor.value() : nullptr,
            nextRunningMeanTensor.has_value() ? &nextRunningMeanTensor.value() : nullptr,
            nextRunningVarianceTensor.has_value() ? &nextRunningVarianceTensor.value() : nullptr);
    }

private:
    GpuBatchnormFwdTrainParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuBatchnormFwdTrainPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using InputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<InputDataTypeEnum>;
    using ScaleBiasDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType
        = hipdnn_test_sdk::utilities::DataTypeToNative<MeanVarianceDataTypeEnum>;
    using OutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
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

        // momentum tensor is required for batchnorm forward training
        if(!nodeAttributes->momentum_tensor_uid().has_value())
        {
            return false;
        }

        // mean and inv_variance tensors must either both be present or both be absent
        if(nodeAttributes->mean_tensor_uid().has_value()
           != nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            return false;
        }

        // running stat tensors must either all be present or all be absent
        if(nodeAttributes->prev_running_mean_tensor_uid().has_value()
               != nodeAttributes->prev_running_variance_tensor_uid().has_value()
           || nodeAttributes->prev_running_mean_tensor_uid().has_value()
                  != nodeAttributes->next_running_mean_tensor_uid().has_value()
           || nodeAttributes->prev_running_mean_tensor_uid().has_value()
                  != nodeAttributes->next_running_variance_tensor_uid().has_value())
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->momentum_tensor_uid().value());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->epsilon_tensor_uid(), ComputeDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->momentum_tensor_uid().value(), ComputeDataTypeEnum);

        std::vector<int64_t> operandUids = {nodeAttributes->x_tensor_uid(),
                                            nodeAttributes->scale_tensor_uid(),
                                            nodeAttributes->bias_tensor_uid(),
                                            nodeAttributes->y_tensor_uid(),
                                            nodeAttributes->epsilon_tensor_uid(),
                                            nodeAttributes->momentum_tensor_uid().value()};

        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid().value(), MeanVarianceDataTypeEnum);
            operandUids.push_back(nodeAttributes->mean_tensor_uid().value());
        }
        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->inv_variance_tensor_uid().value(),
                              MeanVarianceDataTypeEnum);
            operandUids.push_back(nodeAttributes->inv_variance_tensor_uid().value());
        }
        if(nodeAttributes->prev_running_mean_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->prev_running_mean_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->prev_running_mean_tensor_uid().value(),
                              MeanVarianceDataTypeEnum);
            operandUids.push_back(nodeAttributes->prev_running_mean_tensor_uid().value());
        }
        if(nodeAttributes->prev_running_variance_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap,
                                nodeAttributes->prev_running_variance_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->prev_running_variance_tensor_uid().value(),
                              MeanVarianceDataTypeEnum);
            operandUids.push_back(nodeAttributes->prev_running_variance_tensor_uid().value());
        }
        if(nodeAttributes->next_running_mean_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->next_running_mean_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->next_running_mean_tensor_uid().value(),
                              MeanVarianceDataTypeEnum);
            operandUids.push_back(nodeAttributes->next_running_mean_tensor_uid().value());
        }
        if(nodeAttributes->next_running_variance_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap,
                                nodeAttributes->next_running_variance_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->next_running_variance_tensor_uid().value(),
                              MeanVarianceDataTypeEnum);
            operandUids.push_back(nodeAttributes->next_running_variance_tensor_uid().value());
        }

        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type BatchnormAttributes");
        }

        if(!nodeAttributes->momentum_tensor_uid().has_value())
        {
            throw std::runtime_error(
                "Batchnorm forward training node is missing required momentum tensor");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuBatchnormFwdTrainParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
            *tensorMap.at(nodeAttributes->momentum_tensor_uid().value()),
            nodeAttributes->mean_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->mean_tensor_uid().value())
                : nullptr,
            nodeAttributes->inv_variance_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value())
                : nullptr,
            nodeAttributes->prev_running_mean_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->prev_running_mean_tensor_uid().value())
                : nullptr,
            nodeAttributes->prev_running_variance_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->prev_running_variance_tensor_uid().value())
                : nullptr,
            nodeAttributes->next_running_mean_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->next_running_mean_tensor_uid().value())
                : nullptr,
            nodeAttributes->next_running_variance_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->next_running_variance_tensor_uid().value())
                : nullptr);

        return std::make_unique<GpuBatchnormFwdTrainPlan<InputDataType,
                                                         ScaleBiasDataType,
                                                         MeanVarianceDataType,
                                                         OutputDataType,
                                                         ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
