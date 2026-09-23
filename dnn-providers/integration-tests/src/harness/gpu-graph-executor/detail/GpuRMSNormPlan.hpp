// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceRMSNorm.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <optional>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// ===========================================================================
// GPU RMSNorm forward and backward plans
// ===========================================================================

struct GpuRMSNormFwdParams
{
    GpuRMSNormFwdParams() = default;
    GpuRMSNormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& inputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& epsilonAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invRmsAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* biasAttributes = nullptr)
        : inputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(inputAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , outputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(outputAttributes))
        , epsilonTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(epsilonAttributes))
        , invRmsTensor(invRmsAttributes != nullptr
                           ? std::make_optional(
                                 hipdnn_test_sdk::detail::unpackTensorAttributes(*invRmsAttributes))
                           : std::nullopt)
        , biasTensor(biasAttributes != nullptr
                         ? std::make_optional(
                               hipdnn_test_sdk::detail::unpackTensorAttributes(*biasAttributes))
                         : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT inputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invRmsTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> biasTensor;
};

struct GpuRMSNormBwdParams
{
    GpuRMSNormBwdParams() = default;
    GpuRMSNormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& gradOutputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& inputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& invRmsAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& gradInputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& gradScaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* gradBiasAttributes = nullptr)
        : gradOutputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(gradOutputAttributes))
        , inputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(inputAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , invRmsTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(invRmsAttributes))
        , gradInputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(gradInputAttributes))
        , gradScaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(gradScaleAttributes))
        , gradBiasTensor(gradBiasAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *gradBiasAttributes))
                             : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT gradOutputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT inputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT invRmsTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT gradInputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT gradScaleTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> gradBiasTensor;
};

// ===========================================================================
// GPU RMSNorm forward and backward plans
// ===========================================================================

template <typename InputDataType,
          typename ScaleDataType,
          typename OutputDataType,
          typename ComputeDataType>
class GpuRMSNormFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuRMSNormFwdPlan(GpuRMSNormFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<InputDataType> inputTensor(
            variantPack.at(_params.inputTensor.uid),
            _params.inputTensor.dims,
            _params.inputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleDataType> scaleTensor(
            variantPack.at(_params.scaleTensor.uid),
            _params.scaleTensor.dims,
            _params.scaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> outputTensor(
            variantPack.at(_params.outputTensor.uid),
            _params.outputTensor.dims,
            _params.outputTensor.strides);
        const auto epsilonValue = static_cast<double>(
            hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<ComputeDataType>(
                _params.epsilonTensor, variantPack, "Epsilon"));

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<ComputeDataType>> invRmsTensor;
        if(_params.invRmsTensor.has_value())
        {
            invRmsTensor.emplace(variantPack.at(_params.invRmsTensor->uid),
                                 _params.invRmsTensor->dims,
                                 _params.invRmsTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<ScaleDataType>> biasTensor;
        if(_params.biasTensor.has_value())
        {
            biasTensor.emplace(variantPack.at(_params.biasTensor->uid),
                               _params.biasTensor->dims,
                               _params.biasTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceRMSNorm::
            fprop<InputDataType, ScaleDataType, OutputDataType, ComputeDataType>(
                inputTensor,
                scaleTensor,
                outputTensor,
                epsilonValue,
                invRmsTensor.has_value() ? &invRmsTensor.value() : nullptr,
                biasTensor.has_value() ? &biasTensor.value() : nullptr);
    }

private:
    GpuRMSNormFwdParams _params;
};

template <typename GradOutputDataType,
          typename InputDataType,
          typename ScaleDataType,
          typename GradInputDataType,
          typename ComputeDataType>
class GpuRMSNormBwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuRMSNormBwdPlan(GpuRMSNormBwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<GradOutputDataType> gradOutputTensor(
            variantPack.at(_params.gradOutputTensor.uid),
            _params.gradOutputTensor.dims,
            _params.gradOutputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<InputDataType> inputTensor(
            variantPack.at(_params.inputTensor.uid),
            _params.inputTensor.dims,
            _params.inputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleDataType> scaleTensor(
            variantPack.at(_params.scaleTensor.uid),
            _params.scaleTensor.dims,
            _params.scaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ComputeDataType> invRmsTensor(
            variantPack.at(_params.invRmsTensor.uid),
            _params.invRmsTensor.dims,
            _params.invRmsTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<GradInputDataType> gradInputTensor(
            variantPack.at(_params.gradInputTensor.uid),
            _params.gradInputTensor.dims,
            _params.gradInputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleDataType> gradScaleTensor(
            variantPack.at(_params.gradScaleTensor.uid),
            _params.gradScaleTensor.dims,
            _params.gradScaleTensor.strides);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<ScaleDataType>> gradBiasTensor;
        if(_params.gradBiasTensor.has_value())
        {
            gradBiasTensor.emplace(variantPack.at(_params.gradBiasTensor->uid),
                                   _params.gradBiasTensor->dims,
                                   _params.gradBiasTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceRMSNorm::bprop<GradOutputDataType,
                                                     InputDataType,
                                                     ScaleDataType,
                                                     GradInputDataType,
                                                     ComputeDataType>(
            gradOutputTensor,
            inputTensor,
            scaleTensor,
            invRmsTensor,
            gradInputTensor,
            gradScaleTensor,
            gradBiasTensor.has_value() ? &gradBiasTensor.value() : nullptr);
    }

private:
    GpuRMSNormBwdParams _params;
};

// ==========================================================================
// GPU RMSNorm forward and backward plan builders
// ==========================================================================

template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuRMSNormFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using InputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<InputDataTypeEnum>;
    using ScaleDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using OutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->epsilon_tensor_uid(), ComputeDataTypeEnum);

        std::vector<int64_t> operandUids = {nodeAttributes->x_tensor_uid(),
                                            nodeAttributes->scale_tensor_uid(),
                                            nodeAttributes->y_tensor_uid(),
                                            nodeAttributes->epsilon_tensor_uid()};

        if(nodeAttributes->inv_rms_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_rms_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_rms_tensor_uid().value(), ComputeDataTypeEnum);
            operandUids.push_back(nodeAttributes->inv_rms_tensor_uid().value());
        }

        if(nodeAttributes->bias_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->bias_tensor_uid().value(), ScaleDataTypeEnum);
            operandUids.push_back(nodeAttributes->bias_tensor_uid().value());
        }

        // Reject if any operand is runtime pass-by-value
        // The plan cannot resolve a PBV host scalar
        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type RMSNormAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuRMSNormFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
                                   nodeAttributes->inv_rms_tensor_uid().has_value()
                                       ? tensorMap.at(nodeAttributes->inv_rms_tensor_uid().value())
                                       : nullptr,
                                   nodeAttributes->bias_tensor_uid().has_value()
                                       ? tensorMap.at(nodeAttributes->bias_tensor_uid().value())
                                       : nullptr);

        return std::make_unique<
            GpuRMSNormFwdPlan<InputDataType, ScaleDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType GradOutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType GradInputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuRMSNormBwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using GradOutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<GradOutputDataTypeEnum>;
    using InputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<InputDataTypeEnum>;
    using ScaleDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using GradInputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<GradInputDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_rms_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dscale_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), GradOutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->inv_rms_tensor_uid(), ComputeDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), GradInputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleDataTypeEnum);

        std::vector<int64_t> operandUids = {nodeAttributes->dy_tensor_uid(),
                                            nodeAttributes->x_tensor_uid(),
                                            nodeAttributes->scale_tensor_uid(),
                                            nodeAttributes->inv_rms_tensor_uid(),
                                            nodeAttributes->dx_tensor_uid(),
                                            nodeAttributes->dscale_tensor_uid()};

        if(nodeAttributes->dbias_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dbias_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->dbias_tensor_uid().value(), ScaleDataTypeEnum);
            operandUids.push_back(nodeAttributes->dbias_tensor_uid().value());
        }

        // Reject if any operand is runtime pass-by-value
        // The plan cannot resolve a PBV host scalar
        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type RMSNormBackwardAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuRMSNormBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->inv_rms_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                                   nodeAttributes->dbias_tensor_uid().has_value()
                                       ? tensorMap.at(nodeAttributes->dbias_tensor_uid().value())
                                       : nullptr);

        return std::make_unique<GpuRMSNormBwdPlan<GradOutputDataType,
                                                  InputDataType,
                                                  ScaleDataType,
                                                  GradInputDataType,
                                                  ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
