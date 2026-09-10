// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceReduction.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
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

struct GpuReductionParams
{
    GpuReductionParams() = default;
    GpuReductionParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& inputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::ReductionMode reductionMode)
        : inputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(inputAttributes))
        , outputTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(outputAttributes))
        , reductionMode(reductionMode)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT inputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    hipdnn_flatbuffers_sdk::data_objects::ReductionMode reductionMode;
};

template <typename InputDataType, typename OutputDataType, typename ComputeDataType>
class GpuReductionPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuReductionPlan(GpuReductionParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<InputDataType> inputTensor(
            variantPack.at(_params.inputTensor.uid),
            _params.inputTensor.dims,
            _params.inputTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> outputTensor(
            variantPack.at(_params.outputTensor.uid),
            _params.outputTensor.dims,
            _params.outputTensor.strides);

        hipdnn_gpu_ref::GpuFpReferenceReduction::
            reduce<InputDataType, OutputDataType, ComputeDataType>(
                inputTensor, outputTensor, _params.reductionMode);
    }

private:
    GpuReductionParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuReductionPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using InputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<InputDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_ReductionAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->in_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->out_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->in_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->out_tensor_uid(), OutputDataTypeEnum);

        using hipdnn_flatbuffers_sdk::data_objects::ReductionMode;
        auto mode = nodeAttributes->mode();
        switch(mode)
        {
        case ReductionMode::ADD:
        case ReductionMode::AVG:
        case ReductionMode::AMAX:
        case ReductionMode::NORM1:
        case ReductionMode::NORM2:
        case ReductionMode::MUL:
        case ReductionMode::MUL_NO_ZEROS:
        case ReductionMode::MIN_OP:
        case ReductionMode::MAX_OP:
            break;
        default:
            return false;
        }

        return !anyOperandIsRuntimePassByValue(
            tensorMap, {nodeAttributes->in_tensor_uid(), nodeAttributes->out_tensor_uid()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ReductionAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::invalid_argument("Node attributes are not of type ReductionAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuReductionParams params(*tensorMap.at(nodeAttributes->in_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->out_tensor_uid()),
                                  nodeAttributes->mode());
        return std::make_unique<GpuReductionPlan<InputDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
