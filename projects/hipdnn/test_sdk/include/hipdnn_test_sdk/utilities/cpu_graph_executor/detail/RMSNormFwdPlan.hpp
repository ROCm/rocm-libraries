// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <optional>
#include <variant>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceRMSNorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct RMSNormFwdParams
{
    RMSNormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& epsilonAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invRmsAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* biasAttributes = nullptr)
        : xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , epsilonTensor(unpackTensorAttributes(epsilonAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , invRmsTensor(invRmsAttributes != nullptr
                           ? std::make_optional(unpackTensorAttributes(*invRmsAttributes))
                           : std::nullopt)
        , biasTensor(biasAttributes != nullptr
                         ? std::make_optional(unpackTensorAttributes(*biasAttributes))
                         : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invRmsTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> biasTensor;
};

template <typename XDataType,
          typename ScaleDataType,
          typename OutputDataType,
          typename ComputeDataType>
class RMSNormFwdPlan : public IGraphNodePlanExecutor
{
public:
    RMSNormFwdPlan(RMSNormFwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids = {_params.yTensor.uid};
        if(_params.invRmsTensor.has_value())
        {
            ids.push_back(_params.invRmsTensor.value().uid);
        }
        return ids;
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        const double epsilon = hipdnn_flatbuffers_sdk::utilities::extractDoubleFromTensorValue(
            _params.epsilonTensor, "Epsilon");

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>> shallowBiasTensor;
        if(_params.biasTensor.has_value())
        {
            shallowBiasTensor = createShallowTensor<ScaleDataType>(
                *_params.biasTensor, variantPack.at(_params.biasTensor->uid));
        }

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<ComputeDataType>>
            shallowInvRmsTensor;
        if(_params.invRmsTensor.has_value())
        {
            shallowInvRmsTensor = createShallowTensor<ComputeDataType>(
                *_params.invRmsTensor, variantPack.at(_params.invRmsTensor->uid));
        }

        utilities::CpuFpReferenceRMSNorm::
            forward<XDataType, ScaleDataType, OutputDataType, ComputeDataType>(
                *shallowXTensor,
                *shallowScaleTensor,
                *shallowYTensor,
                epsilon,
                shallowInvRmsTensor.get(),
                shallowBiasTensor.get());
    }

private:
    RMSNormFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class RMSNormFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleDataType = utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

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

        const auto* nodeAttributes = node.attributes_as_RMSNormAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleDataTypeEnum);

        // bias is optional
        if(nodeAttributes->bias_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->bias_tensor_uid().value(), ScaleDataTypeEnum);
        }

        // inv_rms is optional
        if(nodeAttributes->inv_rms_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_rms_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_rms_tensor_uid().value(), ComputeDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type RMSNormAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const auto* invRmsPtr = nodeAttributes->inv_rms_tensor_uid().has_value()
                                    ? tensorMap.at(nodeAttributes->inv_rms_tensor_uid().value())
                                    : nullptr;
        const auto* biasPtr = nodeAttributes->bias_tensor_uid().has_value()
                                  ? tensorMap.at(nodeAttributes->bias_tensor_uid().value())
                                  : nullptr;

        RMSNormFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
                                *tensorMap.at(nodeAttributes->y_tensor_uid()),
                                invRmsPtr,
                                biasPtr);

        return std::make_unique<
            RMSNormFwdPlan<XDataType, ScaleDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};
} // namespace hipdnn_test_sdk::detail
