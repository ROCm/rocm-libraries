// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/FlatbufferUtils.hpp>
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
    RMSNormFwdParams() = default;
    RMSNormFwdParams(const hipdnn_data_sdk::data_objects::TensorAttributes& xAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& scaleAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& epsilonAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& yAttributes)
        : xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , epsilonTensor(unpackTensorAttributes(epsilonAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
    {
    }

    RMSNormFwdParams(const hipdnn_data_sdk::data_objects::TensorAttributes& xAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& scaleAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& epsilonAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& yAttributes,
                     const hipdnn_data_sdk::data_objects::TensorAttributes& invRmsAttributes)
        : xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , epsilonTensor(unpackTensorAttributes(epsilonAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , invRmsTensor(unpackTensorAttributes(invRmsAttributes))
        , hasInvRms(true)
    {
    }

    hipdnn_data_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT epsilonTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT invRmsTensor;
    bool hasInvRms = false;
    hipdnn_data_sdk::data_objects::TensorAttributesT biasTensor;
    bool hasBias = false;
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

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        double epsilon = hipdnn_data_sdk::utilities::extractDoubleFromTensorValue(
            _params.epsilonTensor, "Epsilon");

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<ScaleDataType>> shallowBiasTensor;
        if(_params.hasBias)
        {
            shallowBiasTensor = createShallowTensor<ScaleDataType>(
                _params.biasTensor, variantPack.at(_params.biasTensor.uid));
        }

        if(_params.hasInvRms)
        {
            auto shallowInvRmsTensor = createShallowTensor<ComputeDataType>(
                _params.invRmsTensor, variantPack.at(_params.invRmsTensor.uid));

            utilities::CpuFpReferenceRMSNorm::
                forward<XDataType, ScaleDataType, OutputDataType, ComputeDataType>(
                    *shallowXTensor,
                    *shallowScaleTensor,
                    *shallowYTensor,
                    epsilon,
                    shallowInvRmsTensor.get(),
                    shallowBiasTensor.get());
        }
        else
        {
            utilities::CpuFpReferenceRMSNorm::
                forward<XDataType, ScaleDataType, OutputDataType, ComputeDataType>(
                    *shallowXTensor,
                    *shallowScaleTensor,
                    *shallowYTensor,
                    epsilon,
                    nullptr,
                    shallowBiasTensor.get());
        }
    }

private:
    RMSNormFwdParams _params;
};

template <hipdnn_data_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType ScaleDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType ComputeDataTypeEnum>
class RMSNormFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleDataType = utilities::DataTypeToNative<ScaleDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_data_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
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
        }

        // inv_rms is optional
        if(nodeAttributes->inv_rms_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_rms_tensor_uid().value());
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_data_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_data_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_RMSNormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type RMSNormAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        RMSNormFwdParams params(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
                                *tensorMap.at(nodeAttributes->y_tensor_uid()));

        if(nodeAttributes->inv_rms_tensor_uid().has_value())
        {
            params.invRmsTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->inv_rms_tensor_uid().value()));
            params.hasInvRms = true;
        }

        if(nodeAttributes->bias_tensor_uid().has_value())
        {
            params.biasTensor
                = unpackTensorAttributes(*tensorMap.at(nodeAttributes->bias_tensor_uid().value()));
            params.hasBias = true;
        }

        return std::make_unique<
            RMSNormFwdPlan<XDataType, ScaleDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};
} // namespace hipdnn_test_sdk::detail
