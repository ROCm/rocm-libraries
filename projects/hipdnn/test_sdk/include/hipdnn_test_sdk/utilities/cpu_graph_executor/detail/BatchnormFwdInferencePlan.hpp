// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct BatchnormFwdInferenceParams
{
    BatchnormFwdInferenceParams() = default;
    BatchnormFwdInferenceParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& meanAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& invVarianceAttributes)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , biasTensor(unpackTensorAttributes(biasAttributes))
        , meanTensor(unpackTensorAttributes(meanAttributes))
        , invVarianceTensor(unpackTensorAttributes(invVarianceAttributes))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT meanTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT invVarianceTensor;
};

template <typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename OutputDataType,
          typename ComputeDataType>
class BatchnormFwdPlan : public IGraphNodePlanExecutor
{
public:
    BatchnormFwdPlan(BatchnormFwdInferenceParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.yTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        auto shallowBiasTensor = createShallowTensor<ScaleBiasDataType>(
            _params.biasTensor, variantPack.at(_params.biasTensor.uid));

        auto shallowMeanTensor = createShallowTensor<MeanVarianceDataType>(
            _params.meanTensor, variantPack.at(_params.meanTensor.uid));

        auto shallowInvVarianceTensor = createShallowTensor<MeanVarianceDataType>(
            _params.invVarianceTensor, variantPack.at(_params.invVarianceTensor.uid));

        utilities::CpuFpReferenceBatchnorm::fwdInference(*shallowXTensor,
                                                         *shallowScaleTensor,
                                                         *shallowBiasTensor,
                                                         *shallowMeanTensor,
                                                         *shallowInvVarianceTensor,
                                                         *shallowYTensor);
    }

private:
    BatchnormFwdInferenceParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class BatchnormFwdInferencePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleBiasDataType = utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType = utilities::DataTypeToNative<MeanVarianceDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->mean_tensor_uid(), MeanVarianceDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanVarianceDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type BatchnormInferenceAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        BatchnormFwdInferenceParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
            *tensorMap.at(nodeAttributes->mean_tensor_uid()),
            *tensorMap.at(nodeAttributes->inv_variance_tensor_uid()));

        return std::make_unique<BatchnormFwdPlan<XDataType,
                                                 ScaleBiasDataType,
                                                 MeanVarianceDataType,
                                                 OutputDataType,
                                                 ComputeDataType>>(std::move(params));
    }
};
} // namespace hipdnn_test_sdk::detail
