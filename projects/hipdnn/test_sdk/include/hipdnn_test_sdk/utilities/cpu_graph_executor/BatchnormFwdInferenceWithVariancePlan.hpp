// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/PlanUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

struct BatchnormFwdInferenceWithVarianceParams
{
    BatchnormFwdInferenceWithVarianceParams() = default;
    BatchnormFwdInferenceWithVarianceParams(
        const hipdnn_data_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_data_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_data_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_data_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_data_sdk::data_objects::TensorAttributes& meanAttributes,
        const hipdnn_data_sdk::data_objects::TensorAttributes& varianceAttributes)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , biasTensor(unpackTensorAttributes(biasAttributes))
        , meanTensor(unpackTensorAttributes(meanAttributes))
        , varianceTensor(unpackTensorAttributes(varianceAttributes))
    {
    }

    hipdnn_data_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT meanTensor;
    hipdnn_data_sdk::data_objects::TensorAttributesT varianceTensor;
};

template <typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename OutputDataType,
          typename ComputeDataType>
class BatchnormFwdInferenceWithVariancePlan : public IGraphNodePlanExecutor
{
public:
    BatchnormFwdInferenceWithVariancePlan(BatchnormFwdInferenceWithVarianceParams&& params)
        : _params(std::move(params))
    {
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

        auto shallowVarianceTensor = createShallowTensor<MeanVarianceDataType>(
            _params.varianceTensor, variantPack.at(_params.varianceTensor.uid));

        CpuFpReferenceBatchnorm::fwdInferenceWithVariance(*shallowXTensor,
                                                          *shallowScaleTensor,
                                                          *shallowBiasTensor,
                                                          *shallowMeanTensor,
                                                          *shallowVarianceTensor,
                                                          *shallowYTensor);
    }

private:
    BatchnormFwdInferenceWithVarianceParams _params;
};

template <hipdnn_data_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_data_sdk::data_objects::DataType ComputeDataTypeEnum>
class BatchnormFwdInferenceWithVariancePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = DataTypeToNative<XDataTypeEnum>;
    using ScaleBiasDataType = DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType = DataTypeToNative<MeanVarianceDataTypeEnum>;
    using OutputDataType = DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_data_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->variance_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->mean_tensor_uid(), MeanVarianceDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->variance_tensor_uid(), MeanVarianceDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin_sdk::IGraph& graph,
                      const hipdnn_data_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type BatchnormInferenceAttributesVarianceExt");
        }

        const auto& tensorMap = graph.getTensorMap();
        BatchnormFwdInferenceWithVarianceParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
            *tensorMap.at(nodeAttributes->mean_tensor_uid()),
            *tensorMap.at(nodeAttributes->variance_tensor_uid()));

        return std::make_unique<BatchnormFwdInferenceWithVariancePlan<XDataType,
                                                                      ScaleBiasDataType,
                                                                      MeanVarianceDataType,
                                                                      OutputDataType,
                                                                      ComputeDataType>>(
            std::move(params));
    }
};
}
