// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceBatchnorm.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanUtils.hpp>

namespace hipdnn_sdk::test_utilities
{

struct BatchnormBwdParams
{
    BatchnormBwdParams() = default;
    BatchnormBwdParams(const hipdnn_sdk::data_objects::TensorAttributes& dyAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& xAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& meanAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& invVarianceAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& scaleAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& dxAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& dscaleAttributes,
                       const hipdnn_sdk::data_objects::TensorAttributes& dbiasAttributes)
        : dyTensor(unpackTensorAttributes(dyAttributes))
        , xTensor(unpackTensorAttributes(xAttributes))
        , meanTensor(unpackTensorAttributes(meanAttributes))
        , invVarianceTensor(unpackTensorAttributes(invVarianceAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , dxTensor(unpackTensorAttributes(dxAttributes))
        , dscaleTensor(unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(unpackTensorAttributes(dbiasAttributes))
    {
    }

    hipdnn_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_sdk::data_objects::TensorAttributesT meanTensor;
    hipdnn_sdk::data_objects::TensorAttributesT invVarianceTensor;
    hipdnn_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_sdk::data_objects::TensorAttributesT dxTensor;
    hipdnn_sdk::data_objects::TensorAttributesT dscaleTensor;
    hipdnn_sdk::data_objects::TensorAttributesT dbiasTensor;
};

template <typename InputDataType, typename ScaleBiasDataType, typename MeanVarianceDataType>
class BatchnormBwdPlan : public IGraphNodePlanExecutor
{
public:
    BatchnormBwdPlan(BatchnormBwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowDyTensor = createShallowTensor<InputDataType>(
            _params.dyTensor, variantPack.at(_params.dyTensor.uid));

        auto shallowXTensor = createShallowTensor<InputDataType>(
            _params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowMeanTensor = createShallowTensor<MeanVarianceDataType>(
            _params.meanTensor, variantPack.at(_params.meanTensor.uid));

        auto shallowInvVarianceTensor = createShallowTensor<MeanVarianceDataType>(
            _params.invVarianceTensor, variantPack.at(_params.invVarianceTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        auto shallowDxTensor = createShallowTensor<InputDataType>(
            _params.dxTensor, variantPack.at(_params.dxTensor.uid));

        auto shallowDscaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.dscaleTensor, variantPack.at(_params.dscaleTensor.uid));

        auto shallowDbiasTensor = createShallowTensor<ScaleBiasDataType>(
            _params.dbiasTensor, variantPack.at(_params.dbiasTensor.uid));

        CpuFpReferenceBatchnormImpl<InputDataType, ScaleBiasDataType, MeanVarianceDataType>::
            batchnormBwd(*shallowDyTensor,
                         *shallowXTensor,
                         *shallowMeanTensor,
                         *shallowInvVarianceTensor,
                         *shallowScaleTensor,
                         *shallowDxTensor,
                         *shallowDscaleTensor,
                         *shallowDbiasTensor);
    }

private:
    BatchnormBwdParams _params;
};

template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_sdk::data_objects::DataType MeanVarianceDataTypeEnum>
class BatchnormBwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using InputDataType = DataTypeToNative<InputDataTypeEnum>;
    using ScaleBiasDataType = DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType = DataTypeToNative<MeanVarianceDataTypeEnum>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        if(!nodeAttributes->mean_tensor_uid().has_value()
           || !nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            throw std::runtime_error(
                "BatchnormBackwardAttributes mean or inv_variance tensor is optional.  Cpu ref "
                "implementation currently doesnt support optional tensors");
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid().value());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid().value());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dscale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dbias_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->mean_tensor_uid().value(), MeanVarianceDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->inv_variance_tensor_uid().value(), MeanVarianceDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dbias_tensor_uid(), ScaleBiasDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type BatchnormBackwardAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        BatchnormBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->mean_tensor_uid().value()),
                                  *tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value()),
                                  *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->dbias_tensor_uid()));

        return std::make_unique<
            BatchnormBwdPlan<InputDataType, ScaleBiasDataType, MeanVarianceDataType>>(
            std::move(params));
    }
};

}
