// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

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

struct BatchnormBwdParams
{
    BatchnormBwdParams() = default;
    BatchnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dscaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dbiasAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
        = nullptr)
        : dyTensor(unpackTensorAttributes(dyAttributes))
        , xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , dxTensor(unpackTensorAttributes(dxAttributes))
        , dscaleTensor(unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(unpackTensorAttributes(dbiasAttributes))
    {
        if(meanAttributes != nullptr && invVarianceAttributes != nullptr)
        {
            meanTensor = unpackTensorAttributes(*meanAttributes);
            invVarianceTensor = unpackTensorAttributes(*invVarianceAttributes);
        }
    }

    BatchnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& meanAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& invVarianceAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dscaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dbiasAttributes)
        : dyTensor(unpackTensorAttributes(dyAttributes))
        , xTensor(unpackTensorAttributes(xAttributes))
        , scaleTensor(unpackTensorAttributes(scaleAttributes))
        , dxTensor(unpackTensorAttributes(dxAttributes))
        , dscaleTensor(unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(unpackTensorAttributes(dbiasAttributes))
        , meanTensor(unpackTensorAttributes(meanAttributes))
        , invVarianceTensor(unpackTensorAttributes(invVarianceAttributes))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dscaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dbiasTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invVarianceTensor;
};

template <typename DyDataType,
          typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename OutputDataType,
          typename ComputeDataType>
class BatchnormBwdPlan : public IGraphNodePlanExecutor
{
public:
    BatchnormBwdPlan(BatchnormBwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.dxTensor.uid, _params.dscaleTensor.uid, _params.dbiasTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowDyTensor = createShallowTensor<DyDataType>(
            _params.dyTensor, variantPack.at(_params.dyTensor.uid));

        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowScaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.scaleTensor, variantPack.at(_params.scaleTensor.uid));

        auto shallowDxTensor = createShallowTensor<OutputDataType>(
            _params.dxTensor, variantPack.at(_params.dxTensor.uid));

        auto shallowDscaleTensor = createShallowTensor<ScaleBiasDataType>(
            _params.dscaleTensor, variantPack.at(_params.dscaleTensor.uid));

        auto shallowDbiasTensor = createShallowTensor<ScaleBiasDataType>(
            _params.dbiasTensor, variantPack.at(_params.dbiasTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>>
            shallowMeanTensor;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<MeanVarianceDataType>>
            shallowInvVarianceTensor;
        if(_params.meanTensor.has_value() && _params.invVarianceTensor.has_value())
        {
            shallowMeanTensor = createShallowTensor<MeanVarianceDataType>(
                _params.meanTensor.value(), variantPack.at(_params.meanTensor.value().uid));

            shallowInvVarianceTensor = createShallowTensor<MeanVarianceDataType>(
                _params.invVarianceTensor.value(),
                variantPack.at(_params.invVarianceTensor.value().uid));
        }

        utilities::CpuFpReferenceBatchnorm::backward(*shallowDyTensor,
                                                     *shallowXTensor,
                                                     *shallowScaleTensor,
                                                     *shallowDxTensor,
                                                     *shallowDscaleTensor,
                                                     *shallowDbiasTensor,
                                                     shallowMeanTensor.get(),
                                                     shallowInvVarianceTensor.get());
    }

private:
    BatchnormBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class BatchnormBwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DyDataType = utilities::DataTypeToNative<DyDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dscale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dbias_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dbias_tensor_uid(), ScaleBiasDataTypeEnum);

        const bool hasMean = nodeAttributes->mean_tensor_uid().has_value();
        const bool hasInvVariance = nodeAttributes->inv_variance_tensor_uid().has_value();
        if(hasMean != hasInvVariance)
        {
            return false;
        }

        if(hasMean)
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid(), MeanVarianceDataTypeEnum);
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanVarianceDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type BatchnormBackwardAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttr = nullptr;
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarAttr = nullptr;

        const bool hasMean = nodeAttributes->mean_tensor_uid().has_value();
        const bool hasInvVariance = nodeAttributes->inv_variance_tensor_uid().has_value();
        if(hasMean && hasInvVariance)
        {
            meanAttr = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            invVarAttr = tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value());
        }

        BatchnormBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->dbias_tensor_uid()),
                                  meanAttr,
                                  invVarAttr);

        return std::make_unique<BatchnormBwdPlan<DyDataType,
                                                 XDataType,
                                                 ScaleBiasDataType,
                                                 MeanVarianceDataType,
                                                 OutputDataType,
                                                 ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
