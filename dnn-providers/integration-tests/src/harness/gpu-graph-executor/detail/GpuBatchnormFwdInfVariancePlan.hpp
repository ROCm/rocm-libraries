// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <vector>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBatchnormFwdInfVarianceParams
{
    GpuBatchnormFwdInfVarianceParams() = default;
    GpuBatchnormFwdInfVarianceParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& meanAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& varianceAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& epsilonAttributes)
        : xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , biasTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(biasAttributes))
        , meanTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(meanAttributes))
        , varianceTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(varianceAttributes))
        , yTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(yAttributes))
        , epsilonTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(epsilonAttributes))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT meanTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT varianceTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
};

template <typename XDataType,
          typename ScaleBiasDataType,
          typename MeanVarianceDataType,
          typename YDataType,
          typename ComputeDataType>
class GpuBatchnormFwdInfVariancePlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuBatchnormFwdInfVariancePlan(GpuBatchnormFwdInfVarianceParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XDataType> xTensor(
            variantPack.at(_params.xTensor.uid), _params.xTensor.dims, _params.xTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> scaleTensor(
            variantPack.at(_params.scaleTensor.uid),
            _params.scaleTensor.dims,
            _params.scaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> biasTensor(
            variantPack.at(_params.biasTensor.uid),
            _params.biasTensor.dims,
            _params.biasTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType> meanTensor(
            variantPack.at(_params.meanTensor.uid),
            _params.meanTensor.dims,
            _params.meanTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType> varianceTensor(
            variantPack.at(_params.varianceTensor.uid),
            _params.varianceTensor.dims,
            _params.varianceTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<YDataType> yTensor(
            variantPack.at(_params.yTensor.uid), _params.yTensor.dims, _params.yTensor.strides);
        const auto epsilonValue = static_cast<double>(
            hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<ComputeDataType>(
                _params.epsilonTensor, variantPack, "Epsilon"));

        hipdnn_gpu_ref::GpuFpReferenceBatchnorm::fwdInferenceWithVariance<XDataType,
                                                                          ScaleBiasDataType,
                                                                          MeanVarianceDataType,
                                                                          YDataType,
                                                                          ComputeDataType>(
            xTensor, scaleTensor, biasTensor, meanTensor, varianceTensor, yTensor, epsilonValue);
    }

private:
    GpuBatchnormFwdInfVarianceParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuBatchnormFwdInfVariancePlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using XDataType = hipdnn_test_sdk::utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleBiasDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType
        = hipdnn_test_sdk::utilities::DataTypeToNative<MeanVarianceDataTypeEnum>;
    using YDataType = hipdnn_test_sdk::utilities::DataTypeToNative<YDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->variance_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->mean_tensor_uid(), MeanVarianceDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->variance_tensor_uid(), MeanVarianceDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), YDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->epsilon_tensor_uid(), ComputeDataTypeEnum);

        return !anyOperandIsRuntimePassByValue(tensorMap,
                                               {nodeAttributes->x_tensor_uid(),
                                                nodeAttributes->scale_tensor_uid(),
                                                nodeAttributes->bias_tensor_uid(),
                                                nodeAttributes->mean_tensor_uid(),
                                                nodeAttributes->variance_tensor_uid(),
                                                nodeAttributes->y_tensor_uid(),
                                                nodeAttributes->epsilon_tensor_uid()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormInferenceAttributesVarianceExt();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type BatchnormInferenceAttributesVarianceExt");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuBatchnormFwdInfVarianceParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
            *tensorMap.at(nodeAttributes->mean_tensor_uid()),
            *tensorMap.at(nodeAttributes->variance_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            *tensorMap.at(nodeAttributes->epsilon_tensor_uid()));

        return std::make_unique<GpuBatchnormFwdInfVariancePlan<XDataType,
                                                               ScaleBiasDataType,
                                                               MeanVarianceDataType,
                                                               YDataType,
                                                               ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
