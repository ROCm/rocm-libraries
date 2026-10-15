// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuBatchnormBwdParams
{
    GpuBatchnormBwdParams() = default;
    GpuBatchnormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dscaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dbiasAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
        = nullptr)
        : dyTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dyAttributes))
        , xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , dxTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dxAttributes))
        , dscaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dbiasAttributes))
    {
        if(meanAttributes != nullptr && invVarianceAttributes != nullptr)
        {
            meanTensor = hipdnn_test_sdk::detail::unpackTensorAttributes(*meanAttributes);
            invVarianceTensor
                = hipdnn_test_sdk::detail::unpackTensorAttributes(*invVarianceAttributes);
        }
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
          typename DxDataType,
          typename ComputeDataType>
class GpuBatchnormBwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuBatchnormBwdPlan(GpuBatchnormBwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<DyDataType> dyTensor(
            variantPack.at(_params.dyTensor.uid), _params.dyTensor.dims, _params.dyTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<XDataType> xTensor(
            variantPack.at(_params.xTensor.uid), _params.xTensor.dims, _params.xTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> scaleTensor(
            variantPack.at(_params.scaleTensor.uid),
            _params.scaleTensor.dims,
            _params.scaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<DxDataType> dxTensor(
            variantPack.at(_params.dxTensor.uid), _params.dxTensor.dims, _params.dxTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> dscaleTensor(
            variantPack.at(_params.dscaleTensor.uid),
            _params.dscaleTensor.dims,
            _params.dscaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> dbiasTensor(
            variantPack.at(_params.dbiasTensor.uid),
            _params.dbiasTensor.dims,
            _params.dbiasTensor.strides);

        std::unique_ptr<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>> meanTensor;
        std::unique_ptr<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>> invVarianceTensor;
        if(_params.meanTensor.has_value() && _params.invVarianceTensor.has_value())
        {
            meanTensor = std::make_unique<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>>(
                variantPack.at(_params.meanTensor->uid),
                _params.meanTensor->dims,
                _params.meanTensor->strides);
            invVarianceTensor
                = std::make_unique<hipdnn_gpu_ref::ShallowGpuTensor<MeanVarianceDataType>>(
                    variantPack.at(_params.invVarianceTensor->uid),
                    _params.invVarianceTensor->dims,
                    _params.invVarianceTensor->strides);
        }

        hipdnn_gpu_ref::GpuFpReferenceBatchnorm::backward<DyDataType,
                                                          XDataType,
                                                          ScaleBiasDataType,
                                                          MeanVarianceDataType,
                                                          DxDataType,
                                                          ComputeDataType>(dyTensor,
                                                                           xTensor,
                                                                           scaleTensor,
                                                                           dxTensor,
                                                                           dscaleTensor,
                                                                           dbiasTensor,
                                                                           meanTensor.get(),
                                                                           invVarianceTensor.get());
    }

private:
    GpuBatchnormBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuBatchnormBwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using DyDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DyDataTypeEnum>;
    using XDataType = hipdnn_test_sdk::utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleBiasDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanVarianceDataType
        = hipdnn_test_sdk::utilities::DataTypeToNative<MeanVarianceDataTypeEnum>;
    using DxDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DxDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        if(nodeAttributes->peer_stats_tensor_uid() != nullptr
           && !nodeAttributes->peer_stats_tensor_uid()->empty())
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
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dbias_tensor_uid(), ScaleBiasDataTypeEnum);

        const bool hasMean = nodeAttributes->mean_tensor_uid().has_value();
        const bool hasInvVariance = nodeAttributes->inv_variance_tensor_uid().has_value();
        if(hasMean != hasInvVariance)
        {
            return false;
        }

        std::vector<int64_t> tensorUids{nodeAttributes->dy_tensor_uid(),
                                        nodeAttributes->x_tensor_uid(),
                                        nodeAttributes->scale_tensor_uid(),
                                        nodeAttributes->dx_tensor_uid(),
                                        nodeAttributes->dscale_tensor_uid(),
                                        nodeAttributes->dbias_tensor_uid()};
        if(hasMean)
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid(), MeanVarianceDataTypeEnum);
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanVarianceDataTypeEnum);
            tensorUids.push_back(nodeAttributes->mean_tensor_uid().value());
            tensorUids.push_back(nodeAttributes->inv_variance_tensor_uid().value());
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        if(anyOperandIsRuntimePassByValue(tensorMap, tensorUids))
        {
            return false;
        }

        const auto* meanAttributes
            = hasMean ? tensorMap.at(nodeAttributes->mean_tensor_uid().value()) : nullptr;
        const auto* invVarianceAttributes
            = hasMean ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value()) : nullptr;
        GpuBatchnormBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->dbias_tensor_uid()),
                                     meanAttributes,
                                     invVarianceAttributes);

        using TensorProps = hipdnn_gpu_ref::GpuFpReferenceBatchnorm::TensorProps;
        const TensorProps dyProps("dy", params.dyTensor.dims, params.dyTensor.strides);
        const TensorProps xProps("input", params.xTensor.dims, params.xTensor.strides);
        const TensorProps scaleProps("scale", params.scaleTensor.dims, params.scaleTensor.strides);
        const TensorProps dxProps("dx", params.dxTensor.dims, params.dxTensor.strides);
        const TensorProps dscaleProps(
            "dscale", params.dscaleTensor.dims, params.dscaleTensor.strides);
        const TensorProps dbiasProps("dbias", params.dbiasTensor.dims, params.dbiasTensor.strides);
        std::optional<TensorProps> meanProps;
        std::optional<TensorProps> invVarianceProps;
        if(params.meanTensor.has_value())
        {
            meanProps.emplace("mean", params.meanTensor->dims, params.meanTensor->strides);
            invVarianceProps.emplace(
                "invVariance", params.invVarianceTensor->dims, params.invVarianceTensor->strides);
        }

        try
        {
            hipdnn_gpu_ref::GpuFpReferenceBatchnorm::validateBackwardMetadata(
                dyProps,
                xProps,
                scaleProps,
                dxProps,
                dscaleProps,
                dbiasProps,
                meanProps ? &*meanProps : nullptr,
                invVarianceProps ? &*invVarianceProps : nullptr);
        }
        catch(const std::invalid_argument&)
        {
            return false;
        }

        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_BatchnormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type BatchnormBackwardAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr;
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
            = nullptr;
        if(nodeAttributes->mean_tensor_uid().has_value()
           && nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            meanAttributes = tensorMap.at(nodeAttributes->mean_tensor_uid().value());
            invVarianceAttributes = tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value());
        }

        GpuBatchnormBwdParams params(*tensorMap.at(nodeAttributes->dy_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->x_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->scale_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->dx_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
                                     *tensorMap.at(nodeAttributes->dbias_tensor_uid()),
                                     meanAttributes,
                                     invVarianceAttributes);

        return std::make_unique<GpuBatchnormBwdPlan<DyDataType,
                                                    XDataType,
                                                    ScaleBiasDataType,
                                                    MeanVarianceDataType,
                                                    DxDataType,
                                                    ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
