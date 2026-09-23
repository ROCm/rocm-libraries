// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn-gpu-ref/GpuFpReferenceLayernorm.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <stdexcept>
#include <unordered_map>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuLayernormBwdParams
{
    GpuLayernormBwdParams() = default;
    GpuLayernormBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dscaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dbiasAttributes,
        const int64_t normalizedDimCount,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* epsilonAttributes = nullptr)
        : dyTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dyAttributes))
        , xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , dxTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dxAttributes))
        , dscaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dscaleAttributes))
        , dbiasTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dbiasAttributes))
        , normalizedDimCount(normalizedDimCount)
        , meanTensor(meanAttributes != nullptr
                         ? std::make_optional(
                               hipdnn_test_sdk::detail::unpackTensorAttributes(*meanAttributes))
                         : std::nullopt)
        , invVarianceTensor(
              invVarianceAttributes != nullptr
                  ? std::make_optional(
                        hipdnn_test_sdk::detail::unpackTensorAttributes(*invVarianceAttributes))
                  : std::nullopt)
        , epsilonTensor(epsilonAttributes != nullptr
                            ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                  *epsilonAttributes))
                            : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dscaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dbiasTensor;
    int64_t normalizedDimCount;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invVarianceTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> epsilonTensor;
};

template <typename DyDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType,
          typename DxDataType,
          typename ComputeDataType>
class GpuLayernormBwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuLayernormBwdPlan(GpuLayernormBwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<DyDataType> dyTensor(
            variantPack.at(_params.dyTensor.uid), _params.dyTensor.dims, _params.dyTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<DxDataType> xTensor(
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
        std::unique_ptr<hipdnn_gpu_ref::ShallowGpuTensor<MeanInvVarianceDataType>> meanTensor;
        hipdnn_gpu_ref::ShallowGpuTensor<MeanInvVarianceDataType>* meanTensorPtr = nullptr;
        if(_params.meanTensor.has_value())
        {
            meanTensor
                = std::make_unique<hipdnn_gpu_ref::ShallowGpuTensor<MeanInvVarianceDataType>>(
                    variantPack.at(_params.meanTensor.value().uid),
                    _params.meanTensor.value().dims,
                    _params.meanTensor.value().strides);
            meanTensorPtr = meanTensor.get();
        }
        std::unique_ptr<hipdnn_gpu_ref::ShallowGpuTensor<MeanInvVarianceDataType>>
            invVarianceTensor;
        hipdnn_gpu_ref::ShallowGpuTensor<MeanInvVarianceDataType>* invVarianceTensorPtr = nullptr;
        if(_params.invVarianceTensor.has_value())
        {
            invVarianceTensor
                = std::make_unique<hipdnn_gpu_ref::ShallowGpuTensor<MeanInvVarianceDataType>>(
                    variantPack.at(_params.invVarianceTensor.value().uid),
                    _params.invVarianceTensor.value().dims,
                    _params.invVarianceTensor.value().strides);
            invVarianceTensorPtr = invVarianceTensor.get();
        }
        double epsilon = hipdnn_data_sdk::utilities::LAYERNORM_DEFAULT_EPSILON;
        if(_params.epsilonTensor.has_value())
        {
            epsilon = static_cast<double>(
                hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<ComputeDataType>(
                    _params.epsilonTensor.value(), variantPack, "Epsilon"));
        }

        hipdnn_gpu_ref::GpuFpReferenceLayernorm::bprop<DyDataType,
                                                       ScaleBiasDataType,
                                                       DxDataType,
                                                       MeanInvVarianceDataType,
                                                       ComputeDataType>(dyTensor,
                                                                        xTensor,
                                                                        scaleTensor,
                                                                        dxTensor,
                                                                        dscaleTensor,
                                                                        dbiasTensor,
                                                                        epsilon,
                                                                        meanTensorPtr,
                                                                        invVarianceTensorPtr,
                                                                        _params.normalizedDimCount);
    }

private:
    GpuLayernormBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuLayernormBwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using DyDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DyDataTypeEnum>;
    using ScaleBiasDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanInvVarianceDataType
        = hipdnn_test_sdk::utilities::DataTypeToNative<MeanInvVarianceDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_LayernormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        std::vector<int64_t> operandUids = {nodeAttributes->dy_tensor_uid(),
                                            nodeAttributes->x_tensor_uid(),
                                            nodeAttributes->scale_tensor_uid(),
                                            nodeAttributes->dx_tensor_uid(),
                                            nodeAttributes->dscale_tensor_uid(),
                                            nodeAttributes->dbias_tensor_uid()};

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dscale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dbias_tensor_uid());
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid().value());
            operandUids.push_back(nodeAttributes->mean_tensor_uid().value());
        }
        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid().value());
            operandUids.push_back(nodeAttributes->inv_variance_tensor_uid().value());
        }
        if(nodeAttributes->epsilon_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid().value());
            operandUids.push_back(nodeAttributes->epsilon_tensor_uid().value());
        }

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), DxDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dscale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dbias_tensor_uid(), ScaleBiasDataTypeEnum);
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid().value(), MeanInvVarianceDataTypeEnum);
        }
        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->inv_variance_tensor_uid().value(),
                              MeanInvVarianceDataTypeEnum);
        }
        if(nodeAttributes->epsilon_tensor_uid().has_value())
        {
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->epsilon_tensor_uid().value(), ComputeDataTypeEnum);
        }

        return !anyOperandIsRuntimePassByValue(tensorMap, operandUids);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_LayernormBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type LayernormBackwardAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuLayernormBwdParams params(
            *tensorMap.at(nodeAttributes->dy_tensor_uid()),
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->dx_tensor_uid()),
            *tensorMap.at(nodeAttributes->dscale_tensor_uid()),
            *tensorMap.at(nodeAttributes->dbias_tensor_uid()),
            nodeAttributes->normalized_dim_count(),
            nodeAttributes->mean_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->mean_tensor_uid().value())
                : nullptr,
            nodeAttributes->inv_variance_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value())
                : nullptr,
            nodeAttributes->epsilon_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->epsilon_tensor_uid().value())
                : nullptr);

        return std::make_unique<GpuLayernormBwdPlan<DyDataType,
                                                    ScaleBiasDataType,
                                                    MeanInvVarianceDataType,
                                                    DxDataType,
                                                    ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
