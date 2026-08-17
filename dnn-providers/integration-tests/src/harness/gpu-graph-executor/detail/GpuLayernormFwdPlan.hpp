// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <hipdnn-gpu-ref/GpuFpReferenceLayernorm.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
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

struct GpuLayernormFwdParams
{
    GpuLayernormFwdParams() = default;
    GpuLayernormFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& scaleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& biasAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& epsilonAttributes,
        const int64_t normalizedDimCount,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* meanAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* invVarianceAttributes
        = nullptr)
        : xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , yTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(yAttributes))
        , scaleTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(scaleAttributes))
        , biasTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(biasAttributes))
        , epsilonTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(epsilonAttributes))
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
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT scaleTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT biasTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT epsilonTensor;
    int64_t normalizedDimCount;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> meanTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> invVarianceTensor;
};

template <typename XDataType,
          typename ScaleBiasDataType,
          typename MeanInvVarianceDataType,
          typename YDataType,
          typename ComputeDataType>
class GpuLayernormFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuLayernormFwdPlan(GpuLayernormFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XDataType> xTensor(
            variantPack.at(_params.xTensor.uid), _params.xTensor.dims, _params.xTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<YDataType> yTensor(
            variantPack.at(_params.yTensor.uid), _params.yTensor.dims, _params.yTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> scaleTensor(
            variantPack.at(_params.scaleTensor.uid),
            _params.scaleTensor.dims,
            _params.scaleTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ScaleBiasDataType> biasTensor(
            variantPack.at(_params.biasTensor.uid),
            _params.biasTensor.dims,
            _params.biasTensor.strides);
        const double epsilon
            = hipdnn_flatbuffers_sdk::utilities::resolveDoubleScalarFromVariantPack(
                _params.epsilonTensor, variantPack, "Epsilon");
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

        hipdnn_gpu_ref::GpuFpReferenceLayernorm::fprop<XDataType,
                                                       ScaleBiasDataType,
                                                       YDataType,
                                                       MeanInvVarianceDataType,
                                                       ComputeDataType>(xTensor,
                                                                        &scaleTensor,
                                                                        &biasTensor,
                                                                        yTensor,
                                                                        epsilon,
                                                                        _params.normalizedDimCount,
                                                                        meanTensorPtr,
                                                                        invVarianceTensorPtr);
    }

private:
    GpuLayernormFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ScaleBiasDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MeanInvVarianceDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuLayernormFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using XDataType = hipdnn_test_sdk::utilities::DataTypeToNative<XDataTypeEnum>;
    using ScaleBiasDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ScaleBiasDataTypeEnum>;
    using MeanInvVarianceDataType
        = hipdnn_test_sdk::utilities::DataTypeToNative<MeanInvVarianceDataTypeEnum>;
    using YDataType = hipdnn_test_sdk::utilities::DataTypeToNative<YDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_LayernormAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->bias_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->epsilon_tensor_uid());
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->mean_tensor_uid());
        }
        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->inv_variance_tensor_uid());
        }

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->scale_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->bias_tensor_uid(), ScaleBiasDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), YDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->epsilon_tensor_uid(), ComputeDataTypeEnum);
        if(nodeAttributes->mean_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->mean_tensor_uid(), MeanInvVarianceDataTypeEnum);
        }
        if(nodeAttributes->inv_variance_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->inv_variance_tensor_uid(), MeanInvVarianceDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_LayernormAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type LayernormAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuLayernormFwdParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            *tensorMap.at(nodeAttributes->scale_tensor_uid()),
            *tensorMap.at(nodeAttributes->bias_tensor_uid()),
            *tensorMap.at(nodeAttributes->epsilon_tensor_uid()),
            nodeAttributes->normalized_dim_count(),
            nodeAttributes->mean_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->mean_tensor_uid().value())
                : nullptr,
            nodeAttributes->inv_variance_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->inv_variance_tensor_uid().value())
                : nullptr);

        return std::make_unique<GpuLayernormFwdPlan<XDataType,
                                                    ScaleBiasDataType,
                                                    MeanInvVarianceDataType,
                                                    YDataType,
                                                    ComputeDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
