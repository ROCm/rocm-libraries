// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceConvolution.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuConvolutionWrwPlanParams
{
    GpuConvolutionWrwPlanParams() = default;
    GpuConvolutionWrwPlanParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dwAttributes,
        const std::vector<int64_t>& prePadding,
        const std::vector<int64_t>& postPadding,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& dilation,
        const hipdnn_flatbuffers_sdk::data_objects::ConvMode convolutionMode)
        : xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , dyTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dyAttributes))
        , dwTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dwAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , dilation(dilation)
        , convMode(convolutionMode)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dwTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    hipdnn_flatbuffers_sdk::data_objects::ConvMode convMode;
};

template <typename XDataType, typename DyDataType, typename DwDataType, typename ComputeDataType>
class GpuConvolutionWrwPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuConvolutionWrwPlan(GpuConvolutionWrwPlanParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XDataType> xTensor(
            variantPack.at(_params.xTensor.uid), _params.xTensor.dims, _params.xTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<DyDataType> dyTensor(
            variantPack.at(_params.dyTensor.uid), _params.dyTensor.dims, _params.dyTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<DwDataType> dwTensor(
            variantPack.at(_params.dwTensor.uid), _params.dwTensor.dims, _params.dwTensor.strides);

        hipdnn_gpu_ref::GpuFpReferenceConvolution::
            wgrad<XDataType, DwDataType, DyDataType, ComputeDataType>(xTensor,
                                                                      dwTensor,
                                                                      dyTensor,
                                                                      _params.stride,
                                                                      _params.dilation,
                                                                      _params.prePadding,
                                                                      _params.postPadding);
    }

private:
    GpuConvolutionWrwPlanParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DwDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuConvolutionWrwPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using XDataType = hipdnn_test_sdk::utilities::DataTypeToNative<XDataTypeEnum>;
    using DyDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DyDataTypeEnum>;
    using DwDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DwDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_ConvolutionWrwAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ConvolutionWrwAttributes");
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dw_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dw_tensor_uid(), DwDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return !anyOperandIsRuntimePassByValue(tensorMap,
                                               {nodeAttributes->x_tensor_uid(),
                                                nodeAttributes->dy_tensor_uid(),
                                                nodeAttributes->dw_tensor_uid()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionWrwAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ConvolutionWrwAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuConvolutionWrwPlanParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->dy_tensor_uid()),
            *tensorMap.at(nodeAttributes->dw_tensor_uid()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->pre_padding()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->post_padding()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->stride()),
            hipdnn_flatbuffers_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->dilation()),
            nodeAttributes->conv_mode());

        return std::make_unique<
            GpuConvolutionWrwPlan<XDataType, DyDataType, DwDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
