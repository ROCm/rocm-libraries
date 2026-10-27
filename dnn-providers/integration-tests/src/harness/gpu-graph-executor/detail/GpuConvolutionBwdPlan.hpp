// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn-gpu-ref/GpuFpReferenceConvolution.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
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

struct GpuConvolutionBwdPlanParams
{
    GpuConvolutionBwdPlanParams() = default;
    GpuConvolutionBwdPlanParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dyAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& wAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dxAttributes,
        const std::vector<int64_t>& prePadding,
        const std::vector<int64_t>& postPadding,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& dilation,
        const hipdnn_flatbuffers_sdk::data_objects::ConvMode convolutionMode)
        : dyTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dyAttributes))
        , wTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(wAttributes))
        , dxTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(dxAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , dilation(dilation)
        , convMode(convolutionMode)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dyTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT wTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dxTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    hipdnn_flatbuffers_sdk::data_objects::ConvMode convMode;
};

template <typename DyDataType, typename WDataType, typename DxDataType, typename ComputeDataType>
class GpuConvolutionBwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuConvolutionBwdPlan(GpuConvolutionBwdPlanParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<DyDataType> dyTensor(
            variantPack.at(_params.dyTensor.uid), _params.dyTensor.dims, _params.dyTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<WDataType> wTensor(
            variantPack.at(_params.wTensor.uid), _params.wTensor.dims, _params.wTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<DxDataType> dxTensor(
            variantPack.at(_params.dxTensor.uid), _params.dxTensor.dims, _params.dxTensor.strides);

        hipdnn_gpu_ref::GpuFpReferenceConvolution::
            dgrad<DxDataType, WDataType, DyDataType, ComputeDataType>(dxTensor,
                                                                      wTensor,
                                                                      dyTensor,
                                                                      _params.stride,
                                                                      _params.dilation,
                                                                      _params.prePadding,
                                                                      _params.postPadding);
    }

private:
    GpuConvolutionBwdPlanParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType WDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DxDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuConvolutionBwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using DyDataType = hipdnn_test_sdk::utilities::DataTypeToNative<DyDataTypeEnum>;
    using WDataType = hipdnn_test_sdk::utilities::DataTypeToNative<WDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_ConvolutionBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->w_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dx_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->w_tensor_uid(), WDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dx_tensor_uid(), DxDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return !anyOperandIsRuntimePassByValue(tensorMap,
                                               {nodeAttributes->dy_tensor_uid(),
                                                nodeAttributes->w_tensor_uid(),
                                                nodeAttributes->dx_tensor_uid()});
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ConvolutionBwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuConvolutionBwdPlanParams params(
            *tensorMap.at(nodeAttributes->dy_tensor_uid()),
            *tensorMap.at(nodeAttributes->w_tensor_uid()),
            *tensorMap.at(nodeAttributes->dx_tensor_uid()),
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
            GpuConvolutionBwdPlan<DyDataType, WDataType, DxDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
