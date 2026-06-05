// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuConvolutionFwdParams
{
    GpuConvolutionFwdParams() = default;
    GpuConvolutionFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& wAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const std::vector<int64_t>& prePadding,
        const std::vector<int64_t>& postPadding,
        const std::vector<int64_t>& stride,
        const std::vector<int64_t>& dilation,
        const hipdnn_flatbuffers_sdk::data_objects::ConvMode convolutionMode)
        : xTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(xAttributes))
        , wTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(wAttributes))
        , yTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(yAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , dilation(dilation)
        , convMode(convolutionMode)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT wTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    hipdnn_flatbuffers_sdk::data_objects::ConvMode convMode;
};

template <typename XDataType, typename WDataType, typename OutputDataType, typename ComputeDataType>
class GpuConvolutionFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuConvolutionFwdPlan(GpuConvolutionFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<XDataType> xTensor(
            variantPack.at(_params.xTensor.uid), _params.xTensor.dims, _params.xTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<WDataType> wTensor(
            variantPack.at(_params.wTensor.uid), _params.wTensor.dims, _params.wTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<OutputDataType> yTensor(
            variantPack.at(_params.yTensor.uid), _params.yTensor.dims, _params.yTensor.strides);

        hipdnn_gpu_ref::GpuFpReferenceConvolution::
            fprop<XDataType, WDataType, OutputDataType, ComputeDataType>(xTensor,
                                                                         wTensor,
                                                                         yTensor,
                                                                         _params.stride,
                                                                         _params.dilation,
                                                                         _params.prePadding,
                                                                         _params.postPadding);
    }

private:
    GpuConvolutionFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType WDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GpuConvolutionFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using XDataType = hipdnn_test_sdk::utilities::DataTypeToNative<XDataTypeEnum>;
    using WDataType = hipdnn_test_sdk::utilities::DataTypeToNative<WDataTypeEnum>;
    using OutputDataType = hipdnn_test_sdk::utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = hipdnn_test_sdk::utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->w_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->w_tensor_uid(), WDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), OutputDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ConvolutionFwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        GpuConvolutionFwdParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->w_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
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
            GpuConvolutionFwdPlan<XDataType, WDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
