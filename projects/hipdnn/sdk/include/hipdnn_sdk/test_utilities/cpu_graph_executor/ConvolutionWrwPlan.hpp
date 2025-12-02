// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PlanUtils.hpp>

namespace hipdnn_sdk::test_utilities
{

struct ConvolutionWrwParams
{
    ConvolutionWrwParams() = default;
    ConvolutionWrwParams(const hipdnn_sdk::data_objects::TensorAttributes& xAttributes,
                         const hipdnn_sdk::data_objects::TensorAttributes& dwAttributes,
                         const hipdnn_sdk::data_objects::TensorAttributes& dyAttributes,
                         const std::vector<int64_t>& prePadding,
                         const std::vector<int64_t>& postPadding,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& dilation,
                         const hipdnn_sdk::data_objects::ConvMode convolutionMode)
        : xTensor(unpackTensorAttributes(xAttributes))
        , dwTensor(unpackTensorAttributes(dwAttributes))
        , dyTensor(unpackTensorAttributes(dyAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , dilation(dilation)
        , convMode(convolutionMode)
    {
    }

    hipdnn_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_sdk::data_objects::TensorAttributesT dwTensor;
    hipdnn_sdk::data_objects::TensorAttributesT dyTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    hipdnn_sdk::data_objects::ConvMode convMode;
};

template <typename XDataType,
          typename DyDataType,
          typename OutputDataType,
          typename ComputeDataType>
class ConvolutionWrwPlan : public IGraphNodePlanExecutor
{
public:
    ConvolutionWrwPlan(ConvolutionWrwParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowDWTensor = createShallowTensor<OutputDataType>(
            _params.dwTensor, variantPack.at(_params.dwTensor.uid));

        auto shallowDYTensor = createShallowTensor<DyDataType>(
            _params.dyTensor, variantPack.at(_params.dyTensor.uid));

        CpuFpReferenceConvolution::wgrad(*shallowXTensor,
                                         *shallowDWTensor,
                                         *shallowDYTensor,
                                         _params.stride,
                                         _params.dilation,
                                         _params.prePadding,
                                         _params.postPadding);
    }

private:
    ConvolutionWrwParams _params;
};

template <hipdnn_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_sdk::data_objects::DataType DyDataTypeEnum,
          hipdnn_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum>
class ConvolutionWrwPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = DataTypeToNative<XDataTypeEnum>;
    using DyDataType = DataTypeToNative<DyDataTypeEnum>;
    using OutputDataType = DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionWrwAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dw_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dy_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dw_tensor_uid(), OutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), DyDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionWrwAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ConvolutionWrwAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        ConvolutionWrwParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->dw_tensor_uid()),
            *tensorMap.at(nodeAttributes->dy_tensor_uid()),
            utilities::convertFlatBufferVectorToStdVector(nodeAttributes->pre_padding()),
            utilities::convertFlatBufferVectorToStdVector(nodeAttributes->post_padding()),
            utilities::convertFlatBufferVectorToStdVector(nodeAttributes->stride()),
            utilities::convertFlatBufferVectorToStdVector(nodeAttributes->dilation()),
            nodeAttributes->conv_mode());

        return std::make_unique<
            ConvolutionWrwPlan<XDataType, DyDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

}
