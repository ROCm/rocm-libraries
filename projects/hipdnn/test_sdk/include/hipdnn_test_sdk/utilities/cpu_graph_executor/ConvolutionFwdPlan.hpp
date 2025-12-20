// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/PlanUtils.hpp>

namespace hipdnn_test_sdk::utilities
{

struct ConvolutionFwdParams
{
    ConvolutionFwdParams() = default;
    ConvolutionFwdParams(const hipdnn_sdk::data_objects::TensorAttributes& xAttributes,
                         const hipdnn_sdk::data_objects::TensorAttributes& wAttributes,
                         const hipdnn_sdk::data_objects::TensorAttributes& yAttributes,
                         const std::vector<int64_t>& prePadding,
                         const std::vector<int64_t>& postPadding,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& dilation,
                         const hipdnn_sdk::data_objects::ConvMode convolutionMode)
        : xTensor(unpackTensorAttributes(xAttributes))
        , wTensor(unpackTensorAttributes(wAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , prePadding(prePadding)
        , postPadding(postPadding)
        , stride(stride)
        , dilation(dilation)
        , convMode(convolutionMode)
    {
    }

    hipdnn_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_sdk::data_objects::TensorAttributesT wTensor;
    hipdnn_sdk::data_objects::TensorAttributesT yTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> postPadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    hipdnn_sdk::data_objects::ConvMode convMode;
};

template <typename XDataType, typename WDataType, typename OutputDataType, typename ComputeDataType>
class ConvolutionFwdPlan : public IGraphNodePlanExecutor
{
public:
    ConvolutionFwdPlan(ConvolutionFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowWTensor
            = createShallowTensor<WDataType>(_params.wTensor, variantPack.at(_params.wTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        CpuFpReferenceConvolution::fprop(*shallowXTensor,
                                         *shallowWTensor,
                                         *shallowYTensor,
                                         _params.stride,
                                         _params.dilation,
                                         _params.prePadding,
                                         _params.postPadding);
    }

private:
    ConvolutionFwdParams _params;
};

template <hipdnn_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_sdk::data_objects::DataType WDataTypeEnum,
          hipdnn_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum>
class ConvolutionFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = DataTypeToNative<XDataTypeEnum>;
    using WDataType = DataTypeToNative<WDataTypeEnum>;
    using OutputDataType = DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
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

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin_sdk::IGraph& graph,
                      const hipdnn_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ConvolutionFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ConvolutionFwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        ConvolutionFwdParams params(
            *tensorMap.at(nodeAttributes->x_tensor_uid()),
            *tensorMap.at(nodeAttributes->w_tensor_uid()),
            *tensorMap.at(nodeAttributes->y_tensor_uid()),
            hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->pre_padding()),
            hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(
                nodeAttributes->post_padding()),
            hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(nodeAttributes->stride()),
            hipdnn_sdk::utilities::convertFlatBufferVectorToStdVector(nodeAttributes->dilation()),
            nodeAttributes->conv_mode());

        return std::make_unique<
            ConvolutionFwdPlan<XDataType, WDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

}
