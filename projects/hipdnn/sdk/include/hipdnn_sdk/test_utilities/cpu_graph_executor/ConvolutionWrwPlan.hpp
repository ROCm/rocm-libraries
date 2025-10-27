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

template <typename InputDataType, typename AccumulatorType>
class ConvolutionWrwPlan : public IGraphNodePlanExecutor
{
public:
    ConvolutionWrwPlan(ConvolutionWrwParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor = createShallowTensor<InputDataType>(
            _params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowDWTensor = createShallowTensor<InputDataType>(
            _params.dwTensor, variantPack.at(_params.dwTensor.uid));

        auto shallowDYTensor = createShallowTensor<InputDataType>(
            _params.dyTensor, variantPack.at(_params.dyTensor.uid));

        CpuFpReferenceConvolutionImpl<InputDataType, AccumulatorType>::convBwdWeight(
            *shallowXTensor,
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

template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_sdk::data_objects::DataType AccumulatorDataTypeEnum>
class ConvolutionWrwPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using InputDataType = DataTypeToNative<InputDataTypeEnum>;
    using AccumulatorDataType = DataTypeToNative<AccumulatorDataTypeEnum>;

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

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dw_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dy_tensor_uid(), InputDataTypeEnum);

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
            convertFlatBufferVectorToStdVector(nodeAttributes->pre_padding()),
            convertFlatBufferVectorToStdVector(nodeAttributes->post_padding()),
            convertFlatBufferVectorToStdVector(nodeAttributes->stride()),
            convertFlatBufferVectorToStdVector(nodeAttributes->dilation()),
            nodeAttributes->conv_mode());

        return std::make_unique<ConvolutionWrwPlan<InputDataType, AccumulatorDataType>>(
            std::move(params));
    }
};

}
