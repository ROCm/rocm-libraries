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

template <typename InputDataType, typename AccumulatorType>
class ConvolutionFwdPlan : public IGraphNodePlanExecutor
{
public:
    ConvolutionFwdPlan(ConvolutionFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor = createShallowTensor<InputDataType>(
            _params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowWTensor = createShallowTensor<InputDataType>(
            _params.wTensor, variantPack.at(_params.wTensor.uid));

        auto shallowYTensor = createShallowTensor<InputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        // TODO: Figure out pre / post padding.  Probably have to re-write ConvolutionFwdInference
        CpuFpReferenceConvolutionImpl<InputDataType, AccumulatorType>::convFwdInference(
            *shallowXTensor,
            *shallowWTensor,
            *shallowYTensor,
            _params.stride,
            _params.dilation,
            _params.prePadding);
    }

private:
    ConvolutionFwdParams _params;
};

template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
          hipdnn_sdk::data_objects::DataType AccumulatorDataTypeEnum>
class ConvolutionFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using InputDataType = DataTypeToNative<InputDataTypeEnum>;
    using AccumulatorDataType = DataTypeToNative<AccumulatorDataTypeEnum>;

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

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->w_tensor_uid(), InputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), InputDataTypeEnum);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_plugin::IGraph& graph,
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
            convertFlatBufferVectorToStdVector(nodeAttributes->pre_padding()),
            convertFlatBufferVectorToStdVector(nodeAttributes->post_padding()),
            convertFlatBufferVectorToStdVector(nodeAttributes->stride()),
            convertFlatBufferVectorToStdVector(nodeAttributes->dilation()),
            nodeAttributes->conv_mode());

        return std::make_unique<ConvolutionFwdPlan<InputDataType, AccumulatorDataType>>(
            std::move(params));
    }
};

}
