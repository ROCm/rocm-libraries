// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <variant>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct ConvolutionFwdParams
{
    ConvolutionFwdParams() = default;
    ConvolutionFwdParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
                         const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& wAttributes,
                         const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
                         const std::vector<int64_t>& prePadding,
                         const std::vector<int64_t>& postPadding,
                         const std::vector<int64_t>& stride,
                         const std::vector<int64_t>& dilation,
                         const hipdnn_flatbuffers_sdk::data_objects::ConvMode convolutionMode)
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
class ConvolutionFwdPlan : public IGraphNodePlanExecutor
{
public:
    ConvolutionFwdPlan(ConvolutionFwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.yTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));

        auto shallowWTensor
            = createShallowTensor<WDataType>(_params.wTensor, variantPack.at(_params.wTensor.uid));

        auto shallowYTensor = createShallowTensor<OutputDataType>(
            _params.yTensor, variantPack.at(_params.yTensor.uid));

        utilities::CpuFpReferenceConvolution::fprop(*shallowXTensor,
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

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType WDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class ConvolutionFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using WDataType = utilities::DataTypeToNative<WDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

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

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
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
            ConvolutionFwdPlan<XDataType, WDataType, OutputDataType, ComputeDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
