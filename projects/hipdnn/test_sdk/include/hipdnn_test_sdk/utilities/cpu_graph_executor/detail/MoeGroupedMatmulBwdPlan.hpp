// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmulBwd.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct MoeGroupedMatmulBwdParams
{
    MoeGroupedMatmulBwdParams() = default;
    MoeGroupedMatmulBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& doutputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tokenAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& firstTokenOffsetAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dweightAttributes)
        : doutputTensor(unpackTensorAttributes(doutputAttributes))
        , tokenTensor(unpackTensorAttributes(tokenAttributes))
        , firstTokenOffsetTensor(unpackTensorAttributes(firstTokenOffsetAttributes))
        , dweightTensor(unpackTensorAttributes(dweightAttributes))
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT doutputTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT firstTokenOffsetTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dweightTensor;
};

template <typename DoutputDataType,
          typename TokenDataType,
          typename DweightDataType,
          typename ComputeDataType>
class MoeGroupedMatmulBwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit MoeGroupedMatmulBwdPlan(MoeGroupedMatmulBwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.dweightTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto doutput = createShallowTensor<DoutputDataType>(
            _params.doutputTensor, variantPack.at(_params.doutputTensor.uid));
        auto token = createShallowTensor<TokenDataType>(_params.tokenTensor,
                                                        variantPack.at(_params.tokenTensor.uid));
        auto firstTokenOffset = createShallowTensor<int32_t>(
            _params.firstTokenOffsetTensor, variantPack.at(_params.firstTokenOffsetTensor.uid));
        auto dweight = createShallowTensor<DweightDataType>(
            _params.dweightTensor, variantPack.at(_params.dweightTensor.uid));

        utilities::CpuFpReferenceMoeGroupedMatmulBwd::
            backward<DoutputDataType, TokenDataType, DweightDataType, ComputeDataType>(
                *doutput, *token, *firstTokenOffset, *dweight);
    }

private:
    MoeGroupedMatmulBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType DoutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType TokenDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DweightDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class MoeGroupedMatmulBwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using DoutputDataType = utilities::DataTypeToNative<DoutputDataTypeEnum>;
    using TokenDataType = utilities::DataTypeToNative<TokenDataTypeEnum>;
    using DweightDataType = utilities::DataTypeToNative<DweightDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

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

        const auto* nodeAttributes = node.attributes_as_MoeGroupedMatmulBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->doutput_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->token_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->first_token_offset_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dweight_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->doutput_tensor_uid(), DoutputDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->token_tensor_uid(), TokenDataTypeEnum);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->first_token_offset_tensor_uid(), DataType::INT32);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dweight_tensor_uid(), DweightDataTypeEnum);

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_MoeGroupedMatmulBwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error(
                "Node attributes are not of type MoeGroupedMatmulBwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        return std::make_unique<MoeGroupedMatmulBwdPlan<DoutputDataType,
                                                        TokenDataType,
                                                        DweightDataType,
                                                        ComputeDataType>>(
            MoeGroupedMatmulBwdParams(
                *tensorMap.at(nodeAttributes->doutput_tensor_uid()),
                *tensorMap.at(nodeAttributes->token_tensor_uid()),
                *tensorMap.at(nodeAttributes->first_token_offset_tensor_uid()),
                *tensorMap.at(nodeAttributes->dweight_tensor_uid())));
    }

private:
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
};

} // namespace hipdnn_test_sdk::detail
