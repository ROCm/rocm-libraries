// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeRouting.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <cstdint>
#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct MoeRoutingParams
{
    MoeRoutingParams() = default;
    MoeRoutingParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& topkIdsAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& firstTokenOffsetAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tokenIndexAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tokenKsAttributes,
        int64_t routingExperts)
        : topkIdsTensor(unpackTensorAttributes(topkIdsAttributes))
        , firstTokenOffsetTensor(unpackTensorAttributes(firstTokenOffsetAttributes))
        , tokenIndexTensor(unpackTensorAttributes(tokenIndexAttributes))
        , tokenKsTensor(unpackTensorAttributes(tokenKsAttributes))
        , numExperts(routingExperts)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT topkIdsTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT firstTokenOffsetTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenIndexTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenKsTensor;
    int64_t numExperts = 0;
};

template <typename OffsetType>
class MoeRoutingPlan : public IGraphNodePlanExecutor
{
public:
    explicit MoeRoutingPlan(MoeRoutingParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.firstTokenOffsetTensor.uid,
                _params.tokenIndexTensor.uid,
                _params.tokenKsTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto topkIds = createShallowTensor<int32_t>(_params.topkIdsTensor,
                                                    variantPack.at(_params.topkIdsTensor.uid));
        auto firstTokenOffset = createShallowTensor<OffsetType>(
            _params.firstTokenOffsetTensor, variantPack.at(_params.firstTokenOffsetTensor.uid));
        auto tokenIndex = createShallowTensor<int32_t>(
            _params.tokenIndexTensor, variantPack.at(_params.tokenIndexTensor.uid));
        auto tokenKs = createShallowTensor<int32_t>(_params.tokenKsTensor,
                                                    variantPack.at(_params.tokenKsTensor.uid));

        utilities::CpuFpReferenceMoeRouting::route<OffsetType>(
            *topkIds, _params.numExperts, *firstTokenOffset, *tokenIndex, *tokenKs);
    }

private:
    MoeRoutingParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType OffsetDataTypeEnum>
class MoeRoutingPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using OffsetType = utilities::DataTypeToNative<OffsetDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_MoeRoutingAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->topk_ids_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->first_token_offset_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->token_index_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->token_ks_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap,
                          nodeAttributes->topk_ids_tensor_uid(),
                          hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
        CHECK_TENSOR_TYPE(
            tensorMap, nodeAttributes->first_token_offset_tensor_uid(), OffsetDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap,
                          nodeAttributes->token_index_tensor_uid(),
                          hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
        CHECK_TENSOR_TYPE(tensorMap,
                          nodeAttributes->token_ks_tensor_uid(),
                          hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);

        if(nodeAttributes->num_experts() < 1)
        {
            return false;
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_MoeRoutingAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type MoeRoutingAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        return std::make_unique<MoeRoutingPlan<OffsetType>>(
            MoeRoutingParams(*tensorMap.at(nodeAttributes->topk_ids_tensor_uid()),
                             *tensorMap.at(nodeAttributes->first_token_offset_tensor_uid()),
                             *tensorMap.at(nodeAttributes->token_index_tensor_uid()),
                             *tensorMap.at(nodeAttributes->token_ks_tensor_uid()),
                             nodeAttributes->num_experts()));
    }
};

} // namespace hipdnn_test_sdk::detail
