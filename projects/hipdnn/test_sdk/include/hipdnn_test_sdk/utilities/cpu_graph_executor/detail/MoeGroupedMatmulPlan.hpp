// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/MoeGroupedMatmulValidation.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmul.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <optional>
#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct MoeGroupedMatmulParams
{
    MoeGroupedMatmulParams() = default;
    MoeGroupedMatmulParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tokenAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& weightAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& firstTokenOffsetAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& outputAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tokenIndexAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tokenKsAttributes,
        hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode moeMode,
        int32_t moeTopK)
        : tokenTensor(unpackTensorAttributes(tokenAttributes))
        , weightTensor(unpackTensorAttributes(weightAttributes))
        , firstTokenOffsetTensor(unpackTensorAttributes(firstTokenOffsetAttributes))
        , outputTensor(unpackTensorAttributes(outputAttributes))
        , tokenIndexTensor(tokenIndexAttributes != nullptr
                               ? std::make_optional(unpackTensorAttributes(*tokenIndexAttributes))
                               : std::nullopt)
        , tokenKsTensor(tokenKsAttributes != nullptr
                            ? std::make_optional(unpackTensorAttributes(*tokenKsAttributes))
                            : std::nullopt)
        , mode(moeMode)
        , topK(moeTopK)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT tokenTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT weightTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT firstTokenOffsetTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT outputTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> tokenIndexTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> tokenKsTensor;
    hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode mode
        = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode::NONE;
    int32_t topK = 0;
};

template <typename TokenDataType,
          typename WeightDataType,
          typename OutputDataType,
          typename ComputeDataType>
class MoeGroupedMatmulPlan : public IGraphNodePlanExecutor
{
public:
    explicit MoeGroupedMatmulPlan(MoeGroupedMatmulParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.outputTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto token = bindShallowTensor<TokenDataType>(_params.tokenTensor, variantPack);
        auto weight = bindShallowTensor<WeightDataType>(_params.weightTensor, variantPack);
        auto firstTokenOffset
            = bindShallowTensor<int32_t>(_params.firstTokenOffsetTensor, variantPack);
        auto output = bindShallowTensor<OutputDataType>(_params.outputTensor, variantPack);
        auto tokenIndex = bindOptionalShallowTensor<int32_t>(_params.tokenIndexTensor, variantPack);
        auto tokenKs = bindOptionalShallowTensor<int32_t>(_params.tokenKsTensor, variantPack);

        utilities::CpuFpReferenceMoeGroupedMatmul::
            forward<TokenDataType, WeightDataType, OutputDataType, ComputeDataType>(
                *token,
                *weight,
                *firstTokenOffset,
                *output,
                _params.mode,
                _params.topK,
                tokenIndex.get(),
                tokenKs.get());
    }

private:
    MoeGroupedMatmulParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType TokenDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType WeightDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class MoeGroupedMatmulPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using TokenDataType = utilities::DataTypeToNative<TokenDataTypeEnum>;
    using WeightDataType = utilities::DataTypeToNative<WeightDataTypeEnum>;
    using OutputDataType = utilities::DataTypeToNative<OutputDataTypeEnum>;
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

        const auto* nodeAttributes = node.attributes_as_MoeGroupedMatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->token_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->weight_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->first_token_offset_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->output_tensor_uid());

        // Do not hand-check first_token_offset's dtype here: rule 1 of the shared
        // routing contract (evaluated below) covers it.
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->token_tensor_uid(), TokenDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->weight_tensor_uid(), WeightDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->output_tensor_uid(), OutputDataTypeEnum);

        const bool hasTokenIndex = nodeAttributes->token_index_tensor_uid().has_value();
        const bool hasTokenKs = nodeAttributes->token_ks_tensor_uid().has_value();

        if(hasTokenIndex)
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->token_index_tensor_uid().value());
        }
        if(hasTokenKs)
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->token_ks_tensor_uid().value());
        }

        const auto tokenIndexDataTypeOrUnset
            = hasTokenIndex
                  ? tensorMap.at(nodeAttributes->token_index_tensor_uid().value())->data_type()
                  : hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
        const auto tokenKsDataTypeOrUnset
            = hasTokenKs ? tensorMap.at(nodeAttributes->token_ks_tensor_uid().value())->data_type()
                         : hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;

        const auto* weightTensor = tensorMap.at(nodeAttributes->weight_tensor_uid());
        const hipdnn_flatbuffers_sdk::utilities::MoeGroupedMatmulRouting routing{
            nodeAttributes->mode(),
            hasTokenIndex,
            hasTokenKs,
            tensorMap.at(nodeAttributes->first_token_offset_tensor_uid())->data_type(),
            tokenIndexDataTypeOrUnset,
            tokenKsDataTypeOrUnset,
            nodeAttributes->top_k(),
            weightTensor->dims() == nullptr || weightTensor->dims()->empty()
                ? 0
                : weightTensor->dims()->Get(0)};
        if(hipdnn_flatbuffers_sdk::utilities::checkMoeGroupedMatmulRouting(routing) != nullptr)
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
        const auto* nodeAttributes = node.attributes_as_MoeGroupedMatmulAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type MoeGroupedMatmulAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tokenIndexAttributes
            = nodeAttributes->token_index_tensor_uid().has_value()
                  ? tensorMap.at(nodeAttributes->token_index_tensor_uid().value())
                  : nullptr;
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* tokenKsAttributes
            = nodeAttributes->token_ks_tensor_uid().has_value()
                  ? tensorMap.at(nodeAttributes->token_ks_tensor_uid().value())
                  : nullptr;

        return std::make_unique<
            MoeGroupedMatmulPlan<TokenDataType, WeightDataType, OutputDataType, ComputeDataType>>(
            MoeGroupedMatmulParams(*tensorMap.at(nodeAttributes->token_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->weight_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->first_token_offset_tensor_uid()),
                                   *tensorMap.at(nodeAttributes->output_tensor_uid()),
                                   tokenIndexAttributes,
                                   tokenKsAttributes,
                                   nodeAttributes->mode(),
                                   nodeAttributes->top_k()));
    }
};

} // namespace hipdnn_test_sdk::detail
