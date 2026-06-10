// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct SdpaFwdParams
{
    SdpaFwdParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
                  std::optional<float> attnScaleValue,
                  int64_t leftBound,
                  int64_t rightBound,
                  bool topLeftAlignment,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attnMaskAttributes
                  = nullptr)
        : qTensor(unpackTensorAttributes(qAttributes))
        , kTensor(unpackTensorAttributes(kAttributes))
        , vTensor(unpackTensorAttributes(vAttributes))
        , oTensor(unpackTensorAttributes(oAttributes))
        , attnScaleValue(attnScaleValue)
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , attnMaskTensor(attnMaskAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*attnMaskAttributes))
                             : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    std::optional<float> attnScaleValue;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> attnMaskTensor;
};

template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
class SdpaFwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaFwdPlan(SdpaFwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.oTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowQTensor
            = createShallowTensor<QDataType>(_params.qTensor, variantPack.at(_params.qTensor.uid));
        auto shallowKTensor
            = createShallowTensor<KDataType>(_params.kTensor, variantPack.at(_params.kTensor.uid));
        auto shallowVTensor
            = createShallowTensor<VDataType>(_params.vTensor, variantPack.at(_params.vTensor.uid));
        auto shallowOTensor
            = createShallowTensor<ODataType>(_params.oTensor, variantPack.at(_params.oTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowAttnMaskTensor;
        if(_params.attnMaskTensor.has_value())
        {
            shallowAttnMaskTensor = createShallowTensor<float>(
                *_params.attnMaskTensor, variantPack.at(_params.attnMaskTensor->uid));
        }

        utilities::CpuFpReferenceSdpa::forward<QDataType, KDataType, VDataType, ODataType, float>(
            *shallowQTensor,
            *shallowKTensor,
            *shallowVTensor,
            *shallowOTensor,
            _params.attnScaleValue,
            shallowAttnMaskTensor.get(),
            _params.leftBound,
            _params.rightBound,
            _params.topLeftAlignment);
    }

private:
    SdpaFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class SdpaFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using QDataType = utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = utilities::DataTypeToNative<ODataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->q_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->k_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->v_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->o_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);

        // Unsupported mask modes
        if(nodeAttributes->alibi_mask() || nodeAttributes->padding_mask())
        {
            return false;
        }

        // Unsupported: variable sequence lengths
        if(nodeAttributes->seq_len_q_tensor_uid().has_value()
           || nodeAttributes->seq_len_kv_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: dropout
        if(nodeAttributes->dropout_probability().has_value()
           || nodeAttributes->seed_tensor_uid().has_value()
           || nodeAttributes->offset_tensor_uid().has_value()
           || nodeAttributes->dropout_mask_tensor_uid().has_value()
           || nodeAttributes->dropout_scale_tensor_uid().has_value()
           || nodeAttributes->rng_dump_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: paged KV cache
        if(nodeAttributes->page_table_k_tensor_uid().has_value()
           || nodeAttributes->page_table_v_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: block sparse attention
        if(nodeAttributes->block_mask_tensor_uid().has_value()
           || nodeAttributes->sink_token_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: FP8 quantization / descaling
        if(nodeAttributes->descale_q_tensor_uid().has_value()
           || nodeAttributes->descale_k_tensor_uid().has_value()
           || nodeAttributes->descale_v_tensor_uid().has_value()
           || nodeAttributes->descale_s_tensor_uid().has_value()
           || nodeAttributes->scale_s_tensor_uid().has_value()
           || nodeAttributes->scale_o_tensor_uid().has_value()
           || nodeAttributes->amax_s_tensor_uid().has_value()
           || nodeAttributes->amax_o_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: softmax stats outputs
        if(nodeAttributes->stats_tensor_uid().has_value()
           || nodeAttributes->max_tensor_uid().has_value()
           || nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            return false;
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }

        const auto* attnMaskPtr = nodeAttributes->attn_mask_tensor_uid().has_value()
                                      ? tensorMap.at(nodeAttributes->attn_mask_tensor_uid().value())
                                      : nullptr;

        int64_t leftBound = (nodeAttributes->left_bound().has_value())
                                ? nodeAttributes->left_bound().value()
                                : -1;
        int64_t rightBound = (nodeAttributes->right_bound().has_value())
                                 ? nodeAttributes->right_bound().value()
                                 : -1;

        if(leftBound < -1 || rightBound < -1)
        {
            throw std::invalid_argument("SdpaFwdPlan: left_bound and right_bound must be >= -1 "
                                        "(got left_bound="
                                        + std::to_string(leftBound)
                                        + ", right_bound=" + std::to_string(rightBound) + ")");
        }

        bool isTopLeft = nodeAttributes->diagonal_alignment()
                         == hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;

        // Validate mutually exclusive deprecated attributes
        if(nodeAttributes->causal_mask() && nodeAttributes->causal_mask_bottom_right())
        {
            throw std::invalid_argument("Cannot set both causal_mask and causal_mask_bottom_right. "
                                        "Use diagonal_alignment={TOP_LEFT|BOTTOM_RIGHT} with "
                                        "left_bound=-1, right_bound=0 instead.");
        }

        // Check deprecated attributes
        if(nodeAttributes->causal_mask())
        {
            leftBound = -1;
            rightBound = 0;
            isTopLeft = true;
        }
        if(nodeAttributes->causal_mask_bottom_right())
        {
            leftBound = -1;
            rightBound = 0;
            isTopLeft = false;
        }

        return std::make_unique<SdpaFwdPlan<QDataType, KDataType, VDataType, ODataType>>(
            SdpaFwdParams(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                          *tensorMap.at(nodeAttributes->k_tensor_uid()),
                          *tensorMap.at(nodeAttributes->v_tensor_uid()),
                          *tensorMap.at(nodeAttributes->o_tensor_uid()),
                          attnScaleValue,
                          leftBound,
                          rightBound,
                          isTopLeft,
                          attnMaskPtr));
    }
};

} // namespace hipdnn_test_sdk::detail
