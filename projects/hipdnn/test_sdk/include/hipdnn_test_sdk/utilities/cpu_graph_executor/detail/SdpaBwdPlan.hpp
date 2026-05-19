// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct SdpaBwdParams
{
    SdpaBwdParams(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& doAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& statsAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dqAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dkAttributes,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dvAttributes,
                  std::optional<float> attnScaleValue,
                  bool causalMask,
                  const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attnMaskAttributes
                  = nullptr)
        : qTensor(unpackTensorAttributes(qAttributes))
        , kTensor(unpackTensorAttributes(kAttributes))
        , vTensor(unpackTensorAttributes(vAttributes))
        , oTensor(unpackTensorAttributes(oAttributes))
        , doTensor(unpackTensorAttributes(doAttributes))
        , statsTensor(unpackTensorAttributes(statsAttributes))
        , dqTensor(unpackTensorAttributes(dqAttributes))
        , dkTensor(unpackTensorAttributes(dkAttributes))
        , dvTensor(unpackTensorAttributes(dvAttributes))
        , attnScaleValue(attnScaleValue)
        , causalMask(causalMask)
        , attnMaskTensor(attnMaskAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*attnMaskAttributes))
                             : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT doTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT statsTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dqTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dkTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT dvTensor;
    std::optional<float> attnScaleValue;
    bool causalMask;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> attnMaskTensor;
};

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename DODataType,
          typename DQDataType,
          typename DKDataType,
          typename DVDataType>
class SdpaBwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaBwdPlan(SdpaBwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.dqTensor.uid, _params.dkTensor.uid, _params.dvTensor.uid};
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
        auto shallowDOTensor = createShallowTensor<DODataType>(
            _params.doTensor, variantPack.at(_params.doTensor.uid));
        auto shallowStatsTensor = createShallowTensor<float>(
            _params.statsTensor, variantPack.at(_params.statsTensor.uid));
        auto shallowDQTensor = createShallowTensor<DQDataType>(
            _params.dqTensor, variantPack.at(_params.dqTensor.uid));
        auto shallowDKTensor = createShallowTensor<DKDataType>(
            _params.dkTensor, variantPack.at(_params.dkTensor.uid));
        auto shallowDVTensor = createShallowTensor<DVDataType>(
            _params.dvTensor, variantPack.at(_params.dvTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowAttnMaskTensor;
        if(_params.attnMaskTensor.has_value())
        {
            shallowAttnMaskTensor = createShallowTensor<float>(
                *_params.attnMaskTensor, variantPack.at(_params.attnMaskTensor->uid));
        }

        utilities::CpuFpReferenceSdpa::backward<QDataType,
                                                KDataType,
                                                VDataType,
                                                ODataType,
                                                DODataType,
                                                DQDataType,
                                                DKDataType,
                                                DVDataType,
                                                float>(*shallowQTensor,
                                                       *shallowKTensor,
                                                       *shallowVTensor,
                                                       *shallowOTensor,
                                                       *shallowDOTensor,
                                                       *shallowDQTensor,
                                                       *shallowDKTensor,
                                                       *shallowDVTensor,
                                                       _params.attnScaleValue,
                                                       shallowStatsTensor.get(),
                                                       shallowAttnMaskTensor.get(),
                                                       _params.causalMask);
    }

private:
    SdpaBwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DODataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DQDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DKDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType DVDataTypeEnum>
class SdpaBwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using QDataType = utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = utilities::DataTypeToNative<ODataTypeEnum>;
    using DODataType = utilities::DataTypeToNative<DODataTypeEnum>;
    using DQDataType = utilities::DataTypeToNative<DQDataTypeEnum>;
    using DKDataType = utilities::DataTypeToNative<DKDataTypeEnum>;
    using DVDataType = utilities::DataTypeToNative<DVDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->q_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->k_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->v_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->o_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->do_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dq_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dk_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->dv_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->do_tensor_uid(), DODataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dq_tensor_uid(), DQDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dk_tensor_uid(), DKDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dv_tensor_uid(), DVDataTypeEnum);

        // Unsupported mask modes
        if(nodeAttributes->causal_mask_bottom_right())
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
        if(nodeAttributes->seed_tensor_uid().has_value()
           || nodeAttributes->offset_tensor_uid().has_value()
           || nodeAttributes->dropout_mask_tensor_uid().has_value()
           || nodeAttributes->dropout_scale_tensor_uid().has_value()
           || nodeAttributes->dropout_scale_inv_tensor_uid().has_value())
        {
            return false;
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaBackwardAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaBackwardAttributes");
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

        SdpaBwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                             *tensorMap.at(nodeAttributes->k_tensor_uid()),
                             *tensorMap.at(nodeAttributes->v_tensor_uid()),
                             *tensorMap.at(nodeAttributes->o_tensor_uid()),
                             *tensorMap.at(nodeAttributes->do_tensor_uid()),
                             *tensorMap.at(nodeAttributes->stats_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dq_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dk_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dv_tensor_uid()),
                             attnScaleValue,
                             nodeAttributes->causal_mask(),
                             attnMaskPtr);

        return std::make_unique<SdpaBwdPlan<QDataType,
                                            KDataType,
                                            VDataType,
                                            ODataType,
                                            DODataType,
                                            DQDataType,
                                            DKDataType,
                                            DVDataType>>(std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
