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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaSinkOperand.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

/// Stats may be rank-3 [B, H, Sq] (as in the checked-in backward bundles) or rank-4 [B, H, Sq, 1];
/// the reference reads rank-4, so a rank-3 tensor gets a trailing unit dim.
inline hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT
    withTrailingUnitDim(hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT attrs)
{
    if(attrs.dims.size() == 3)
    {
        attrs.dims.push_back(1);
        attrs.strides.push_back(1);
    }
    return attrs;
}

struct SdpaBwdParams
{
    SdpaBwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& doAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& statsAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dqAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dkAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& dvAttributes,
        std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scale,
        int64_t leftBound,
        int64_t rightBound,
        bool topLeftAlignment,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attnMaskAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* dBiasAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* seqLenQAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* seqLenKvAttributes = nullptr)
        : qTensor(unpackTensorAttributes(qAttributes))
        , kTensor(unpackTensorAttributes(kAttributes))
        , vTensor(unpackTensorAttributes(vAttributes))
        , oTensor(unpackTensorAttributes(oAttributes))
        , doTensor(unpackTensorAttributes(doAttributes))
        , statsTensor(withTrailingUnitDim(unpackTensorAttributes(statsAttributes)))
        , dqTensor(unpackTensorAttributes(dqAttributes))
        , dkTensor(unpackTensorAttributes(dkAttributes))
        , dvTensor(unpackTensorAttributes(dvAttributes))
        , scaleTensor(std::move(scale))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , attnMaskTensor(attnMaskAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*attnMaskAttributes))
                             : std::nullopt)
        , dBiasTensor(dBiasAttributes != nullptr
                          ? std::make_optional(unpackTensorAttributes(*dBiasAttributes))
                          : std::nullopt)
        , seqLenQTensor(seqLenQAttributes != nullptr
                            ? std::make_optional(unpackTensorAttributes(*seqLenQAttributes))
                            : std::nullopt)
        , seqLenKvTensor(seqLenKvAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*seqLenKvAttributes))
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
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> attnMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dBiasTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> seqLenQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> seqLenKvTensor;
    // Explicit-mask dropout (cuDNN custom dropout): mask, scale, and optional scale_inv.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutScaleTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutScaleInvTensor;
    // Philox dropout checked through RNG_DUMP: dropoutMaskTensor is the rng_dump tensor, and these
    // are 1 / (1 - p) and (1 - p).
    std::optional<float> philoxDropoutScale;
    std::optional<float> philoxDropoutScaleInv;
    // Attention sink (input) and its gradient (output): FLOAT or the Q dtype [1, H, 1, 1].
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> sinkTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dSinkTensor;
    bool alibi = false;
};

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename DODataType,
          typename DQDataType,
          typename DKDataType,
          typename DVDataType,
          typename Mask = float>
class SdpaBwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaBwdPlan(SdpaBwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids
            = {_params.dqTensor.uid, _params.dkTensor.uid, _params.dvTensor.uid};
        if(_params.dBiasTensor.has_value())
        {
            ids.push_back(_params.dBiasTensor->uid);
        }
        if(_params.dSinkTensor.has_value())
        {
            ids.push_back(_params.dSinkTensor->uid);
        }
        return ids;
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

        // Padding lengths (INT32, one per batch). Applicability guarantees both or neither.
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<int32_t>> shallowSeqLenQ;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<int32_t>> shallowSeqLenKv;
        if(_params.seqLenQTensor.has_value() && _params.seqLenKvTensor.has_value())
        {
            shallowSeqLenQ = createShallowTensor<int32_t>(
                *_params.seqLenQTensor, variantPack.at(_params.seqLenQTensor->uid));
            shallowSeqLenKv = createShallowTensor<int32_t>(
                *_params.seqLenKvTensor, variantPack.at(_params.seqLenKvTensor->uid));
        }

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowDropoutMask;
        std::optional<float> dropoutScale;
        std::optional<float> dropoutScaleInv;
        if(_params.dropoutMaskTensor.has_value() && _params.philoxDropoutScale.has_value())
        {
            shallowDropoutMask = createShallowTensor<float>(
                *_params.dropoutMaskTensor, variantPack.at(_params.dropoutMaskTensor->uid));
            dropoutScale = _params.philoxDropoutScale;
            dropoutScaleInv = _params.philoxDropoutScaleInv;
        }
        else if(_params.dropoutMaskTensor.has_value() && _params.dropoutScaleTensor.has_value())
        {
            shallowDropoutMask = createShallowTensor<float>(
                *_params.dropoutMaskTensor, variantPack.at(_params.dropoutMaskTensor->uid));
            dropoutScale = resolveHostScalarOperand(
                *_params.dropoutScaleTensor, variantPack, "SDPA dropout scale");
        }
        if(_params.dropoutScaleInvTensor.has_value())
        {
            dropoutScaleInv = resolveHostScalarOperand(
                *_params.dropoutScaleInvTensor, variantPack, "SDPA dropout scale inverse");
        }

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<Mask>> shallowAttnMaskTensor;
        if(_params.attnMaskTensor.has_value())
        {
            shallowAttnMaskTensor = createShallowTensor<Mask>(
                *_params.attnMaskTensor, variantPack.at(_params.attnMaskTensor->uid));
        }
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<Mask>> shallowDBiasTensor;
        if(_params.dBiasTensor.has_value())
        {
            shallowDBiasTensor = createShallowTensor<Mask>(
                *_params.dBiasTensor, variantPack.at(_params.dBiasTensor->uid));
        }

        // Attention sink and its gradient: FLOAT or Q dtype, read and written through FLOAT views.
        std::optional<SdpaSinkOperand<QDataType>> sink;
        if(_params.sinkTensor.has_value())
        {
            sink.emplace(*_params.sinkTensor, variantPack.at(_params.sinkTensor->uid));
        }
        std::optional<SdpaDSinkOperand<QDataType>> dSink;
        if(_params.dSinkTensor.has_value())
        {
            dSink.emplace(*_params.dSinkTensor, variantPack.at(_params.dSinkTensor->uid));
        }

        std::optional<float> effectiveScale;
        if(_params.scaleTensor.has_value())
        {
            effectiveScale
                = readHostFloatOperand(_params.scaleTensor.value(), variantPack, "SDPA scale");
        }

        utilities::CpuFpReferenceSdpa::backward<QDataType,
                                                KDataType,
                                                VDataType,
                                                ODataType,
                                                DODataType,
                                                DQDataType,
                                                DKDataType,
                                                DVDataType,
                                                float,
                                                Mask>(
            *shallowQTensor,
            *shallowKTensor,
            *shallowVTensor,
            *shallowOTensor,
            *shallowDOTensor,
            *shallowDQTensor,
            *shallowDKTensor,
            *shallowDVTensor,
            utilities::SdpaBwdOptions<Mask>{}
                .setScale(effectiveScale)
                .setLse(rawPointer(shallowStatsTensor))
                .setMask(rawPointer(shallowAttnMaskTensor))
                .setDBias(rawPointer(shallowDBiasTensor))
                .setBand(_params.leftBound, _params.rightBound, _params.topLeftAlignment)
                .setSeqLens(rawPointer(shallowSeqLenQ), rawPointer(shallowSeqLenKv))
                .setDropout(rawPointer(shallowDropoutMask), dropoutScale, dropoutScaleInv)
                .setSink(sink.has_value() ? sink->tensor() : nullptr,
                         dSink.has_value() ? dSink->tensor() : nullptr)
                .setAlibi(_params.alibi));
        if(dSink.has_value())
        {
            dSink->commit();
        }
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
          hipdnn_flatbuffers_sdk::data_objects::DataType DVDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum
          = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
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
    using MaskType = utilities::MaskDataTypeToNative<MaskDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        // The dense gate is shared with the GPU reference (isSupportedSdpaBwdNode).
        const auto* nodeAttributes = node.attributes_as_SdpaBackwardAttributes();
        if(nodeAttributes == nullptr
           || !isSupportedSdpaBwdNode(*nodeAttributes, tensorMap, MaskDataTypeEnum))
        {
            return false;
        }

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->do_tensor_uid(), DODataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dq_tensor_uid(), DQDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dk_tensor_uid(), DKDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->dv_tensor_uid(), DVDataTypeEnum);

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

        const auto* scalePtr = nodeAttributes->scale_tensor_uid().has_value()
                                   ? tensorMap.at(nodeAttributes->scale_tensor_uid().value())
                                   : nullptr;
        auto scale = foldSdpaScale(scalePtr, attnScaleValue);

        auto [leftBound, rightBound, isTopLeft]
            = extractDiagonalBandParams(*nodeAttributes, "SdpaBwdPlan");

        SdpaBwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                             *tensorMap.at(nodeAttributes->k_tensor_uid()),
                             *tensorMap.at(nodeAttributes->v_tensor_uid()),
                             *tensorMap.at(nodeAttributes->o_tensor_uid()),
                             *tensorMap.at(nodeAttributes->do_tensor_uid()),
                             *tensorMap.at(nodeAttributes->stats_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dq_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dk_tensor_uid()),
                             *tensorMap.at(nodeAttributes->dv_tensor_uid()),
                             std::move(scale),
                             leftBound,
                             rightBound,
                             isTopLeft,
                             attnMaskPtr,
                             nodeAttributes->dbias_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->dbias_tensor_uid().value())
                                 : nullptr,
                             nodeAttributes->seq_len_q_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->seq_len_q_tensor_uid().value())
                                 : nullptr,
                             nodeAttributes->seq_len_kv_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->seq_len_kv_tensor_uid().value())
                                 : nullptr);
        params.alibi = nodeAttributes->alibi_mask();
        for(auto [uid, field] :
            {std::pair{nodeAttributes->dropout_mask_tensor_uid(), &params.dropoutMaskTensor},
             std::pair{nodeAttributes->dropout_scale_tensor_uid(), &params.dropoutScaleTensor},
             std::pair{nodeAttributes->dropout_scale_inv_tensor_uid(),
                       &params.dropoutScaleInvTensor}})
        {
            if(uid.has_value())
            {
                *field = unpackTensorAttributes(*tensorMap.at(uid.value()));
            }
        }
        // Philox dropout checked through the engine's backward RNG_DUMP: the dump is the explicit
        // mask, with scale 1 / (1 - p) and scale_inv (1 - p) (cuDNN's composition).
        if(isSdpaBwdPhiloxDropoutWithDump(*nodeAttributes, tensorMap))
        {
            const auto keep = 1.0F - nodeAttributes->dropout_probability().value();
            params.dropoutMaskTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->rng_dump_tensor_uid().value()));
            params.philoxDropoutScale = 1.0F / keep;
            params.philoxDropoutScaleInv = keep;
        }
        for(auto [uid, field] :
            {std::pair{nodeAttributes->sink_token_tensor_uid(), &params.sinkTensor},
             std::pair{nodeAttributes->dsink_token_tensor_uid(), &params.dSinkTensor}})
        {
            if(uid.has_value())
            {
                *field = unpackTensorAttributes(*tensorMap.at(uid.value()));
            }
        }

        return std::make_unique<SdpaBwdPlan<QDataType,
                                            KDataType,
                                            VDataType,
                                            ODataType,
                                            DODataType,
                                            DQDataType,
                                            DKDataType,
                                            DVDataType,
                                            MaskType>>(std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
