// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <optional>
#include <stdexcept>
#include <vector>

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
    SdpaFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
        std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scale,
        int64_t leftBound,
        int64_t rightBound,
        bool topLeftAlignment,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attnMaskAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* lseAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* descaleQAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* descaleKAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* descaleVAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* seqLenQAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* seqLenKvAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* pageTableKAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* pageTableVAttributes
        = nullptr,
        int64_t pagedMaxSeqKv = 0,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* sinkAttributes = nullptr)
        : qTensor(unpackTensorAttributes(qAttributes))
        , kTensor(unpackTensorAttributes(kAttributes))
        , vTensor(unpackTensorAttributes(vAttributes))
        , oTensor(unpackTensorAttributes(oAttributes))
        , scaleTensor(std::move(scale))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , attnMaskTensor(attnMaskAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*attnMaskAttributes))
                             : std::nullopt)
        , lseTensor(lseAttributes != nullptr
                        ? std::make_optional(unpackTensorAttributes(*lseAttributes))
                        : std::nullopt)
        , descaleQTensor(descaleQAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*descaleQAttributes))
                             : std::nullopt)
        , descaleKTensor(descaleKAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*descaleKAttributes))
                             : std::nullopt)
        , descaleVTensor(descaleVAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*descaleVAttributes))
                             : std::nullopt)
        , seqLenQTensor(seqLenQAttributes != nullptr
                            ? std::make_optional(unpackTensorAttributes(*seqLenQAttributes))
                            : std::nullopt)
        , seqLenKvTensor(seqLenKvAttributes != nullptr
                             ? std::make_optional(unpackTensorAttributes(*seqLenKvAttributes))
                             : std::nullopt)
        , pageTableKTensor(pageTableKAttributes != nullptr
                               ? std::make_optional(unpackTensorAttributes(*pageTableKAttributes))
                               : std::nullopt)
        , pageTableVTensor(pageTableVAttributes != nullptr
                               ? std::make_optional(unpackTensorAttributes(*pageTableVAttributes))
                               : std::nullopt)
        , pagedMaxSeqKv(pagedMaxSeqKv)
        , sinkTensor(sinkAttributes != nullptr
                         ? std::make_optional(unpackTensorAttributes(*sinkAttributes))
                         : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> attnMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> lseTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleVTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> seqLenQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> seqLenKvTensor;
    // Paged KV cache: page tables for K and/or V and the logical key extent (0 = not paged).
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> pageTableKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> pageTableVTensor;
    int64_t pagedMaxSeqKv;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> sinkTensor;
    // Explicit-mask dropout (cuDNN custom dropout): mask and single-element scale.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutScaleTensor;
};

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename Mask = float>
class SdpaFwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaFwdPlan(SdpaFwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids = {_params.oTensor.uid};
        if(_params.lseTensor.has_value())
        {
            ids.push_back(_params.lseTensor->uid);
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

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<Mask>> shallowAttnMaskTensor;
        if(_params.attnMaskTensor.has_value())
        {
            shallowAttnMaskTensor = createShallowTensor<Mask>(
                *_params.attnMaskTensor, variantPack.at(_params.attnMaskTensor->uid));
        }

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowLseTensor;
        if(_params.lseTensor.has_value())
        {
            shallowLseTensor = createShallowTensor<float>(*_params.lseTensor,
                                                          variantPack.at(_params.lseTensor->uid));
        }

        // FP8 descale tensors (float32). Absent for non-fp8 graphs.
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowDescaleQ;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowDescaleK;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowDescaleV;
        if(_params.descaleQTensor.has_value())
        {
            shallowDescaleQ = createShallowTensor<float>(
                *_params.descaleQTensor, variantPack.at(_params.descaleQTensor->uid));
        }
        if(_params.descaleKTensor.has_value())
        {
            shallowDescaleK = createShallowTensor<float>(
                *_params.descaleKTensor, variantPack.at(_params.descaleKTensor->uid));
        }
        if(_params.descaleVTensor.has_value())
        {
            shallowDescaleV = createShallowTensor<float>(
                *_params.descaleVTensor, variantPack.at(_params.descaleVTensor->uid));
        }

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

        // Paged KV cache: gather each paged container into a dense [B, H, maxSeqKv, D] tensor
        // through its page table, then run the dense padded reference on it.
        std::optional<hipdnn_data_sdk::utilities::Tensor<KDataType>> denseK;
        std::optional<hipdnn_data_sdk::utilities::Tensor<VDataType>> denseV;
        if(_params.pageTableKTensor.has_value())
        {
            auto table = createShallowTensor<int32_t>(
                *_params.pageTableKTensor, variantPack.at(_params.pageTableKTensor->uid));
            denseK.emplace(utilities::CpuFpReferenceSdpa::gatherPagedKv(
                *shallowKTensor, *table, *shallowSeqLenKv, _params.pagedMaxSeqKv));
        }
        if(_params.pageTableVTensor.has_value())
        {
            auto table = createShallowTensor<int32_t>(
                *_params.pageTableVTensor, variantPack.at(_params.pageTableVTensor->uid));
            denseV.emplace(utilities::CpuFpReferenceSdpa::gatherPagedKv(
                *shallowVTensor, *table, *shallowSeqLenKv, _params.pagedMaxSeqKv));
        }
        const hipdnn_data_sdk::utilities::TensorBase<KDataType>& kForRef
            = denseK.has_value()
                  ? static_cast<const hipdnn_data_sdk::utilities::TensorBase<KDataType>&>(*denseK)
                  : *shallowKTensor;
        const hipdnn_data_sdk::utilities::TensorBase<VDataType>& vForRef
            = denseV.has_value()
                  ? static_cast<const hipdnn_data_sdk::utilities::TensorBase<VDataType>&>(*denseV)
                  : *shallowVTensor;

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowSink;
        if(_params.sinkTensor.has_value())
        {
            shallowSink = createShallowTensor<float>(*_params.sinkTensor,
                                                     variantPack.at(_params.sinkTensor->uid));
        }

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> shallowDropoutMask;
        std::optional<float> dropoutScale;
        if(_params.dropoutMaskTensor.has_value() && _params.dropoutScaleTensor.has_value())
        {
            shallowDropoutMask = createShallowTensor<float>(
                *_params.dropoutMaskTensor, variantPack.at(_params.dropoutMaskTensor->uid));
            dropoutScale = resolveHostScalarOperand(
                *_params.dropoutScaleTensor, variantPack, "SDPA dropout scale");
        }

        // Runtime attention scale: resolve the folded scale operand (runtime tensor
        // or baked attn_scale_value) to a scalar; nullopt => 1.0, no scaling (cuDNN's default).
        std::optional<float> effectiveScale;
        if(_params.scaleTensor.has_value())
        {
            effectiveScale = hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
                _params.scaleTensor.value(), variantPack, "SDPA scale");
        }

        utilities::CpuFpReferenceSdpa::
            forward<QDataType, KDataType, VDataType, ODataType, float, Mask>(
                *shallowQTensor,
                kForRef,
                vForRef,
                *shallowOTensor,
                effectiveScale,
                shallowAttnMaskTensor.get(),
                _params.leftBound,
                _params.rightBound,
                _params.topLeftAlignment,
                shallowLseTensor.get(),
                shallowDescaleQ.get(),
                shallowDescaleK.get(),
                shallowDescaleV.get(),
                shallowSeqLenQ.get(),
                shallowSeqLenKv.get(),
                shallowSink.get(),
                shallowDropoutMask.get(),
                dropoutScale);
    }

private:
    SdpaFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum
          = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
class SdpaFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using QDataType = utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = utilities::DataTypeToNative<ODataTypeEnum>;
    using MaskType = utilities::MaskDataTypeToNative<MaskDataTypeEnum>;

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

        // The builder's mask dtype must match the graph: UNSET builds only mask-free graphs,
        // every other builder only graphs whose mask tensor has exactly that dtype.
        if constexpr(MaskDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        {
            if(nodeAttributes->attn_mask_tensor_uid().has_value())
            {
                return false;
            }
        }
        else
        {
            if(!nodeAttributes->attn_mask_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->attn_mask_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->attn_mask_tensor_uid().value(), MaskDataTypeEnum);
        }

        // Unsupported mask modes
        if(nodeAttributes->alibi_mask())
        {
            return false;
        }

        // Padding: both seq_len tensors (INT32, one value per batch) or neither.
        if(!isSupportedSdpaPadding(*nodeAttributes, tensorMap))
        {
            return false;
        }

        // Dropout: only cuDNN's explicit-mask form; Philox dropout and RNG_DUMP are declined.
        if(!isSupportedSdpaDropout(*nodeAttributes, tensorMap)
           || nodeAttributes->rng_dump_tensor_uid().has_value())
        {
            return false;
        }

        // Paged KV cache: cuDNN layout, which needs padding; see sdpaPagedMaxSeqKv.
        if(sdpaPagedMaxSeqKv(*nodeAttributes, tensorMap) < 0)
        {
            return false;
        }

        // Unsupported: block sparse attention
        if(nodeAttributes->block_mask_tensor_uid().has_value())
        {
            return false;
        }

        // Attention sink: FLOAT [1, H, 1, 1].
        if(!isSupportedSdpaSink(*nodeAttributes, tensorMap))
        {
            return false;
        }

        // FP8 quantization: descales are required for fp8 inputs (mirrors AITER's
        // requirement that q/k/v descales are all provided for fp8) and rejected
        // for non-fp8 inputs.
        constexpr bool IS_FP8_INPUT
            = QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3
              || QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3_FNUZ
              || QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2
              || QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2_FNUZ;
        const int numQkvDescales
            = static_cast<int>(nodeAttributes->descale_q_tensor_uid().has_value())
              + static_cast<int>(nodeAttributes->descale_k_tensor_uid().has_value())
              + static_cast<int>(nodeAttributes->descale_v_tensor_uid().has_value());
        if((IS_FP8_INPUT && numQkvDescales != 3) || (!IS_FP8_INPUT && numQkvDescales != 0))
        {
            return false;
        }

        // Unsupported regardless of dtype: softmax/output (de)quantization. The
        // forward reference models only q/k/v descales.
        if(nodeAttributes->descale_s_tensor_uid().has_value()
           || nodeAttributes->scale_s_tensor_uid().has_value()
           || nodeAttributes->scale_o_tensor_uid().has_value()
           || nodeAttributes->amax_s_tensor_uid().has_value()
           || nodeAttributes->amax_o_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported: max / running-sum softmax stats outputs (the reference does not
        // produce these). The log-sum-exp stats tensor IS supported and handled below.
        if(nodeAttributes->max_tensor_uid().has_value()
           || nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            return false;
        }

        // Supported: log-sum-exp output via the stats tensor. It must exist in the map and
        // be FLOAT (LSE is always float). The rank reconciliation to [B, H, Sq] is enforced
        // in execute().
        if(nodeAttributes->stats_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->stats_tensor_uid().value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

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

        const auto* scalePtr = nodeAttributes->scale_tensor_uid().has_value()
                                   ? tensorMap.at(nodeAttributes->scale_tensor_uid().value())
                                   : nullptr;
        auto scale = foldSdpaScale(scalePtr, attnScaleValue);

        const auto* lsePtr = nodeAttributes->stats_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->stats_tensor_uid().value())
                                 : nullptr;

        const auto* descaleQPtr = nodeAttributes->descale_q_tensor_uid().has_value()
                                      ? tensorMap.at(nodeAttributes->descale_q_tensor_uid().value())
                                      : nullptr;
        const auto* descaleKPtr = nodeAttributes->descale_k_tensor_uid().has_value()
                                      ? tensorMap.at(nodeAttributes->descale_k_tensor_uid().value())
                                      : nullptr;
        const auto* descaleVPtr = nodeAttributes->descale_v_tensor_uid().has_value()
                                      ? tensorMap.at(nodeAttributes->descale_v_tensor_uid().value())
                                      : nullptr;

        auto [leftBound, rightBound, isTopLeft]
            = extractDiagonalBandParams(*nodeAttributes, "SdpaFwdPlan");

        SdpaFwdParams params(*tensorMap.at(nodeAttributes->q_tensor_uid()),
                             *tensorMap.at(nodeAttributes->k_tensor_uid()),
                             *tensorMap.at(nodeAttributes->v_tensor_uid()),
                             *tensorMap.at(nodeAttributes->o_tensor_uid()),
                             std::move(scale),
                             leftBound,
                             rightBound,
                             isTopLeft,
                             attnMaskPtr,
                             lsePtr,
                             descaleQPtr,
                             descaleKPtr,
                             descaleVPtr,
                             nodeAttributes->seq_len_q_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->seq_len_q_tensor_uid().value())
                                 : nullptr,
                             nodeAttributes->seq_len_kv_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->seq_len_kv_tensor_uid().value())
                                 : nullptr,
                             nodeAttributes->page_table_k_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->page_table_k_tensor_uid().value())
                                 : nullptr,
                             nodeAttributes->page_table_v_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->page_table_v_tensor_uid().value())
                                 : nullptr,
                             sdpaPagedMaxSeqKv(*nodeAttributes, tensorMap),
                             nodeAttributes->sink_token_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->sink_token_tensor_uid().value())
                                 : nullptr);
        if(nodeAttributes->dropout_mask_tensor_uid().has_value())
        {
            params.dropoutMaskTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->dropout_mask_tensor_uid().value()));
            params.dropoutScaleTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->dropout_scale_tensor_uid().value()));
        }
        return std::make_unique<SdpaFwdPlan<QDataType, KDataType, VDataType, ODataType, MaskType>>(
            std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
