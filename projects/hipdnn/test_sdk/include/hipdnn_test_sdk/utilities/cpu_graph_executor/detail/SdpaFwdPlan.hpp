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
    // Philox dropout checked through RNG_DUMP: dropoutMaskTensor is the rng_dump tensor and this is
    // 1 / (1 - p).
    std::optional<float> philoxDropoutScale;
    // cuDNN Max / Sum_exp row statistics outputs, FLOAT [B, H, Sq, 1].
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> maxTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> sumExpTensor;
    bool alibi = false;
    // cuDNN sdpa_fp8 softmax / output scaling: one-element FLOAT scales and amax outputs.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleSTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleSTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleOTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> amaxSTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> amaxOTensor;
    // cuDNN block-sparse mask, UINT8 (SEM-025).
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> blockMaskTensor;
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
        for(const auto* stats : {&_params.lseTensor,
                                 &_params.maxTensor,
                                 &_params.sumExpTensor,
                                 &_params.amaxSTensor,
                                 &_params.amaxOTensor})
        {
            if(stats->has_value())
            {
                ids.push_back((*stats)->uid);
            }
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

        const auto bindStats = [&variantPack](const auto& stats) {
            return stats.has_value()
                       ? createShallowTensor<float>(*stats, variantPack.at(stats->uid))
                       : std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>>{};
        };
        auto shallowLseTensor = bindStats(_params.lseTensor);
        auto shallowMaxTensor = bindStats(_params.maxTensor);
        auto shallowSumExpTensor = bindStats(_params.sumExpTensor);

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
        if(_params.dropoutMaskTensor.has_value() && _params.philoxDropoutScale.has_value())
        {
            shallowDropoutMask = createShallowTensor<float>(
                *_params.dropoutMaskTensor, variantPack.at(_params.dropoutMaskTensor->uid));
            dropoutScale = _params.philoxDropoutScale;
        }
        else if(_params.dropoutMaskTensor.has_value() && _params.dropoutScaleTensor.has_value())
        {
            shallowDropoutMask = createShallowTensor<float>(
                *_params.dropoutMaskTensor, variantPack.at(_params.dropoutMaskTensor->uid));
            dropoutScale = resolveHostScalarOperand(
                *_params.dropoutScaleTensor, variantPack, "SDPA dropout scale");
        }

        const auto scalar = [&variantPack](const auto& attrs, const char* name) {
            return attrs.has_value()
                       ? std::optional<float>(resolveHostScalarOperand(*attrs, variantPack, name))
                       : std::nullopt;
        };
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<uint8_t>> shallowBlockMask;
        if(_params.blockMaskTensor.has_value())
        {
            shallowBlockMask = createShallowTensor<uint8_t>(
                *_params.blockMaskTensor, variantPack.at(_params.blockMaskTensor->uid));
        }
        auto shallowAmaxS = bindStats(_params.amaxSTensor);
        auto shallowAmaxO = bindStats(_params.amaxOTensor);

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
                utilities::SdpaFwdOptions<Mask>{}
                    .setScale(effectiveScale)
                    .setMask(rawPointer(shallowAttnMaskTensor))
                    .setBand(_params.leftBound, _params.rightBound, _params.topLeftAlignment)
                    .setSeqLens(rawPointer(shallowSeqLenQ), rawPointer(shallowSeqLenKv))
                    .setSink(rawPointer(shallowSink))
                    .setDropout(rawPointer(shallowDropoutMask), dropoutScale)
                    .setAlibi(_params.alibi)
                    .setBlockMask(rawPointer(shallowBlockMask))
                    .setLse(rawPointer(shallowLseTensor))
                    .setRowStats(rawPointer(shallowMaxTensor), rawPointer(shallowSumExpTensor))
                    .setDescales(rawPointer(shallowDescaleQ),
                                 rawPointer(shallowDescaleK),
                                 rawPointer(shallowDescaleV))
                    .setSoftmaxScales(scalar(_params.scaleSTensor, "SDPA scale_s"),
                                      scalar(_params.descaleSTensor, "SDPA descale_s"))
                    .setOutputScale(scalar(_params.scaleOTensor, "SDPA scale_o"))
                    .setAmax(rawPointer(shallowAmaxS), rawPointer(shallowAmaxO)));
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

        if(!isSupportedSdpaAlibi(*nodeAttributes))
        {
            return false;
        }

        // Padding: both seq_len tensors (INT32, one value per batch) or neither.
        if(!isSupportedSdpaPadding(*nodeAttributes, tensorMap))
        {
            return false;
        }

        // Dropout: cuDNN's explicit-mask form, or Philox dropout checked through the engine's
        // RNG_DUMP keep mask. Philox without RNG_DUMP has no exact oracle and is declined.
        const bool philoxWithDump = isSdpaPhiloxDropoutWithDump(*nodeAttributes, tensorMap);
        if(!philoxWithDump
           && (!isSupportedSdpaDropout(*nodeAttributes, tensorMap)
               || nodeAttributes->rng_dump_tensor_uid().has_value()))
        {
            return false;
        }

        // Paged KV cache: cuDNN layout, which needs padding; see sdpaPagedMaxSeqKv.
        if(sdpaPagedMaxSeqKv(*nodeAttributes, tensorMap) < 0)
        {
            return false;
        }

        // cuDNN block-sparse mask: UINT8, one bit per 128 x 128 score tile.
        if(!isSupportedSdpaBlockMask(*nodeAttributes, tensorMap))
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
        // Descales are read as float.
        for(const auto uid : {nodeAttributes->descale_q_tensor_uid(),
                              nodeAttributes->descale_k_tensor_uid(),
                              nodeAttributes->descale_v_tensor_uid()})
        {
            if(uid.has_value())
            {
                CHECK_TENSOR_EXISTS(tensorMap, uid.value());
                CHECK_TENSOR_TYPE(
                    tensorMap, uid.value(), hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
            }
        }

        // cuDNN sdpa_fp8 softmax / output scaling and amax outputs (fp8 inputs only).
        if(!isSupportedSdpaFp8Scaling(*nodeAttributes, tensorMap, IS_FP8_INPUT))
        {
            return false;
        }

        // cuDNN Max / Sum_exp row statistics: FLOAT outputs (the reference checks [B, H, Sq, 1]).
        for(const auto uid :
            {nodeAttributes->max_tensor_uid(), nodeAttributes->sum_exp_tensor_uid()})
        {
            if(uid.has_value())
            {
                CHECK_TENSOR_EXISTS(tensorMap, uid.value());
                CHECK_TENSOR_TYPE(
                    tensorMap, uid.value(), hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
            }
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
        params.alibi = nodeAttributes->alibi_mask();
        for(auto [uid, bound] :
            {std::pair{nodeAttributes->scale_s_tensor_uid(), &params.scaleSTensor},
             std::pair{nodeAttributes->descale_s_tensor_uid(), &params.descaleSTensor},
             std::pair{nodeAttributes->scale_o_tensor_uid(), &params.scaleOTensor},
             std::pair{nodeAttributes->amax_s_tensor_uid(), &params.amaxSTensor},
             std::pair{nodeAttributes->amax_o_tensor_uid(), &params.amaxOTensor},
             std::pair{nodeAttributes->block_mask_tensor_uid(), &params.blockMaskTensor}})
        {
            if(uid.has_value())
            {
                *bound = unpackTensorAttributes(*tensorMap.at(uid.value()));
            }
        }
        if(nodeAttributes->max_tensor_uid().has_value())
        {
            params.maxTensor
                = unpackTensorAttributes(*tensorMap.at(nodeAttributes->max_tensor_uid().value()));
        }
        if(nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            params.sumExpTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->sum_exp_tensor_uid().value()));
        }
        if(nodeAttributes->dropout_mask_tensor_uid().has_value())
        {
            params.dropoutMaskTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->dropout_mask_tensor_uid().value()));
            params.dropoutScaleTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->dropout_scale_tensor_uid().value()));
        }
        else if(isSdpaPhiloxDropoutWithDump(*nodeAttributes, tensorMap))
        {
            // The engine's dumped keep mask is the reference's explicit mask.
            params.dropoutMaskTensor = unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->rng_dump_tensor_uid().value()));
            params.philoxDropoutScale
                = 1.0F / (1.0F - nodeAttributes->dropout_probability().value());
        }
        return std::make_unique<SdpaFwdPlan<QDataType, KDataType, VDataType, ODataType, MaskType>>(
            std::move(params));
    }
};

} // namespace hipdnn_test_sdk::detail
