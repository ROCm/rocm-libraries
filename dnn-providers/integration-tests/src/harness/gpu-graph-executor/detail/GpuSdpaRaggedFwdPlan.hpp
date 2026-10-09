// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "GpuScalarOperand.hpp"
#include "GpuSdpaFwdPlan.hpp" // reuse sdpaProbabilityMode<>()
#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Unpacked attributes and resolved parameters for a ragged SDPA node (RFC-0014 packed
// [B,S,H,D] + ragged_offset). q/k/v/o each carry an int32 offset aux [B+1,1,1,1], scaled to
// elements by the tensor's ragged_offset_multiplier.
// The optional LSE [B,Sq,H,1] is packed if it has its own ragged_offset aux, else dense.
struct GpuSdpaRaggedFwdParams
{
    GpuSdpaRaggedFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& raggedOffsetQAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& raggedOffsetKAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& raggedOffsetVAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& raggedOffsetOAttributes,
        std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor,
        int64_t leftBound,
        int64_t rightBound,
        bool topLeftAlignment,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* lseAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* raggedOffsetLseAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* descaleQAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* descaleKAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* descaleVAttributes = nullptr)
        : qTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(qAttributes))
        , kTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(kAttributes))
        , vTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(vAttributes))
        , oTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(oAttributes))
        , raggedOffsetQTensor(
              hipdnn_test_sdk::detail::unpackTensorAttributes(raggedOffsetQAttributes))
        , raggedOffsetKTensor(
              hipdnn_test_sdk::detail::unpackTensorAttributes(raggedOffsetKAttributes))
        , raggedOffsetVTensor(
              hipdnn_test_sdk::detail::unpackTensorAttributes(raggedOffsetVAttributes))
        , raggedOffsetOTensor(
              hipdnn_test_sdk::detail::unpackTensorAttributes(raggedOffsetOAttributes))
        , scaleTensor(std::move(scaleTensor))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , lseTensor(lseAttributes != nullptr
                        ? std::make_optional(
                              hipdnn_test_sdk::detail::unpackTensorAttributes(*lseAttributes))
                        : std::nullopt)
        , raggedOffsetLseTensor(
              raggedOffsetLseAttributes != nullptr
                  ? std::make_optional(
                        hipdnn_test_sdk::detail::unpackTensorAttributes(*raggedOffsetLseAttributes))
                  : std::nullopt)
        , descaleQTensor(descaleQAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *descaleQAttributes))
                             : std::nullopt)
        , descaleKTensor(descaleKAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *descaleKAttributes))
                             : std::nullopt)
        , descaleVTensor(descaleVAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *descaleVAttributes))
                             : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetQTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetKTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetVTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT raggedOffsetOTensor;
    // Folded scale operand (scale tensor, else baked attn_scale_value), resolved at execute time.
    // Absent means 1.0 (no scaling), as in cuDNN.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> lseTensor;
    // Set only for a packed LSE. Absent means a dense [B,Sq,H,1] LSE.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> raggedOffsetLseTensor;
    // Optional fp8 descales (float): scalar [1] or per-KV-head [B, H_kv, 1, 1].
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleVTensor;
};

// Runs the ragged SDPA GPU reference on the variant-pack device buffers.
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float>
class GpuSdpaRaggedFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuSdpaRaggedFwdPlan(GpuSdpaRaggedFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<QDataType> qTensor(
            variantPack.at(_params.qTensor.uid), _params.qTensor.dims, _params.qTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<KDataType> kTensor(
            variantPack.at(_params.kTensor.uid), _params.kTensor.dims, _params.kTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<VDataType> vTensor(
            variantPack.at(_params.vTensor.uid), _params.vTensor.dims, _params.vTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ODataType> oTensor(
            variantPack.at(_params.oTensor.uid), _params.oTensor.dims, _params.oTensor.strides);

        const auto bindOffsets
            = [&variantPack](const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& t) {
                  return hipdnn_gpu_ref::ShallowGpuTensor<int32_t>(
                      variantPack.at(t.uid), t.dims, t.strides);
              };
        auto raggedOffsetQTensor = bindOffsets(_params.raggedOffsetQTensor);
        auto raggedOffsetKTensor = bindOffsets(_params.raggedOffsetKTensor);
        auto raggedOffsetVTensor = bindOffsets(_params.raggedOffsetVTensor);
        auto raggedOffsetOTensor = bindOffsets(_params.raggedOffsetOTensor);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> lseTensor;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> raggedOffsetLseTensor;
        if(_params.lseTensor.has_value())
        {
            lseTensor.emplace(variantPack.at(_params.lseTensor->uid),
                              _params.lseTensor->dims,
                              _params.lseTensor->strides);
        }
        if(_params.raggedOffsetLseTensor.has_value())
        {
            raggedOffsetLseTensor.emplace(variantPack.at(_params.raggedOffsetLseTensor->uid),
                                          _params.raggedOffsetLseTensor->dims,
                                          _params.raggedOffsetLseTensor->strides);
        }

        // A host-scalar descale (baked or pass-by-value) is staged in a one-element tensor.
        // A device descale is viewed in place. fpropRagged checks the shape.
        const auto bindDescale
            = [&variantPack](
                  const std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>& d,
                  std::optional<hipdnn_data_sdk::utilities::Tensor<float>>& staged,
                  std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>>& view)
            -> hipdnn_data_sdk::utilities::TensorBase<float>* {
            if(!d.has_value())
            {
                return nullptr;
            }
            if(isHostScalarOperand(*d))
            {
                staged.emplace(std::vector<int64_t>{1});
                staged->fillWithValue(resolveScalarOperand(*d, variantPack, "SDPA descale"));
                return &staged.value();
            }
            view.emplace(variantPack.at(d->uid), d->dims, d->strides);
            return &view.value();
        };
        std::optional<hipdnn_data_sdk::utilities::Tensor<float>> stagedDescaleQ;
        std::optional<hipdnn_data_sdk::utilities::Tensor<float>> stagedDescaleK;
        std::optional<hipdnn_data_sdk::utilities::Tensor<float>> stagedDescaleV;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> descaleQView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> descaleKView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> descaleVView;
        auto* descaleQ = bindDescale(_params.descaleQTensor, stagedDescaleQ, descaleQView);
        auto* descaleK = bindDescale(_params.descaleKTensor, stagedDescaleK, descaleKView);
        auto* descaleV = bindDescale(_params.descaleVTensor, stagedDescaleV, descaleVView);

        std::optional<float> attnScale;
        if(_params.scaleTensor.has_value())
        {
            attnScale = resolveScalarOperand(*_params.scaleTensor, variantPack, "SDPA scale");
        }

        // Each tensor's ragged_offset_multiplier scales its stored offsets to elements.
        hipdnn_gpu_ref::RaggedOffsetMultipliers multipliers;
        multipliers.q = _params.qTensor.ragged_offset_multiplier;
        multipliers.k = _params.kTensor.ragged_offset_multiplier;
        multipliers.v = _params.vTensor.ragged_offset_multiplier;
        multipliers.o = _params.oTensor.ragged_offset_multiplier;
        if(_params.lseTensor.has_value())
        {
            multipliers.lse = _params.lseTensor->ragged_offset_multiplier;
        }

        hipdnn_gpu_ref::GpuFpReferenceSdpaRagged::
            fpropRagged<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                qTensor,
                kTensor,
                vTensor,
                oTensor,
                raggedOffsetQTensor,
                raggedOffsetKTensor,
                raggedOffsetVTensor,
                raggedOffsetOTensor,
                attnScale,
                _params.leftBound,
                _params.rightBound,
                _params.topLeftAlignment,
                lseTensor.has_value() ? &lseTensor.value() : nullptr,
                raggedOffsetLseTensor.has_value() ? &raggedOffsetLseTensor.value() : nullptr,
                sdpaProbabilityMode<QDataType, KDataType, VDataType, ODataType>(),
                descaleQ,
                descaleK,
                descaleV,
                multipliers);
    }

private:
    GpuSdpaRaggedFwdParams _params;
};

// Same unsupported-feature gates as the dense GpuSdpaFwdPlanBuilder, plus: q/k/v/o (and a packed
// LSE) must each carry a ragged_offset aux with a multiplier >= 1 and use the RFC-0014
// [B, S, H, D] token-major layout, and seq_len_q/kv must be absent (the padded variant is not
// supported).
// GpuReferenceGraphExecutor::buildSignatureKey picks dense vs ragged.
template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class GpuSdpaRaggedFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using QDataType = hipdnn_test_sdk::utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = hipdnn_test_sdk::utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = hipdnn_test_sdk::utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = hipdnn_test_sdk::utilities::DataTypeToNative<ODataTypeEnum>;

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

        // Each primary needs an INT32 ragged_offset aux (without them this is a dense SDPA node),
        // a ragged_offset_multiplier >= 1 and the RFC-0014 [B, S, H, D] token-major layout.
        for(const auto primaryUid : {nodeAttributes->q_tensor_uid(),
                                     nodeAttributes->k_tensor_uid(),
                                     nodeAttributes->v_tensor_uid(),
                                     nodeAttributes->o_tensor_uid()})
        {
            const auto* primary = tensorMap.at(primaryUid);
            if(!primary->ragged_offset_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, primary->ragged_offset_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              primary->ragged_offset_tensor_uid().value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
            if(!isSupportedRaggedLayout(*primary))
            {
                return false;
            }
        }

        // The padded seq-lens variant is not supported. Lengths come from ragged_offset alone.
        if(nodeAttributes->seq_len_q_tensor_uid().has_value()
           || nodeAttributes->seq_len_kv_tensor_uid().has_value())
        {
            return false;
        }

        // Unsupported mask modes
        if(nodeAttributes->alibi_mask() || nodeAttributes->padding_mask())
        {
            return false;
        }

        // Unsupported: additive attention bias (gated off on the ASM v3 path)
        if(nodeAttributes->attn_mask_tensor_uid().has_value())
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

        // Unsupported: paged KV cache (a separate task on top of ragged)
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

        // AITER fp8 fwd descales Q/K/V only, with no softmax/output requant. Q/K/V descales must
        // be FLOAT, and a host-scalar descale must be a single element.
        if(nodeAttributes->descale_s_tensor_uid().has_value()
           || nodeAttributes->scale_s_tensor_uid().has_value()
           || nodeAttributes->scale_o_tensor_uid().has_value()
           || nodeAttributes->amax_s_tensor_uid().has_value()
           || nodeAttributes->amax_o_tensor_uid().has_value())
        {
            return false;
        }
        for(const auto descaleUid : {nodeAttributes->descale_q_tensor_uid(),
                                     nodeAttributes->descale_k_tensor_uid(),
                                     nodeAttributes->descale_v_tensor_uid()})
        {
            if(descaleUid.has_value())
            {
                CHECK_TENSOR_EXISTS(tensorMap, descaleUid.value());
                CHECK_TENSOR_TYPE(tensorMap,
                                  descaleUid.value(),
                                  hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
                const auto descale = hipdnn_test_sdk::detail::unpackTensorAttributes(
                    *tensorMap.at(descaleUid.value()));
                if(isHostScalarOperand(descale) && elementCount(descale) != 1)
                {
                    return false;
                }
            }
        }

        // The scale may be baked, pass-by-value or device-resident. It must be one element, and a
        // device scale must be FLOAT since it is read back as one float.
        if(nodeAttributes->scale_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid().value());
            const auto scale = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->scale_tensor_uid().value()));
            if(elementCount(scale) != 1
               || (!isHostScalarOperand(scale)
                   && scale.data_type != hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT))
            {
                return false;
            }
        }

        // The reference does not produce max / sum-exp stats. LSE is handled below.
        if(nodeAttributes->max_tensor_uid().has_value()
           || nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            return false;
        }

        // LSE (stats tensor) must be FLOAT. With an INT32 ragged_offset aux it is packed,
        // otherwise dense [B,Sq,H,1].
        if(nodeAttributes->stats_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->stats_tensor_uid().value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
            const auto statsRaggedOffsetUid
                = tensorMap.at(nodeAttributes->stats_tensor_uid().value())
                      ->ragged_offset_tensor_uid();
            if(statsRaggedOffsetUid.has_value())
            {
                CHECK_TENSOR_EXISTS(tensorMap, statsRaggedOffsetUid.value());
                CHECK_TENSOR_TYPE(tensorMap,
                                  statsRaggedOffsetUid.value(),
                                  hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
                if(!isSupportedRaggedLayout(
                       *tensorMap.at(nodeAttributes->stats_tensor_uid().value())))
                {
                    return false;
                }
            }
        }

        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const auto* scalePtr = nodeAttributes->scale_tensor_uid().has_value()
                                   ? tensorMap.at(nodeAttributes->scale_tensor_uid().value())
                                   : nullptr;
        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }
        auto scaleTensor = hipdnn_test_sdk::detail::foldSdpaScale(scalePtr, attnScaleValue);

        const auto* lsePtr = nodeAttributes->stats_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->stats_tensor_uid().value())
                                 : nullptr;
        const auto* raggedOffsetLsePtr
            = (lsePtr != nullptr && lsePtr->ragged_offset_tensor_uid().has_value())
                  ? tensorMap.at(lsePtr->ragged_offset_tensor_uid().value())
                  : nullptr;

        const auto descalePtr = [&tensorMap](::flatbuffers::Optional<int64_t> uid)
            -> const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* {
            return uid.has_value() ? tensorMap.at(uid.value()) : nullptr;
        };
        const auto* descaleQPtr = descalePtr(nodeAttributes->descale_q_tensor_uid());
        const auto* descaleKPtr = descalePtr(nodeAttributes->descale_k_tensor_uid());
        const auto* descaleVPtr = descalePtr(nodeAttributes->descale_v_tensor_uid());

        int64_t leftBound = (nodeAttributes->left_bound().has_value())
                                ? nodeAttributes->left_bound().value()
                                : -1;
        int64_t rightBound = (nodeAttributes->right_bound().has_value())
                                 ? nodeAttributes->right_bound().value()
                                 : -1;

        if(leftBound < -1 || rightBound < -1)
        {
            throw std::invalid_argument(
                "GpuSdpaRaggedFwdPlan: left_bound and right_bound must be >= -1 (got left_bound="
                + std::to_string(leftBound) + ", right_bound=" + std::to_string(rightBound) + ")");
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

        // Deprecated causal flags override the bounds and alignment.
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

        // isApplicable checked that every primary has a ragged_offset aux. fpropRagged checks
        // that o and v share the per-batch lengths of q and k.
        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* kAttr = tensorMap.at(nodeAttributes->k_tensor_uid());
        const auto* vAttr = tensorMap.at(nodeAttributes->v_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());

        return std::make_unique<
            GpuSdpaRaggedFwdPlan<QDataType, KDataType, VDataType, ODataType, float>>(
            GpuSdpaRaggedFwdParams(*qAttr,
                                   *kAttr,
                                   *vAttr,
                                   *oAttr,
                                   *tensorMap.at(qAttr->ragged_offset_tensor_uid().value()),
                                   *tensorMap.at(kAttr->ragged_offset_tensor_uid().value()),
                                   *tensorMap.at(vAttr->ragged_offset_tensor_uid().value()),
                                   *tensorMap.at(oAttr->ragged_offset_tensor_uid().value()),
                                   std::move(scaleTensor),
                                   leftBound,
                                   rightBound,
                                   isTopLeft,
                                   lsePtr,
                                   raggedOffsetLsePtr,
                                   descaleQPtr,
                                   descaleKPtr,
                                   descaleVPtr));
    }

private:
    // RFC-0014 requires ragged_offset_multiplier >= 1 and a token-major [B, S, H, D] layout.
    static bool
        isSupportedRaggedLayout(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attr)
    {
        const auto unpacked = hipdnn_test_sdk::detail::unpackTensorAttributes(attr);
        return unpacked.ragged_offset_multiplier >= 1
               && hipdnn_test_sdk::detail::isTokenMajorRaggedLayout(unpacked.dims,
                                                                    unpacked.strides);
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
