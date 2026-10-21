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
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

#include "GpuScalarOperand.hpp"
#include "GpuSdpaFwdPlan.hpp" // reuse sdpaProbabilityMode<>()
#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Unpacked attributes and resolved parameters for a paged-KV SDPA node (vLLM block tables).
// K/V are block containers [num_blocks, H, block_size, D] read through page_table_k/v
// [B, 1, M, 1]. Q/O are dense [B, H, Sq, D], or ragged (RFC-0014 offset aux on each).
// seq_len_kv (and optional seq_len_q) are INT32 [B, 1, 1, 1] device tensors.
struct GpuSdpaPagedFwdParams
{
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorAttributesT = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

    static std::optional<TensorAttributesT> unpackOptional(const TensorAttributes* attributes)
    {
        return attributes != nullptr
                   ? std::make_optional(
                         hipdnn_test_sdk::detail::unpackTensorAttributes(*attributes))
                   : std::nullopt;
    }

    GpuSdpaPagedFwdParams(const TensorAttributes& qAttributes,
                          const TensorAttributes& kAttributes,
                          const TensorAttributes& vAttributes,
                          const TensorAttributes& oAttributes,
                          const TensorAttributes* raggedOffsetQAttributes,
                          const TensorAttributes* raggedOffsetOAttributes,
                          const TensorAttributes& pageTableKAttributes,
                          const TensorAttributes& pageTableVAttributes,
                          const TensorAttributes& seqLenKvAttributes,
                          const TensorAttributes* seqLenQAttributes,
                          int64_t maxSeqLenKv,
                          std::optional<TensorAttributesT> scaleTensor,
                          int64_t leftBound,
                          int64_t rightBound,
                          bool topLeftAlignment,
                          const TensorAttributes* lseAttributes = nullptr,
                          const TensorAttributes* raggedOffsetLseAttributes = nullptr,
                          const TensorAttributes* descaleQAttributes = nullptr,
                          const TensorAttributes* descaleKAttributes = nullptr,
                          const TensorAttributes* descaleVAttributes = nullptr)
        : qTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(qAttributes))
        , kTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(kAttributes))
        , vTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(vAttributes))
        , oTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(oAttributes))
        , raggedOffsetQTensor(unpackOptional(raggedOffsetQAttributes))
        , raggedOffsetOTensor(unpackOptional(raggedOffsetOAttributes))
        , pageTableKTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(pageTableKAttributes))
        , pageTableVTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(pageTableVAttributes))
        , seqLenKvTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(seqLenKvAttributes))
        , seqLenQTensor(unpackOptional(seqLenQAttributes))
        , maxSeqLenKv(maxSeqLenKv)
        , scaleTensor(std::move(scaleTensor))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , lseTensor(unpackOptional(lseAttributes))
        , raggedOffsetLseTensor(unpackOptional(raggedOffsetLseAttributes))
        , descaleQTensor(unpackOptional(descaleQAttributes))
        , descaleKTensor(unpackOptional(descaleKAttributes))
        , descaleVTensor(unpackOptional(descaleVAttributes))
    {
    }

    TensorAttributesT qTensor;
    TensorAttributesT kTensor;
    TensorAttributesT vTensor;
    TensorAttributesT oTensor;
    // Set for ragged Q/O, absent for dense.
    std::optional<TensorAttributesT> raggedOffsetQTensor;
    std::optional<TensorAttributesT> raggedOffsetOTensor;
    TensorAttributesT pageTableKTensor;
    TensorAttributesT pageTableVTensor;
    TensorAttributesT seqLenKvTensor;
    std::optional<TensorAttributesT> seqLenQTensor;
    int64_t maxSeqLenKv;
    // Folded scale operand (scale tensor, else baked attn_scale_value), resolved at execute time.
    // Absent means 1.0 (no scaling), as in cuDNN.
    std::optional<TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<TensorAttributesT> lseTensor;
    // Set only for a packed LSE (ragged Q). Absent means a dense [B,H,Sq,1] LSE.
    std::optional<TensorAttributesT> raggedOffsetLseTensor;
    // Optional fp8 descales (float): scalar [1] or per-KV-head [B, H_kv, 1, 1].
    std::optional<TensorAttributesT> descaleQTensor;
    std::optional<TensorAttributesT> descaleKTensor;
    std::optional<TensorAttributesT> descaleVTensor;
};

// Runs the paged SDPA GPU reference (GpuFpReferenceSdpaRagged::fpropPaged) on the variant-pack
// device buffers.
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float>
class GpuSdpaPagedFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuSdpaPagedFwdPlan(GpuSdpaPagedFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        using TensorAttributesT = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

        hipdnn_gpu_ref::ShallowGpuTensor<QDataType> qTensor(
            variantPack.at(_params.qTensor.uid), _params.qTensor.dims, _params.qTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<KDataType> kTensor(
            variantPack.at(_params.kTensor.uid), _params.kTensor.dims, _params.kTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<VDataType> vTensor(
            variantPack.at(_params.vTensor.uid), _params.vTensor.dims, _params.vTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ODataType> oTensor(
            variantPack.at(_params.oTensor.uid), _params.oTensor.dims, _params.oTensor.strides);

        const auto bindInt32 = [&variantPack](const TensorAttributesT& t) {
            return hipdnn_gpu_ref::ShallowGpuTensor<int32_t>(
                variantPack.at(t.uid), t.dims, t.strides);
        };
        const auto bindOptionalInt32
            = [&variantPack](const std::optional<TensorAttributesT>& t,
                             std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>>& view)
            -> hipdnn_data_sdk::utilities::TensorBase<int32_t>* {
            if(!t.has_value())
            {
                return nullptr;
            }
            view.emplace(variantPack.at(t->uid), t->dims, t->strides);
            return &view.value();
        };

        // A shared page table (same uid) gets two views of the same buffer.
        auto pageTableKTensor = bindInt32(_params.pageTableKTensor);
        auto pageTableVTensor = bindInt32(_params.pageTableVTensor);
        auto seqLenKvTensor = bindInt32(_params.seqLenKvTensor);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> raggedOffsetQView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> raggedOffsetOView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> seqLenQView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> raggedOffsetLseView;
        auto* raggedOffsetQ = bindOptionalInt32(_params.raggedOffsetQTensor, raggedOffsetQView);
        auto* raggedOffsetO = bindOptionalInt32(_params.raggedOffsetOTensor, raggedOffsetOView);
        auto* seqLenQ = bindOptionalInt32(_params.seqLenQTensor, seqLenQView);
        auto* raggedOffsetLse
            = bindOptionalInt32(_params.raggedOffsetLseTensor, raggedOffsetLseView);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> lseTensor;
        if(_params.lseTensor.has_value())
        {
            lseTensor.emplace(variantPack.at(_params.lseTensor->uid),
                              _params.lseTensor->dims,
                              _params.lseTensor->strides);
        }

        // A host-scalar descale (baked or pass-by-value) is staged in a one-element tensor.
        // A device descale is viewed in place. fpropPaged checks the shape.
        const auto bindDescale
            = [&variantPack](const std::optional<TensorAttributesT>& d,
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

        // Ragged Q/O/LSE offsets are scaled to elements by each tensor's multiplier. K/V are
        // paged, so their multipliers are unused.
        hipdnn_gpu_ref::RaggedOffsetMultipliers multipliers;
        multipliers.q = _params.qTensor.ragged_offset_multiplier;
        multipliers.o = _params.oTensor.ragged_offset_multiplier;
        if(_params.lseTensor.has_value())
        {
            multipliers.lse = _params.lseTensor->ragged_offset_multiplier;
        }

        hipdnn_gpu_ref::GpuFpReferenceSdpaRagged::
            fpropPaged<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                qTensor,
                kTensor,
                vTensor,
                oTensor,
                raggedOffsetQ,
                raggedOffsetO,
                pageTableKTensor,
                pageTableVTensor,
                seqLenKvTensor,
                seqLenQ,
                _params.maxSeqLenKv,
                attnScale,
                _params.leftBound,
                _params.rightBound,
                _params.topLeftAlignment,
                lseTensor.has_value() ? &lseTensor.value() : nullptr,
                raggedOffsetLse,
                sdpaProbabilityMode<QDataType, KDataType, VDataType, ODataType>(),
                descaleQ,
                descaleK,
                descaleV,
                multipliers);
    }

private:
    GpuSdpaPagedFwdParams _params;
};

// Applicable iff both page tables, seq_len_kv and max_seq_len_kv are present and K/V carry no
// ragged_offset. Q and O are both dense, or both ragged (RFC-0014 token-major layout, multiplier
// >= 1) with seq_len_q. Otherwise the same unsupported-feature gates as GpuSdpaRaggedFwdPlanBuilder
// apply, except that padding_mask may be either value (seq lens are always honored).
// GpuReferenceGraphExecutor::buildSignatureKey routes page-table graphs here.
template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class GpuSdpaPagedFwdPlanBuilder : public IGpuGraphNodePlanBuilder
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
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

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

        // Paged KV: both page tables, seq_len_kv and max_seq_len_kv.
        if(!nodeAttributes->page_table_k_tensor_uid().has_value()
           || !nodeAttributes->page_table_v_tensor_uid().has_value()
           || !nodeAttributes->seq_len_kv_tensor_uid().has_value()
           || !nodeAttributes->max_seq_len_kv().has_value())
        {
            return false;
        }
        for(const auto uid : {nodeAttributes->page_table_k_tensor_uid().value(),
                              nodeAttributes->page_table_v_tensor_uid().value(),
                              nodeAttributes->seq_len_kv_tensor_uid().value()})
        {
            CHECK_TENSOR_EXISTS(tensorMap, uid);
            CHECK_TENSOR_TYPE(tensorMap, uid, DataType::INT32);
        }
        if(nodeAttributes->seq_len_q_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->seq_len_q_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->seq_len_q_tensor_uid().value(), DataType::INT32);
        }

        // K/V are block containers, never ragged.
        if(tensorMap.at(nodeAttributes->k_tensor_uid())->ragged_offset_tensor_uid().has_value()
           || tensorMap.at(nodeAttributes->v_tensor_uid())->ragged_offset_tensor_uid().has_value())
        {
            return false;
        }

        // Q and O are both dense or both ragged. Ragged Q needs seq_len_q.
        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());
        const bool raggedQ = qAttr->ragged_offset_tensor_uid().has_value();
        if(raggedQ != oAttr->ragged_offset_tensor_uid().has_value())
        {
            return false;
        }
        if(raggedQ)
        {
            if(!nodeAttributes->seq_len_q_tensor_uid().has_value())
            {
                return false;
            }
            for(const auto* primary : {qAttr, oAttr})
            {
                CHECK_TENSOR_EXISTS(tensorMap, primary->ragged_offset_tensor_uid().value());
                CHECK_TENSOR_TYPE(
                    tensorMap, primary->ragged_offset_tensor_uid().value(), DataType::INT32);
                if(!isSupportedRaggedLayout(*primary))
                {
                    return false;
                }
            }
        }

        // Unsupported mask modes. padding_mask is fine: seq lens are always honored.
        if(nodeAttributes->alibi_mask())
        {
            return false;
        }

        // Unsupported: additive attention bias
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

        // Unsupported: block sparse attention and sinks
        if(nodeAttributes->block_mask_tensor_uid().has_value()
           || nodeAttributes->sink_token_tensor_uid().has_value())
        {
            return false;
        }

        // fp8 descales Q/K/V only, with no softmax/output requant, as in the ragged plan.
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
                CHECK_TENSOR_TYPE(tensorMap, descaleUid.value(), DataType::FLOAT);
                const auto descale = hipdnn_test_sdk::detail::unpackTensorAttributes(
                    *tensorMap.at(descaleUid.value()));
                if(isHostScalarOperand(descale) && elementCount(descale) != 1)
                {
                    return false;
                }
            }
        }

        // The scale must be one element, and a device scale must be FLOAT.
        if(nodeAttributes->scale_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid().value());
            const auto scale = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->scale_tensor_uid().value()));
            if(elementCount(scale) != 1
               || (!isHostScalarOperand(scale) && scale.data_type != DataType::FLOAT))
            {
                return false;
            }
        }

        // The reference does not produce max / sum-exp stats.
        if(nodeAttributes->max_tensor_uid().has_value()
           || nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            return false;
        }

        // LSE must be FLOAT. A packed LSE needs a ragged Q.
        if(nodeAttributes->stats_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->stats_tensor_uid().value(), DataType::FLOAT);
            const auto* statsAttr = tensorMap.at(nodeAttributes->stats_tensor_uid().value());
            const auto statsRaggedOffsetUid = statsAttr->ragged_offset_tensor_uid();
            if(statsRaggedOffsetUid.has_value())
            {
                if(!raggedQ)
                {
                    return false;
                }
                CHECK_TENSOR_EXISTS(tensorMap, statsRaggedOffsetUid.value());
                CHECK_TENSOR_TYPE(tensorMap, statsRaggedOffsetUid.value(), DataType::INT32);
                if(!isSupportedRaggedLayout(*statsAttr))
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
        const auto optionalPtr = [&tensorMap](::flatbuffers::Optional<int64_t> uid)
            -> const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* {
            return uid.has_value() ? tensorMap.at(uid.value()) : nullptr;
        };

        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }
        auto scaleTensor = hipdnn_test_sdk::detail::foldSdpaScale(
            optionalPtr(nodeAttributes->scale_tensor_uid()), attnScaleValue);

        const auto* lsePtr = optionalPtr(nodeAttributes->stats_tensor_uid());
        const auto* raggedOffsetLsePtr
            = lsePtr != nullptr ? optionalPtr(lsePtr->ragged_offset_tensor_uid()) : nullptr;

        // The deprecated causal flags compose with the bounds and alignment as in the CPU
        // and dense GPU plans (a causal sliding window keeps its left bound).
        const auto band = extractDiagonalBandParams(*nodeAttributes, "GpuSdpaPagedFwdPlan");

        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());

        return std::make_unique<
            GpuSdpaPagedFwdPlan<QDataType, KDataType, VDataType, ODataType, float>>(
            GpuSdpaPagedFwdParams(*qAttr,
                                  *tensorMap.at(nodeAttributes->k_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->v_tensor_uid()),
                                  *oAttr,
                                  optionalPtr(qAttr->ragged_offset_tensor_uid()),
                                  optionalPtr(oAttr->ragged_offset_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->page_table_k_tensor_uid().value()),
                                  *tensorMap.at(nodeAttributes->page_table_v_tensor_uid().value()),
                                  *tensorMap.at(nodeAttributes->seq_len_kv_tensor_uid().value()),
                                  optionalPtr(nodeAttributes->seq_len_q_tensor_uid()),
                                  static_cast<int64_t>(nodeAttributes->max_seq_len_kv().value()),
                                  std::move(scaleTensor),
                                  band.leftBound,
                                  band.rightBound,
                                  band.topLeftAlignment,
                                  lsePtr,
                                  raggedOffsetLsePtr,
                                  optionalPtr(nodeAttributes->descale_q_tensor_uid()),
                                  optionalPtr(nodeAttributes->descale_k_tensor_uid()),
                                  optionalPtr(nodeAttributes->descale_v_tensor_uid())));
    }

private:
    // RFC-0014 requires ragged_offset_multiplier >= 1 and a token-major [B, H, S, D] layout.
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
