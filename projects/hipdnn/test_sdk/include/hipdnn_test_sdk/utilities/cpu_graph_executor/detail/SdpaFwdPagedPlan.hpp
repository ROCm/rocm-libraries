// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/ApplicabilityUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaSinkUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedTokenBoundaries.hpp>

namespace hipdnn_test_sdk::detail
{

// Unpacked attributes of a paged SDPA forward node (cuDNN paged attention): K/V are block
// containers [num_blocks, H_kv, block_size, D] read through INT32 [B, 1, M, 1] page tables, with
// per-batch lengths from seq_len_kv (and seq_len_q). Q, O and the optional LSE are dense, or ragged
// (RFC-0014) with their own INT32 ragged_offset aux.
struct SdpaFwdPagedParams
{
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorAttributesT = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

    static std::optional<TensorAttributesT> unpackOptional(const TensorAttributes* attributes)
    {
        return attributes != nullptr ? std::make_optional(unpackTensorAttributes(*attributes))
                                     : std::nullopt;
    }

    SdpaFwdPagedParams(const TensorAttributes& qAttributes,
                       const TensorAttributes& kAttributes,
                       const TensorAttributes& vAttributes,
                       const TensorAttributes& oAttributes,
                       const TensorAttributes& pageTableKAttributes,
                       const TensorAttributes& pageTableVAttributes,
                       const TensorAttributes& seqLenKvAttributes,
                       const TensorAttributes* seqLenQAttributes,
                       int64_t maxSeqLenKv,
                       std::optional<TensorAttributesT> scale,
                       int64_t leftBound,
                       int64_t rightBound,
                       bool topLeftAlignment,
                       const TensorAttributes* lseAttributes = nullptr,
                       const TensorAttributes* raggedOffsetQAttributes = nullptr,
                       const TensorAttributes* raggedOffsetOAttributes = nullptr,
                       const TensorAttributes* raggedOffsetLseAttributes = nullptr,
                       const TensorAttributes* sinkAttributes = nullptr)
        : qTensor(unpackTensorAttributes(qAttributes))
        , kTensor(unpackTensorAttributes(kAttributes))
        , vTensor(unpackTensorAttributes(vAttributes))
        , oTensor(unpackTensorAttributes(oAttributes))
        , pageTableKTensor(unpackTensorAttributes(pageTableKAttributes))
        , pageTableVTensor(unpackTensorAttributes(pageTableVAttributes))
        , seqLenKvTensor(unpackTensorAttributes(seqLenKvAttributes))
        , seqLenQTensor(unpackOptional(seqLenQAttributes))
        , maxSeqLenKv(maxSeqLenKv)
        , scaleTensor(std::move(scale))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , lseTensor(unpackOptional(lseAttributes))
        , raggedOffsetQTensor(unpackOptional(raggedOffsetQAttributes))
        , raggedOffsetOTensor(unpackOptional(raggedOffsetOAttributes))
        , raggedOffsetLseTensor(unpackOptional(raggedOffsetLseAttributes))
        , sinkTensor(unpackOptional(sinkAttributes))
    {
    }

    TensorAttributesT qTensor;
    TensorAttributesT kTensor;
    TensorAttributesT vTensor;
    TensorAttributesT oTensor;
    TensorAttributesT pageTableKTensor;
    TensorAttributesT pageTableVTensor;
    TensorAttributesT seqLenKvTensor;
    std::optional<TensorAttributesT> seqLenQTensor;
    int64_t maxSeqLenKv;
    // Folded scale operand (scale tensor, else baked attn_scale_value). Absent means 1.0.
    std::optional<TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<TensorAttributesT> lseTensor;
    // Set exactly for the ragged tensors among Q, O and LSE.
    std::optional<TensorAttributesT> raggedOffsetQTensor;
    std::optional<TensorAttributesT> raggedOffsetOTensor;
    std::optional<TensorAttributesT> raggedOffsetLseTensor;
    // Attention sink [1, H_q, 1, 1] (FLOAT, HALF or BFLOAT16).
    std::optional<TensorAttributesT> sinkTensor;
};

// Runs CpuFpReferenceSdpaRagged::forwardPaged on the variant-pack host buffers.
template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
class SdpaFwdPagedPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaFwdPagedPlan(SdpaFwdPagedParams&& params)
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
        auto q
            = bindQueryLayout<QDataType>(_params.qTensor, _params.raggedOffsetQTensor, variantPack);
        auto k
            = createShallowTensor<KDataType>(_params.kTensor, variantPack.at(_params.kTensor.uid));
        auto v
            = createShallowTensor<VDataType>(_params.vTensor, variantPack.at(_params.vTensor.uid));
        auto o
            = bindQueryLayout<ODataType>(_params.oTensor, _params.raggedOffsetOTensor, variantPack);

        auto pageTableK = createShallowTensor<int32_t>(
            _params.pageTableKTensor, variantPack.at(_params.pageTableKTensor.uid));
        auto pageTableV = createShallowTensor<int32_t>(
            _params.pageTableVTensor, variantPack.at(_params.pageTableVTensor.uid));
        auto seqLenKv = createShallowTensor<int32_t>(_params.seqLenKvTensor,
                                                     variantPack.at(_params.seqLenKvTensor.uid));
        auto seqLenQ = bindOptionalShallowTensor<int32_t>(_params.seqLenQTensor, variantPack);

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> lse;
        if(_params.lseTensor.has_value())
        {
            lse = bindQueryLayout<float>(
                *_params.lseTensor, _params.raggedOffsetLseTensor, variantPack);
        }

        // Resolve the folded scale operand (runtime tensor or baked attn_scale_value) to a scalar;
        // nullopt => 1.0, no scaling (cuDNN's default).
        std::optional<float> effectiveScale;
        if(_params.scaleTensor.has_value())
        {
            effectiveScale = hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
                _params.scaleTensor.value(), variantPack, "SDPA scale");
        }

        std::optional<std::vector<float>> sinkPerHead;
        if(_params.sinkTensor.has_value())
        {
            sinkPerHead
                = readSdpaSinkPerHead(*_params.sinkTensor, variantPack, _params.qTensor.dims.at(1));
        }

        utilities::CpuFpReferenceSdpaRagged::
            forwardPaged<QDataType, KDataType, VDataType, ODataType, float>(
                *q,
                *k,
                *v,
                *o,
                *pageTableK,
                *pageTableV,
                *seqLenKv,
                seqLenQ.get(),
                _params.maxSeqLenKv,
                effectiveScale,
                _params.leftBound,
                _params.rightBound,
                _params.topLeftAlignment,
                lse.get(),
                nullptr,
                nullptr,
                nullptr,
                sinkPerHead.has_value() ? &*sinkPerHead : nullptr);
    }

private:
    // Q, O and LSE are ragged when their attributes carry a ragged_offset aux. The ragged view
    // reads the aux from the variant pack and scales it by the ragged_offset_multiplier.
    template <typename T>
    static std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<T>> bindQueryLayout(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensor,
        const std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>& raggedOffset,
        const std::unordered_map<int64_t, void*>& variantPack)
    {
        if(!raggedOffset.has_value())
        {
            return createShallowTensor<T>(tensor, variantPack.at(tensor.uid));
        }
        std::shared_ptr<hipdnn_data_sdk::utilities::ITensor> aux
            = createShallowTensor<int32_t>(*raggedOffset, variantPack.at(raggedOffset->uid));
        return std::make_unique<hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>>(
            variantPack.at(tensor.uid),
            tensor.dims,
            tensor.strides,
            hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
            std::move(aux),
            std::nullopt,
            tensor.ragged_offset_multiplier);
    }

    SdpaFwdPagedParams _params;
};

// Paged SDPA forward: applicable iff both page tables are present (the dense SdpaFwdPlanBuilder
// declines them). Also requires seq_len_kv and max_seq_len_kv, and seq_len_q when Q is ragged.
// Supports the diagonal band mask, the scale operand, padding_mask either way (seq lens are always
// honored) and an FLOAT LSE. Q, O and LSE may be ragged (Q and O together); K, V, the page tables
// and the seq lens may not. Attention sinks follow the frontend contract. No attention bias,
// alibi, dropout, block mask, descales, softmax/output quantization or max / sum-exp outputs.
template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class SdpaFwdPagedPlanBuilder : public IGraphNodePlanBuilder
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

        // Paged K/V: both page tables, seq_len_kv (all INT32) and max_seq_len_kv.
        constexpr auto INT32_DATA_TYPE = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
        CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->page_table_k_tensor_uid());
        CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->page_table_v_tensor_uid());
        CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->seq_len_kv_tensor_uid());
        CHECK_OPTIONAL_TENSOR_TYPE(
            tensorMap, nodeAttributes->page_table_k_tensor_uid(), INT32_DATA_TYPE);
        CHECK_OPTIONAL_TENSOR_TYPE(
            tensorMap, nodeAttributes->page_table_v_tensor_uid(), INT32_DATA_TYPE);
        CHECK_OPTIONAL_TENSOR_TYPE(
            tensorMap, nodeAttributes->seq_len_kv_tensor_uid(), INT32_DATA_TYPE);
        if(!nodeAttributes->max_seq_len_kv().has_value())
        {
            return false;
        }
        if(nodeAttributes->seq_len_q_tensor_uid().has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, nodeAttributes->seq_len_q_tensor_uid());
            CHECK_OPTIONAL_TENSOR_TYPE(
                tensorMap, nodeAttributes->seq_len_q_tensor_uid(), INT32_DATA_TYPE);
        }

        // Q and O are ragged together; a ragged Q needs seq_len_q. A ragged LSE needs a ragged Q.
        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());
        const bool raggedQ = qAttr->ragged_offset_tensor_uid().has_value();
        if(raggedQ != oAttr->ragged_offset_tensor_uid().has_value())
        {
            return false;
        }
        if(raggedQ && !nodeAttributes->seq_len_q_tensor_uid().has_value())
        {
            return false;
        }
        std::unordered_set<int64_t> raggedIds;
        if(raggedQ)
        {
            if(!isSupportedRaggedTensor(*qAttr, tensorMap)
               || !isSupportedRaggedTensor(*oAttr, tensorMap))
            {
                return false;
            }
            raggedIds.insert(qAttr->uid());
            raggedIds.insert(oAttr->uid());
        }

        // Unsupported mask modes. padding_mask may be either value: seq lens are always honored.
        if(nodeAttributes->alibi_mask() || nodeAttributes->attn_mask_tensor_uid().has_value())
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

        // Unsupported: block sparse attention
        if(nodeAttributes->block_mask_tensor_uid().has_value())
        {
            return false;
        }

        // Supported: an attention sink of the frontend-contract shape and data type. It is not
        // ragged, so it never joins raggedIds.
        if(!isSdpaSinkSupported(*nodeAttributes, tensorMap))
        {
            return false;
        }

        // Unsupported: fp8 descales and softmax/output (de)quantization.
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

        // Unsupported: max / running-sum softmax stats outputs.
        if(nodeAttributes->max_tensor_uid().has_value()
           || nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            return false;
        }

        // Supported: a FLOAT log-sum-exp (stats) output, dense or ragged like Q.
        if(nodeAttributes->stats_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->stats_tensor_uid().value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
            const auto* lseAttr = tensorMap.at(nodeAttributes->stats_tensor_uid().value());
            if(lseAttr->ragged_offset_tensor_uid().has_value())
            {
                if(!raggedQ || !isSupportedRaggedTensor(*lseAttr, tensorMap))
                {
                    return false;
                }
                raggedIds.insert(lseAttr->uid());
            }
        }

        // Only Q, O and LSE may be ragged.
        return hipdnn_flatbuffers_sdk::utilities::listUnsupportedRaggedTensorIds(tensorMap,
                                                                                 raggedIds)
            .empty();
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
        if(!nodeAttributes->page_table_k_tensor_uid().has_value()
           || !nodeAttributes->page_table_v_tensor_uid().has_value()
           || !nodeAttributes->seq_len_kv_tensor_uid().has_value()
           || !nodeAttributes->max_seq_len_kv().has_value())
        {
            throw std::runtime_error("SdpaFwdPagedPlan: page_table_k, page_table_v, seq_len_kv and "
                                     "max_seq_len_kv are required");
        }

        const auto& tensorMap = graph.getTensorMap();
        const auto optionalAttr = [&tensorMap](::flatbuffers::Optional<int64_t> uid)
            -> const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* {
            return uid.has_value() ? tensorMap.at(uid.value()) : nullptr;
        };

        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }
        auto scale
            = foldSdpaScale(optionalAttr(nodeAttributes->scale_tensor_uid()), attnScaleValue);

        auto [leftBound, rightBound, isTopLeft]
            = extractDiagonalBandParams(*nodeAttributes, "SdpaFwdPagedPlan");

        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());
        const auto* lseAttr = optionalAttr(nodeAttributes->stats_tensor_uid());

        return std::make_unique<SdpaFwdPagedPlan<QDataType, KDataType, VDataType, ODataType>>(
            SdpaFwdPagedParams(
                *qAttr,
                *tensorMap.at(nodeAttributes->k_tensor_uid()),
                *tensorMap.at(nodeAttributes->v_tensor_uid()),
                *oAttr,
                *tensorMap.at(nodeAttributes->page_table_k_tensor_uid().value()),
                *tensorMap.at(nodeAttributes->page_table_v_tensor_uid().value()),
                *tensorMap.at(nodeAttributes->seq_len_kv_tensor_uid().value()),
                optionalAttr(nodeAttributes->seq_len_q_tensor_uid()),
                nodeAttributes->max_seq_len_kv().value(),
                std::move(scale),
                leftBound,
                rightBound,
                isTopLeft,
                lseAttr,
                optionalAttr(qAttr->ragged_offset_tensor_uid()),
                optionalAttr(oAttr->ragged_offset_tensor_uid()),
                lseAttr != nullptr ? optionalAttr(lseAttr->ragged_offset_tensor_uid()) : nullptr,
                optionalAttr(nodeAttributes->sink_token_tensor_uid())));
    }

private:
    // A ragged Q/O/LSE needs an INT32 ragged_offset aux in the graph, a multiplier >= 1 and the
    // RFC-0014 [B, H, S, D] token-major layout.
    static bool isSupportedRaggedTensor(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& attr,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto auxUid = attr.ragged_offset_tensor_uid();
        if(!auxUid.has_value())
        {
            return false;
        }
        const auto aux = tensorMap.find(auxUid.value());
        if(aux == tensorMap.end()
           || aux->second->data_type() != hipdnn_flatbuffers_sdk::data_objects::DataType::INT32)
        {
            return false;
        }
        const auto unpacked = unpackTensorAttributes(attr);
        return unpacked.ragged_offset_multiplier >= 1
               && isTokenMajorRaggedLayout(unpacked.dims, unpacked.strides);
    }
};

} // namespace hipdnn_test_sdk::detail
