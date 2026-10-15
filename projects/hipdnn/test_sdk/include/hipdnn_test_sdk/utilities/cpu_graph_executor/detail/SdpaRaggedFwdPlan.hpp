// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

// Unpacked attributes and resolved parameters for a ragged SDPA node (RFC-0014 logical
// [B,H,S,D], packed BSHD + ragged_offset). q/k/v/o each carry an int32 offset aux [B+1,1,1,1],
// scaled to elements by the tensor's ragged_offset_multiplier.
// The optional LSE [B,H,Sq,1] is packed if it has its own ragged_offset aux, else dense.
struct SdpaRaggedFwdParams
{
    SdpaRaggedFwdParams(
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
        : qTensor(unpackTensorAttributes(qAttributes))
        , kTensor(unpackTensorAttributes(kAttributes))
        , vTensor(unpackTensorAttributes(vAttributes))
        , oTensor(unpackTensorAttributes(oAttributes))
        , raggedOffsetQTensor(unpackTensorAttributes(raggedOffsetQAttributes))
        , raggedOffsetKTensor(unpackTensorAttributes(raggedOffsetKAttributes))
        , raggedOffsetVTensor(unpackTensorAttributes(raggedOffsetVAttributes))
        , raggedOffsetOTensor(unpackTensorAttributes(raggedOffsetOAttributes))
        , scaleTensor(std::move(scaleTensor))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , lseTensor(lseAttributes != nullptr
                        ? std::make_optional(unpackTensorAttributes(*lseAttributes))
                        : std::nullopt)
        , raggedOffsetLseTensor(
              raggedOffsetLseAttributes != nullptr
                  ? std::make_optional(unpackTensorAttributes(*raggedOffsetLseAttributes))
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
    // Set only for a packed LSE. Absent means a dense [B,H,Sq,1] LSE.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> raggedOffsetLseTensor;
    // Optional fp8 descales (float): scalar [1] or per-KV-head [B, H_kv, 1, 1].
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleVTensor;
};

// Runs CpuFpReferenceSdpaRagged on the variant-pack host buffers. Every ragged operand is a
// ShallowRaggedTensor view, so the reference addresses the packed data through its offset table.
template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
class SdpaRaggedFwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaRaggedFwdPlan(SdpaRaggedFwdParams&& params)
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
        auto qTensor
            = bindRagged<QDataType>(_params.qTensor, _params.raggedOffsetQTensor, variantPack);
        auto kTensor
            = bindRagged<KDataType>(_params.kTensor, _params.raggedOffsetKTensor, variantPack);
        auto vTensor
            = bindRagged<VDataType>(_params.vTensor, _params.raggedOffsetVTensor, variantPack);
        auto oTensor
            = bindRagged<ODataType>(_params.oTensor, _params.raggedOffsetOTensor, variantPack);

        // A packed LSE is ragged on its own offset table. A dense LSE is a plain view.
        std::optional<hipdnn_data_sdk::utilities::ShallowRaggedTensor<float>> raggedLseTensor;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> denseLseTensor;
        hipdnn_data_sdk::utilities::TensorBase<float>* lseTensor = nullptr;
        if(_params.lseTensor.has_value())
        {
            const auto& lse = *_params.lseTensor;
            if(_params.raggedOffsetLseTensor.has_value())
            {
                lseTensor = &raggedLseTensor.emplace(
                    variantPack.at(lse.uid),
                    lse.dims,
                    lse.strides,
                    hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
                    bindOffsets(*_params.raggedOffsetLseTensor, variantPack),
                    std::nullopt,
                    lse.ragged_offset_multiplier);
            }
            else
            {
                denseLseTensor = createShallowTensor<float>(lse, variantPack.at(lse.uid));
                lseTensor = denseLseTensor.get();
            }
        }

        const auto descaleQ = bindDescale(_params.descaleQTensor, variantPack);
        const auto descaleK = bindDescale(_params.descaleKTensor, variantPack);
        const auto descaleV = bindDescale(_params.descaleVTensor, variantPack);

        std::optional<float> attnScale;
        if(_params.scaleTensor.has_value())
        {
            attnScale = resolveScalarOperand(*_params.scaleTensor, variantPack, "SDPA scale");
        }

        utilities::CpuFpReferenceSdpaRagged::
            forward<QDataType, KDataType, VDataType, ODataType, float>(qTensor,
                                                                       kTensor,
                                                                       vTensor,
                                                                       oTensor,
                                                                       attnScale,
                                                                       _params.leftBound,
                                                                       _params.rightBound,
                                                                       _params.topLeftAlignment,
                                                                       lseTensor,
                                                                       descaleQ.get(),
                                                                       descaleK.get(),
                                                                       descaleV.get());
    }

private:
    // The int32 ragged_offset aux viewed in place. Views are non-owning, so Q/O and K/V may share
    // one table.
    static std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
        bindOffsets(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& raggedOffset,
                    const std::unordered_map<int64_t, void*>& variantPack)
    {
        return std::make_shared<hipdnn_data_sdk::utilities::ShallowTensor<int32_t>>(
            variantPack.at(raggedOffset.uid), raggedOffset.dims, raggedOffset.strides);
    }

    // Views a packed primary through its ragged_offset aux, scaled by the primary's
    // ragged_offset_multiplier.
    template <typename T>
    static hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>
        bindRagged(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensor,
                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& raggedOffset,
                   const std::unordered_map<int64_t, void*>& variantPack)
    {
        return hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>(
            variantPack.at(tensor.uid),
            tensor.dims,
            tensor.strides,
            hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
            bindOffsets(raggedOffset, variantPack),
            std::nullopt,
            tensor.ragged_offset_multiplier);
    }

    // A host-scalar descale (baked or pass-by-value) is staged in a one-element tensor. An
    // ordinary descale tensor is viewed in place. CpuFpReferenceSdpaRagged checks the shape.
    static std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> bindDescale(
        const std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>& descale,
        const std::unordered_map<int64_t, void*>& variantPack)
    {
        if(!descale.has_value())
        {
            return nullptr;
        }
        if(isHostScalarOperand(*descale))
        {
            auto staged = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(
                std::vector<int64_t>{1});
            staged->fillWithValue(resolveScalarOperand(*descale, variantPack, "SDPA descale"));
            return staged;
        }
        return createShallowTensor<float>(*descale, variantPack.at(descale->uid));
    }

    // A host scalar is baked or read from its pass-by-value slot. An ordinary tensor's slot is
    // a host buffer on the CPU reference; isApplicable admits it only as one element.
    static float
        resolveScalarOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& operand,
                             const std::unordered_map<int64_t, void*>& variantPack,
                             const char* name)
    {
        if(isHostScalarOperand(operand))
        {
            return hipdnn_flatbuffers_sdk::utilities::resolveScalarFromVariantPack<float>(
                operand, variantPack, name);
        }
        return hipdnn_flatbuffers_sdk::utilities::readHostScalarAs<float>(
            variantPack.at(operand.uid), operand.data_type, name);
    }

    SdpaRaggedFwdParams _params;
};

// Accepts exactly what GpuSdpaRaggedFwdPlanBuilder accepts: q/k/v/o (and a packed LSE) must each
// carry an INT32 ragged_offset aux with a multiplier >= 1 and use the RFC-0014 [B, H, S, D]
// token-major layout, and seq_len_q/kv must be absent (the padded variant is not supported).
// Unlike the dense SdpaFwdPlanBuilder, an additive attn_mask is rejected and Q/K/V descales are
// optional. CpuReferenceGraphExecutor::buildSignatureKey picks dense vs ragged.
template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class SdpaRaggedFwdPlanBuilder : public IGraphNodePlanBuilder
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

        // Each primary needs an INT32 ragged_offset aux (without them this is a dense SDPA node),
        // a ragged_offset_multiplier >= 1 and the RFC-0014 [B, H, S, D] token-major layout.
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
            if(!isSupportedRaggedSdpaLayout(*primary))
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
                const auto descale = unpackTensorAttributes(*tensorMap.at(descaleUid.value()));
                if(isHostScalarOperand(descale) && elementCount(descale) != 1)
                {
                    return false;
                }
            }
        }

        // The scale may be baked, pass-by-value or an ordinary tensor. It must be one element,
        // and an ordinary scale tensor must be FLOAT, matching the GPU reference.
        if(nodeAttributes->scale_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->scale_tensor_uid().value());
            const auto scale
                = unpackTensorAttributes(*tensorMap.at(nodeAttributes->scale_tensor_uid().value()));
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
        // otherwise dense [B,H,Sq,1].
        if(nodeAttributes->stats_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->stats_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap,
                              nodeAttributes->stats_tensor_uid().value(),
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);
            const auto* stats = tensorMap.at(nodeAttributes->stats_tensor_uid().value());
            if(stats->ragged_offset_tensor_uid().has_value())
            {
                CHECK_TENSOR_EXISTS(tensorMap, stats->ragged_offset_tensor_uid().value());
                CHECK_TENSOR_TYPE(tensorMap,
                                  stats->ragged_offset_tensor_uid().value(),
                                  hipdnn_flatbuffers_sdk::data_objects::DataType::INT32);
                if(!isSupportedRaggedSdpaLayout(*stats))
                {
                    return false;
                }
            }
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
        const auto optionalTensor = [&tensorMap](::flatbuffers::Optional<int64_t> uid)
            -> const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* {
            return uid.has_value() ? tensorMap.at(uid.value()) : nullptr;
        };

        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }
        auto scale
            = foldSdpaScale(optionalTensor(nodeAttributes->scale_tensor_uid()), attnScaleValue);

        const auto* lsePtr = optionalTensor(nodeAttributes->stats_tensor_uid());
        const auto* raggedOffsetLsePtr
            = lsePtr != nullptr ? optionalTensor(lsePtr->ragged_offset_tensor_uid()) : nullptr;

        auto [leftBound, rightBound, isTopLeft]
            = extractDiagonalBandParams(*nodeAttributes, "SdpaRaggedFwdPlan");

        // isApplicable checked that every primary has a ragged_offset aux. The reference checks
        // that o and v share the per-batch lengths of q and k.
        const auto* qAttr = tensorMap.at(nodeAttributes->q_tensor_uid());
        const auto* kAttr = tensorMap.at(nodeAttributes->k_tensor_uid());
        const auto* vAttr = tensorMap.at(nodeAttributes->v_tensor_uid());
        const auto* oAttr = tensorMap.at(nodeAttributes->o_tensor_uid());

        return std::make_unique<SdpaRaggedFwdPlan<QDataType, KDataType, VDataType, ODataType>>(
            SdpaRaggedFwdParams(*qAttr,
                                *kAttr,
                                *vAttr,
                                *oAttr,
                                *tensorMap.at(qAttr->ragged_offset_tensor_uid().value()),
                                *tensorMap.at(kAttr->ragged_offset_tensor_uid().value()),
                                *tensorMap.at(vAttr->ragged_offset_tensor_uid().value()),
                                *tensorMap.at(oAttr->ragged_offset_tensor_uid().value()),
                                std::move(scale),
                                leftBound,
                                rightBound,
                                isTopLeft,
                                lsePtr,
                                raggedOffsetLsePtr,
                                optionalTensor(nodeAttributes->descale_q_tensor_uid()),
                                optionalTensor(nodeAttributes->descale_k_tensor_uid()),
                                optionalTensor(nodeAttributes->descale_v_tensor_uid())));
    }
};

} // namespace hipdnn_test_sdk::detail
