// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

// Unpacks a finalized HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR into
// frontend SdpaAttributes. Required tensors are registered in tensorMap; optional
// tensors and scalars are set only when present in the descriptor.
[[nodiscard]] inline Error unpackSdpaFwdOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::SdpaAttributes& attributes)
{
    // Unpack q tensor
    std::shared_ptr<graph::TensorAttributes> qTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_FWD_QDESC, tensorMap, qTensor, "sdpa Q_EXT tensor"));
    attributes.set_q(qTensor);

    // Unpack k tensor
    std::shared_ptr<graph::TensorAttributes> kTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_FWD_KDESC, tensorMap, kTensor, "sdpa K_EXT tensor"));
    attributes.set_k(kTensor);

    // Unpack v tensor
    std::shared_ptr<graph::TensorAttributes> vTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_FWD_VDESC, tensorMap, vTensor, "sdpa V_EXT tensor"));
    attributes.set_v(vTensor);

    // Unpack o tensor
    std::shared_ptr<graph::TensorAttributes> oTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_SDPA_FWD_ODESC, tensorMap, oTensor, "sdpa O_EXT tensor"));
    attributes.set_o(oTensor);

    // Unpack bias tensor
    std::shared_ptr<graph::TensorAttributes> biasTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_ATTN_MASK_EXT,
                                            tensorMap,
                                            biasTensor,
                                            "sdpa ATTN_MASK_EXT tensor"));
    if(biasTensor)
    {
        attributes.set_bias(biasTensor);
    }

    // Unpack attn_scale tensor
    std::shared_ptr<graph::TensorAttributes> attnScaleTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                                            tensorMap,
                                            attnScaleTensor,
                                            "sdpa SCALE_EXT tensor"));
    if(attnScaleTensor)
    {
        attributes.set_attn_scale(attnScaleTensor);
    }

    // Unpack seq_len_q tensor
    std::shared_ptr<graph::TensorAttributes> seqLenQTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                                            tensorMap,
                                            seqLenQTensor,
                                            "sdpa SEQ_LEN_Q_EXT tensor"));
    if(seqLenQTensor)
    {
        attributes.set_seq_len_q(seqLenQTensor);
    }

    // Unpack seq_len_kv tensor
    std::shared_ptr<graph::TensorAttributes> seqLenKvTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
                                            tensorMap,
                                            seqLenKvTensor,
                                            "sdpa SEQ_LEN_KV_EXT tensor"));
    if(seqLenKvTensor)
    {
        attributes.set_seq_len_kv(seqLenKvTensor);
    }

    // Unpack seed tensor
    std::shared_ptr<graph::TensorAttributes> seedTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SEED_EXT,
                                            tensorMap,
                                            seedTensor,
                                            "sdpa SEED_EXT tensor"));
    if(seedTensor)
    {
        attributes.set_seed(seedTensor);
    }

    // Unpack offset tensor
    std::shared_ptr<graph::TensorAttributes> offsetTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_OFFSET_EXT,
                                            tensorMap,
                                            offsetTensor,
                                            "sdpa OFFSET_EXT tensor"));
    if(offsetTensor)
    {
        attributes.set_offset(offsetTensor);
    }

    // Unpack dropout_mask tensor
    std::shared_ptr<graph::TensorAttributes> dropoutMaskTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_MASK_EXT,
                                            tensorMap,
                                            dropoutMaskTensor,
                                            "sdpa DROPOUT_MASK_EXT tensor"));
    if(dropoutMaskTensor)
    {
        attributes.set_dropout_mask(dropoutMaskTensor);
    }

    // Unpack dropout_scale tensor
    std::shared_ptr<graph::TensorAttributes> dropoutScaleTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SCALE_EXT,
                                            tensorMap,
                                            dropoutScaleTensor,
                                            "sdpa DROPOUT_SCALE_EXT tensor"));
    if(dropoutScaleTensor)
    {
        attributes.set_dropout_scale(dropoutScaleTensor);
    }

    // Unpack page_table_k tensor
    std::shared_ptr<graph::TensorAttributes> pageTableKTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC,
                                            tensorMap,
                                            pageTableKTensor,
                                            "sdpa PAGE_TABLE_K_EXT tensor"));
    if(pageTableKTensor)
    {
        attributes.set_page_table_k(pageTableKTensor);
    }

    // Unpack page_table_v tensor
    std::shared_ptr<graph::TensorAttributes> pageTableVTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC,
                                            tensorMap,
                                            pageTableVTensor,
                                            "sdpa PAGE_TABLE_V_EXT tensor"));
    if(pageTableVTensor)
    {
        attributes.set_page_table_v(pageTableVTensor);
    }

    // Unpack block_mask tensor
    std::shared_ptr<graph::TensorAttributes> blockMaskTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC,
                                            tensorMap,
                                            blockMaskTensor,
                                            "sdpa BLOCK_MASK_EXT tensor"));
    if(blockMaskTensor)
    {
        attributes.set_block_mask(blockMaskTensor);
    }

    // Unpack sink_token tensor
    std::shared_ptr<graph::TensorAttributes> sinkTokenTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SINK_TOKEN_EXT,
                                            tensorMap,
                                            sinkTokenTensor,
                                            "sdpa SINK_TOKEN_EXT tensor"));
    if(sinkTokenTensor)
    {
        attributes.set_sink_token(sinkTokenTensor);
    }

    // Unpack descale_q tensor
    std::shared_ptr<graph::TensorAttributes> descaleQTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_Q_EXT,
                                            tensorMap,
                                            descaleQTensor,
                                            "sdpa DESCALE_Q_EXT tensor"));
    if(descaleQTensor)
    {
        attributes.set_descale_q(descaleQTensor);
    }

    // Unpack descale_k tensor
    std::shared_ptr<graph::TensorAttributes> descaleKTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_K_EXT,
                                            tensorMap,
                                            descaleKTensor,
                                            "sdpa DESCALE_K_EXT tensor"));
    if(descaleKTensor)
    {
        attributes.set_descale_k(descaleKTensor);
    }

    // Unpack descale_v tensor
    std::shared_ptr<graph::TensorAttributes> descaleVTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_V_EXT,
                                            tensorMap,
                                            descaleVTensor,
                                            "sdpa DESCALE_V_EXT tensor"));
    if(descaleVTensor)
    {
        attributes.set_descale_v(descaleVTensor);
    }

    // Unpack descale_s tensor
    std::shared_ptr<graph::TensorAttributes> descaleSTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_S_EXT,
                                            tensorMap,
                                            descaleSTensor,
                                            "sdpa DESCALE_S_EXT tensor"));
    if(descaleSTensor)
    {
        attributes.set_descale_s(descaleSTensor);
    }

    // Unpack scale_s tensor
    std::shared_ptr<graph::TensorAttributes> scaleSTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_S_EXT,
                                            tensorMap,
                                            scaleSTensor,
                                            "sdpa SCALE_S_EXT tensor"));
    if(scaleSTensor)
    {
        attributes.set_scale_s(scaleSTensor);
    }

    // Unpack scale_o tensor
    std::shared_ptr<graph::TensorAttributes> scaleOTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_O_EXT,
                                            tensorMap,
                                            scaleOTensor,
                                            "sdpa SCALE_O_EXT tensor"));
    if(scaleOTensor)
    {
        attributes.set_scale_o(scaleOTensor);
    }

    // Unpack stats tensor
    std::shared_ptr<graph::TensorAttributes> statsTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
                                            tensorMap,
                                            statsTensor,
                                            "sdpa STATS_EXT tensor"));
    if(statsTensor)
    {
        attributes.set_stats(statsTensor);
    }

    // Unpack max_output tensor
    std::shared_ptr<graph::TensorAttributes> maxOutputTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_MAX_EXT,
                                            tensorMap,
                                            maxOutputTensor,
                                            "sdpa MAX_EXT tensor"));
    if(maxOutputTensor)
    {
        attributes.set_max(maxOutputTensor);
    }

    // Unpack sum_exp tensor
    std::shared_ptr<graph::TensorAttributes> sumExpTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_SUM_EXP_EXT,
                                            tensorMap,
                                            sumExpTensor,
                                            "sdpa SUM_EXP_EXT tensor"));
    if(sumExpTensor)
    {
        attributes.set_sum_exp(sumExpTensor);
    }

    // Unpack rng_dump tensor
    std::shared_ptr<graph::TensorAttributes> rngDumpTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_RNG_DUMP_EXT,
                                            tensorMap,
                                            rngDumpTensor,
                                            "sdpa RNG_DUMP_EXT tensor"));
    if(rngDumpTensor)
    {
        attributes.set_rng_dump(rngDumpTensor);
    }

    // Unpack amax_s tensor
    std::shared_ptr<graph::TensorAttributes> amaxSTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_S_EXT,
                                            tensorMap,
                                            amaxSTensor,
                                            "sdpa AMAX_S_EXT tensor"));
    if(amaxSTensor)
    {
        attributes.set_amax_s(amaxSTensor);
    }

    // Unpack amax_o tensor
    std::shared_ptr<graph::TensorAttributes> amaxOTensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_O_EXT,
                                            tensorMap,
                                            amaxOTensor,
                                            "sdpa AMAX_O_EXT tensor"));
    if(amaxOTensor)
    {
        attributes.set_amax_o(amaxOTensor);
    }

    // Unpack mma_core_mode — the packer omits this attribute when NOT_SET because
    // toHipdnnDataType() has no mapping for NOT_SET, so setDescriptorAttrDataType()
    // would return an error. The attribute is therefore absent when not explicitly set.
    {
        auto [mmaCoreMode, mmaCoreModeErr] = unpackGraphDataType(
            opDesc, HIPDNN_ATTR_SDPA_FWD_MMA_CORE_MODE_EXT, "sdpa mma_core_mode");
        if(!mmaCoreModeErr.is_bad())
        {
            attributes.set_mma_core_mode(mmaCoreMode);
        }
    }

    // Unpack generate_stats (optional)
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_FWD_GENERATE_STATS_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "sdpa generate_stats"));
        attributes.generate_stats = opt;
    }

    // Unpack alibi_mask (optional)
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_FWD_ALIBI_MASK_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "sdpa alibi_mask"));
        attributes.alibi_mask = opt.value_or(false);
    }

    // Unpack padding_mask (optional)
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_FWD_PADDING_MASK_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "sdpa padding_mask"));
        attributes.padding_mask = opt.value_or(false);
    }

    // Unpack causal_mask (optional)
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_SDPA_FWD_CAUSAL_MASK_EXT,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           opt,
                                                           "sdpa causal_mask"));
        attributes.causal_mask = opt.value_or(false);
    }

    // Unpack causal_mask_bottom_right (optional)
    {
        std::optional<bool> opt;
        HIPDNN_CHECK_ERROR(
            getDescriptorAttrOptionalScalar(opDesc,
                                            HIPDNN_ATTR_SDPA_FWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT,
                                            HIPDNN_TYPE_BOOLEAN,
                                            opt,
                                            "sdpa causal_mask_bottom_right"));
        attributes.causal_mask_bottom_right = opt.value_or(false);
    }

    // Unpack dropout_probability (optional)
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_FWD_DROPOUT_PROBABILITY_EXT,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.dropout_probability,
                                                       "sdpa dropout_probability"));

    // Unpack attn_scale_value (optional)
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_FWD_ATTN_SCALE_VALUE_EXT,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.attn_scale_value,
                                                       "sdpa attn_scale_value"));

    // Unpack left_bound (optional)
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_FWD_LEFT_BOUND_EXT,
                                                       HIPDNN_TYPE_INT64,
                                                       attributes.left_bound,
                                                       "sdpa left_bound"));

    // Unpack right_bound (optional)
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_FWD_RIGHT_BOUND_EXT,
                                                       HIPDNN_TYPE_INT64,
                                                       attributes.right_bound,
                                                       "sdpa right_bound"));

    // Unpack max_seq_len_kv (optional)
    HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                       HIPDNN_ATTR_SDPA_FWD_MAX_SEQ_LEN_KV_EXT,
                                                       HIPDNN_TYPE_INT32,
                                                       attributes.max_seq_len_kv,
                                                       "sdpa max_seq_len_kv"));

    // Unpack diagonal_alignment
    hipdnnDiagonalAlignment_t diagonalAlignment{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_SDPA_FWD_DIAGONAL_ALIGNMENT_EXT,
                                               HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                               diagonalAlignment,
                                               "sdpa diagonal_alignment"));
    auto [diagonalAlignmentResult, diagonalAlignmentErr]
        = fromHipdnnDiagonalAlignment(diagonalAlignment);
    if(diagonalAlignmentErr.is_bad())
    {
        return diagonalAlignmentErr;
    }
    attributes.set_diagonal_alignment(diagonalAlignmentResult);

    // Unpack implementation
    hipdnnAttentionImplementation_t implementation{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_SDPA_FWD_IMPLEMENTATION_EXT,
                                               HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT,
                                               implementation,
                                               "sdpa implementation"));
    auto [implementationResult, implementationErr]
        = fromHipdnnAttentionImplementation(implementation);
    if(implementationErr.is_bad())
    {
        return implementationErr;
    }
    attributes.set_implementation(implementationResult);

    // Unpack compute data type
    auto [dt, dtErr]
        = unpackGraphDataType(opDesc, HIPDNN_ATTR_SDPA_FWD_COMP_TYPE_EXT, "sdpa compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
