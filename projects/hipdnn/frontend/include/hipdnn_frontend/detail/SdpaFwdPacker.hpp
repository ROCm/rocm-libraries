// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

#include "HipdnnAttentionImplementation.h"
#include "HipdnnDiagonalAlignment.h"

namespace hipdnn_frontend::detail
{

// Builds an SDPA fprop operation descriptor from SdpaAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createSdpaFwdOperation(const graph::SdpaAttributes& attributes,
                           std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                           std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_SDPA_FWD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create SDPA fprop operation descriptor"};
    }

    // Required tensors
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_FWD_QDESC,
                                             attributes.get_q(),
                                             tensorDescs,
                                             "SDPA fprop Q"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_FWD_KDESC,
                                             attributes.get_k(),
                                             tensorDescs,
                                             "SDPA fprop K"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_FWD_VDESC,
                                             attributes.get_v(),
                                             tensorDescs,
                                             "SDPA fprop V"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_FWD_ODESC,
                                             attributes.get_o(),
                                             tensorDescs,
                                             "SDPA fprop O"));

    // Optional input tensors
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_ATTN_MASK_EXT,
                                                     attributes.get_bias(),
                                                     tensorDescs,
                                                     "SDPA fprop ATTN_MASK"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALEDESC,
                                                     attributes.get_attn_scale(),
                                                     tensorDescs,
                                                     "SDPA fprop SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_QDESC,
                                                     attributes.get_seq_len_q(),
                                                     tensorDescs,
                                                     "SDPA fprop SEQ_LEN_Q"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SEQ_LEN_KVDESC,
                                                     attributes.get_seq_len_kv(),
                                                     tensorDescs,
                                                     "SDPA fprop SEQ_LEN_KV"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SEED_EXT,
                                                     attributes.get_seed(),
                                                     tensorDescs,
                                                     "SDPA fprop SEED"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_OFFSET_EXT,
                                                     attributes.get_offset(),
                                                     tensorDescs,
                                                     "SDPA fprop OFFSET"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_MASK_EXT,
                                      attributes.get_dropout_mask(),
                                      tensorDescs,
                                      "SDPA fprop DROPOUT_MASK"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_FWD_DROPOUT_SCALE_EXT,
                                      attributes.get_dropout_scale(),
                                      tensorDescs,
                                      "SDPA fprop DROPOUT_SCALE"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_KDESC,
                                      attributes.get_page_table_k(),
                                      tensorDescs,
                                      "SDPA fprop PAGE_TABLE_K"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_FWD_PAGE_TABLE_VDESC,
                                      attributes.get_page_table_v(),
                                      tensorDescs,
                                      "SDPA fprop PAGE_TABLE_V"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_BLOCK_MASK_DESC,
                                                     attributes.get_block_mask(),
                                                     tensorDescs,
                                                     "SDPA fprop BLOCK_MASK"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SINK_TOKEN_EXT,
                                                     attributes.get_sink_token(),
                                                     tensorDescs,
                                                     "SDPA fprop SINK_TOKEN"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_Q_EXT,
                                                     attributes.get_descale_q(),
                                                     tensorDescs,
                                                     "SDPA fprop DESCALE_Q"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_K_EXT,
                                                     attributes.get_descale_k(),
                                                     tensorDescs,
                                                     "SDPA fprop DESCALE_K"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_V_EXT,
                                                     attributes.get_descale_v(),
                                                     tensorDescs,
                                                     "SDPA fprop DESCALE_V"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_DESCALE_S_EXT,
                                                     attributes.get_descale_s(),
                                                     tensorDescs,
                                                     "SDPA fprop DESCALE_S"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_S_EXT,
                                                     attributes.get_scale_s(),
                                                     tensorDescs,
                                                     "SDPA fprop SCALE_S"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SCALE_O_EXT,
                                                     attributes.get_scale_o(),
                                                     tensorDescs,
                                                     "SDPA fprop SCALE_O"));

    // Optional output tensors
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_STATSDESC,
                                                     attributes.get_stats(),
                                                     tensorDescs,
                                                     "SDPA fprop STATS"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_MAX_EXT,
                                                     attributes.get_max(),
                                                     tensorDescs,
                                                     "SDPA fprop MAX"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_SUM_EXP_EXT,
                                                     attributes.get_sum_exp(),
                                                     tensorDescs,
                                                     "SDPA fprop SUM_EXP"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_RNG_DUMP_EXT,
                                                     attributes.get_rng_dump(),
                                                     tensorDescs,
                                                     "SDPA fprop RNG_DUMP"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_S_EXT,
                                                     attributes.get_amax_s(),
                                                     tensorDescs,
                                                     "SDPA fprop AMAX_S"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_FWD_AMAX_O_EXT,
                                                     attributes.get_amax_o(),
                                                     tensorDescs,
                                                     "SDPA fprop AMAX_O"));

    // Set boolean scalar parameters
    if(attributes.generate_stats.has_value())
    {
        const bool val = attributes.generate_stats.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_FWD_GENERATE_STATS_EXT,
                                                   HIPDNN_TYPE_BOOLEAN,
                                                   val,
                                                   "SDPA fprop generate_stats"));
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_FWD_ALIBI_MASK_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.alibi_mask,
                                               "SDPA fprop alibi_mask"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_FWD_PADDING_MASK_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.padding_mask,
                                               "SDPA fprop padding_mask"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_FWD_CAUSAL_MASK_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.causal_mask,
                                               "SDPA fprop causal_mask"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_FWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.causal_mask_bottom_right,
                                               "SDPA fprop causal_mask_bottom_right"));

    // Set optional float scalar parameters
    if(attributes.dropout_probability.has_value())
    {
        const float val = attributes.dropout_probability.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_FWD_DROPOUT_PROBABILITY_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   val,
                                                   "SDPA fprop dropout_probability"));
    }
    if(attributes.attn_scale_value.has_value())
    {
        const float val = attributes.attn_scale_value.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_FWD_ATTN_SCALE_VALUE_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   val,
                                                   "SDPA fprop attn_scale_value"));
    }

    // Set optional int64 scalar parameters
    if(attributes.left_bound.has_value())
    {
        const int64_t val = attributes.left_bound.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_FWD_LEFT_BOUND_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   val,
                                                   "SDPA fprop left_bound"));
    }
    if(attributes.right_bound.has_value())
    {
        const int64_t val = attributes.right_bound.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_FWD_RIGHT_BOUND_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   val,
                                                   "SDPA fprop right_bound"));
    }
    if(attributes.max_seq_len_kv.has_value())
    {
        auto val = attributes.max_seq_len_kv.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_FWD_MAX_SEQ_LEN_KV_EXT,
                                                   HIPDNN_TYPE_INT32,
                                                   val,
                                                   "SDPA fprop max_seq_len_kv"));
    }

    // Set enum parameters using dedicated backend enum types
    auto diagonalAlignment = toBackendDiagonalAlignment(attributes.diagonal_alignment);
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_FWD_DIAGONAL_ALIGNMENT_EXT,
                                               HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                               diagonalAlignment,
                                               "SDPA fprop diagonal_alignment"));

    if(attributes.mma_core_mode != DataType::NOT_SET)
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                     HIPDNN_ATTR_SDPA_FWD_MMA_CORE_MODE_EXT,
                                                     attributes.mma_core_mode,
                                                     "SDPA fprop mma_core_mode"));
    }

    auto implementationVal = toBackendAttentionImplementation(attributes.implementation);
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_FWD_IMPLEMENTATION_EXT,
                                               HIPDNN_TYPE_ATTENTION_IMPLEMENTATION_EXT,
                                               implementationVal,
                                               "SDPA fprop implementation"));

    // Set compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_SDPA_FWD_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "SDPA fprop compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "SDPA forward operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "SDPA fprop operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
