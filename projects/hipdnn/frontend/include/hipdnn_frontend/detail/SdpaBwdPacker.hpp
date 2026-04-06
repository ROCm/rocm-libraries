// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

#include "HipdnnDiagonalAlignment.h"

namespace hipdnn_frontend::detail
{

// Builds an SDPA bprop operation descriptor from SdpaBackwardAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createSdpaBwdOperation(const graph::SdpaBackwardAttributes& attributes,
                           std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                           std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_SDPA_BWD_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create SDPA bprop operation descriptor"};
    }

    // Required input tensors
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_Q_EXT,
                                             attributes.get_q(),
                                             tensorDescs,
                                             "SDPA bprop Q"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_K_EXT,
                                             attributes.get_k(),
                                             tensorDescs,
                                             "SDPA bprop K"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_V_EXT,
                                             attributes.get_v(),
                                             tensorDescs,
                                             "SDPA bprop V"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_O_EXT,
                                             attributes.get_o(),
                                             tensorDescs,
                                             "SDPA bprop O"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_DO_EXT,
                                             attributes.get_do(),
                                             tensorDescs,
                                             "SDPA bprop dO"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_STATS_EXT,
                                             attributes.get_stats(),
                                             tensorDescs,
                                             "SDPA bprop Stats"));

    // Required output tensors
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_DQ_EXT,
                                             attributes.get_dq(),
                                             tensorDescs,
                                             "SDPA bprop dQ"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_DK_EXT,
                                             attributes.get_dk(),
                                             tensorDescs,
                                             "SDPA bprop dK"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SDPA_BWD_DV_EXT,
                                             attributes.get_dv(),
                                             tensorDescs,
                                             "SDPA bprop dV"));

    // Optional input tensors
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_SCALE_EXT,
                                                     attributes.get_attn_scale(),
                                                     tensorDescs,
                                                     "SDPA bprop SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_ATTN_MASK_EXT,
                                                     attributes.get_bias(),
                                                     tensorDescs,
                                                     "SDPA bprop ATTN_MASK"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_Q_EXT,
                                                     attributes.get_seq_len_q(),
                                                     tensorDescs,
                                                     "SDPA bprop SEQ_LEN_Q"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_SEQ_LEN_KV_EXT,
                                                     attributes.get_seq_len_kv(),
                                                     tensorDescs,
                                                     "SDPA bprop SEQ_LEN_KV"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_SEED_EXT,
                                                     attributes.get_seed(),
                                                     tensorDescs,
                                                     "SDPA bprop SEED"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_OFFSET_EXT,
                                                     attributes.get_offset(),
                                                     tensorDescs,
                                                     "SDPA bprop OFFSET"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_MASK_EXT,
                                      attributes.get_dropout_mask(),
                                      tensorDescs,
                                      "SDPA bprop DROPOUT_MASK"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_EXT,
                                      attributes.get_dropout_scale(),
                                      tensorDescs,
                                      "SDPA bprop DROPOUT_SCALE"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_SDPA_BWD_DROPOUT_SCALE_INV_EXT,
                                      attributes.get_dropout_scale_inv(),
                                      tensorDescs,
                                      "SDPA bprop DROPOUT_SCALE_INV"));

    // Optional output tensors
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SDPA_BWD_DBIAS_EXT,
                                                     attributes.get_dbias(),
                                                     tensorDescs,
                                                     "SDPA bprop DBIAS"));

    // Set boolean scalar parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_BWD_ALIBI_MASK_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.alibi_mask,
                                               "SDPA bprop alibi_mask"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_BWD_PADDING_MASK_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.padding_mask,
                                               "SDPA bprop padding_mask"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.causal_mask,
                                               "SDPA bprop causal_mask"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_BWD_CAUSAL_MASK_BOTTOM_RIGHT_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.causal_mask_bottom_right,
                                               "SDPA bprop causal_mask_bottom_right"));

    // Set optional float scalar parameters
    if(attributes.dropout_probability.has_value())
    {
        const float val = attributes.dropout_probability.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_BWD_DROPOUT_PROBABILITY_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   val,
                                                   "SDPA bprop dropout_probability"));
    }
    if(attributes.attn_scale_value.has_value())
    {
        const float val = attributes.attn_scale_value.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_BWD_ATTN_SCALE_VALUE_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   val,
                                                   "SDPA bprop attn_scale_value"));
    }

    // Set optional int64 scalar parameters
    if(attributes.left_bound.has_value())
    {
        const int64_t val = attributes.left_bound.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_BWD_LEFT_BOUND_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   val,
                                                   "SDPA bprop left_bound"));
    }
    if(attributes.right_bound.has_value())
    {
        const int64_t val = attributes.right_bound.value();
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_SDPA_BWD_RIGHT_BOUND_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   val,
                                                   "SDPA bprop right_bound"));
    }

    // Set enum parameters using dedicated backend enum types
    auto diagonalAlignment = toBackendDiagonalAlignment(attributes.diagonal_alignment);
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_SDPA_BWD_DIAGONAL_ALIGNMENT_EXT,
                                               HIPDNN_TYPE_DIAGONAL_ALIGNMENT_EXT,
                                               diagonalAlignment,
                                               "SDPA bprop diagonal_alignment"));

    // Set compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_SDPA_BWD_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "SDPA bprop compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "SDPA bprop operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
