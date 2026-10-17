// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnGateDomain.h"
#include <hipdnn_frontend/attributes/KdaAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a KDA operation descriptor from KdaAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createKdaOperation(const graph::KdaAttributes& attributes,
                       std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                       std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_KDA_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create KDA operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_Q_DESC_EXT,
                                             attributes.get_q(),
                                             tensorDescs,
                                             "KDA Q_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_K_DESC_EXT,
                                             attributes.get_k(),
                                             tensorDescs,
                                             "KDA K_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_V_DESC_EXT,
                                             attributes.get_v(),
                                             tensorDescs,
                                             "KDA V_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_G_DESC_EXT,
                                             attributes.get_g(),
                                             tensorDescs,
                                             "KDA G_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_BETA_DESC_EXT,
                                             attributes.get_beta(),
                                             tensorDescs,
                                             "KDA BETA_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_CU_SEQLENS_DESC_EXT,
                                             attributes.get_cu_seqlens(),
                                             tensorDescs,
                                             "KDA CU_SEQLENS_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_KDA_INITIAL_STATE_DESC_EXT,
                                      attributes.get_initial_state(),
                                      tensorDescs,
                                      "KDA INITIAL_STATE_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_KDA_A_LOG_DESC_EXT,
                                                     attributes.get_a_log(),
                                                     tensorDescs,
                                                     "KDA A_LOG_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_KDA_DT_BIAS_DESC_EXT,
                                                     attributes.get_dt_bias(),
                                                     tensorDescs,
                                                     "KDA DT_BIAS_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_KDA_STATE_INDICES_DESC_EXT,
                                      attributes.get_state_indices(),
                                      tensorDescs,
                                      "KDA STATE_INDICES_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_KDA_O_DESC_EXT,
                                             attributes.get_o(),
                                             tensorDescs,
                                             "KDA O_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_KDA_FINAL_STATE_DESC_EXT,
                                                     attributes.get_final_state(),
                                                     tensorDescs,
                                                     "KDA FINAL_STATE_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_KDA_STATE_CHECKPOINTS_DESC_EXT,
                                      attributes.get_state_checkpoints(),
                                      tensorDescs,
                                      "KDA STATE_CHECKPOINTS_DESC_EXT"));

    // Set KDA parameters
    if(attributes.get_scale().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_KDA_SCALE_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   *attributes.get_scale(),
                                                   "KDA scale"));
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_OUTPUT_FINAL_STATE_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_output_final_state(),
                                               "KDA output_final_state"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_USE_QK_L2NORM_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_use_qk_l2norm(),
                                               "KDA use_qk_l2norm"));
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
                                HIPDNN_ATTR_OPERATION_KDA_CHECKPOINT_EVERY_N_TOKENS_EXT,
                                HIPDNN_TYPE_INT64,
                                attributes.get_checkpoint_every_n_tokens(),
                                "KDA checkpoint_every_n_tokens"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_USE_BETA_SIGMOID_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_use_beta_sigmoid(),
                                               "KDA use_beta_sigmoid"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_ALLOW_NEG_EIGVAL_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_allow_neg_eigval(),
                                               "KDA allow_neg_eigval"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_SAFE_GATE_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_safe_gate(),
                                               "KDA safe_gate"));

    // Set KDA mode
    auto gateDomain = hipdnn_frontend::toBackendGateDomain(attributes.get_gate_domain());
    if(!gateDomain.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported gate domain"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_GATE_DOMAIN_EXT,
                                               HIPDNN_TYPE_GATE_DOMAIN_EXT,
                                               *gateDomain,
                                               "KDA mode"));
    if(attributes.get_gate_lower_bound().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_KDA_GATE_LOWER_BOUND_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   *attributes.get_gate_lower_bound(),
                                                   "KDA gate_lower_bound"));
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_KDA_BATCH_INVARIANT_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_batch_invariant(),
                                               "KDA batch_invariant"));
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
                                HIPDNN_ATTR_OPERATION_KDA_OVERWRITE_INITIAL_STATE_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                attributes.get_overwrite_initial_state(),
                                "KDA overwrite_initial_state"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_KDA_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "KDA compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "KDA operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
