// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnGateDomain.h"
#include <hipdnn_frontend/attributes/GdnAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a GDN operation descriptor from GdnAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createGdnOperation(const graph::GdnAttributes& attributes,
                       std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                       std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_GDN_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create GDN operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_Q_DESC_EXT,
                                             attributes.get_q(),
                                             tensorDescs,
                                             "GDN Q_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_K_DESC_EXT,
                                             attributes.get_k(),
                                             tensorDescs,
                                             "GDN K_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_V_DESC_EXT,
                                             attributes.get_v(),
                                             tensorDescs,
                                             "GDN V_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_G_DESC_EXT,
                                             attributes.get_g(),
                                             tensorDescs,
                                             "GDN G_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_BETA_DESC_EXT,
                                             attributes.get_beta(),
                                             tensorDescs,
                                             "GDN BETA_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_CU_SEQLENS_DESC_EXT,
                                             attributes.get_cu_seqlens(),
                                             tensorDescs,
                                             "GDN CU_SEQLENS_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_GDN_INITIAL_STATE_DESC_EXT,
                                      attributes.get_initial_state(),
                                      tensorDescs,
                                      "GDN INITIAL_STATE_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_GDN_A_LOG_DESC_EXT,
                                                     attributes.get_a_log(),
                                                     tensorDescs,
                                                     "GDN A_LOG_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_GDN_DT_BIAS_DESC_EXT,
                                                     attributes.get_dt_bias(),
                                                     tensorDescs,
                                                     "GDN DT_BIAS_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_GDN_STATE_INDICES_DESC_EXT,
                                      attributes.get_state_indices(),
                                      tensorDescs,
                                      "GDN STATE_INDICES_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GDN_O_DESC_EXT,
                                             attributes.get_o(),
                                             tensorDescs,
                                             "GDN O_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_GDN_FINAL_STATE_DESC_EXT,
                                                     attributes.get_final_state(),
                                                     tensorDescs,
                                                     "GDN FINAL_STATE_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_GDN_STATE_CHECKPOINTS_DESC_EXT,
                                      attributes.get_state_checkpoints(),
                                      tensorDescs,
                                      "GDN STATE_CHECKPOINTS_DESC_EXT"));

    // Set GDN parameters
    if(attributes.get_scale().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_GDN_SCALE_EXT,
                                                   HIPDNN_TYPE_FLOAT,
                                                   *attributes.get_scale(),
                                                   "GDN scale"));
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_OUTPUT_FINAL_STATE_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_output_final_state(),
                                               "GDN output_final_state"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_USE_QK_L2NORM_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_use_qk_l2norm(),
                                               "GDN use_qk_l2norm"));
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
                                HIPDNN_ATTR_OPERATION_GDN_CHECKPOINT_EVERY_N_TOKENS_EXT,
                                HIPDNN_TYPE_INT64,
                                attributes.get_checkpoint_every_n_tokens(),
                                "GDN checkpoint_every_n_tokens"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_USE_BETA_SIGMOID_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_use_beta_sigmoid(),
                                               "GDN use_beta_sigmoid"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_ALLOW_NEG_EIGVAL_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_allow_neg_eigval(),
                                               "GDN allow_neg_eigval"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_SAFE_GATE_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_safe_gate(),
                                               "GDN safe_gate"));

    // Set GDN mode
    auto gateDomain = hipdnn_frontend::toBackendGateDomain(attributes.get_gate_domain());
    if(!gateDomain.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported gate domain"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_GATE_DOMAIN_EXT,
                                               HIPDNN_TYPE_GATE_DOMAIN_EXT,
                                               *gateDomain,
                                               "GDN mode"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GDN_BATCH_INVARIANT_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_batch_invariant(),
                                               "GDN batch_invariant"));
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
                                HIPDNN_ATTR_OPERATION_GDN_OVERWRITE_INITIAL_STATE_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                attributes.get_overwrite_initial_state(),
                                "GDN overwrite_initial_state"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_GDN_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "GDN compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "GDN operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
