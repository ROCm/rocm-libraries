// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/MoeRoutingAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a MoE routing operation descriptor from MoeRoutingAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createMoeRoutingOperation(
    const graph::MoeRoutingAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_MOE_ROUTING_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create MoE routing operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOPK_IDS_DESC_EXT,
                                             attributes.get_topk_ids(),
                                             tensorDescs,
                                             "MoE routing TOPK_IDS_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_MOE_ROUTING_FIRST_TOKEN_OFFSET_DESC_EXT,
                              attributes.get_first_token_offset(),
                              tensorDescs,
                              "MoE routing FIRST_TOKEN_OFFSET_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_INDEX_DESC_EXT,
                                             attributes.get_token_index(),
                                             tensorDescs,
                                             "MoE routing TOKEN_INDEX_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_MOE_ROUTING_TOKEN_KS_DESC_EXT,
                                             attributes.get_token_ks(),
                                             tensorDescs,
                                             "MoE routing TOKEN_KS_DESC_EXT"));

    // Set MoE routing parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_MOE_ROUTING_NUM_EXPERTS_EXT,
                                               HIPDNN_TYPE_INT64,
                                               attributes.get_num_experts(),
                                               "MoE routing num_experts"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "MoE routing operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
