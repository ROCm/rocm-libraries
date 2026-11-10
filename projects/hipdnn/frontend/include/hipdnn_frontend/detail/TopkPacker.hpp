// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/TopkAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a top-k operation descriptor from TopkAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createTopkOperation(const graph::TopkAttributes& attributes,
                        std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                        std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_TOPK_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create top-k operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_TOPK_XDESC_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "Top-k XDESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_TOPK_INDICES_DESC_EXT,
                                             attributes.get_indices(),
                                             tensorDescs,
                                             "Top-k INDICES_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_TOPK_VALUES_DESC_EXT,
                                                     attributes.get_values(),
                                                     tensorDescs,
                                                     "Top-k VALUES_DESC_EXT"));

    // Set top-k parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_TOPK_K_EXT,
                                               HIPDNN_TYPE_INT64,
                                               attributes.get_k(),
                                               "Top-k k"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_TOPK_LARGEST_EXT,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_largest(),
                                               "Top-k largest"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "top-k operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
