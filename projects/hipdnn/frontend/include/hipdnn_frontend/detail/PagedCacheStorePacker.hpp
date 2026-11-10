// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/PagedCacheStoreAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a paged cache store operation descriptor from PagedCacheStoreAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createPagedCacheStoreOperation(
    const graph::PagedCacheStoreAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_PAGED_CACHE_STORE_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create paged cache store operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_XDESC_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "Paged cache store XDESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SLOT_MAPPING_DESC_EXT,
                              attributes.get_slot_mapping(),
                              tensorDescs,
                              "Paged cache store SLOT_MAPPING_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_CONTAINER_DESC_EXT,
                              attributes.get_container(),
                              tensorDescs,
                              "Paged cache store CONTAINER_DESC_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_PAGED_CACHE_STORE_SCALE_DESC_EXT,
                                      attributes.get_scale(),
                                      tensorDescs,
                                      "Paged cache store SCALE_DESC_EXT"));

    // Set paged cache store parameters

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "paged cache store operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
