// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/GatherAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a gather operation descriptor from GatherAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createGatherOperation(const graph::GatherAttributes& attributes,
                          std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                          std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_GATHER_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create gather operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GATHER_XDESC_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "gather XDESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GATHER_INDICES_DESC_EXT,
                                             attributes.get_indices(),
                                             tensorDescs,
                                             "gather INDICES_DESC_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_GATHER_YDESC_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "gather YDESC_EXT"));

    // Set gather parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_GATHER_AXIS_EXT,
                                               HIPDNN_TYPE_INT64,
                                               attributes.get_axis(),
                                               "gather axis"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "gather operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
