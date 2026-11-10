// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/RopeAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a RoPE operation descriptor from RopeAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createRopeOperation(const graph::RopeAttributes& attributes,
                        std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                        std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_ROPE_FWD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create RoPE operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_ROPE_FWD_XDESC,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "RoPE XDESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_ROPE_FWD_FREQSDESC,
                                             attributes.get_freqs(),
                                             tensorDescs,
                                             "RoPE FREQSDESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_ROPE_FWD_YDESC,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "RoPE YDESC"));

    // Set RoPE parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_ROPE_FWD_OUTPUT_SCALE,
                                               HIPDNN_TYPE_FLOAT,
                                               attributes.get_output_scale(),
                                               "RoPE output_scale"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_DIM,
                                               HIPDNN_TYPE_INT64,
                                               attributes.get_rope_dim(),
                                               "RoPE rope_dim"));
    if(attributes.get_rope_offset().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_ROPE_FWD_ROPE_OFFSET_EXT,
                                                   HIPDNN_TYPE_INT64,
                                                   *attributes.get_rope_offset(),
                                                   "RoPE rope_offset"));
    }

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "RoPE operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
