// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/SoftmaxAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a softmax operation descriptor from SoftmaxAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createSoftmaxOperation(const graph::SoftmaxAttributes& attributes,
                           std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                           std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_SOFTMAX_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create softmax operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SOFTMAX_XDESC,
                                             attributes.get_p(),
                                             tensorDescs,
                                             "softmax XDESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_SOFTMAX_YDESC,
                                             attributes.get_s(),
                                             tensorDescs,
                                             "softmax YDESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_DESC,
                                                     attributes.get_stats(),
                                                     tensorDescs,
                                                     "softmax STATS_DESC"));

    // Set softmax parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_SOFTMAX_STATS_LOG2,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_stats_use_log2(),
                                               "softmax stats_use_log2"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_SOFTMAX_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "softmax compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "softmax operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
