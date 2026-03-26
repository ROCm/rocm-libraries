// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/PoolingBwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a pooling backward operation descriptor from PoolingBwdAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createPoolingBwdOperation(
    const graph::PoolingBwdAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_POOLING_BACKWARD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create pooling backward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DY,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "pooling DY"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_POOLING_BACKWARD_DX,
                                             attributes.get_dx(),
                                             tensorDescs,
                                             "pooling DX"));

    // Set pooling parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "pooling pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "pooling post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "pooling stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_WINDOW_SIZE,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_window_size(),
                                            "pooling window_size"));

    // Set pooling mode
    auto poolingMode = hipdnn_frontend::toBackendPoolingMode(attributes.get_pooling_mode());
    if(!poolingMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported pooling mode"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(
        opDesc.get(), HIPDNN_ATTR_POOLING_MODE, HIPDNN_TYPE_POOLING_MODE, *poolingMode, "pooling mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_POOLING_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "pooling compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "pooling operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
