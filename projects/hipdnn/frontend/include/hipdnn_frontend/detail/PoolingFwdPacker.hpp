// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/PoolingFwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a pooling forward operation descriptor from PoolingFwdAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createPoolingFwdOperation(
    const graph::PoolingFwdAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_POOLING_FORWARD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create pooling forward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_POOLING_FORWARD_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "pooling X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_POOLING_FORWARD_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "pooling Y"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_POOLING_FORWARD_INDEX_EXT,
                                      attributes.get_index(),
                                      tensorDescs,
                                      "pooling INDEX"));

    // Set pooling parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_PRE_PADDINGS_EXT,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "pooling pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_POST_PADDINGS_EXT,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "pooling post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_STRIDES_EXT,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "pooling stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_POOLING_WINDOW_EXT,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_window(),
                                            "pooling window"));

    // Set pooling mode
    auto poolingMode = hipdnn_frontend::toBackendPoolingMode(attributes.get_pooling_mode());
    if(!poolingMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported pooling mode"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_POOLING_MODE_EXT,
                                               HIPDNN_TYPE_POOLING_MODE,
                                               *poolingMode,
                                               "pooling mode"));

    // Set pooling mode
    auto paddingMode = hipdnn_frontend::toBackendPaddingMode(attributes.get_padding_mode());
    if(!paddingMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported padding mode"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_POOLING_PADDING_MODE_EXT,
                                               HIPDNN_TYPE_PADDING_MODE,
                                               *paddingMode,
                                               "pooling mode"));
    if(attributes.get_generate_index().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_POOLING_GENERATE_INDEX_EXT,
                                                   HIPDNN_TYPE_BOOLEAN,
                                                   *attributes.get_generate_index(),
                                                   "pooling generate_index"));
    }

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
