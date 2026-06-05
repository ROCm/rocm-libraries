// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a resample forward operation descriptor from ResampleFwdAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createResampleFwdOperation(
    const graph::ResampleFwdAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_RESAMPLE_FWD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create resample forward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_XDESC,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "resample X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_YDESC,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "resample Y"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_RESAMPLE_FWD_IDXDESC,
                                                     attributes.get_index(),
                                                     tensorDescs,
                                                     "resample INDEX"));

    // Set resample parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_RESAMPLE_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "resample pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_RESAMPLE_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "resample post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_RESAMPLE_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "resample stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_RESAMPLE_WINDOW_DIMS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_window(),
                                            "resample window"));

    // Set resample mode
    auto resampleMode = hipdnn_frontend::toBackendResampleMode(attributes.get_resample_mode());
    if(!resampleMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported resample mode"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_RESAMPLE_MODE,
                                               HIPDNN_TYPE_RESAMPLE_MODE,
                                               *resampleMode,
                                               "resample mode"));

    // Set padding mode (optional)
    if(attributes.get_padding_mode() != PaddingMode::NOT_SET)
    {
        auto paddingMode = hipdnn_frontend::toBackendPaddingMode(attributes.get_padding_mode());
        if(!paddingMode.has_value())
        {
            return {ErrorCode::INVALID_VALUE, "Unsupported padding mode"};
        }
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_RESAMPLE_PADDING_MODE,
                                                   HIPDNN_TYPE_PADDING_MODE,
                                                   *paddingMode,
                                                   "padding mode"));
    }
    if(attributes.get_generate_index().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_RESAMPLE_GENERATE_INDEX_EXT,
                                                   HIPDNN_TYPE_BOOLEAN,
                                                   *attributes.get_generate_index(),
                                                   "resample generate_index"));
    }

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_RESAMPLE_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "resample compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "resample operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
