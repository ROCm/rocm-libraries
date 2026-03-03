// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a convbprop operation descriptor from ConvDgradAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createConvDgradOperation(
    const graph::ConvDgradAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create convbprop operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DY,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "convbprop DY"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_W,
                                             attributes.get_w(),
                                             tensorDescs,
                                             "convbprop W"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DX,
                                             attributes.get_dx(),
                                             tensorDescs,
                                             "convbprop DX"));

    // Set convbprop parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "convbprop pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "convbprop post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "convbprop stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_dilation(),
                                            "convbprop dilation"));

    // Set convbprop mode
    auto convMode = hipdnn_frontend::toBackendConvMode(attributes.get_convolution_mode());
    if(!convMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE,
                std::string("Unsupported convolution mode: ")
                    + to_string(attributes.get_convolution_mode())};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               *convMode,
                                               "convbprop mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "convbprop compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "convbprop operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
