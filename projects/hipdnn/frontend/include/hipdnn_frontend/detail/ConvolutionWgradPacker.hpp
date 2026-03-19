// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a conv wgrad operation descriptor from ConvWgradAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createConvWgradOperation(
    const graph::ConvWgradAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
        HIPDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create conv wgrad operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_X,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "conv wgrad X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DY,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "conv wgrad DY"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DW,
                                             attributes.get_dw(),
                                             tensorDescs,
                                             "conv wgrad DW"));

    // Set conv wgrad parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "conv wgrad pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "conv wgrad post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "conv wgrad stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_dilation(),
                                            "conv wgrad dilation"));

    // Set conv wgrad mode and compute data type
    auto convMode = hipdnn_frontend::toBackendConvMode(attributes.get_convolution_mode());
    if(!convMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE,
                std::string("Unsupported conv wgrad mode: ")
                    + to_string(attributes.get_convolution_mode())};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               *convMode,
                                               "conv wgrad mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "conv wgrad compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "conv wgrad operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
