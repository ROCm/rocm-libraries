// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a conv fprop operation descriptor from ConvFpropAttributes.
// Tensor descriptors are created/deduplicated via createOrFindTensorDesc.
inline Error createConvFpropOperation(
    const graph::ConvFpropAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create conv fprop operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "conv fprop X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                             attributes.get_w(),
                                             tensorDescs,
                                             "conv fprop W"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "conv fprop Y"));

    // Set conv fprop parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "conv fprop pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "conv fprop post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "conv fprop stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_dilation(),
                                            "conv fprop dilation"));

    // Set conv fprop mode and compute data type
    auto convMode = hipdnn_frontend::toBackendConvMode(attributes.get_convolution_mode());
    if(!convMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE,
                std::string("Unsupported conv fprop mode: ")
                    + to_string(attributes.get_convolution_mode())};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               *convMode,
                                               "conv fprop mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "conv fprop compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "conv fprop operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
