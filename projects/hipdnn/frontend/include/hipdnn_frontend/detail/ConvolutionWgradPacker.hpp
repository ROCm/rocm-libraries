// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a convolutionwrw operation descriptor from ConvWgradAttributes.
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
                "Failed to create convolutionwrw operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_X,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "convolutionwrw X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DY,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "convolutionwrw DY"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DW,
                                             attributes.get_dw(),
                                             tensorDescs,
                                             "convolutionwrw DW"));

    // Set convolutionwrw parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            "convolutionwrw pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            "convolutionwrw post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            "convolutionwrw stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                            HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_dilation(),
                                            "convolutionwrw dilation"));

    // Set convolutionwrw mode and compute data type
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
                                               "convolutionwrw mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "convolutionwrw compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "convolutionwrw operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
