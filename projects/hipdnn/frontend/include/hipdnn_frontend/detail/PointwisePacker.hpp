// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a pointwise operation descriptor from PointwiseAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createPointwiseOperation(
    const graph::PointwiseAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create pointwise operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                             attributes.get_input_0(),
                                             tensorDescs,
                                             "pointwise IN_0"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                                             attributes.get_output_0(),
                                             tensorDescs,
                                             "pointwise OUT_0"));
    if(attributes.get_input_1())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT,
                                                 attributes.get_input_1(),
                                                 tensorDescs,
                                                 "pointwise IN_1"));
    }
    if(attributes.get_input_2())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT,
                                                 attributes.get_input_2(),
                                                 tensorDescs,
                                                 "pointwise IN_2"));
    }
    if(attributes.get_axis().has_value())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_POINTWISE_AXIS,
                                                   HIPDNN_TYPE_INT64,
                                                   *attributes.get_axis(),
                                                   "pointwise AXIS"));
    }

    // Set pointwise parameters

    // Set pointwise mode
    auto operation = hipdnn_frontend::toBackendPointwiseMode(attributes.get_mode());
    if(!operation.has_value())
    {
        return {ErrorCode::INVALID_VALUE,
                "Unsupported pointwise mode: "
                    + std::to_string(static_cast<int>(attributes.get_mode()))};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_POINTWISE_MODE,
                                               HIPDNN_TYPE_POINTWISE_MODE,
                                               *operation,
                                               "pointwise mode"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrOptionalScalar(opDesc.get(),
                                                       HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.get_relu_lower_clip(),
                                                       "pointwise relu_lower_clip"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrOptionalScalar(opDesc.get(),
                                                       HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.get_relu_upper_clip(),
                                                       "pointwise relu_upper_clip"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrOptionalScalar(opDesc.get(),
                                                       HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.get_relu_lower_clip_slope(),
                                                       "pointwise relu_lower_clip_slope"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrOptionalScalar(opDesc.get(),
                                                       HIPDNN_ATTR_POINTWISE_SWISH_BETA,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.get_swish_beta(),
                                                       "pointwise swish_beta"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrOptionalScalar(opDesc.get(),
                                                       HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.get_elu_alpha(),
                                                       "pointwise elu_alpha"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrOptionalScalar(opDesc.get(),
                                                       HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                                       HIPDNN_TYPE_FLOAT,
                                                       attributes.get_softplus_beta(),
                                                       "pointwise softplus_beta"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_POINTWISE_MATH_PREC,
                                                 attributes.compute_data_type,
                                                 "pointwise compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "pointwise operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
