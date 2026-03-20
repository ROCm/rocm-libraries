// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackPointwiseOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::PointwiseAttributes& attributes)
{
    // Unpack input_0 tensor
    std::shared_ptr<graph::TensorAttributes> input0Tensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_POINTWISE_IN_0_EXT,
                                               tensorMap,
                                               input0Tensor,
                                               "pointwise IN_0 tensor"));
    attributes.set_input_0(input0Tensor);

    // Unpack output_0 tensor
    std::shared_ptr<graph::TensorAttributes> output0Tensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_POINTWISE_OUT_0_EXT,
                                               tensorMap,
                                               output0Tensor,
                                               "pointwise OUT_0 tensor"));
    attributes.set_output_0(output0Tensor);

    // Unpack input_1 tensor (optional)
    std::shared_ptr<graph::TensorAttributes> input1Tensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_POINTWISE_IN_1_EXT,
                                            tensorMap,
                                            input1Tensor,
                                            "pointwise IN_1 tensor"));
    if(input1Tensor)
    {
        attributes.set_input_1(input1Tensor);
    }

    // Unpack input_2 tensor (optional)
    std::shared_ptr<graph::TensorAttributes> input2Tensor;
    HIPDNN_CHECK_ERROR(unpackOptionalTensor(opDesc,
                                            HIPDNN_ATTR_OPERATION_POINTWISE_IN_2_EXT,
                                            tensorMap,
                                            input2Tensor,
                                            "pointwise IN_2 tensor"));
    if(input2Tensor)
    {
        attributes.set_input_2(input2Tensor);
    }

    // Unpack operation
    hipdnnPointwiseMode_t operation{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_POINTWISE_MODE,
                                               HIPDNN_TYPE_POINTWISE_MODE,
                                               operation,
                                               "pointwise operation"));
    auto [operationResult, operationErr] = fromHipdnnPointwiseMode(operation);
    if(operationErr.is_bad())
    {
        return operationErr;
    }
    attributes.set_mode(operationResult);

    // Unpack relu_lower_clip (optional)
    {
        std::optional<float> reluLowerClip;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP,
                                                           HIPDNN_TYPE_FLOAT,
                                                           reluLowerClip,
                                                           "pointwise relu_lower_clip"));
        if(reluLowerClip.has_value())
        {
            attributes.set_relu_lower_clip(*reluLowerClip);
        }
    }

    // Unpack relu_upper_clip (optional)
    {
        std::optional<float> reluUpperClip;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_POINTWISE_RELU_UPPER_CLIP,
                                                           HIPDNN_TYPE_FLOAT,
                                                           reluUpperClip,
                                                           "pointwise relu_upper_clip"));
        if(reluUpperClip.has_value())
        {
            attributes.set_relu_upper_clip(*reluUpperClip);
        }
    }

    // Unpack relu_lower_clip_slope (optional)
    {
        std::optional<float> reluLowerClipSlope;
        HIPDNN_CHECK_ERROR(
            getDescriptorAttrOptionalScalar(opDesc,
                                            HIPDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE,
                                            HIPDNN_TYPE_FLOAT,
                                            reluLowerClipSlope,
                                            "pointwise relu_lower_clip_slope"));
        if(reluLowerClipSlope.has_value())
        {
            attributes.set_relu_lower_clip_slope(*reluLowerClipSlope);
        }
    }

    // Unpack swish_beta (optional)
    {
        std::optional<float> swishBeta;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_POINTWISE_SWISH_BETA,
                                                           HIPDNN_TYPE_FLOAT,
                                                           swishBeta,
                                                           "pointwise swish_beta"));
        if(swishBeta.has_value())
        {
            attributes.set_swish_beta(*swishBeta);
        }
    }

    // Unpack elu_alpha (optional)
    {
        std::optional<float> eluAlpha;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_POINTWISE_ELU_ALPHA,
                                                           HIPDNN_TYPE_FLOAT,
                                                           eluAlpha,
                                                           "pointwise elu_alpha"));
        if(eluAlpha.has_value())
        {
            attributes.set_elu_alpha(*eluAlpha);
        }
    }

    // Unpack softplus_beta (optional)
    {
        std::optional<float> softplusBeta;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_POINTWISE_SOFTPLUS_BETA,
                                                           HIPDNN_TYPE_FLOAT,
                                                           softplusBeta,
                                                           "pointwise softplus_beta"));
        if(softplusBeta.has_value())
        {
            attributes.set_softplus_beta(*softplusBeta);
        }
    }

    // Unpack axis (optional)
    {
        std::optional<int64_t> axis;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(
            opDesc, HIPDNN_ATTR_POINTWISE_AXIS, HIPDNN_TYPE_INT64, axis, "pointwise axis"));
        if(axis.has_value())
        {
            attributes.set_axis(*axis);
        }
    }

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_POINTWISE_MATH_PREC, "pointwise compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
