// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Packs the convolution parameters shared across all convolution operation
// variants (fprop, dgrad, wgrad): padding, stride, dilation, mode, and
// compute data type.  The ConvAttrs template parameter must provide
// get_pre_padding(), get_post_padding(), get_stride(), get_dilation(),
// get_convolution_mode(), and the public member compute_data_type.
template <typename ConvAttrs>
[[nodiscard]] inline Error packConvolutionParams(hipdnnBackendDescriptor_t opDesc,
                                                 const ConvAttrs& attributes,
                                                 const std::string& errorLabel)
{
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc,
                                            HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_pre_padding(),
                                            errorLabel + " pre_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc,
                                            HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_post_padding(),
                                            errorLabel + " post_padding"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc,
                                            HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_stride(),
                                            errorLabel + " stride"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc,
                                            HIPDNN_ATTR_CONVOLUTION_DILATIONS,
                                            HIPDNN_TYPE_INT64,
                                            attributes.get_dilation(),
                                            errorLabel + " dilation"));

    auto convMode = hipdnn_frontend::toBackendConvMode(attributes.get_convolution_mode());
    if(!convMode.has_value())
    {
        return {ErrorCode::INVALID_VALUE,
                std::string("Unsupported ") + errorLabel
                    + " mode: " + to_string(attributes.get_convolution_mode())};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               *convMode,
                                               errorLabel + " mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc,
                                                 HIPDNN_ATTR_CONVOLUTION_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 errorLabel + " compute data type"));

    return {};
}

// Unpacks the convolution parameters shared across all convolution operation
// variants.  The ConvAttrs template parameter must provide set_pre_padding(),
// set_post_padding(), set_stride(), set_dilation(), set_convolution_mode(),
// and set_compute_data_type().
template <typename ConvAttrs>
[[nodiscard]] inline Error unpackConvolutionParams(hipdnnBackendDescriptor_t opDesc,
                                                   ConvAttrs& attributes,
                                                   const std::string& errorLabel)
{
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, prePadding, errorLabel + " pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, postPadding, errorLabel + " post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, stride, errorLabel + " stride"));
    attributes.set_stride(std::move(stride));

    std::vector<int64_t> dilation;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_DILATIONS, dilation, errorLabel + " dilation"));
    attributes.set_dilation(std::move(dilation));

    hipdnnConvolutionMode_t convMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               convMode,
                                               errorLabel + " conv_mode"));
    auto [convModeResult, convModeErr] = fromHipdnnConvMode(convMode);
    if(convModeErr.is_bad())
    {
        return convModeErr;
    }
    attributes.set_convolution_mode(convModeResult);

    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, errorLabel + " compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    return {};
}

} // namespace hipdnn_frontend::detail
