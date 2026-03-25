// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackConvWgradOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::ConvWgradAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_X,
                                               tensorMap,
                                               xTensor,
                                               "convolutionwrw X tensor"));
    attributes.set_x(xTensor);

    // Unpack dy tensor
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DY,
                                               tensorMap,
                                               dyTensor,
                                               "convolutionwrw DY tensor"));
    attributes.set_dy(dyTensor);

    // Unpack dw tensor
    std::shared_ptr<graph::TensorAttributes> dwTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_FILTER_DW,
                                               tensorMap,
                                               dwTensor,
                                               "convolutionwrw DW tensor"));
    attributes.set_dw(dwTensor);

    // Unpack pre_padding
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, prePadding, "convolutionwrw pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    // Unpack post_padding
    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, postPadding, "convolutionwrw post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    // Unpack stride
    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, stride, "convolutionwrw stride"));
    attributes.set_stride(std::move(stride));

    // Unpack dilation
    std::vector<int64_t> dilation;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_DILATIONS, dilation, "convolutionwrw dilation"));
    attributes.set_dilation(std::move(dilation));

    // Unpack conv_mode
    hipdnnConvolutionMode_t convMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               convMode,
                                               "convolutionwrw conv_mode"));
    auto [convModeResult, convModeErr] = fromHipdnnConvMode(convMode);
    if(convModeErr.is_bad())
    {
        return convModeErr;
    }
    attributes.set_convolution_mode(convModeResult);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, "convolutionwrw compute data type");
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
