// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackConvBpropOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::ConvDgradAttributes& attributes)
{
    // Unpack dy tensor
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DY,
                                               tensorMap,
                                               dyTensor,
                                               "convolutionbwd DY tensor"));
    attributes.set_dy(dyTensor);

    // Unpack w tensor
    std::shared_ptr<graph::TensorAttributes> wTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_W,
                                               tensorMap,
                                               wTensor,
                                               "convolutionbwd W tensor"));
    attributes.set_w(wTensor);

    // Unpack dx tensor
    std::shared_ptr<graph::TensorAttributes> dxTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BACKWARD_DX,
                                               tensorMap,
                                               dxTensor,
                                               "convolutionbwd DX tensor"));
    attributes.set_dx(dxTensor);

    // Unpack pre_padding
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, prePadding, "convolutionbwd pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    // Unpack post_padding
    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, postPadding, "convolutionbwd post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    // Unpack stride
    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, stride, "convolutionbwd stride"));
    attributes.set_stride(std::move(stride));

    // Unpack dilation
    std::vector<int64_t> dilation;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_DILATIONS, dilation, "convolutionbwd dilation"));
    attributes.set_dilation(std::move(dilation));

    // Unpack conv_mode
    hipdnnConvolutionMode_t convMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               convMode,
                                               "convolutionbwd conv_mode"));
    auto [convModeResult, convModeErr] = fromHipdnnConvMode(convMode);
    if(convModeErr.is_bad())
    {
        return convModeErr;
    }
    attributes.set_convolution_mode(convModeResult);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, "convolutionbwd compute data type");
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
