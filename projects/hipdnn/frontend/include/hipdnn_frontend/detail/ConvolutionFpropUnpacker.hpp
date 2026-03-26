// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks a convolution forward operation from a backend operation descriptor.
/// Populates the ConvFpropAttributes with tensors (using tensorMap for sharing)
/// and convolution parameters.
[[nodiscard]] inline Error unpackConvFpropOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::ConvFpropAttributes& attributes)
{
    // Unpack X (input) tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, tensorMap, xTensor, "conv X tensor"));
    attributes.set_x(xTensor);

    // Unpack W (weights) tensor
    std::shared_ptr<graph::TensorAttributes> wTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, tensorMap, wTensor, "conv W tensor"));
    attributes.set_w(wTensor);

    // Unpack Y (output) tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(
        opDesc, HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, tensorMap, yTensor, "conv Y tensor"));
    attributes.set_y(yTensor);

    // Unpack convolution parameters
    std::vector<int64_t> prePadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_PRE_PADDINGS, prePadding, "conv pre_padding"));
    attributes.set_pre_padding(std::move(prePadding));

    std::vector<int64_t> postPadding;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_POST_PADDINGS, postPadding, "conv post_padding"));
    attributes.set_post_padding(std::move(postPadding));

    std::vector<int64_t> stride;
    HIPDNN_CHECK_ERROR(getDescriptorAttrVec(
        opDesc, HIPDNN_ATTR_CONVOLUTION_FILTER_STRIDES, stride, "conv stride"));
    attributes.set_stride(std::move(stride));

    std::vector<int64_t> dilation;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrVec(opDesc, HIPDNN_ATTR_CONVOLUTION_DILATIONS, dilation, "conv dilation"));
    attributes.set_dilation(std::move(dilation));

    // Unpack convolution mode
    hipdnnConvolutionMode_t convMode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_CONVOLUTION_CONV_MODE,
                                               HIPDNN_TYPE_CONVOLUTION_MODE,
                                               convMode,
                                               "conv mode"));
    auto [mode, modeErr] = fromHipdnnConvMode(convMode);
    if(modeErr.is_bad())
    {
        return modeErr;
    }
    attributes.set_convolution_mode(mode);

    // Unpack compute data type
    auto [dt, dtErr]
        = unpackGraphDataType(opDesc, HIPDNN_ATTR_CONVOLUTION_COMP_TYPE, "conv compute data type");
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
