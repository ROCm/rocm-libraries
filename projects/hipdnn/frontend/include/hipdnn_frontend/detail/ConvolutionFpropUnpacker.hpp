// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/detail/ConvolutionPackerHelpers.hpp>
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
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                               tensorMap,
                                               xTensor,
                                               "conv fprop X tensor"));
    attributes.set_x(xTensor);

    // Unpack W (weights) tensor
    std::shared_ptr<graph::TensorAttributes> wTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                               tensorMap,
                                               wTensor,
                                               "conv fprop W tensor"));
    attributes.set_w(wTensor);

    // Unpack Y (output) tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                               tensorMap,
                                               yTensor,
                                               "conv fprop Y tensor"));
    attributes.set_y(yTensor);

    // Unpack shared convolution parameters
    HIPDNN_CHECK_ERROR(unpackConvolutionParams(opDesc, attributes, "conv fprop"));

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
