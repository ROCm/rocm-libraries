// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/detail/ConvolutionPackerHelpers.hpp>
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
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X,
                                               tensorMap,
                                               xTensor,
                                               "conv wgrad X tensor"));
    attributes.set_x(xTensor);

    // Unpack dy tensor
    std::shared_ptr<graph::TensorAttributes> dyTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY,
                                               tensorMap,
                                               dyTensor,
                                               "conv wgrad DY tensor"));
    attributes.set_dy(dyTensor);

    // Unpack dw tensor
    std::shared_ptr<graph::TensorAttributes> dwTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW,
                                               tensorMap,
                                               dwTensor,
                                               "conv wgrad DW tensor"));
    attributes.set_dw(dwTensor);

    // Unpack shared convolution parameters
    HIPDNN_CHECK_ERROR(unpackConvolutionParams(opDesc, attributes, "conv wgrad"));

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
