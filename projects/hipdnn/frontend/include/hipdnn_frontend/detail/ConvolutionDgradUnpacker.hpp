// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/detail/ConvolutionPackerHelpers.hpp>
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
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY,
                                               tensorMap,
                                               dyTensor,
                                               "conv dgrad DY tensor"));
    attributes.set_dy(dyTensor);

    // Unpack w tensor
    std::shared_ptr<graph::TensorAttributes> wTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W,
                                               tensorMap,
                                               wTensor,
                                               "conv dgrad W tensor"));
    attributes.set_w(wTensor);

    // Unpack dx tensor
    std::shared_ptr<graph::TensorAttributes> dxTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX,
                                               tensorMap,
                                               dxTensor,
                                               "conv dgrad DX tensor"));
    attributes.set_dx(dxTensor);

    // Unpack shared convolution parameters
    HIPDNN_CHECK_ERROR(unpackConvolutionParams(opDesc, attributes, "conv dgrad"));

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
