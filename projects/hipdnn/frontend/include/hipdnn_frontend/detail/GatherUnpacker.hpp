// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/GatherAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackGatherOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::GatherAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_GATHER_XDESC_EXT,
                                               tensorMap,
                                               xTensor,
                                               "gather XDESC_EXT tensor"));
    attributes.set_x(xTensor);

    // Unpack indices tensor
    std::shared_ptr<graph::TensorAttributes> indicesTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_GATHER_INDICES_DESC_EXT,
                                               tensorMap,
                                               indicesTensor,
                                               "gather INDICES_DESC_EXT tensor"));
    attributes.set_indices(indicesTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_GATHER_YDESC_EXT,
                                               tensorMap,
                                               yTensor,
                                               "gather YDESC_EXT tensor"));
    attributes.set_y(yTensor);

    // Unpack axis
    int64_t axis = 0;
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(
        opDesc, HIPDNN_ATTR_OPERATION_GATHER_AXIS_EXT, HIPDNN_TYPE_INT64, axis, "gather axis"));
    attributes.set_axis(axis);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
