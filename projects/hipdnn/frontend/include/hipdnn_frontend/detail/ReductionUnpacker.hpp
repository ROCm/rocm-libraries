// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

[[nodiscard]] inline Error unpackReductionOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::ReductionAttributes& attributes)
{
    // Unpack x tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                               tensorMap,
                                               xTensor,
                                               "reduction XDESC tensor"));
    attributes.set_x(xTensor);

    // Unpack y tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                               tensorMap,
                                               yTensor,
                                               "reduction YDESC tensor"));
    attributes.set_y(yTensor);

    // Unpack mode
    hipdnnReduceTensorOp_t mode{};
    HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(opDesc,
                                               HIPDNN_ATTR_REDUCTION_OPERATOR,
                                               HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                               mode,
                                               "reduction mode"));
    auto [modeResult, modeErr] = fromHipdnnReduceTensorOp(mode);
    if(modeErr.is_bad())
    {
        return modeErr;
    }
    attributes.set_mode(modeResult);

    // Unpack is_deterministic (optional)
    {
        std::optional<bool> isDeterministic;
        HIPDNN_CHECK_ERROR(getDescriptorAttrOptionalScalar(opDesc,
                                                           HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                                                           HIPDNN_TYPE_BOOLEAN,
                                                           isDeterministic,
                                                           "reduction is_deterministic"));
        if(isDeterministic.has_value())
        {
            attributes.set_is_deterministic(*isDeterministic);
        }
    }

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_REDUCTION_COMP_TYPE, "reduction compute data type");
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
