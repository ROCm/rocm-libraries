// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks a custom op operation from a backend operation descriptor.
/// Populates the CustomOpAttributes with tensor arrays (using tensorMap for sharing),
/// custom_op_id, opaque data payload, compute data type, and operation name.
[[nodiscard]] inline Error unpackCustomOpOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::CustomOpAttributes& attributes)
{
    // Unpack input tensor array
    std::vector<std::shared_ptr<graph::TensorAttributes>> inputTensors;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensorArray(opDesc,
                                                    HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                                    tensorMap,
                                                    inputTensors,
                                                    "custom op input tensors"));
    attributes.set_inputs(std::move(inputTensors));

    // Unpack output tensor array
    std::vector<std::shared_ptr<graph::TensorAttributes>> outputTensors;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensorArray(opDesc,
                                                    HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                                                    tensorMap,
                                                    outputTensors,
                                                    "custom op output tensors"));
    attributes.set_outputs(std::move(outputTensors));

    // Unpack custom_op_id string
    std::string customOpId;
    HIPDNN_CHECK_ERROR(getDescriptorAttrString(
        opDesc, HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT, customOpId, "custom op id"));
    attributes.set_custom_op_id(customOpId);

    // Unpack opaque data payload
    std::vector<uint8_t> opaqueData;
    HIPDNN_CHECK_ERROR(getDescriptorAttrByteArray(
        opDesc, HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT, opaqueData, "custom op data"));
    attributes.set_data(std::move(opaqueData));

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(
        opDesc, HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT, "custom op compute data type");
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
