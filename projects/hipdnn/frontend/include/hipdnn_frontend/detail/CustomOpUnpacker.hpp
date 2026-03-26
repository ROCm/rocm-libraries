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

/// Unpacks an array of tensor descriptors from a backend operation descriptor.
/// For each tensor descriptor, finds it in the tensorMap (by UID, for sharing) or
/// creates a new one and registers it.
[[nodiscard]] inline Error unpackAndRegisterTensorArray(
    hipdnnBackendDescriptor_t opDesc,
    hipdnnBackendAttributeName_t tensorAttrName,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    std::vector<std::shared_ptr<graph::TensorAttributes>>& outTensors,
    const std::string& errorContext)
{
    auto [tensorDescs, descErr] = getDescriptorAttrDescArray(opDesc, tensorAttrName, errorContext);
    if(descErr.is_bad())
    {
        return descErr;
    }

    outTensors.clear();
    outTensors.reserve(tensorDescs.size());
    for(size_t i = 0; i < tensorDescs.size(); ++i)
    {
        if(tensorDescs[i].get() == nullptr)
        {
            return {ErrorCode::HIPDNN_BACKEND_ERROR,
                    "Null tensor descriptor at index " + std::to_string(i) + " for "
                        + errorContext};
        }

        // Read the UID to check if we already have this tensor
        int64_t uid = 0;
        HIPDNN_CHECK_ERROR(getDescriptorAttrScalar(tensorDescs[i].get(),
                                                   HIPDNN_ATTR_TENSOR_UNIQUE_ID,
                                                   HIPDNN_TYPE_INT64,
                                                   uid,
                                                   "tensor UID for " + errorContext + "["
                                                       + std::to_string(i) + "]"));

        auto it = tensorMap.find(uid);
        if(it != tensorMap.end())
        {
            outTensors.push_back(it->second);
        }
        else
        {
            std::shared_ptr<graph::TensorAttributes> tensor;
            HIPDNN_CHECK_ERROR(unpackTensorAttributes(tensorDescs[i].get(), tensor));
            tensorMap[uid] = tensor;
            outTensors.push_back(std::move(tensor));
        }
    }

    return {};
}

/// Unpacks a byte array attribute (HIPDNN_TYPE_CHAR) from a backend descriptor.
/// Returns an empty vector if the attribute is not supported or has no elements.
[[nodiscard]] inline Error getDescriptorAttrByteArray(hipdnnBackendDescriptor_t desc,
                                                      hipdnnBackendAttributeName_t attrName,
                                                      std::vector<uint8_t>& value,
                                                      const std::string& errorContext)
{
    int64_t count = 0;
    auto countStatus = hipdnnBackend()->backendGetAttribute(
        desc, attrName, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    if(countStatus == HIPDNN_STATUS_NOT_SUPPORTED)
    {
        value.clear();
        return {};
    }
    if(countStatus != HIPDNN_STATUS_SUCCESS)
    {
        std::array<char, HIPDNN_ERROR_STRING_MAX_LENGTH> backendErrMsg{};
        hipdnnBackend()->getLastErrorString(backendErrMsg.data(), backendErrMsg.size());
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to get count for " + errorContext
                    + " Backend error: " + backendErrMsg.data()};
    }
    if(count <= 0)
    {
        value.clear();
        return {};
    }

    value.resize(static_cast<size_t>(count));
    int64_t actualCount = 0;
    HIPDNN_RETURN_ON_BACKEND_FAILURE(
        hipdnnBackend()->backendGetAttribute(
            desc, attrName, HIPDNN_TYPE_CHAR, count, &actualCount, value.data()),
        "Failed to get " + errorContext);

    return {};
}

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
    attributes.set_custom_op_id(std::move(customOpId));

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
