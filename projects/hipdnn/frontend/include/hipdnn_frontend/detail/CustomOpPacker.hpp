// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a custom op operation descriptor from CustomOpAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorArrayRef.
inline Error
    createCustomOpOperation(const graph::CustomOpAttributes& attributes,
                            std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                            std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_CUSTOM_OP_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create custom op operation descriptor"};
    }

    // Set input tensor array (skip if empty — zero-input custom ops are valid)
    if(!attributes.get_inputs().empty())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorArrayRef(opDesc.get(),
                                                      HIPDNN_ATTR_OPERATION_CUSTOM_OP_INPUTS_EXT,
                                                      attributes.get_inputs(),
                                                      tensorDescs,
                                                      "custom op inputs"));
    }

    // Set output tensor array (skip if empty — zero-output custom ops are valid)
    if(!attributes.get_outputs().empty())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorArrayRef(opDesc.get(),
                                                      HIPDNN_ATTR_OPERATION_CUSTOM_OP_OUTPUTS_EXT,
                                                      attributes.get_outputs(),
                                                      tensorDescs,
                                                      "custom op outputs"));
    }

    // Set custom_op_id string
    HIPDNN_CHECK_ERROR(setDescriptorAttrString(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_CUSTOM_OP_ID_EXT,
                                               attributes.get_custom_op_id(),
                                               "custom op id"));

    // Set opaque data payload
    if(!attributes.get_data().empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrVec(opDesc.get(),
                                                HIPDNN_ATTR_OPERATION_CUSTOM_OP_DATA_EXT,
                                                HIPDNN_TYPE_CHAR,
                                                attributes.get_data(),
                                                "custom op data"));
    }

    // Set compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_CUSTOM_OP_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "custom op compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "custom op operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "custom op operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
