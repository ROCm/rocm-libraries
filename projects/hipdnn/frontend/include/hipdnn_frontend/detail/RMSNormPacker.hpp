// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a rmsnorm operation descriptor from RMSNormAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error
    createRMSNormOperation(const graph::RMSNormAttributes& attributes,
                           std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
                           std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create rmsnorm operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    // Required tensors
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "rmsnorm X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "rmsnorm SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT,
                                             attributes.get_epsilon(),
                                             tensorDescs,
                                             "rmsnorm EPSILON"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "rmsnorm Y"));

    // Optional tensors — only set if present
    if(attributes.get_bias())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT,
                                                 attributes.get_bias(),
                                                 tensorDescs,
                                                 "rmsnorm BIAS"));
    }
    if(attributes.get_inv_rms())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                                                 attributes.get_inv_rms(),
                                                 tensorDescs,
                                                 "rmsnorm INV_RMS"));
    }

    // Set forward phase
    auto forwardPhase = toBackendNormFwdPhase(attributes.get_forward_phase());
    if(!forwardPhase.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported rmsnorm forward phase"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                               HIPDNN_TYPE_NORM_FWD_PHASE,
                                               *forwardPhase,
                                               "rmsnorm forward phase"));

    // Set compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "rmsnorm compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "rmsnorm operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "rmsnorm operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
