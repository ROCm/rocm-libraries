// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a rmsnormbackward operation descriptor from RMSNormBackwardAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createRMSNormBackwardOperation(
    const graph::RMSNormBackwardAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create rmsnormbackward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "rmsnormbackward DY_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "rmsnormbackward X_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "rmsnormbackward SCALE_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                                             attributes.get_inv_rms(),
                                             tensorDescs,
                                             "rmsnormbackward INV_RMS_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                                             attributes.get_dx(),
                                             tensorDescs,
                                             "rmsnormbackward DX_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT,
                                             attributes.get_dscale(),
                                             tensorDescs,
                                             "rmsnormbackward DSCALE_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT,
                                      attributes.get_dbias(),
                                      tensorDescs,
                                      "rmsnormbackward DBIAS_EXT"));

    // Set rmsnormbackward parameters

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "rmsnormbackward compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "rmsnormbackward operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
