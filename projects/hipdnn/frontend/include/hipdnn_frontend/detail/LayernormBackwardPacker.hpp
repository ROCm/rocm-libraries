// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a layernormbackward operation descriptor from LayernormBackwardAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createLayernormBackwardOperation(
    const graph::LayernormBackwardAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create layernormbackward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "layernormbackward DY"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "layernormbackward X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "layernormbackward SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                                                     attributes.get_mean(),
                                                     tensorDescs,
                                                     "layernormbackward MEAN"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                                      attributes.get_inv_variance(),
                                      tensorDescs,
                                      "layernormbackward INV_VARIANCE"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                                      attributes.get_epsilon(),
                                      tensorDescs,
                                      "layernormbackward EPSILON"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                                             attributes.get_dx(),
                                             tensorDescs,
                                             "layernormbackward DX"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                                             attributes.get_dscale(),
                                             tensorDescs,
                                             "layernormbackward DSCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                                             attributes.get_dbias(),
                                             tensorDescs,
                                             "layernormbackward DBIAS"));

    // Set layernormbackward parameters
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_LAYERNORM_BACKWARD_NORMALIZED_DIM_COUNT,
                                               HIPDNN_TYPE_INT64,
                                               attributes.get_normalized_dim_count(),
                                               "layernormbackward normalized_dim_count"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "layernormbackward compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "layernormbackward operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
