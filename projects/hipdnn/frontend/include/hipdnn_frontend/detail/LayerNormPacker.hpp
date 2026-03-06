// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a layernorm operation descriptor from LayernormAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createLayernormOperation(
    const graph::LayernormAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create layernorm operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "layernorm X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "layernorm SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT,
                                             attributes.get_bias(),
                                             tensorDescs,
                                             "layernorm BIAS"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT,
                                             attributes.get_epsilon(),
                                             tensorDescs,
                                             "layernorm EPSILON"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "layernorm Y"));

    // Optional tensors: only set if present
    auto mean = attributes.get_mean();
    if(mean)
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT,
                                                 mean,
                                                 tensorDescs,
                                                 "layernorm MEAN"));
    }

    auto invVariance = attributes.get_inv_variance();
    if(invVariance)
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                                                 invVariance,
                                                 tensorDescs,
                                                 "layernorm INV_VARIANCE"));
    }

    // Set forward phase as int64_t
    auto forwardPhase = static_cast<int64_t>(toSdkType(attributes.get_forward_phase()));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_LAYERNORM_FORWARD_PHASE_EXT,
                                               HIPDNN_TYPE_INT64,
                                               forwardPhase,
                                               "layernorm forward phase"));

    // Set compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "layernorm compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "layernorm operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
