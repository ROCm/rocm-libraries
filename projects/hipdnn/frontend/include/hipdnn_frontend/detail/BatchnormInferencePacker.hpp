// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a batchnorminference operation descriptor from BatchnormInferenceAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createBatchnormInferenceOperation(
    const graph::BatchnormInferenceAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
        HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create batchnorminference operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "batchnorminference X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                                             attributes.get_mean(),
                                             tensorDescs,
                                             "batchnorminference MEAN"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                              attributes.get_inv_variance(),
                              tensorDescs,
                              "batchnorminference INV_VARIANCE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "batchnorminference SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                                             attributes.get_bias(),
                                             tensorDescs,
                                             "batchnorminference BIAS"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "batchnorminference Y"));

    // Set batchnorminference parameters

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "batchnorminference compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_NAME_EXT,
                                                   opName,
                                                   "batchnorminference operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "batchnorminference operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
