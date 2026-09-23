// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a batchnorm inference variance ext operation descriptor from
// BatchnormInferenceAttributesVarianceExt.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createBatchnormInferenceVarianceExtOperation(
    const graph::BatchnormInferenceAttributesVarianceExt& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
        HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_VARIANCE_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create batchnorm inference variance ext operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_X_EXT,
                              attributes.get_x(),
                              tensorDescs,
                              "batchnorm inference variance ext X"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_MEAN_EXT,
                              attributes.get_mean(),
                              tensorDescs,
                              "batchnorm inference variance ext MEAN"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_VARIANCE_EXT,
                              attributes.get_variance(),
                              tensorDescs,
                              "batchnorm inference variance ext VARIANCE"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_SCALE_EXT,
                              attributes.get_scale(),
                              tensorDescs,
                              "batchnorm inference variance ext SCALE"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_BIAS_EXT,
                              attributes.get_bias(),
                              tensorDescs,
                              "batchnorm inference variance ext BIAS"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_Y_EXT,
                              attributes.get_y(),
                              tensorDescs,
                              "batchnorm inference variance ext Y"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_VARIANCE_EPSILON_EXT,
                              attributes.get_epsilon(),
                              tensorDescs,
                              "batchnorm inference variance ext EPSILON"));

    HIPDNN_CHECK_ERROR(
        setDescriptorAttrDataType(opDesc.get(),
                                  HIPDNN_ATTR_BATCHNORM_INF_VAR_COMP_TYPE_EXT,
                                  attributes.compute_data_type,
                                  "batchnorm inference variance ext compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_NAME_EXT,
                                                   opName,
                                                   "batchnorminferencevarianceext operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(
        finalizeDescriptor(opDesc.get(), "batchnorm inference variance ext operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
