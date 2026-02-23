// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a batchnorminference operation descriptor from BatchnormInferenceAttributes.
// Tensor descriptors are created/deduplicated via createOrFindTensorDesc.
inline Error createBatchnormInferenceOperation(
    const graph::BatchnormInferenceAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Ensure tensor descriptors exist for X, MEAN, INV_VARIANCE, SCALE, BIAS, Y
    auto [errX, xUid] = createOrFindTensorDesc(tensorDescs, attributes.get_x());
    HIPDNN_CHECK_ERROR(errX);
    auto [errMean, meanUid] = createOrFindTensorDesc(tensorDescs, attributes.get_mean());
    HIPDNN_CHECK_ERROR(errMean);
    auto [errInvVariance, invVarianceUid]
        = createOrFindTensorDesc(tensorDescs, attributes.get_inv_variance());
    HIPDNN_CHECK_ERROR(errInvVariance);
    auto [errScale, scaleUid] = createOrFindTensorDesc(tensorDescs, attributes.get_scale());
    HIPDNN_CHECK_ERROR(errScale);
    auto [errBias, biasUid] = createOrFindTensorDesc(tensorDescs, attributes.get_bias());
    HIPDNN_CHECK_ERROR(errBias);
    auto [errY, yUid] = createOrFindTensorDesc(tensorDescs, attributes.get_y());
    HIPDNN_CHECK_ERROR(errY);

    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create batchnorminference operation descriptor"};
    }

    // Set tensor references
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X,
                                                  xUid,
                                                  tensorDescs,
                                                  "batchnorminference X"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN,
                                                  meanUid,
                                                  tensorDescs,
                                                  "batchnorminference MEAN"));
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrTensorRef(opDesc.get(),
                                   HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE,
                                   invVarianceUid,
                                   tensorDescs,
                                   "batchnorminference INV_VARIANCE"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE,
                                                  scaleUid,
                                                  tensorDescs,
                                                  "batchnorminference SCALE"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS,
                                                  biasUid,
                                                  tensorDescs,
                                                  "batchnorminference BIAS"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrTensorRef(opDesc.get(),
                                                  HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y,
                                                  yUid,
                                                  tensorDescs,
                                                  "batchnorminference Y"));

    // Set batchnorminference parameters

    auto computeDataType = hipdnn_frontend::toSdkType(attributes.compute_data_type);
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE,
                                               HIPDNN_TYPE_DATA_TYPE,
                                               computeDataType,
                                               "batchnorminference compute data type"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "batchnorminference operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
