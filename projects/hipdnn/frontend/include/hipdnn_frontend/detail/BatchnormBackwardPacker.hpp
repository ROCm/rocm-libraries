// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a batchnorm backward operation descriptor from BatchnormBackwardAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createBatchnormBackwardOperation(
    const graph::BatchnormBackwardAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
        HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create batchnorm backward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                             attributes.get_dy(),
                                             tensorDescs,
                                             "batchnorm backward DY"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "batchnorm backward X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "batchnorm backward SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                                             attributes.get_dx(),
                                             tensorDescs,
                                             "batchnorm backward DX"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                                             attributes.get_dscale(),
                                             tensorDescs,
                                             "batchnorm backward DSCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                                             attributes.get_dbias(),
                                             tensorDescs,
                                             "batchnorm backward DBIAS"));

    // Set optional tensors (mean and inv_variance)
    if(attributes.get_mean())
    {
        HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                                                 attributes.get_mean(),
                                                 tensorDescs,
                                                 "batchnorm backward MEAN"));
    }
    if(attributes.get_inv_variance())
    {
        HIPDNN_CHECK_ERROR(
            ensureAndSetTensorRef(opDesc.get(),
                                  HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                                  attributes.get_inv_variance(),
                                  tensorDescs,
                                  "batchnorm backward INV_VARIANCE"));
    }

    // Set compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "batchnorm backward compute data type"));

    // Set peer_stats tensor array
    {
        auto& peerStatsAttrs = attributes.get_peer_stats();
        if(!peerStatsAttrs.empty())
        {
            HIPDNN_CHECK_ERROR(
                ensureAndSetTensorArrayRef(opDesc.get(),
                                           HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                                           peerStatsAttrs,
                                           tensorDescs,
                                           "batchnorm backward peer_stats"));
        }
    }

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "batchnorm backward operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
