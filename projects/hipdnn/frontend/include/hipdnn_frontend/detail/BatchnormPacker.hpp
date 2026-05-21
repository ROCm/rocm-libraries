// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a batchnorm training forward operation descriptor from BatchnormAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createBatchnormOperation(
    const graph::BatchnormAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create batchnorm operation descriptor"};
    }

    // Required tensors
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "batchnorm X"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "batchnorm SCALE"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT,
                                             attributes.get_bias(),
                                             tensorDescs,
                                             "batchnorm BIAS"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                                             attributes.get_epsilon(),
                                             tensorDescs,
                                             "batchnorm EPSILON"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "batchnorm Y"));

    // Optional tensors
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                                      attributes.get_prev_running_mean(),
                                      tensorDescs,
                                      "batchnorm PREV_RUNNING_MEAN"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                                      attributes.get_prev_running_variance(),
                                      tensorDescs,
                                      "batchnorm PREV_RUNNING_VARIANCE"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                                                     attributes.get_momentum(),
                                                     tensorDescs,
                                                     "batchnorm MOMENTUM"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT,
                                                     attributes.get_mean(),
                                                     tensorDescs,
                                                     "batchnorm MEAN"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                                      attributes.get_inv_variance(),
                                      tensorDescs,
                                      "batchnorm INV_VARIANCE"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                                      attributes.get_next_running_mean(),
                                      tensorDescs,
                                      "batchnorm NEXT_RUNNING_MEAN"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                                      attributes.get_next_running_variance(),
                                      tensorDescs,
                                      "batchnorm NEXT_RUNNING_VARIANCE"));

    // Compute data type
    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "batchnorm compute data type"));

    // Peer stats tensor array
    {
        auto& peerStatsAttrs = attributes.get_peer_stats();
        if(!peerStatsAttrs.empty())
        {
            HIPDNN_CHECK_ERROR(
                ensureAndSetTensorArrayRef(opDesc.get(),
                                           HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                                           peerStatsAttrs,
                                           tensorDescs,
                                           "batchnorm peer_stats"));
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
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "batchnorm operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
