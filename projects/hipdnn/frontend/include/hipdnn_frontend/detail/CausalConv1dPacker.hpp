// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnCausalConv1dActivation.h"
#include <hipdnn_frontend/attributes/CausalConv1dAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a causal conv1d operation descriptor from CausalConv1dAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createCausalConv1dOperation(
    const graph::CausalConv1dAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_CAUSAL_CONV1D_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create causal conv1d operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "causal conv1d X_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                                             attributes.get_weight(),
                                             tensorDescs,
                                             "causal conv1d WEIGHT_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                                             attributes.get_cu_seqlens(),
                                             tensorDescs,
                                             "causal conv1d CU_SEQLENS_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "causal conv1d Y_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetOptionalTensorRef(opDesc.get(),
                                                     HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                                                     attributes.get_bias(),
                                                     tensorDescs,
                                                     "causal conv1d BIAS_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                                      attributes.get_initial_state(),
                                      tensorDescs,
                                      "causal conv1d INITIAL_STATE_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                                      attributes.get_state_indices(),
                                      tensorDescs,
                                      "causal conv1d STATE_INDICES_EXT"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetOptionalTensorRef(opDesc.get(),
                                      HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                                      attributes.get_final_state(),
                                      tensorDescs,
                                      "causal conv1d FINAL_STATE_EXT"));

    // Set causal conv1d parameters

    // Set causal conv1d mode
    auto activation = hipdnn_frontend::toBackendCausalConv1dActivation(attributes.get_activation());
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                                               HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                                               activation,
                                               "causal conv1d mode"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT,
                                                 attributes.compute_data_type,
                                                 "causal conv1d compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "causal conv1d operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
