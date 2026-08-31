// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "HipdnnMoeGroupedMatmulMode.h"
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a MoE grouped matmul operation descriptor from MoeGroupedMatmulAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createMoeGroupedMatmulOperation(
    const graph::MoeGroupedMatmulAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create MoE grouped matmul operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                             attributes.get_token(),
                                             tensorDescs,
                                             "MoE grouped matmul TOKEN_DESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                                             attributes.get_weight(),
                                             tensorDescs,
                                             "MoE grouped matmul WEIGHT_DESC"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                              attributes.get_first_token_offset(),
                              tensorDescs,
                              "MoE grouped matmul FIRST_TOKEN_OFFSET_DESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                                             attributes.get_output(),
                                             tensorDescs,
                                             "MoE grouped matmul OUTPUT_DESC"));
    // Mode selects the canonical optional descriptor footprint.
    const auto frontendMode = attributes.get_mode();
    auto mode = hipdnn_frontend::toBackendMoeGroupedMatmulMode(frontendMode);
    if(!mode.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported mode"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                               HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                               *mode,
                                               "MoE grouped matmul mode"));

    switch(frontendMode)
    {
    case MoeGroupedMatmulMode::NONE:
        break;
    case MoeGroupedMatmulMode::GATHER:
        HIPDNN_CHECK_ERROR(
            ensureAndSetTensorRef(opDesc.get(),
                                  HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                  attributes.get_token_index(),
                                  tensorDescs,
                                  "MoE grouped matmul TOKEN_INDEX_DESC"));
        break;
    case MoeGroupedMatmulMode::SCATTER:
        HIPDNN_CHECK_ERROR(
            ensureAndSetTensorRef(opDesc.get(),
                                  HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                  attributes.get_token_index(),
                                  tensorDescs,
                                  "MoE grouped matmul TOKEN_INDEX_DESC"));
        HIPDNN_CHECK_ERROR(
            ensureAndSetTensorRef(opDesc.get(),
                                  HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                  attributes.get_token_ks(),
                                  tensorDescs,
                                  "MoE grouped matmul TOKEN_KS_DESC"));
        HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                                   HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K,
                                                   HIPDNN_TYPE_INT32,
                                                   attributes.get_top_k(),
                                                   "MoE grouped matmul top_k"));
        break;
    default:
        return {ErrorCode::INVALID_VALUE, "MoE grouped matmul has an unknown routing mode"};
    }

    // Set MoE grouped matmul parameters

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                                 attributes.compute_data_type,
                                                 "MoE grouped matmul compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "MoE grouped matmul operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
