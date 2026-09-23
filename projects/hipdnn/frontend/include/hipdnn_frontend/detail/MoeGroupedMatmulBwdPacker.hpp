// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/MoeGroupedMatmulBwdAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a MoE grouped matmul backward operation descriptor from MoeGroupedMatmulBwdAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createMoeGroupedMatmulBwdOperation(
    const graph::MoeGroupedMatmulBwdAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
        HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_BWD_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create MoE grouped matmul backward operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DOUTPUT_DESC,
                              attributes.get_doutput(),
                              tensorDescs,
                              "MoE grouped matmul backward DOUTPUT_DESC"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_TOKEN_DESC,
                              attributes.get_token(),
                              tensorDescs,
                              "MoE grouped matmul backward TOKEN_DESC"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_FIRST_TOKEN_OFFSET_DESC,
                              attributes.get_first_token_offset(),
                              tensorDescs,
                              "MoE grouped matmul backward FIRST_TOKEN_OFFSET_DESC"));
    HIPDNN_CHECK_ERROR(
        ensureAndSetTensorRef(opDesc.get(),
                              HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_DWEIGHT_DESC,
                              attributes.get_dweight(),
                              tensorDescs,
                              "MoE grouped matmul backward DWEIGHT_DESC"));

    // Set MoE grouped matmul backward parameters

    HIPDNN_CHECK_ERROR(
        setDescriptorAttrDataType(opDesc.get(),
                                  HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_BWD_MATH_PREC,
                                  attributes.compute_data_type,
                                  "MoE grouped matmul backward compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(
        finalizeDescriptor(opDesc.get(), "MoE grouped matmul backward operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
