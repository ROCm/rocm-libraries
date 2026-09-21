// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a reduction operation descriptor from ReductionAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createReductionOperation(
    const graph::ReductionAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(HIPDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR, "Failed to create reduction operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_REDUCTION_XDESC,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "reduction XDESC"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_REDUCTION_YDESC,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "reduction YDESC"));

    // Set reduction parameters

    // Set reduction mode
    auto modeOpt = attributes.get_mode();
    if(!modeOpt.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Reduction mode not set"};
    }
    auto mode = hipdnn_frontend::toBackendReductionMode(*modeOpt);
    if(!mode.has_value())
    {
        return {ErrorCode::INVALID_VALUE, "Unsupported reduction mode"};
    }
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_REDUCTION_OPERATOR,
                                               HIPDNN_TYPE_REDUCTION_OPERATOR_TYPE,
                                               *mode,
                                               "reduction mode"));
    HIPDNN_CHECK_ERROR(setDescriptorAttrScalar(opDesc.get(),
                                               HIPDNN_ATTR_REDUCTION_IS_DETERMINISTIC,
                                               HIPDNN_TYPE_BOOLEAN,
                                               attributes.get_is_deterministic(),
                                               "reduction is_deterministic"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_REDUCTION_COMP_TYPE,
                                                 attributes.compute_data_type,
                                                 "reduction compute data type"));

    // Set operation name if provided
    auto& opName = attributes.get_name();
    if(!opName.empty())
    {
        HIPDNN_CHECK_ERROR(setDescriptorAttrString(
            opDesc.get(), HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    }

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "reduction operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
