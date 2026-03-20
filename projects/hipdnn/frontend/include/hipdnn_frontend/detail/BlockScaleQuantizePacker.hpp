// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

namespace hipdnn_frontend::detail
{

// Builds a blockscalequantize operation descriptor from BlockScaleQuantizeAttributes.
// Tensor descriptors are created/deduplicated via ensureAndSetTensorRef.
inline Error createBlockScaleQuantizeOperation(
    const graph::BlockScaleQuantizeAttributes& attributes,
    std::unordered_map<int64_t, ScopedHipdnnBackendDescriptor>& tensorDescs,
    std::vector<ScopedHipdnnBackendDescriptor>& operations)
{
    // Create operation descriptor
    ScopedHipdnnBackendDescriptor opDesc(
        HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR_EXT);
    if(!opDesc.valid())
    {
        return {ErrorCode::HIPDNN_BACKEND_ERROR,
                "Failed to create blockscalequantize operation descriptor"};
    }

    // Create tensor descriptors (if needed) and set them on the operation
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_X_EXT,
                                             attributes.get_x(),
                                             tensorDescs,
                                             "blockscalequantize X_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_Y_EXT,
                                             attributes.get_y(),
                                             tensorDescs,
                                             "blockscalequantize Y_EXT"));
    HIPDNN_CHECK_ERROR(ensureAndSetTensorRef(opDesc.get(),
                                             HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_EXT,
                                             attributes.get_scale(),
                                             tensorDescs,
                                             "blockscalequantize SCALE_EXT"));

    // Set block scale quantize parameters
    if(!attributes.get_block_size().has_value())
    {
        return {ErrorCode::ATTRIBUTE_NOT_SET, "blockscalequantize block_size is required"};
    }
    const int32_t blockSize = attributes.get_block_size().value();
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE_EXT,
                                HIPDNN_TYPE_INT32,
                                blockSize,
                                "blockscalequantize block_size"));

    if(attributes.get_axis().has_value())
    {
        const int64_t axis = attributes.get_axis().value();
        HIPDNN_CHECK_ERROR(
            setDescriptorAttrScalar(opDesc.get(),
                                    HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT,
                                    HIPDNN_TYPE_INT64,
                                    axis,
                                    "blockscalequantize axis"));
    }

    const bool transpose = attributes.get_transpose();
    HIPDNN_CHECK_ERROR(
        setDescriptorAttrScalar(opDesc.get(),
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                transpose,
                                "blockscalequantize transpose"));

    HIPDNN_CHECK_ERROR(setDescriptorAttrDataType(opDesc.get(),
                                                 HIPDNN_ATTR_BLOCK_SCALE_QUANTIZE_MATH_PREC_EXT,
                                                 attributes.compute_data_type,
                                                 "blockscalequantize MATH_PREC"));

    // Finalize operation descriptor
    HIPDNN_CHECK_ERROR(finalizeDescriptor(opDesc.get(), "blockscalequantize operation descriptor"));

    operations.push_back(std::move(opDesc));
    return {};
}

} // namespace hipdnn_frontend::detail
