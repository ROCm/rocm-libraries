// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorUnpackHelpers.hpp>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::detail
{

/// Unpacks a block scale quantize operation from a backend operation descriptor.
/// Populates the BlockScaleQuantizeAttributes with tensors (using tensorMap for sharing)
/// and block scale quantize parameters.
[[nodiscard]] inline Error unpackBlockScaleQuantizeOperation(
    hipdnnBackendDescriptor_t opDesc,
    std::unordered_map<int64_t, std::shared_ptr<graph::TensorAttributes>>& tensorMap,
    graph::BlockScaleQuantizeAttributes& attributes)
{
    // Unpack X (input) tensor
    std::shared_ptr<graph::TensorAttributes> xTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                               tensorMap,
                                               xTensor,
                                               "block_scale_quantize X tensor"));
    attributes.set_x(xTensor);

    // Unpack Y (output) tensor
    std::shared_ptr<graph::TensorAttributes> yTensor;
    HIPDNN_CHECK_ERROR(unpackAndRegisterTensor(opDesc,
                                               HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
                                               tensorMap,
                                               yTensor,
                                               "block_scale_quantize Y tensor"));
    attributes.set_y(yTensor);

    // Unpack Scale tensor
    std::shared_ptr<graph::TensorAttributes> scaleTensor;
    HIPDNN_CHECK_ERROR(
        unpackAndRegisterTensor(opDesc,
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
                                tensorMap,
                                scaleTensor,
                                "block_scale_quantize Scale tensor"));
    attributes.set_scale(scaleTensor);

    // Unpack block_size (INT32)
    int32_t blockSize = 0;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrScalar(opDesc,
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                                HIPDNN_TYPE_INT32,
                                blockSize,
                                "block_scale_quantize block_size"));
    attributes.set_block_size(blockSize);

    // Unpack optional axis (INT64)
    std::optional<int64_t> axis;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrOptionalScalar(opDesc,
                                        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT,
                                        HIPDNN_TYPE_INT64,
                                        axis,
                                        "block_scale_quantize axis"));
    if(axis.has_value())
    {
        attributes.set_axis(axis.value());
    }

    // Unpack transpose (BOOLEAN)
    bool transpose = false;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrScalar(opDesc,
                                HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                                HIPDNN_TYPE_BOOLEAN,
                                transpose,
                                "block_scale_quantize transpose"));
    attributes.set_transpose(transpose);

    // Unpack compute data type
    auto [dt, dtErr] = unpackGraphDataType(opDesc,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                                           "block_scale_quantize compute data type");
    if(dtErr.is_bad())
    {
        return dtErr;
    }
    attributes.set_compute_data_type(dt);

    // Unpack operation name
    std::string opName;
    HIPDNN_CHECK_ERROR(
        getDescriptorAttrString(opDesc, HIPDNN_ATTR_OPERATION_NAME_EXT, opName, "operation name"));
    attributes.set_name(opName);

    return {};
}

} // namespace hipdnn_frontend::detail
