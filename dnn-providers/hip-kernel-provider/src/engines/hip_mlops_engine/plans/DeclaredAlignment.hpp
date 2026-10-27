// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <unordered_map>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>

// What the HIP_MLOPS engines whose kernels only access whole elements (LayerNorm, RMSNorm,
// Resample) need from the alignment a graph declares for its tensors
// (TensorAttributes::alignment(), in bytes).
//
// The backend rejects a device pointer whose address is not a multiple of the declared alignment,
// so the declaration is a multiple guarantee: declared 16 guarantees 16-byte alignment, declared 4
// only 4. A kernel that loads one `T` at a time (no vector access) needs every device pointer to
// be a multiple of sizeof(T), so the declaration has to be a multiple of sizeof(T): 4 or 8 for a
// float, 2 for a half, but not 1 and not 6 for a float.
namespace hip_kernel_provider
{

/**
 * Size in bytes of one element of a data type that occupies a whole number of bytes, 0 for any
 * other type (sub-byte or unset).
 */
inline int64_t wholeElementBytes(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    switch(type)
    {
    case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT64:
        return 8;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
        return 4;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
        return 2;
    case hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::BOOLEAN:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E8M0:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3_FNUZ:
    case hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2_FNUZ:
        return 1;
    default:
        return 0;
    }
}

/**
 * True when the declared alignment of a tensor with elements of `elementBytes` bytes guarantees
 * that every element address is a valid typed address: it is a positive multiple of the element
 * size.
 */
inline bool declaredAlignmentCoversElement(int64_t declaredBytes, int64_t elementBytes)
{
    return declaredBytes > 0 && elementBytes > 0 && declaredBytes % elementBytes == 0;
}

/**
 * The uid of a tensor of the graph whose declared alignment does not cover its own element size,
 * or nullopt when every tensor does. Only tensors backed by a device buffer are looked at:
 * virtual tensors have no buffer, and pass-by-value scalars are host values that the backend does
 * not alignment-check. Tensors of a type without a whole-byte element size are left to the
 * engines' data type checks.
 */
inline std::optional<int64_t> findTensorWithUncoveredElementAlignment(
    const std::unordered_map<int64_t,
                             const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
        tensorMap)
{
    std::optional<int64_t> offender;
    for(const auto& [uid, tensor] : tensorMap)
    {
        if(tensor == nullptr || tensor->virtual_()
           || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(tensor))
        {
            continue;
        }

        const int64_t elementBytes = wholeElementBytes(tensor->data_type());
        if(elementBytes == 0 || declaredAlignmentCoversElement(tensor->alignment(), elementBytes))
        {
            continue;
        }

        // Report the lowest uid so the decline message is stable across hash orderings.
        if(!offender.has_value() || uid < *offender)
        {
            offender = uid;
        }
    }
    return offender;
}

} // namespace hip_kernel_provider
