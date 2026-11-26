// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/**
 * @file PackFeatures.hpp
 * @brief The cost and tensor features a native pack binds for UHD models.
 *
 * Every native pack publishes the same per-tensor fields and the same overflow-checked
 * work counts, so the definitions live here once: two copies of a feature drift apart,
 * and a model trained on one pack's spelling then reads another's.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::pack_features
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::BoundTokens;

/// The dtype facts the binding publishes.
///
/// `spelling` mirrors frontend `to_string(DataType)`, the vocabulary UHD
/// `categorical_encoding` uses (not `EnumNameDataType`'s "FLOAT"); restated because this
/// provider does not link the frontend. Empty spelling or `bytes == 0` (e.g. sub-byte
/// types) leaves the dependent token absent rather than wrong (RFC 0019 §13.6).
struct DataTypeFacts
{
    std::string_view spelling;
    int64_t bytes;
};

inline DataTypeFacts dataTypeFacts(data_objects::DataType dataType)
{
    switch(dataType)
    {
    case data_objects::DataType::FLOAT:
        return {"fp32", 4};
    case data_objects::DataType::HALF:
        return {"fp16", 2};
    case data_objects::DataType::BFLOAT16:
        return {"bf16", 2};
    case data_objects::DataType::DOUBLE:
        return {"fp64", 8};
    case data_objects::DataType::UINT8:
        return {"uint8", 1};
    case data_objects::DataType::INT32:
        return {"int32", 4};
    case data_objects::DataType::INT8:
        return {"int8", 1};
    case data_objects::DataType::FP8_E4M3:
        return {"fp8_e4m3", 1};
    case data_objects::DataType::FP8_E5M2:
        return {"fp8_e5m2", 1};
    case data_objects::DataType::FP8_E8M0:
        return {"fp8_e8m0", 1};
    case data_objects::DataType::FP8_E4M3_FNUZ:
        return {"fp8_e4m3_fnuz", 1};
    case data_objects::DataType::FP8_E5M2_FNUZ:
        return {"fp8_e5m2_fnuz", 1};
    case data_objects::DataType::INT64:
        return {"int64", 8};
    case data_objects::DataType::BOOLEAN:
        return {"boolean", 1};
    case data_objects::DataType::FP4_E2M1:
        return {"fp4_e2m1", 0};
    case data_objects::DataType::FP6_E2M3:
        return {"fp6_e2m3", 0};
    case data_objects::DataType::FP6_E3M2:
        return {"fp6_e3m2", 0};
    case data_objects::DataType::INT4:
        return {"int4", 0};
    case data_objects::DataType::UNSET:
    default:
        return {{}, 0};
    }
}

/// @p left * @p right, or nullopt on overflow or a non-positive factor, so a wrapped value
/// is never published as a cost feature.
inline std::optional<int64_t> checkedMultiply(int64_t left, int64_t right)
{
    if(left <= 0 || right <= 0 || left > std::numeric_limits<int64_t>::max() / right)
    {
        return std::nullopt;
    }
    return left * right;
}

inline std::optional<int64_t> checkedProduct(std::initializer_list<int64_t> factors)
{
    int64_t result = 1;
    for(const auto factor : factors)
    {
        const auto next = checkedMultiply(result, factor);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        result = *next;
    }
    return result;
}

inline std::optional<int64_t> elementCount(const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    int64_t count = 1;
    for(const auto dim : *dims)
    {
        const auto next = checkedMultiply(count, dim);
        if(!next.has_value())
        {
            return std::nullopt;
        }
        count = *next;
    }
    return count;
}

/// Sum over operands of element count x that operand's own dtype size (correct for mixed
/// precision). nullopt when any operand has no statable width, so the token is absent.
inline std::optional<int64_t>
    movedBytes(std::initializer_list<const data_objects::TensorAttributes*> operands)
{
    int64_t total = 0;
    for(const auto* operand : operands)
    {
        const auto count = elementCount(*operand);
        const auto width = dataTypeFacts(operand->data_type()).bytes;
        if(!count.has_value() || width == 0)
        {
            return std::nullopt;
        }
        const auto operandBytes = checkedMultiply(*count, width);
        if(!operandBytes.has_value() || total > std::numeric_limits<int64_t>::max() - *operandBytes)
        {
            return std::nullopt;
        }
        total += *operandBytes;
    }
    return total;
}

/// Publishes @p tensor's RFC 0020 §6.1 `dims[i]` and `dtype` under @p root. dtype is the
/// spelling string, never a code: the model's `categorical_encoding` encodes it downstream.
inline void bindTensorFields(BoundTokens& bound,
                             std::string_view root,
                             const data_objects::TensorAttributes& tensor)
{
    const auto* dims = tensor.dims();
    const std::string prefix(root);
    for(flatbuffers::uoffset_t axis = 0; axis < dims->size(); ++axis)
    {
        bound[prefix + ".dims[" + std::to_string(axis) + "]"] = dims->Get(axis);
    }

    const auto spelling = dataTypeFacts(tensor.data_type()).spelling;
    if(!spelling.empty())
    {
        bound[prefix + ".dtype"] = std::string(spelling);
    }
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::pack_features
