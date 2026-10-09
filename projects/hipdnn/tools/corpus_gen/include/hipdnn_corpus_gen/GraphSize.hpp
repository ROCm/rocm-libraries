// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/GraphBuilders.hpp>

#include <cstdint>
#include <limits>

/// @file GraphSize.hpp
/// @brief Memory a generated problem needs, computed from the serialized graph alone.
namespace hipdnn_corpus_gen
{

/// Bytes one element of @p type occupies. Sub-byte types round up to one so the result
/// stays an upper bound.
inline int64_t elementBytes(hipdnn_flatbuffers_sdk::data_objects::DataType type)
{
    using hipdnn_flatbuffers_sdk::data_objects::DataType;
    switch(type)
    {
    case DataType::DOUBLE:
    case DataType::INT64:
        return 8;
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    case DataType::INT8:
    case DataType::UINT8:
    case DataType::BOOLEAN:
    case DataType::FP8_E4M3:
    case DataType::FP8_E5M2:
    case DataType::FP8_E8M0:
    case DataType::FP8_E4M3_FNUZ:
    case DataType::FP8_E5M2_FNUZ:
    case DataType::FP4_E2M1:
    case DataType::FP6_E2M3:
    case DataType::FP6_E3M2:
    case DataType::INT4:
        return 1;
    case DataType::UNSET:
    default:
        // Charge the widest so an unknown type cannot slip past the ceiling.
        return 8;
    }
}

/// @brief Total bytes the tensors of @p bytes need allocated (the §4.3.2 bench ceiling).
///
/// Each tensor is charged its strided span, `1 + sum((dim_i - 1) * stride_i)` elements, to
/// match hipdnn_bench::elementSpan. Saturates at INT64_MAX on overflow or on a tensor that
/// cannot be sized (non-positive extent, negative stride, rank-mismatched strides).
inline int64_t graphBytes(const builders::GraphBytes& bytes)
{
    constexpr auto SATURATED = std::numeric_limits<int64_t>::max();

    const auto* graph = hipdnn_flatbuffers_sdk::data_objects::GetGraph(bytes.data());
    if(graph == nullptr || graph->tensors() == nullptr)
    {
        return 0;
    }

    int64_t total = 0;
    for(const auto* tensor : *graph->tensors())
    {
        const auto* dims = tensor->dims();
        if(dims == nullptr)
        {
            continue;
        }
        const auto* strides = tensor->strides();
        if(strides == nullptr || strides->size() != dims->size())
        {
            return SATURATED;
        }

        int64_t furthest = 0;
        for(flatbuffers::uoffset_t i = 0; i < dims->size(); ++i)
        {
            const auto dim = dims->Get(i);
            const auto stride = strides->Get(i);
            if(dim <= 0 || stride < 0)
            {
                return SATURATED;
            }
            if(stride > 0 && dim - 1 > (SATURATED - 1 - furthest) / stride)
            {
                return SATURATED;
            }
            furthest += (dim - 1) * stride;
        }
        const auto span = furthest + 1;

        const auto width = elementBytes(tensor->data_type());
        if(span > SATURATED / width || total > SATURATED - (span * width))
        {
            return SATURATED;
        }
        total += span * width;
    }
    return total;
}

} // namespace hipdnn_corpus_gen
