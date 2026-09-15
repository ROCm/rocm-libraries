// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/GraphBuilders.hpp>

#include <cstdint>
#include <limits>

/// @file GraphSize.hpp
/// @brief How much memory a generated problem would need, from the graph itself.
///
/// Separate from MetadataCorpus.hpp, which owns the engine-backed oracle and therefore links
/// the frontend. This is arithmetic over a serialized graph and nothing else, so it can be
/// tested without a device -- which matters, because the failure mode is a number that is
/// merely too small rather than an error.
namespace hipdnn_corpus_gen
{

/// Bytes one element of @p type occupies, rounded up for the sub-byte types.
///
/// Was a flat four, with a comment asserting that no dtype a corpus uses exceeds it. That was
/// true of every declaration then written and enforced by nothing: adding float64 to one would
/// have halved the ceiling silently, and a ceiling that is too low admits problems too large to
/// benchmark, which is the failure it exists to prevent. Sub-byte types round up to one because
/// their packing is a property of the tensor rather than of the element, and over-counting
/// keeps this a ceiling.
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
        // Charged the widest, so a type this function has not been taught cannot slip a huge
        // problem past the ceiling.
        return 8;
    }
}

/// @brief Total bytes the tensors of @p bytes occupy.
///
/// The benchmarking ceiling §4.3.2 describes: a problem whose tensors do not fit cannot be
/// timed, so it cannot enter a corpus at any budget. Computed rather than declared, because it
/// is a property of the device and the dtype rather than of the operation, and because no
/// per-dimension window can express it. Without it the search faithfully proposes convolutions
/// that are applicable, enormous, and take minutes each.
///
/// Saturates rather than overflowing: a problem large enough to wrap the arithmetic is
/// certainly over any real ceiling, and a wrapped total would read as a small one.
inline int64_t graphBytes(const builders::GraphBytes& bytes)
{
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
        int64_t elements = 1;
        for(const auto dim : *dims)
        {
            if(dim <= 0 || elements > std::numeric_limits<int64_t>::max() / dim)
            {
                return std::numeric_limits<int64_t>::max();
            }
            elements *= dim;
        }

        const auto width = elementBytes(tensor->data_type());
        if(elements > std::numeric_limits<int64_t>::max() / width
           || total > std::numeric_limits<int64_t>::max() - (elements * width))
        {
            return std::numeric_limits<int64_t>::max();
        }
        total += elements * width;
    }
    return total;
}

} // namespace hipdnn_corpus_gen
