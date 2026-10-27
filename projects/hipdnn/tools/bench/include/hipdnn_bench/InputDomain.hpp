// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <cstddef>
#include <cstdint>
#include <set>

/// @file InputDomain.hpp
/// @brief The tensors the validation fill must keep positive.
namespace hipdnn_bench
{

/// @brief Whether a pointwise @p mode is undefined on negative values (and on zero, which the
///        fill never writes): LOG, SQRT and RSQRT.
inline bool undefinedOnNegatives(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode mode)
{
    using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
    return mode == PointwiseMode::LOG || mode == PointwiseMode::SQRT
           || mode == PointwiseMode::RSQRT;
}

/// @brief The uids of the tensors an operation undefined on negative values reads, in the
///        serialized graph (or graph-and-plan container) @p data.
///
/// The fill keeps those it writes positive. One it does not write -- a tensor the graph
/// computes -- can be negative, so a NaN there may be the correct answer.
inline std::set<int64_t> positiveDomainReads(const uint8_t* data, size_t size)
{
    namespace fb = hipdnn_flatbuffers_sdk::data_objects;
    const auto wrapper
        = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper::fromSerializedBlob(data,
                                                                                         size);
    std::set<int64_t> reads;
    const auto* nodes = wrapper.getGraph().nodes();
    if(nodes == nullptr)
    {
        return reads;
    }
    for(const auto* node : *nodes)
    {
        const auto* pointwise = node->attributes_as_PointwiseAttributes();
        if(pointwise != nullptr && undefinedOnNegatives(pointwise->operation()))
        {
            reads.insert(pointwise->in_0_tensor_uid());
        }
    }
    return reads;
}

} // namespace hipdnn_bench
