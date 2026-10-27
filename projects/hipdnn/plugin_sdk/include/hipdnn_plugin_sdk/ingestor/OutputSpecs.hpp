// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Which buffers of a graph a validating benchmark sweep fills and reads back, and which
// buffers only feed the nodes.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/node_operands_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_plugin_sdk/ingestor/BenchmarkValidation.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The buffers a graph's nodes write and the ones they only read.
struct GraphOutputSpecs
{
    /// Non-virtual tensors some node writes, in uid order, with their allocation size.
    std::vector<OutputBufferSpec> outputs;
    /// Non-virtual tensors no node writes, in uid order.
    std::vector<int64_t> inputUids;
};

/// The bytes @p tensor occupies in memory: ceil(span * elementBits / 8), where
/// span = 1 + sum((dims[i] - 1) * strides[i]). Unknown (nullopt) for a zero or negative
/// extent, a negative stride, an unset or unsized data type, dims and strides of different
/// ranks, or a size that overflows.
inline std::optional<size_t>
    tensorByteSize(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& tensor)
{
    const auto bits = heuristics::detail::elementBits(tensor.data_type());
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    // FlatBuffers omits an empty vector, so an absent one is a rank-0 tensor.
    const flatbuffers::uoffset_t rank = dims != nullptr ? dims->size() : 0;
    const flatbuffers::uoffset_t strideRank = strides != nullptr ? strides->size() : 0;
    if(!bits.has_value() || rank != strideRank)
    {
        return std::nullopt;
    }

    constexpr auto LIMIT = std::numeric_limits<uint64_t>::max();
    uint64_t span = 1;
    for(flatbuffers::uoffset_t i = 0; i < rank; ++i)
    {
        const int64_t extent = dims->Get(i);
        const int64_t stride = strides->Get(i);
        if(extent <= 0 || stride < 0)
        {
            return std::nullopt;
        }
        const auto step = static_cast<uint64_t>(extent - 1);
        const auto distance = static_cast<uint64_t>(stride);
        if((distance != 0 && step > LIMIT / distance) || span > LIMIT - step * distance)
        {
            return std::nullopt;
        }
        span += step * distance;
    }

    const auto elementBits = static_cast<uint64_t>(*bits);
    if(span > (LIMIT - 7) / elementBits)
    {
        return std::nullopt;
    }
    const uint64_t bytes = (span * elementBits + 7) / 8;
    return static_cast<size_t>(bytes);
}

/// Splits the non-virtual tensors of @p graph into what its nodes write and what they only
/// read. An output whose size is unknown is left out of `outputs` with a WARN naming it.
inline GraphOutputSpecs
    outputSpecsFor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    std::unordered_set<int64_t> produced;
    const auto* nodes = graph.getGraph().nodes();
    if(nodes != nullptr)
    {
        for(const auto* node : *nodes)
        {
            hipdnn_flatbuffers_sdk::data_objects::node_operands::forEachOutputTensor(
                *node,
                [&produced](std::string_view /*role*/, int64_t uid) { produced.insert(uid); });
        }
    }

    const auto& tensorMap = graph.getTensorMap();
    std::vector<int64_t> uids;
    uids.reserve(tensorMap.size());
    for(const auto& entry : tensorMap)
    {
        uids.push_back(entry.first);
    }
    std::sort(uids.begin(), uids.end());

    GraphOutputSpecs specs;
    for(const int64_t uid : uids)
    {
        const auto& tensor = *tensorMap.at(uid);
        if(tensor.virtual_())
        {
            continue;
        }
        if(produced.count(uid) == 0)
        {
            specs.inputUids.push_back(uid);
            continue;
        }
        const auto byteSize = tensorByteSize(tensor);
        if(!byteSize.has_value())
        {
            HIPDNN_PLUGIN_LOG_WARN("ingestor: output validation: output tensor "
                                   << uid << " has no known size; it is not checked");
            continue;
        }
        specs.outputs.push_back({uid, tensor.data_type(), *byteSize});
    }
    return specs;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
