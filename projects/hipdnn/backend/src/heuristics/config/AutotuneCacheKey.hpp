// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace hipdnn_backend::heuristics::config
{

/// Derives the exact-match cache key for a (graph, device) pair.
///
/// The graph half ignores fields the schema marks `(cache_ignore)` and resolves
/// `(cache_uid)` tensor references positionally. Returns `std::nullopt` when either
/// view is empty or fails verification. A residual 64-bit hash collision is possible
/// and is left to the store layer to resolve.
inline std::optional<std::vector<uint8_t>>
    deriveCacheKey(const hipdnnPluginConstData_t& serializedGraph,
                   const hipdnnPluginConstData_t& deviceProperties)
{
    if(serializedGraph.ptr == nullptr || serializedGraph.size == 0
       || deviceProperties.ptr == nullptr || deviceProperties.size == 0)
    {
        return std::nullopt;
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        serializedGraph.ptr, serializedGraph.size);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graphKey(graphWrapper);
    if(!graphKey.isUsable())
    {
        return std::nullopt;
    }

    const uint64_t graphHash = graphKey.hash();
    const uint64_t deviceHash = hipdnn_data_sdk::utilities::fnv1aHash(
        static_cast<const uint8_t*>(deviceProperties.ptr), deviceProperties.size);

    std::vector<uint8_t> key(sizeof(graphHash) + sizeof(deviceHash));
    std::memcpy(key.data(), &graphHash, sizeof(graphHash));
    std::memcpy(key.data() + sizeof(graphHash), &deviceHash, sizeof(deviceHash));
    return key;
}

} // namespace hipdnn_backend::heuristics::config
