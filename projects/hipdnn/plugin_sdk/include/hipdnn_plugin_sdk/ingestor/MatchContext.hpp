// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device a catalog was built for: a plain HIP device ordinal (hipGetDevice).
using DeviceId = int;

/// Device id meaning "no resolvable device". Negative so it never aliases a real ordinal;
/// matchers decline on it rather than matching against another device's facts.
inline constexpr DeviceId NO_DEVICE = -1;

/// A finalized graph's stable identity, preserved across serialization round trips.
using GraphId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// The catalog cache key. Excludes the handle: a handle's lifetime is unrelated to a plan's
/// validity, so keying on it would be wrong.
///
/// RFC 0017 §8.1 also keys this on engine id and a descriptor-inventory generation (§8.6);
/// both are constant today so they are omitted.
struct CatalogKey
{
    GraphId graphId;
    DeviceId deviceId;

    bool operator==(const CatalogKey& other) const noexcept
    {
        return graphId == other.graphId && deviceId == other.deviceId;
    }
};

/// A user-defined key has no std::hash, so the cache is given this explicitly.
struct CatalogKeyHash
{
    size_t operator()(const CatalogKey& key) const noexcept
    {
        // UUID v4 bytes are already well-distributed; this folds them with the device ordinal.
        size_t hash = 1469598103934665603ULL;
        for(const uint8_t byte : key.graphId)
        {
            hash ^= static_cast<size_t>(byte);
            hash *= 1099511628211ULL;
        }
        hash ^= static_cast<size_t>(key.deviceId) + 0x9e3779b9ULL + (hash << 6U) + (hash >> 2U);
        return hash;
    }
};

/// Token name to MetadataValue map of what matching resolved for one graph.
using BoundTokens = std::unordered_map<std::string, MetadataValue>;

/// Returns nullopt when absent or bound to a non-integer value, so type confusion never reads
/// as a missing token.
inline std::optional<int64_t> tryGetBoundInt(const BoundTokens& bound, std::string_view token)
{
    const auto it = bound.find(std::string(token));
    if(it == bound.end())
    {
        return std::nullopt;
    }
    const auto* value = std::get_if<int64_t>(&it->second);
    if(value == nullptr)
    {
        return std::nullopt;
    }
    return *value;
}

/**
 * @brief Bound token state a matcher, scorer, or dispatch formula reads (`$graph.*`,
 *        `$device.*`); `$kernel.*` arrives separately as a KernelDefinition per candidate.
 *
 * Holds references, not copies: built on the stack for one matching pass and must not
 * outlive the graph it names.
 */
struct MatchContext
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph;
    DeviceId deviceId;
    const hipDeviceProp_t& deviceProperties;
};

/// @brief The graph's stable identity, or nullopt when absent (legacy/unfinalized graphs) or
/// non-v4 (uniqueness is otherwise unguaranteed). Callers must treat both as "cannot cache".
inline std::optional<GraphId>
    tryGetGraphId(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    const auto* id = graph.getGraph().id();
    if(id == nullptr)
    {
        return std::nullopt;
    }
    const auto bytes = hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*id);
    if(!hipdnn_flatbuffers_sdk::utilities::isUuidV4(bytes))
    {
        return std::nullopt;
    }
    return bytes;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
