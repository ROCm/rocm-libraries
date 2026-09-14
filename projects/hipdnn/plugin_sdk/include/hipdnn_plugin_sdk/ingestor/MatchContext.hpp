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

#include <hipdnn_data_sdk/utilities/VersionUtils.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device a catalog was built for: a plain HIP device ordinal.
using DeviceId = int;

/// "No resolvable device"; negative so it never aliases a real ordinal.
inline constexpr DeviceId NO_DEVICE = -1;

/// A finalized graph's stable identity, preserved across serialization round trips.
using GraphId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// The catalog cache key. Excludes the handle: unrelated to a plan's validity.
///
/// RFC 0019 §9.2 keys a cached ranking on `(engine id, graph id, device id)` plus the
/// inventory generation counter. Two of those four are already here by construction: the
/// engine id is implicit in the cache's location -- §9.2, "where the cache already lives on
/// the engine ... the engine id is *implicit* in the cache's location rather than absent
/// from the key" -- and there is no mid-process generation counter to carry, because engines
/// are loaded once at hipDNN startup and nothing re-scans the inventory while a cache is
/// live.
///
/// What the counter exists to catch is still real, though: a ranking outliving the
/// descriptor set that produced it. @ref engineVersion stands in for it, so the question a
/// counter would have answered ("is this entry from the inventory I am running?") is
/// answered by the descriptor set's own authored revision instead of by a number that only
/// ever changes in a re-scan this build does not perform.
struct CatalogKey
{
    GraphId graphId;
    DeviceId deviceId;
    /// `EngineDescriptor::revision` of the descriptor set that ranked this catalog.
    ///
    /// In-process this is constant per engine, so it never separates two live entries; it is
    /// in the key for the case that has no other guard -- an entry reaching a newer engine
    /// version, whether through a future shared cache or through the persisted winner records
    /// that outlive the process (see `EngineIdentity` in WinnerCacheFile.hpp). A stale
    /// ranking is not a wrong answer today only because nothing hands entries across that
    /// boundary; keying it makes that a property of the key rather than of the call graph.
    hipdnn_data_sdk::utilities::Version engineVersion{};

    bool operator==(const CatalogKey& other) const noexcept
    {
        return graphId == other.graphId && deviceId == other.deviceId
               && engineVersion == other.engineVersion;
    }
};

struct CatalogKeyHash
{
    size_t operator()(const CatalogKey& key) const noexcept
    {
        size_t hash = 1469598103934665603ULL;
        for(const uint8_t byte : key.graphId)
        {
            hash ^= static_cast<size_t>(byte);
            hash *= 1099511628211ULL;
        }
        const auto mix = [&hash](size_t value) {
            hash ^= value + 0x9e3779b9ULL + (hash << 6U) + (hash >> 2U);
        };
        mix(static_cast<size_t>(key.deviceId));
        // Each component separately: a packed "major*1000 + minor" style fold makes 1.10.0
        // and 2.0.0 collide, and a revision bump is exactly the event this field exists to
        // separate.
        mix(static_cast<size_t>(key.engineVersion.major));
        mix(static_cast<size_t>(key.engineVersion.minor));
        mix(static_cast<size_t>(key.engineVersion.patch));
        return hash;
    }
};

/// Token name to MetadataValue map of what matching resolved for one graph.
using BoundTokens = std::unordered_map<std::string, MetadataValue>;

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

/// Bound token state a matcher, scorer, or dispatch formula reads. Holds references,
/// not copies: built on the stack for one matching pass, must not outlive the graph.
struct MatchContext
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph;
    DeviceId deviceId;
    const DeviceProperties& deviceProperties;
};

/// nullopt when absent or non-v4 (both mean "cannot cache").
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

/// The graph schema version @p graph's own contents require; unstamped reads as
/// baseline.
inline hipdnn_data_sdk::utilities::Version
    graphSchemaFloor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    return fromEngineApiVersion(graph.getGraph().min_required_engine_api_version());
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
