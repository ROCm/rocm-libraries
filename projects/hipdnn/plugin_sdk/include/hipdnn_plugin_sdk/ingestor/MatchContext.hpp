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

#include <hipdnn_data_sdk/utilities/RankingMetrics.hpp>
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
/// RFC 0019 §9.2's engine id is implicit in the cache's location, and @ref engineVersion
/// replaces its inventory generation counter, since nothing re-scans the inventory.
struct CatalogKey
{
    GraphId graphId;
    DeviceId deviceId;
    /// `EngineDescriptor::revision` of the descriptor set that ranked this catalog. Constant
    /// in-process; keyed so an entry can never be served to a newer engine version.
    // Explicit init avoids -Wmissing-field-initializers on partial CatalogKey{graph, device}.
    // NOLINTNEXTLINE(readability-redundant-member-init) - see above
    hipdnn_data_sdk::utilities::Version engineVersion{};
    /// The ranking metric the order was computed for (RFC 0019 §11.4): every cache of a
    /// ranked order is keyed by metric, or a `time` request is served a `tflops` order.
    /// Always a registered name, so it views the registry's static storage.
    std::string_view rankingMetric = hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC;

    bool operator==(const CatalogKey& other) const noexcept
    {
        return graphId == other.graphId && deviceId == other.deviceId
               && engineVersion == other.engineVersion && rankingMetric == other.rankingMetric;
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
        // Mixed per component: a packed major*1000+minor fold would collide 1.10.0 and 2.0.0.
        mix(static_cast<size_t>(key.engineVersion.major));
        mix(static_cast<size_t>(key.engineVersion.minor));
        mix(static_cast<size_t>(key.engineVersion.patch));
        mix(std::hash<std::string_view>{}(key.rankingMetric));
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
    /// Registered ranking metric for the request (RFC 0019 §11.4); selects the
    /// `sort_kernel_catalog` UHD and keys ranked-order caches. A view: must outlive the
    /// context, and caches key on the registry's copy of the name, never on this view.
    std::string_view rankingMetric = hipdnn_data_sdk::utilities::DEFAULT_RANKING_METRIC;
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
