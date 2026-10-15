// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceProperties.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device a catalog was built for: a plain HIP device ordinal.
using DeviceId = int;

/// "No resolvable device"; negative so it never aliases a real ordinal.
inline constexpr DeviceId NO_DEVICE = -1;

/// The `Graph`-level fields `GraphContentKey` drops as `(cache_ignore)` that a matcher
/// can read; `name` and `id` are left out because no matcher reads them.
struct CatalogGraphFields
{
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType intermediateDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    hipdnn_flatbuffers_sdk::data_objects::DataType ioDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET;
    std::optional<int64_t> preferredEngineId;
    bool isOverrideShapeEnabled = false;
    /// As stamped: absence is kept distinct from an explicit baseline version.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::EngineApiVersion>
        minRequiredEngineApiVersion;

    static CatalogGraphFields of(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph)
    {
        CatalogGraphFields fields;
        fields.computeDataType = graph.compute_data_type();
        fields.intermediateDataType = graph.intermediate_data_type();
        fields.ioDataType = graph.io_data_type();
        fields.preferredEngineId = graph.preferred_engine_id();
        fields.isOverrideShapeEnabled = graph.is_override_shape_enabled();
        if(const auto* version = graph.min_required_engine_api_version(); version != nullptr)
        {
            fields.minRequiredEngineApiVersion = *version;
        }
        return fields;
    }

    bool operator==(const CatalogGraphFields& other) const
    {
        return computeDataType == other.computeDataType
               && intermediateDataType == other.intermediateDataType
               && ioDataType == other.ioDataType && preferredEngineId == other.preferredEngineId
               && isOverrideShapeEnabled == other.isOverrideShapeEnabled
               && minRequiredEngineApiVersion == other.minRequiredEngineApiVersion;
    }
};

/// The catalog cache key. Excludes the handle: unrelated to a plan's validity.
///
/// `graph` alone treats a renumbered graph as the same graph, but a catalog's bound
/// tokens carry raw tensor uids, so `tensorUids` (`Graph.tensors[i].uid`, in order)
/// keeps a renumbered graph from being served another numbering's bindings.
///
/// `graph` also drops the `(cache_ignore)` graph-level fields, which suits the winner
/// cache: they do not change whether a measurement carries over. A catalog's verdict
/// and bound tokens can depend on anything a matcher reads, such as a graph match that
/// declines override-shape graphs, so `graphFields` keys those fields exactly.
/// Only build one from a usable `graph`: an unusable key never equals itself.
struct CatalogKey
{
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graph;
    CatalogGraphFields graphFields;
    std::vector<int64_t> tensorUids;
    DeviceId deviceId;

    bool operator==(const CatalogKey& other) const
    {
        return deviceId == other.deviceId && graphFields == other.graphFields
               && tensorUids == other.tensorUids && graph == other.graph;
    }
};

struct CatalogKeyHash
{
    size_t operator()(const CatalogKey& key) const noexcept
    {
        size_t uidHash = 1469598103934665603ULL;
        for(const int64_t uid : key.tensorUids)
        {
            uidHash ^= static_cast<size_t>(uid);
            uidHash *= 1099511628211ULL;
        }
        size_t hash
            = std::hash<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey>{}(key.graph);
        const auto mix = [&hash](size_t value) {
            hash ^= value + 0x9e3779b9ULL + (hash << 6U) + (hash >> 2U);
        };
        mix(uidHash);
        mix(static_cast<size_t>(key.deviceId));

        const CatalogGraphFields& fields = key.graphFields;
        mix(static_cast<size_t>(fields.computeDataType));
        mix(static_cast<size_t>(fields.intermediateDataType));
        mix(static_cast<size_t>(fields.ioDataType));
        mix(static_cast<size_t>(fields.preferredEngineId.has_value()));
        mix(static_cast<size_t>(fields.preferredEngineId.value_or(0)));
        mix(static_cast<size_t>(fields.isOverrideShapeEnabled));
        mix(static_cast<size_t>(fields.minRequiredEngineApiVersion.has_value()));
        if(fields.minRequiredEngineApiVersion.has_value())
        {
            mix(fields.minRequiredEngineApiVersion->major());
            mix(fields.minRequiredEngineApiVersion->minor());
            mix(fields.minRequiredEngineApiVersion->patch());
        }
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

/// The graph schema version @p graph's own contents require; unstamped reads as
/// baseline.
inline hipdnn_data_sdk::utilities::Version
    graphSchemaFloor(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    return fromEngineApiVersion(graph.getGraph().min_required_engine_api_version());
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
