// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

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

/// The graph half of a catalog cache key: the graph's v4 identity when it carries one,
/// otherwise its exact serialized bytes.
///
/// A v4 id names one finalized graph. Any other id (absent, nil, or content-derived like a
/// corpus graph's v8) is not trusted to be unique, so the graph is keyed by its bytes
/// instead: two graphs with identical bytes have identical catalogs and rankings. Bytes,
/// not the winner cache's logical `GraphContentKey`, because that key ignores fields a
/// ranker may read. A graph with neither is not cacheable (see `tryMake`).
class CatalogGraphKey
{
public:
    /// Keys by @p id.
    explicit CatalogGraphKey(const GraphId& id)
        : _id(id)
    {
    }

    /// @p graph's key, or nullopt when it has no v4 id and supplies no bytes.
    static std::optional<CatalogGraphKey>
        tryMake(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph);

    bool operator==(const CatalogGraphKey& other) const noexcept
    {
        if(_hash != other._hash || _id != other._id
           || (_bytes == nullptr) != (other._bytes == nullptr))
        {
            return false;
        }
        return _bytes == nullptr || _bytes == other._bytes || *_bytes == *other._bytes;
    }

    /// FNV-1a over the id, then the bytes when keyed by them.
    size_t hash() const noexcept
    {
        return _hash;
    }

    /// True when this key holds the graph's bytes rather than a v4 id.
    bool isByContent() const noexcept
    {
        return _bytes != nullptr;
    }

private:
    CatalogGraphKey(const GraphId& id, std::shared_ptr<const std::vector<uint8_t>> bytes)
        : _id(id)
        , _bytes(std::move(bytes))
        , _hash(computeHash())
    {
    }

    size_t computeHash() const noexcept
    {
        uint64_t hash = 1469598103934665603ULL;
        const auto fold = [&hash](uint8_t byte) {
            hash ^= static_cast<uint64_t>(byte);
            hash *= 1099511628211ULL;
        };
        for(const uint8_t byte : _id)
        {
            fold(byte);
        }
        if(_bytes != nullptr)
        {
            for(const uint8_t byte : *_bytes)
            {
                fold(byte);
            }
        }
        return static_cast<size_t>(hash);
    }

    // NOLINTNEXTLINE(readability-redundant-member-init) - zero id when keyed by bytes
    GraphId _id{};
    /// Shared: a key is copied into the cache and outlives the graph it was made from.
    std::shared_ptr<const std::vector<uint8_t>> _bytes;
    size_t _hash = computeHash();
};

/// The catalog cache key. Excludes the handle: unrelated to a plan's validity.
///
/// RFC 0019 §9.2's engine id is implicit in the cache's location, and @ref engineVersion
/// replaces its inventory generation counter, since nothing re-scans the inventory.
struct CatalogKey
{
    CatalogGraphKey graph;
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
        return graph == other.graph && deviceId == other.deviceId
               && engineVersion == other.engineVersion && rankingMetric == other.rankingMetric;
    }
};

struct CatalogKeyHash
{
    size_t operator()(const CatalogKey& key) const noexcept
    {
        size_t hash = key.graph.hash();
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

/// nullopt when absent or non-v4: neither is trusted as a unique identity.
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

inline std::optional<CatalogGraphKey>
    CatalogGraphKey::tryMake(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    if(const auto id = tryGetGraphId(graph); id.has_value())
    {
        return CatalogGraphKey{*id};
    }
    const auto bytes = graph.bytes();
    if(bytes.data == nullptr || bytes.size == 0)
    {
        return std::nullopt;
    }
    return CatalogGraphKey{
        GraphId{},
        std::make_shared<const std::vector<uint8_t>>(bytes.data, bytes.data + bytes.size)};
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
