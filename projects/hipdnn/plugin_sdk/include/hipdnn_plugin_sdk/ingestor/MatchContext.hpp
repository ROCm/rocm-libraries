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

/// The device a catalog was built for. A plain HIP device ordinal, as returned by
/// hipGetDevice; hipDNN has no richer device-identity type today.
using DeviceId = int;

/// The device id meaning "this call has no resolvable device". Negative so it can never
/// alias a real ordinal: a resolver that cannot answer must not return 0, which is a
/// device other calls legitimately resolve and would then share a catalog and property
/// cache entry with. Matchers decline on it, so a call carrying no device finds no
/// applicable kernel rather than matching against another device's facts.
inline constexpr DeviceId NO_DEVICE = -1;

/// A finalized graph's stable identity, minted by hipDNN at finalization and preserved
/// across serialization round trips.
using GraphId = hipdnn_flatbuffers_sdk::utilities::UuidBytes;

/// The catalog cache key. Excludes the handle: a handle can be swapped, rebound, or
/// destroyed while a plan built through it is still live, so keying on it would tie
/// cached work to a lifetime unrelated to that work's validity.
///
/// RFC 0017 §8.1 also keys this on engine id and folds in a descriptor-inventory
/// generation (§8.6); both are constant today (one engine owns one manager, and the
/// pack set is fixed at compile time) so they are omitted until that changes.
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
        // The graph id is already a well-distributed UUID v4; folding its bytes and
        // mixing in the device ordinal is enough for an in-process cache.
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

/**
 * @brief What matching resolved about one graph, carried forward to dispatch.
 *
 * Keyed by the token name a descriptor would write; values are MetadataValue, the same
 * type a KMD field holds, so a criteria expression can compare a graph fact and a kernel
 * fact with one operator (RFC 0017 §8.5, §8.1).
 */
using BoundTokens = std::unordered_map<std::string, MetadataValue>;

/// Reads @p token from @p bound as an integer. Returns nullopt when absent or bound to
/// a non-integer value: conflating "not bound" with "bound to something else" would let
/// a type confusion read as a missing token.
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
 * @brief The bound token state a matcher, scorer, or dispatch formula reads: `$graph.*`
 *        and `$device.*`. `$kernel.*` arrives separately as a KernelDefinition, evaluated
 *        per candidate rather than once per (graph, device).
 *
 * Holds references, not copies: built on the stack for one matching pass, and must not
 * outlive the graph it names.
 */
struct MatchContext
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph;
    DeviceId deviceId;
    const hipDeviceProp_t& deviceProperties;
};

/// @brief The graph's stable identity, or nullopt when it has none.
///
/// Absent on legacy or never-finalized graphs. Callers must treat that as "cannot
/// cache", not an error: matching still produces the right answer, just unmemoized.
///
/// A present but non-v4 id also reads as "no identity" — uniqueness is not otherwise
/// guaranteed, so cache-key correctness depends on isUuidV4()'s exact predicate.
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
