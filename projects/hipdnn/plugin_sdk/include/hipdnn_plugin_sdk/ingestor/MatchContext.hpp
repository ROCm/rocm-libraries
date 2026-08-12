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

/// The catalog cache key. Deliberately excludes the handle: a handle is a caller-side
/// object that can be swapped, rebound, or destroyed while a plan built through it is
/// still live, so keying on it would tie cached work to a lifetime unrelated to that
/// work's validity. Two handles on one device share an entry; one handle rebound to
/// another device does not.
///
/// RFC 0017 §8.1 keys this on (engine id, graph id, device id) and §8.6 folds in a
/// descriptor-inventory generation. Neither appears here, for different reasons:
///
/// - The engine id is constant within any one cache, because GenericEngine owns its
///   state manager outright (std::unique_ptr, not shared) and the state manager owns
///   this cache. One engine, one UED, one catalog: an entry cannot be reached from an
///   engine other than the one that wrote it, so storing the id would only repeat a
///   value every entry already agrees on. This is enforced by the ownership, not by
///   convention -- a second engine cannot be handed this manager.
/// - The generation retires cached verdicts when a discovery scan changes the pack
///   inventory. Nothing is loaded from a file yet, so the inventory is fixed at compile
///   time and the generation would be a constant.
///
/// The generation becomes a real key component once packs can be dropped in; the engine
/// id stays out for as long as the ownership above holds. This is a struct rather than a
/// std::pair so adding a member then does not change how the key is spelled at its use
/// sites.
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
        // The graph id is already a well-distributed 128-bit value (a UUID v4), so
        // folding its bytes and mixing the device ordinal in is sufficient here; this
        // keys an in-process cache and is never persisted or exposed.
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
 * RFC 0017 §8.5: "Matching does double duty: it decides the kernel applies, and it binds
 * the fields the launch will use", and §8.1 keeps that state alongside the catalog so
 * "nothing is re-matched" once a graph has been matched. A dispatch formula reading
 * `$q.uid` is reading a value the matcher already resolved, not re-deriving it from the
 * graph with a second notion of what the graph looks like.
 *
 * Keyed by the token name a descriptor would write. Values are MetadataValue, the same
 * type a KMD field holds: RFC 0017's criteria language puts a graph fact and a kernel
 * fact on either side of one operator (`divisible($q.head_size, $kernel.tile_m)`), so
 * an interpreter needs one value type spanning both namespaces or its first act is a
 * refactor. That is also why the list alternative matters on this side and not only on
 * the kernel side: `stride_order`, the worked example for INT_LIST, is a graph fact.
 */
using BoundTokens = std::unordered_map<std::string, MetadataValue>;

/// Reads @p token from @p bound as an integer.
///
/// Bound tokens are frequently tensor uids and dimensions, so the int64 read is the
/// common one and is worth naming rather than open-coding a std::get_if at every call
/// site. Returns nullopt when the token is absent OR holds a non-integer alternative:
/// a caller that asked for an integer cannot use a list, and conflating "not bound"
/// with "bound to something else" would let a type confusion read as a missing token.
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
 * @brief The bound token state a matcher, scorer, or dispatch formula reads.
 *
 * RFC 0017 names five expression namespaces; this binds the two that do not
 * require a matcher to have run first — `$graph.*` (and the tensor and node fields
 * reached through it) and `$device.*`. The third, `$kernel.*`, arrives separately as a
 * KernelDefinition, because a kernel-scoped check is evaluated once per candidate while
 * this context is bound once per (graph, device).
 *
 * Holds references, not copies: it is built on the stack for the duration of one
 * matching pass and never outlives the graph it names.
 */
struct MatchContext
{
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph;
    DeviceId deviceId;
    const hipDeviceProp_t& deviceProperties;
};

/// @brief The graph's stable identity, or nullopt when it has none.
///
/// Absent on legacy graphs and on locally constructed graphs that were never finalized.
/// Callers must treat that as "cannot cache", never as an error: matching a graph with
/// no id still produces the right answer, it just cannot be memoized (there is no key to
/// memoize it under, and inventing one would alias unrelated graphs).
///
/// A present field is not enough: a nil or non-v4 id (real ids are always v4) is not
/// guaranteed unique, so it reads as "no identity" too. Cache-key uniqueness therefore
/// depends on isUuidV4()'s exact predicate; loosening it reopens this aliasing.
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
