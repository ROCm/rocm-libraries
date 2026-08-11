// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <array>
#include <cstdint>
#include <functional>
#include <optional>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device a catalog was built for. A plain HIP device ordinal, as returned by
/// hipGetDevice; hipDNN has no richer device-identity type today.
using DeviceId = int;

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
/// descriptor-inventory generation. Neither is present, and both omissions are scoped
/// rather than free:
///
/// - The engine id is implicit because one state manager serves one engine and owns this
///   cache, so no other engine's catalog can reach it. That holds only while a state
///   manager is never shared across engines.
/// - The generation retires cached verdicts when a discovery scan changes the pack
///   inventory. Nothing is loaded from a file yet, so the inventory is fixed at compile
///   time and the generation would be a constant.
///
/// Both become real key components once engines are built from UED files and packs can
/// be dropped in. This is a struct rather than a std::pair so adding them then does not
/// change how the key is spelled at its use sites.
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
inline std::optional<GraphId>
    tryGetGraphId(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph)
{
    const auto* id = graph.getGraph().id();
    if(id == nullptr)
    {
        return std::nullopt;
    }
    return hipdnn_flatbuffers_sdk::utilities::toUuidBytes(*id);
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
