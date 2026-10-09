// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One benchmarked candidate. `packId`/`dispatchId` travel with `kernelId` as a staleness
/// cross-check: a pack can be replaced between runs, making the same id a different
/// kernel. `timeMs` is diagnostic only; times are never comparable across runs or
/// records.
struct RankedEntry
{
    DescriptorId kernelId{};
    DescriptorId packId{};
    DescriptorId dispatchId{};
    double timeMs = 0.0;
};

/// Every usable candidate in benchmarked order, best first. Failed candidates are
/// omitted rather than ranked last, so a known-broken kernel is never served as a
/// fallback.
using WinnerRecord = std::vector<RankedEntry>;

/// Graph content plus device. Knobs are absent: a knob filter narrows which candidates
/// a run considers, never what the graph computes, so runs with different filters share
/// a record, which is why the coverage gate exists.
struct WinnerKey
{
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey graph;
    DeviceKey device;

    bool operator==(const WinnerKey& other) const
    {
        return device == other.device && graph == other.graph;
    }

    bool operator!=(const WinnerKey& other) const
    {
        return !(*this == other);
    }
};

struct WinnerKeyHash
{
    size_t operator()(const WinnerKey& key) const noexcept
    {
        const size_t graphHash
            = std::hash<hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphContentKey>{}(key.graph);
        const size_t deviceHash = std::hash<DeviceKey>{}(key.device);
        return graphHash
               ^ (deviceHash + 0x9e3779b97f4a7c15ULL + (graphHash << 6) + (graphHash >> 2));
    }
};

/// Why a benchmarked ranking is being written back, which is what separates the two rules
/// below: a fresh miss adopts an existing entry for the key, and a re-benchmark triggered
/// by a record that did not cover the current candidates appends a superseding line.
enum class WinnerWriteCause
{
    /// No record existed for this key when the plan was built.
    FRESH_MISS,
    /// A record existed but was not usable for the current candidate set, so it was re-measured.
    COVERAGE_REBENCHMARK,
};

/// Does @p record carry a measurement for every kernel in @p kernels? One-directional:
/// entries in @p record absent from @p kernels do not fail coverage. Production
/// decisions go through `orderIfFullyCovered` below, not this directly.
///
/// Linear in `record.size() + kernels.size()`: the record's ids are hashed once rather
/// than scanned per kernel, since a record can cover tens of thousands of candidates.
inline bool recordCovers(const WinnerRecord& record, const std::vector<KernelDefinition>& kernels)
{
    if(kernels.empty())
    {
        return true;
    }
    std::unordered_set<DescriptorId, DescriptorIdHash> recorded;
    recorded.reserve(record.size());
    for(const auto& entry : record)
    {
        recorded.insert(entry.kernelId);
    }
    return std::all_of(kernels.begin(), kernels.end(), [&recorded](const KernelDefinition& kernel) {
        return recorded.find(kernel.kernelId) != recorded.end();
    });
}

namespace detail
{

/// The `(kernelId, packId, dispatchId)` triple `orderByRecord` matches on.
struct RecordedKernelKey
{
    DescriptorId kernelId{};
    DescriptorId packId{};
    DescriptorId dispatchId{};

    bool operator==(const RecordedKernelKey& other) const
    {
        return kernelId == other.kernelId && packId == other.packId
               && dispatchId == other.dispatchId;
    }
};

struct RecordedKernelKeyHash
{
    size_t operator()(const RecordedKernelKey& key) const noexcept
    {
        const DescriptorIdHash hash;
        size_t seed = hash(key.kernelId);
        for(const size_t part : {hash(key.packId), hash(key.dispatchId)})
        {
            seed ^= part + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

} // namespace detail

/// Reorders @p kernels into @p record's ranked order, dropping any kernel the record
/// does not carry and any entry whose `packId`/`dispatchId` no longer agree (distinct
/// from `recordCovers`'s coverage check). An empty result means the caller falls back.
///
/// Each record entry resolves to the FIRST kernel in @p kernels carrying its triple, and
/// an entry repeated in the record repeats its kernel, exactly as a linear scan per entry
/// would; the kernels are indexed once so the whole call is linear rather than quadratic.
inline std::vector<KernelDefinition> orderByRecord(const WinnerRecord& record,
                                                   const std::vector<KernelDefinition>& kernels)
{
    std::vector<KernelDefinition> ordered;
    ordered.reserve(kernels.size());
    if(record.empty() || kernels.empty())
    {
        return ordered;
    }

    std::unordered_map<detail::RecordedKernelKey, size_t, detail::RecordedKernelKeyHash> firstIndex;
    firstIndex.reserve(kernels.size());
    for(size_t index = 0; index < kernels.size(); ++index)
    {
        const auto& kernel = kernels[index];
        // emplace keeps the first index on a repeat, matching find_if's first hit.
        firstIndex.emplace(
            detail::RecordedKernelKey{kernel.kernelId, kernel.packId, kernel.dispatchId}, index);
    }

    for(const auto& entry : record)
    {
        const auto match = firstIndex.find(
            detail::RecordedKernelKey{entry.kernelId, entry.packId, entry.dispatchId});
        if(match != firstIndex.end())
        {
            ordered.push_back(kernels[match->second]);
        }
    }
    return ordered;
}

/// Returns @p record's order over @p kernels only when the record both covers every
/// kernel and orders every one of them; nullopt otherwise. The two can diverge --
/// `recordCovers` matches by `kernelId` alone, `orderByRecord` requires the full
/// `(kernelId, packId, dispatchId)` triple -- so an entry covered by id but whose pack
/// has since moved declines the whole record rather than serving what still resolves.
inline std::optional<std::vector<KernelDefinition>>
    orderIfFullyCovered(const WinnerRecord& record, const std::vector<KernelDefinition>& kernels)
{
    if(!recordCovers(record, kernels))
    {
        return std::nullopt;
    }
    auto ordered = orderByRecord(record, kernels);
    if(ordered.size() != kernels.size())
    {
        return std::nullopt;
    }
    return ordered;
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
