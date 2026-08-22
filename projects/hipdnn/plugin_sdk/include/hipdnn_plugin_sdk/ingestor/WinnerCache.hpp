// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
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

/// Do @p left and @p right rank the same candidates in the same order? Compares the
/// `(kernelId, packId, dispatchId)` sequence positionally and ignores `timeMs`.
///
/// This is the write-back supersession test (see `writeBackToShard()`). The two obvious
/// alternatives are both wrong. Comparing only the kernel-id *set* would treat a genuine
/// reordering -- a driver or firmware change, a different thermal or clock regime, two
/// kernels that actually swapped -- as no change and discard it forever, and here the
/// order IS the payload. Comparing whole records including `timeMs` would never match,
/// because a measured float essentially never repeats bit-for-bit, so every benchmarking
/// run of an unchanged catalog would append a line.
inline bool rankedIdsEqual(const WinnerRecord& left, const WinnerRecord& right)
{
    return std::equal(left.begin(),
                      left.end(),
                      right.begin(),
                      right.end(),
                      [](const RankedEntry& a, const RankedEntry& b) {
                          return a.kernelId == b.kernelId && a.packId == b.packId
                                 && a.dispatchId == b.dispatchId;
                      });
}

/// Does @p record carry a measurement for every kernel in @p kernels? One-directional:
/// entries in @p record absent from @p kernels do not fail coverage. Production
/// decisions go through `orderIfFullyCovered` below, not this directly.
inline bool recordCovers(const WinnerRecord& record, const std::vector<KernelDefinition>& kernels)
{
    return std::all_of(kernels.begin(), kernels.end(), [&record](const KernelDefinition& kernel) {
        return std::any_of(record.begin(), record.end(), [&kernel](const RankedEntry& entry) {
            return entry.kernelId == kernel.kernelId;
        });
    });
}

/// Reorders @p kernels into @p record's ranked order, dropping any kernel the record
/// does not carry and any entry whose `packId`/`dispatchId` no longer agree (distinct
/// from `recordCovers`'s coverage check). An empty result means the caller falls back.
inline std::vector<KernelDefinition> orderByRecord(const WinnerRecord& record,
                                                   const std::vector<KernelDefinition>& kernels)
{
    std::vector<KernelDefinition> ordered;
    ordered.reserve(kernels.size());
    for(const auto& entry : record)
    {
        const auto match = std::find_if(
            kernels.begin(), kernels.end(), [&entry](const KernelDefinition& kernel) {
                return kernel.kernelId == entry.kernelId && kernel.packId == entry.packId
                       && kernel.dispatchId == entry.dispatchId;
            });
        if(match != kernels.end())
        {
            ordered.push_back(*match);
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
