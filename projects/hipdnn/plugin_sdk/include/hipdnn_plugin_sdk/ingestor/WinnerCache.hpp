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

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One benchmarked candidate. `packId`/`dispatchId` travel with `kernelId` as a staleness
/// cross-check: a pack can be replaced between runs, making the same id a different
/// kernel.
///
/// `timeMs` is diagnostic only -- order is the decision. Times are never comparable
/// across runs or records.
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

/// Graph content plus device. Knobs are deliberately absent: a knob filter narrows which
/// candidates a run considers, never what the graph computes, so runs with different
/// filters share a record -- which is why the coverage gate exists.
struct WinnerKey
{
    GraphContentKey graph;
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
        const size_t graphHash = std::hash<GraphContentKey>{}(key.graph);
        const size_t deviceHash = std::hash<DeviceKey>{}(key.device);
        return graphHash
               ^ (deviceHash + 0x9e3779b97f4a7c15ULL + (graphHash << 6) + (graphHash >> 2));
    }
};

/// Does @p record carry a measurement for every kernel in @p kernels?
///
/// One-directional: entries in @p record absent from @p kernels do not fail coverage,
/// so a record wider than the current candidate set still serves. Used directly only by
/// `orderIfFullyCovered` below and by tests; production coverage decisions go through
/// that helper so coverage and orderability cannot be checked independently and drift
/// apart.
inline bool recordCovers(const WinnerRecord& record, const std::vector<KernelDefinition>& kernels)
{
    return std::all_of(kernels.begin(), kernels.end(), [&record](const KernelDefinition& kernel) {
        return std::any_of(record.begin(), record.end(), [&kernel](const RankedEntry& entry) {
            return entry.kernelId == kernel.kernelId;
        });
    });
}

/// Reorders @p kernels into @p record's ranked order, dropping any kernel the record does
/// not carry and any entry whose `packId`/`dispatchId` no longer agree (distinct from
/// `recordCovers`'s coverage check). A stale entry is skipped, not an error; an empty
/// result means the caller falls back to its normal path.
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
/// kernel and orders every one of them; nullopt otherwise. The two conditions can
/// diverge -- `recordCovers` matches by `kernelId` alone, `orderByRecord` requires the
/// full `(kernelId, packId, dispatchId)` triple -- so an entry covered by id but whose
/// pack has since moved must decline the whole record rather than silently serve the
/// entries that still resolve. The sole coverage-plus-order decision point; every
/// caller that needs to know if a record can serve a candidate set goes through this.
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
