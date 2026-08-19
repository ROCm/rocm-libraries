// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/DeviceKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphContentKey.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// One benchmarked candidate. `packId` and `dispatchId` travel with `kernelId` as a
/// staleness cross-check: a descriptor pack can be replaced between runs, and a kernel id
/// that now resolves to a different pack is a different kernel wearing an old name.
///
/// `timeMs` is a **diagnostic only**. Order is the decision. A time measured on a loaded
/// machine is not comparable to one measured idle, so nothing may compare times across
/// runs or across records -- the ranking was fixed when the record was written.
struct RankedEntry
{
    DescriptorId kernelId{};
    DescriptorId packId{};
    DescriptorId dispatchId{};
    double timeMs = 0.0;
};

/// Every usable candidate in benchmarked order, best first. Candidates that failed
/// sampling are omitted rather than ranked last: recording a known-broken kernel as a
/// fallback would let it be served ahead of the normal ranked path.
using WinnerRecord = std::vector<RankedEntry>;

/// Graph content plus device. Knobs are deliberately absent -- a knob filter narrows
/// which candidates a run considers, never what the graph computes, so two runs with
/// different filters share a record. That is what makes the coverage gate necessary, and
/// it is also what makes a narrowed run able to reuse a wider run's measurements.
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
        // The device hash is the cheap discriminator, so mix it into the graph's rather
        // than the other way around; both halves are already folded.
        const size_t graphHash = std::hash<GraphContentKey>{}(key.graph);
        const size_t deviceHash = std::hash<DeviceKey>{}(key.device);
        return graphHash
               ^ (deviceHash + 0x9e3779b97f4a7c15ULL + (graphHash << 6) + (graphHash >> 2));
    }
};

/// Does @p record carry a measurement for every kernel in @p kernels?
///
/// The one coverage predicate, called against two different sets at two different times:
///
///  - **Check 1**, inside `sortedCatalog`, against the whole catalog. Passing means the
///    record can order the catalog outright and the heuristic is never consulted.
///  - **Check 2**, at the lookup site, against the knob-filtered candidates. Passing means
///    the record can be served without re-benchmarking.
///
/// One predicate, deliberately. Two near-identical helpers -- even one that looks like a
/// harmless specialization for the whole-catalog case -- would drift, and the two checks
/// would silently stop agreeing about what "covered" means.
///
/// Note the asymmetry, which is not a bug: entries in the record that are absent from
/// @p kernels do not fail coverage. A record may be wider than the current candidate set
/// (a narrower knob filter, a removed pack), and treating that as uncovered would
/// re-benchmark on every narrowed filter and destroy the reuse the ranked list exists for.
inline bool recordCovers(const WinnerRecord& record, const std::vector<KernelDefinition>& kernels)
{
    return std::all_of(kernels.begin(), kernels.end(), [&record](const KernelDefinition& kernel) {
        return std::any_of(record.begin(), record.end(), [&kernel](const RankedEntry& entry) {
            return entry.kernelId == kernel.kernelId;
        });
    });
}

/// Reorders @p kernels into @p record's ranked order, dropping any kernel the record does
/// not carry and any entry whose `packId`/`dispatchId` no longer agree.
///
/// The staleness check lives here rather than in `recordCovers` because it is a distinct
/// question: coverage asks *was this kernel measured*, agreement asks *is it still the
/// same kernel*. A stale entry is skipped, never an error -- the next ranked entry is
/// tried, and an empty result means the caller falls back to its normal path.
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

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
