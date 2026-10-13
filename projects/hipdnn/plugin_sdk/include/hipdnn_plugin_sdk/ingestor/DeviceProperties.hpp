// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>
#include <hipdnn_plugin_sdk/heuristics/DeviceFeatures.hpp>

namespace hipdnn_plugin_sdk::ingestor
{

/// The device facts matching and dispatch may read, as the `$device.*` namespace.
/// Deliberately not `hipDeviceProp_t`: named fields keep this a closed, reviewable
/// vocabulary instead of every consumer taking a HIP dependency.
struct DeviceProperties
{
    /// Raw GFX target, suffix intact (e.g. `"gfx942:sramecc+:xnack-"`). Compare with
    /// `hipdnn_plugin_sdk::archMatches`, not `==`.
    std::string gcnArchName;
    int warpSize = 0; ///< Threads per wavefront; 0 if unresolved.
    int multiProcessorCount = 0; ///< Compute units; 0 if unresolved.

    // One arch spans several boards (gfx942: MI300X, MI325X, MI308X), and boards with equal
    // CU counts can differ in memory, which changes the winner on bandwidth-bound shapes.
    /// HBM capacity in bytes; 0 if unresolved.
    std::size_t totalGlobalMem = 0;
    /// Memory bus width in bits; 0 if unresolved.
    int memoryBusWidth = 0;
    /// Peak memory clock in kHz; 0 if unresolved.
    int memoryClockRate = 0;
    /// LDS bytes available to one workgroup; 0 if unresolved.
    std::size_t sharedMemPerBlock = 0;
};

/// Theoretical peak HBM bandwidth in bytes/second, or 0 when either input is
/// unresolved. The factor of 2 is double data rate; kHz and bits convert to Hz and bytes.
inline double peakMemoryBandwidth(const DeviceProperties& properties) noexcept
{
    if(properties.memoryClockRate <= 0 || properties.memoryBusWidth <= 0)
    {
        return 0.0;
    }
    return 2.0 * static_cast<double>(properties.memoryClockRate) * 1000.0
           * (static_cast<double>(properties.memoryBusWidth) / 8.0);
}

/// The `$device.*` feature vocabulary as (name, value) pairs. The feature extractor and
/// the benchmark recorder must both use this list, or a model trains on columns the
/// runtime cannot produce. `arch` is absent: it selects which UHD runs (RFC 0019 §3.1).
inline std::vector<std::pair<std::string, std::variant<std::int64_t, double>>>
    deviceFeatureValues(const DeviceProperties& properties)
{
    return heuristics::deviceFeatureValues(properties);
}

/// Does @p arch (a KDP's supported-target list; empty admits everything) admit
/// @p deviceArch? Entries are base ids and the device carries its features, so this is
/// the PREFIX match, not SUBSTRING or equality: `gfx942` admits a device reporting
/// `gfx942:sramecc+:xnack-` and never admits `gfx950`.
inline bool archSupports(const std::vector<std::string>& arch, std::string_view deviceArch)
{
    return arch.empty()
           || std::any_of(arch.begin(), arch.end(), [deviceArch](const std::string& candidate) {
                  return archMatches(deviceArch, candidate, ArchMatchMode::PREFIX);
              });
}

/// Can one device satisfy both @p a and @p b? Empty means "every arch", so it overlaps
/// everything. Both sides are authored base ids, so an entry matches only its twin.
inline bool archOverlaps(const std::vector<std::string>& a, const std::vector<std::string>& b)
{
    if(a.empty() || b.empty())
    {
        return true;
    }
    return std::any_of(a.begin(), a.end(), [&b](const std::string& lhs) {
        return std::any_of(b.begin(), b.end(), [&lhs](const std::string& rhs) {
            return archMatches(lhs, rhs, ArchMatchMode::PREFIX)
                   || archMatches(rhs, lhs, ArchMatchMode::PREFIX);
        });
    });
}

/// Is every device @p inner admits also admitted by @p outer? The asymmetric counterpart
/// to archOverlaps, for asking whether a kernel stays within the pack that binds it.
/// Empty @p outer admits every device, so it covers anything; empty @p inner declares no
/// restriction of its own and is covered by anything. Otherwise every @p inner entry
/// must appear in @p outer: `[gfx942]` is covered by `[gfx942, gfx950]`, not the reverse.
inline bool archCovers(const std::vector<std::string>& outer, const std::vector<std::string>& inner)
{
    if(outer.empty())
    {
        return true;
    }
    return std::all_of(inner.begin(), inner.end(), [&outer](const std::string& entry) {
        return std::any_of(outer.begin(), outer.end(), [&entry](const std::string& candidate) {
            return archMatches(entry, candidate, ArchMatchMode::PREFIX);
        });
    });
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
