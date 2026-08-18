// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

#include <hipdnn_plugin_sdk/ArchMatch.hpp>

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
};

/// Does @p arch (a KDP's supported-target list; empty admits everything) admit
/// @p deviceArch? PREFIX match on the base identifier, not SUBSTRING, so `gfx942`
/// never silently admits `gfx950`.
inline bool archSupports(const std::vector<std::string>& arch, std::string_view deviceArch)
{
    return arch.empty()
           || std::any_of(arch.begin(), arch.end(), [deviceArch](const std::string& candidate) {
                  return archMatches(deviceArch, candidate, ArchMatchMode::PREFIX);
              });
}

/// Can one device satisfy both @p a and @p b? Empty means "every arch", so it overlaps
/// everything. Entries may carry feature suffixes (`gfx942:sramecc+`), so this is the
/// symmetric PREFIX test, not string equality: `gfx942` and `gfx942:sramecc+` are both
/// satisfied by a device reporting `gfx942:sramecc+:xnack-`.
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

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
