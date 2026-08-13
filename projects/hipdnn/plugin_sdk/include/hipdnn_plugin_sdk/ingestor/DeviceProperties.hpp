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

/**
 * @brief The device facts matching and dispatch may read, as the `$device.*` namespace
 *        (RFC 0017 §5).
 *
 * Deliberately not `hipDeviceProp_t`, which would make every ingestor consumer take a
 * HIP dependency and expose ~90 fields as an implied contract where a descriptor
 * selects on a handful. Naming the fields keeps `$device.*` a closed, reviewable
 * vocabulary rather than whatever HIP exposes this release.
 *
 * A provider populates this from whatever it resolves devices with. Adding a field is
 * cheap; removing one is not, since a descriptor may name it once the criteria
 * language lands.
 */
struct DeviceProperties
{
    /**
     * @brief The GFX target, raw and suffix-intact (e.g. `"gfx942:sramecc+:xnack-"`).
     *
     * Raw because `--offload-arch` wants exactly this string. Compare with
     * `hipdnn_plugin_sdk::archMatches` (ArchMatch.hpp), not `==`, or a pack naming
     * `gfx942` never matches a real device.
     */
    std::string gcnArchName;

    /// Threads per wavefront. 0 when no device could be resolved.
    int warpSize = 0;

    /// Compute units. 0 when no device could be resolved.
    int multiProcessorCount = 0;
};

/**
 * @brief Does @p arch admit @p deviceArch?
 *
 * @param arch A KDP's supported-target list. **Empty admits everything**, which is what
 *        an arch-independent pack ships.
 * @param deviceArch A device's raw `gcnArchName`, feature suffix and all.
 *
 * Compares on the base identifier, so a pack listing `gfx942` matches a device
 * reporting `gfx942:sramecc+:xnack-`. PREFIX mode, not SUBSTRING: `gfx942` must never
 * silently admit `gfx950`, and a pack serving a family lists every member. That is the
 * detail this helper exists to keep every caller from getting wrong.
 */
inline bool archSupports(const std::vector<std::string>& arch, std::string_view deviceArch)
{
    return arch.empty()
           || std::any_of(arch.begin(), arch.end(), [deviceArch](const std::string& candidate) {
                  return archMatches(deviceArch, candidate, ArchMatchMode::PREFIX);
              });
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
