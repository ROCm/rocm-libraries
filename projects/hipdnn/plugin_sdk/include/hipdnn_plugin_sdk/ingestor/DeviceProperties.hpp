// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <string>

namespace hipdnn_plugin_sdk::ingestor
{

/**
 * @brief The device facts matching and dispatch may read, as the `$device.*` namespace
 *        (RFC 0017 §5).
 *
 * Deliberately not `hipDeviceProp_t`. That type is a HIP runtime detail, and putting it
 * in this SDK's headers makes every consumer of the ingestor take a HIP dependency to
 * describe a device it may not resolve through HIP at all -- while also exposing ~90
 * fields as an implied contract when a descriptor can select on a handful. Naming the
 * fields is what lets that contract be reviewed, and what lets `$device.*` be a closed
 * vocabulary rather than "whatever HIP happens to expose this release".
 *
 * A provider fills this in from whatever it resolves devices with; the two reference
 * packs' resolver reads `hipDeviceProp_t` and copies across.
 *
 * Grows by adding a field here and populating it, which is a reviewable change to a
 * named surface. Adding one is cheap; removing one is not, since a descriptor may
 * reference it by name once the criteria language lands.
 */
struct DeviceProperties
{
    /**
     * @brief The GFX target, raw and suffix-intact (e.g. `"gfx942:sramecc+:xnack-"`).
     *
     * Kept raw rather than pre-stripped because the suffix is meaningful to a compile:
     * `--offload-arch` wants exactly this string. Compare it with
     * `hipdnn_plugin_sdk::archMatches` (ArchMatch.hpp) rather than `==`, or a pack
     * naming `gfx942` never matches a real device.
     */
    std::string gcnArchName;

    /// Threads per wavefront. 0 when no device could be resolved.
    int warpSize = 0;

    /// Compute units. 0 when no device could be resolved.
    int multiProcessorCount = 0;
};

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
