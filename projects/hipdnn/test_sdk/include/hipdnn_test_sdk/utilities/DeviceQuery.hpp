// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>

#include <hip/hip_runtime.h>

namespace hipdnn_test_sdk::utilities
{

/// Single source of truth for querying the GPU a test runs on.
///
/// Both helpers query the current HIP device at call time (not a hardcoded
/// device 0). To target a specific GPU, set HIP_VISIBLE_DEVICES: the HIP runtime
/// exposes the chosen GPU as device 0, so these queries pick it up automatically
/// (no remapping on our side).

/// Raw gcnArchName string for the current device (e.g. "gfx942:sramecc+:xnack-").
/// Returned verbatim, with no parsing of the suffix flags, so callers can
/// substring-match against bare arches ("gfx942") or fully qualified strings
/// ("gfx942:xnack-") without depending on the exact format ROCm uses today.
///
/// Returns an empty string if the device cannot be queried (e.g. no GPU present,
/// driver missing). Callers should treat empty as "checks disabled" rather than
/// an error.
inline std::string currentDeviceArch()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        return {};
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) != hipSuccess)
    {
        return {};
    }
    return {props.gcnArchName};
}

/// Total VRAM in megabytes for the current device.
/// Returns 0 if the device cannot be queried.
inline std::size_t currentDeviceTotalVramMb()
{
    int device = 0;
    if(hipGetDevice(&device) != hipSuccess)
    {
        return 0;
    }
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device) == hipSuccess)
    {
        return props.totalGlobalMem / (std::size_t{1024} * 1024);
    }
    return 0;
}

} // namespace hipdnn_test_sdk::utilities
