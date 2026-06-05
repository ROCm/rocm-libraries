// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <string>

namespace hipdnn_integration_tests
{

// Returns the raw gcnArchName string for the current HIP device
// (e.g. "gfx942:sramecc+:xnack-"). The string is returned verbatim, with no
// parsing of the suffix flags, so that callers can substring-match against
// either bare arches ("gfx942") or fully qualified strings ("gfx942:xnack-")
// without depending on the exact format ROCm uses today.
//
// Queries the current HIP device at call time (not a hardcoded device 0). To run
// on a specific GPU, set HIP_VISIBLE_DEVICES: the HIP runtime then exposes only
// the chosen GPU, as device 0, so this query picks it up automatically (we do no
// remapping ourselves). Note: TestConfig calls this once at startup and caches
// the result, so the skip system reflects the startup device, not a device
// switched mid-run. This matches the guards today since nothing calls
// hipSetDevice.
//
// Returns an empty string if the device cannot be queried (e.g. no GPU
// present, driver missing). Callers should treat empty as "skip rules
// disabled" rather than as an error — integration tests run unmodified.
inline std::string currentDeviceArchRaw()
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

} // namespace hipdnn_integration_tests
