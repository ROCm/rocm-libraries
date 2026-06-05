// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hip/hip_runtime.h>

#include <string>

namespace hipdnn_integration_tests
{

// Returns the raw gcnArchName string for device 0 (e.g. "gfx942:sramecc+:xnack-").
// The string is returned verbatim, with no parsing of the suffix flags, so that
// callers can substring-match against either bare arches ("gfx942") or fully
// qualified strings ("gfx942:xnack-") without depending on the exact format
// ROCm uses today.
//
// Returns an empty string if the device cannot be queried (e.g. no GPU
// present, driver missing). Callers should treat empty as "skip rules
// disabled" rather than as an error — integration tests run unmodified.
inline std::string currentDeviceArchRaw()
{
    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, 0) != hipSuccess)
    {
        return {};
    }
    return {props.gcnArchName};
}

} // namespace hipdnn_integration_tests
