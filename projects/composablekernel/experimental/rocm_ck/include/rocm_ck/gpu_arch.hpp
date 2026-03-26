// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — GPU architecture detection. Requires HIP runtime.
//
// GPU architecture detection utility for rocm_ck examples.

#pragma once

#include <hip/hip_runtime.h>

#include <string>

namespace rocm_ck {

/// Returns the base GPU architecture name (e.g. "gfx942") for the given device.
/// Strips feature flags from HIP's full ISA string (e.g. "gfx942:sramecc+:xnack-").
/// Returns empty string on failure — caller decides error policy.
inline std::string get_gpu_arch(int device_id = 0)
{
    hipDeviceProp_t device_props;
    if(hipGetDeviceProperties(&device_props, device_id) != hipSuccess)
    {
        return {};
    }

    std::string arch = device_props.gcnArchName;
    size_t colon_pos = arch.find(':');
    if(colon_pos != std::string::npos)
    {
        arch = arch.substr(0, colon_pos);
    }
    return arch;
}

} // namespace rocm_ck
