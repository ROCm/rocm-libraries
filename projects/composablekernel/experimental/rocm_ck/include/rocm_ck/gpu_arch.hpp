// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — GPU architecture detection. Requires HIP runtime.
//
// GPU architecture detection utility for rocm_ck examples.

#pragma once

#include <rocm_ck/types.hpp>

#include <hip/hip_runtime.h>

#include <cstring>
#include <optional>
#include <string>

namespace rocm_ck {

/// Returns the base GPU architecture name (e.g. "gfx942") for the given device.
/// Strips feature flags from HIP's full ISA string (e.g. "gfx942:sramecc+:xnack-").
/// Returns empty string on failure — caller decides error policy.
inline std::string getGpuArch(int device_id = 0)
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

/// Detects the GpuTarget for the given device.
/// Maps HIP's ISA string (e.g. "gfx942:sramecc+:xnack-") to the GpuTarget enum.
/// Returns nullopt for unsupported architectures.
inline std::optional<GpuTarget> detectGpuTarget(int device_id = 0)
{
    hipDeviceProp_t device_props;
    if(hipGetDeviceProperties(&device_props, device_id) != hipSuccess)
        return std::nullopt;

    const char* arch = device_props.gcnArchName;

    if(std::strstr(arch, "gfx90a") != nullptr)
        return GpuTarget::gfx90a;
    if(std::strstr(arch, "gfx942") != nullptr)
        return GpuTarget::gfx942;
    if(std::strstr(arch, "gfx950") != nullptr)
        return GpuTarget::gfx950;
    if(std::strstr(arch, "gfx1100") != nullptr)
        return GpuTarget::gfx1100;
    if(std::strstr(arch, "gfx1101") != nullptr)
        return GpuTarget::gfx1101;
    if(std::strstr(arch, "gfx1102") != nullptr)
        return GpuTarget::gfx1102;
    if(std::strstr(arch, "gfx1150") != nullptr)
        return GpuTarget::gfx1150;
    if(std::strstr(arch, "gfx1151") != nullptr)
        return GpuTarget::gfx1151;
    if(std::strstr(arch, "gfx1200") != nullptr)
        return GpuTarget::gfx1200;
    if(std::strstr(arch, "gfx1201") != nullptr)
        return GpuTarget::gfx1201;

    return std::nullopt;
}

} // namespace rocm_ck
