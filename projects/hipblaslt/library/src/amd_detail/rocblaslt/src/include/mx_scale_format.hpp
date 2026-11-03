// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

// Layout a gfx950 TensileLite kernel must have for these scale types.
// -1 means do not filter. 0 is NoSwizzle, the layout packed Block_32_UE8M0
// (bench scale 3) can use. 1 is HostPreSwizzle, the layout
// Block_32_UE8M0_32_8_EXT (bench scale 1001) uses. -2 means the two sides
// ask for different layouts, so no kernel matches.
//
// Scale types are compared first. A non-packed, non-pre-swizzled GEMM does
// not scan the arch name. The arch is the one cached on the handle. This
// function does not call hipGetDevice or hipGetDeviceProperties.
template <typename Scale>
inline int rocblaslt_gfx950_required_mx_scale_format(const char* arch, Scale scaleA, Scale scaleB)
{
    const bool packedA = scaleA == Scale::Block_32_UE8M0;
    const bool packedB = scaleB == Scale::Block_32_UE8M0;
    const bool preA    = scaleA == Scale::Block_32_UE8M0_32_8_EXT;
    const bool preB    = scaleB == Scale::Block_32_UE8M0_32_8_EXT;
    if(!packedA && !packedB && !preA && !preB)
        return -1;
    if(arch == nullptr || std::string_view(arch).find("gfx950") == std::string_view::npos)
        return -1;
    if((packedA || packedB) && (preA || preB))
        return -2;
    if(packedA || packedB)
        return 0;
    return 1;
}

inline bool rocblaslt_kernel_mx_scale_format_ok(int required, int kernelFormat)
{
    if(required == -1)
        return true;
    return kernelFormat == required;
}

inline const char* rocblaslt_mx_layout_reject_message(int required)
{
    if(required == 0)
        return "gfx950 Block_32_UE8M0 does not use a HostPreSwizzle TensileLite kernel";
    if(required == 1)
        return "gfx950 Block_32_UE8M0_32_8_EXT requires a HostPreSwizzle TensileLite kernel";
    return "gfx950 scale A and scale B ask for different TensileLite scale layouts";
}
