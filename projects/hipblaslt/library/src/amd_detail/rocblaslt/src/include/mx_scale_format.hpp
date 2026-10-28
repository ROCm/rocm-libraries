// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// Library ScalingFormat ordinals. These are not the public
// HIPBLASLT_MATMUL_MATRIX_SCALE_* integers. tensile_host.cpp static_asserts
// them against RocblasltContractionProblem::ScalingFormat.
inline constexpr int kRocblasltScaleBlock32UE8M0    = 3; // Block_32_UE8M0
inline constexpr int kRocblasltScaleBlock32UE8M0Ext = 9; // Block_32_UE8M0_32_8_EXT

// Request codes written onto the Tensile problem. -1 matches every kernel
// layout, -2 matches none, and 1 matches HostPreSwizzle only.
inline constexpr int kMxScaleRequestUnspecified    = -1;
inline constexpr int kMxScaleRequestUnsupported    = -2;
inline constexpr int kMxScaleRequestHostPreSwizzle = 1;

// gfx950 scale format 3 (Block_32_UE8M0) cannot use a shipped HostPreSwizzle
// kernel or a canonical NoSwizzle kernel. Format 1001 still requests
// HostPreSwizzle. Every other architecture, including gfx1250 scale 3, stays
// unspecified so the existing kernels still match.
inline int rocblaslt_mx_scale_format_request(const std::string& arch, int scaleA, int scaleB)
{
    if(arch.find("gfx950") == std::string::npos)
        return kMxScaleRequestUnspecified;

    if(scaleA == kRocblasltScaleBlock32UE8M0 || scaleB == kRocblasltScaleBlock32UE8M0)
        return kMxScaleRequestUnsupported;

    if(scaleA == kRocblasltScaleBlock32UE8M0Ext || scaleB == kRocblasltScaleBlock32UE8M0Ext)
        return kMxScaleRequestHostPreSwizzle;

    return kMxScaleRequestUnspecified;
}
