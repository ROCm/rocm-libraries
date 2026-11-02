// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

// True only for gfx950 packed block-32 E8 scales (Scale::Block_32_UE8M0,
// bench --scaleA/--scaleB 3). Pre-swizzled scales (Scale::Block_32_UE8M0_32_8_EXT,
// bench 1001) return false, as does every other scale and every other arch.
//
// Scale is RocblasltContractionProblem::ScalingFormat. The scale compare is
// first so a non-packed GEMM does not scan the arch name. The caller passes
// the arch already stored on the handle. This function does not call
// hipGetDevice or hipGetDeviceProperties.
template <typename Scale>
inline bool rocblaslt_reject_gfx950_packed_scale(const char* arch, Scale scaleA, Scale scaleB)
{
    if(scaleA != Scale::Block_32_UE8M0 && scaleB != Scale::Block_32_UE8M0)
        return false;
    if(arch == nullptr)
        return false;
    return std::string_view(arch).find("gfx950") != std::string_view::npos;
}
