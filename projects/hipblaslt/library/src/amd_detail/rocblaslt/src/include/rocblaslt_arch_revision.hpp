// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// The GEMM library subtree a device loads. With HSA_DISABLE_GFX12_STRICT unset/1 an
// A0 part reports the same name as B0 (gfx1250), so only hipDeviceProp_t::asicRevision
// tells them apart (A0 -> 0): an A0 part loads library/gfx1250v0/ only (no fallback to
// gfx1250 -- B0 kernels are not valid on A0). gfx1250-strict and every other reported
// name are unchanged. Dependency-free so it can be unit-tested GPU-free.
inline std::string rocblaslt_revisioned_arch_name(const std::string& baseArch, int asicRevision)
{
    if(baseArch == "gfx1250" && asicRevision == 0)
        return "gfx1250v0";
    return baseArch;
}
