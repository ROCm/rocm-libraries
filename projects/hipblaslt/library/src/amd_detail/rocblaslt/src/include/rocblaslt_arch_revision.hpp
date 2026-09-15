// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

// The GEMM library subtree a device loads. gfx1250's two revisions share one ISA
// and compiler target, so only hipDeviceProp_t::asicRevision tells them apart
// (v0 -> 0): a v0 part loads library/gfx1250v0/ only (no fallback to gfx1250 --
// the revisions are independent, not a subset). Everything else is unchanged.
// Dependency-free so it can be unit-tested GPU-free.
inline std::string rocblaslt_revisioned_arch_name(const std::string& baseArch, int asicRevision)
{
    if(baseArch == "gfx1250" && asicRevision == 0)
        return "gfx1250v0";
    return baseArch;
}
