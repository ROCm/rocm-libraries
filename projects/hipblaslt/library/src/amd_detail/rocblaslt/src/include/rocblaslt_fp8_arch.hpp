// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string>

// Whether an architecture can execute FP8 (OCP E4M3FN/E5M2) GEMMs.
//
// CDNA3/CDNA4 (gfx942/gfx950) provide FP8 via MFMA; RDNA4 (gfx1200/gfx1201)
// provides the same OCP formats via WMMA. Split out from the arch-name lookup
// so the list is dependency-free and can be unit-tested GPU-free.
inline bool rocblaslt_arch_supports_fp8(const std::string& archName)
{
    using std::begin;
    using std::end;

    static const std::string fp8Archs[] = {"gfx942", "gfx950", "gfx1200", "gfx1201"};
    return std::find(begin(fp8Archs), end(fp8Archs), archName) != end(fp8Archs);
}
