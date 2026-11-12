// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    // The MX scale layout a kernel reads, stored as ProblemType::mxScaleFormat.
    // Contractions.py writes these values from the solution's MXScaleFormat.
    // HostPreSwizzle reads scales the host has already swizzled. NoSwizzle and
    // InMemorySwizzle read the scale tensor as the caller packs it.
    // HostPreSwizzle128x4 reads scales the host has shuffled in tiles of 128 rows
    // by 4 K blocks. Append only.
    enum class MXScaleFormat : int
    {
        NoSwizzle           = 0,
        HostPreSwizzle      = 1,
        InMemorySwizzle     = 2,
        HostPreSwizzle128x4 = 3,
    };
}
