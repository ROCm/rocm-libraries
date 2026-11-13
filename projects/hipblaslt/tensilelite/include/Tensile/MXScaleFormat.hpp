// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    // The MX scale layout a kernel reads, stored as ProblemType::mxScaleFormat.
    // Contractions.py writes these values from the solution's MXScaleFormat.
    // HostPreSwizzle reads scales the host has already swizzled. NoSwizzle and
    // InMemorySwizzle read the scale tensor as the caller packs it.
    // HostPreSwizzle128x4 reads scales the host shuffled in 128-row by 4-K-block
    // tiles (HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_128_4_EXT). Append only.
    enum class MXScaleFormat : int
    {
        NoSwizzle           = 0,
        HostPreSwizzle      = 1,
        InMemorySwizzle     = 2,
        HostPreSwizzle128x4 = 3,
    };
}
