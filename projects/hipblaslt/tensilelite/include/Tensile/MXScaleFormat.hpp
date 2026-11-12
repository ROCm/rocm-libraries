// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    // The MX scale layout a kernel reads, stored as ProblemType::mxScaleFormat.
    // Contractions.py writes these values from the solution's MXScaleFormat.
    // HostPreSwizzle reads scales the host has already swizzled. NoSwizzle and
    // InMemorySwizzle read the scale tensor as the caller packs it.
    // InMemorySwizzleB128 (MXScaleLRB128 kernels) reads the InMemorySwizzle
    // layout with each K group's M/N padded to 128 and stored as
    // [M/N / 128][lane 32][block 4] dwords (DGen::preSwizzleScalesGFX1250B128).
    enum class MXScaleFormat : int
    {
        NoSwizzle           = 0,
        HostPreSwizzle      = 1,
        InMemorySwizzle     = 2,
        InMemorySwizzleB128 = 3,
    };
}
