// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipblaslt::hostnumerics
{
    // Kernel scale-storage selection shared by the hipBLASLt and TensileLite
    // client adapters. The adapters orient scales and choose a HostNumerics plan.
    enum class MxScaleStorageLayout
    {
        Natural,
        Tile32x8,
        FastDimensionTiles,
    };
} // namespace hipblaslt::hostnumerics
