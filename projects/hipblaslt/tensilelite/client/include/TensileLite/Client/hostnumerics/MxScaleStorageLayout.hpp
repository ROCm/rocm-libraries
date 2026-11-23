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
        // Block_32_UE8M0_128_4_EXT: 128-row by 4-K-block tiles of the natural
        // [MN, K blocks] scales (hipBLASLt mx_tile128x4.hpp). K-contiguous operands only.
        Tile128x4,
    };
} // namespace hipblaslt::hostnumerics
