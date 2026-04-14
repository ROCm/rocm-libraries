// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile::direct_conv
{

// LDS swizzle for global -> LDS -> MFMA register mapping.
//
// Global memory tensor is NHWC with 16-bit elements.
// Global -> LDS loads use uint4 elements (128-bit, 8C) and write to contiguous addresses in LDS.
// LDS -> mfma(16x16) loads 4C (uint2) per thread, row=lane%16, C4=lane/16.
//
// C_ is the number of channels in the tile (BLOCK_C).
template <int C_>
struct SwizzleT
{
    static constexpr int C = C_;

    static constexpr int C16 = C / 16;
    static constexpr int C8  = C / 8;
    static constexpr int C4  = C / 4;

    // Map the (x, c4) coordinates to an offset in LDS.
    //
    // The offset is in units of uint2 (64-bit), appropriate for LDS -> register loading.
    static constexpr int offset_uint2(int x, int c4)
    {
        auto c4m2      = c4 % 2;
        auto c8        = c4 / 2;
        auto offset_x  = x * C4;
        auto offset_c8 = (x + c8) % C8;

        return offset_x + offset_c8 * 2 + c4m2;
    }

    // Map (x, c8) coordinates to a uint4 offset in LDS.
    // Inverse of x() / c8(); consistent with offset_uint2(x, c8*2) / 2.
    static constexpr int offset_uint4(int x, int c8) { return offset_uint2(x, c8 * 2) / 2; }

    // Map an offset in LDS to its x-coordinate.
    //
    // The offset is in units of uint4 (128-bit), appropriate for Global -> LDS loading.
    static constexpr int x(int offset_uint4) { return offset_uint4 / C8; }

    // Map an offset in LDS to its c8-coordinate.
    //
    // The offset is in units of uint4 (128-bit), appropriate for Global -> LDS loading.
    static constexpr int c8(int offset_uint4)
    {
        auto x  = SwizzleT::x(offset_uint4);
        auto c8 = (offset_uint4 + C8 - x) % C8;
        return c8;
    }
};

} // namespace ck_tile::direct_conv
