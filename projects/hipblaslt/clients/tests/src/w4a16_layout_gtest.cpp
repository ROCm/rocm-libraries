// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// Pins the scaleA allocation layout for the asymmetric w4a16 modes
/// (HIPBLASLT_MATMUL_MATRIX_SCALE_VEC{32,64,128}_ZP_EXT): scales first, then the
/// packed int4 zero-points at the next 256-byte boundary.
///
/// The layout is computed in two places that have to agree byte for byte --
/// w4a16::zeroPointOffset here on the client side, which is what the tests and
/// samples write, and c_blockScaleAZeroPointAlignment in tensile_host.cpp, which
/// is what the library reads to derive the pointer it hands the kernel. They
/// cannot share a definition because one is library-internal, so this fixes the
/// client half against the documented contract; a divergence in the other half
/// shows up as wrong results in the matmul_w4a16 cases.

#include "w4a16_datagen.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{
    /// The boundary the public header documents on
    /// HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_ZP_EXT.
    constexpr size_t c_documentedAlignment = 256;

    struct Shape
    {
        int64_t m;
        int64_t k;
        int     groupSize;
    };

    int64_t kGroupsOf(const Shape& s)
    {
        return (s.k + s.groupSize - 1) / s.groupSize;
    }

    size_t scaleRegionBytes(const Shape& s)
    {
        return static_cast<size_t>(s.m) * static_cast<size_t>(kGroupsOf(s)) * 2;
    }

    // Sizes the w4a16 gtest cases use, plus ones chosen to land the scale region
    // just short of, exactly on, and just past a 256-byte boundary.
    const Shape shapes[] = {
        {64, 256, 32},
        {128, 512, 32},
        {192, 256, 64},
        {65, 256, 64},      // odd M: the zero-point rows pad to an even count
        {128, 320, 128},
        {128, 384, 128},    // K an odd number of groups
        {2560, 32896, 128}, // a realistic weight matrix
        {1, 32, 32},        // scales are 2 bytes: far short of one boundary
        {128, 32, 32},      // scales are exactly 256 bytes
        {129, 32, 32},      // one element past a boundary
        {2049, 8192, 128},
    };
}

TEST(w4a16_layout, zero_points_start_on_a_256_byte_boundary)
{
    for(const auto& s : shapes)
    {
        const size_t off = w4a16::zeroPointOffset(s.m, kGroupsOf(s));
        EXPECT_EQ(off % c_documentedAlignment, 0u)
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize
            << ": zero-point offset " << off << " is not " << c_documentedAlignment
            << "-byte aligned";
    }
}

TEST(w4a16_layout, zero_points_start_past_the_scales)
{
    // Rounding up must never overlap the scale region it follows.
    for(const auto& s : shapes)
    {
        const size_t scales = scaleRegionBytes(s);
        const size_t off    = w4a16::zeroPointOffset(s.m, kGroupsOf(s));
        EXPECT_GE(off, scales) << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
        // ...and never waste a whole boundary doing it.
        EXPECT_LT(off - scales, c_documentedAlignment)
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
    }
}

TEST(w4a16_layout, allocation_covers_both_regions)
{
    for(const auto& s : shapes)
    {
        const int64_t kGroups = kGroupsOf(s);
        const size_t  zpBytes = static_cast<size_t>((s.m + 1) / 2) * static_cast<size_t>(kGroups);
        EXPECT_EQ(w4a16::scaleBytes(s.m, kGroups, true),
                  w4a16::zeroPointOffset(s.m, kGroups) + zpBytes)
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
        // Without zero-points there is no padding to account for.
        EXPECT_EQ(w4a16::scaleBytes(s.m, kGroups, false), scaleRegionBytes(s))
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
    }
}
