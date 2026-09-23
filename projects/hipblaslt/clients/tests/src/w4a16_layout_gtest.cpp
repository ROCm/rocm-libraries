// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/// scaleA layout for the asymmetric w4a16 modes: scales first, then packed int4
/// zero-points at the next 256-byte boundary.
///
/// Computed in two places that must agree -- w4a16::zeroPointOffset here and
/// c_blockScaleAZeroPointAlignment in tensile_host.cpp -- which cannot share a
/// definition. This pins the client half; a divergence in the other shows up as
/// wrong matmul_w4a16 results.

#include "w4a16_datagen.hpp"

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <stdexcept>

#include <cstdint>

namespace
{
    /// As documented on HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_ZP_EXT.
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

    // The gtest sizes, plus ones landing the scale region short of, exactly on,
    // and just past a boundary.
    const Shape shapes[] = {
        {64, 256, 32},
        {128, 512, 32},
        {192, 256, 64},
        {65, 256, 64},      // odd M: zero-point rows pad to an even count
        {128, 320, 128},
        {128, 384, 128},    // odd number of K groups
        {2560, 32896, 128}, // a realistic weight matrix
        {1, 32, 32},        // scales far short of one boundary
        {128, 32, 32},      // scales exactly 256 bytes
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
    // Rounding up must not overlap the scales.
    for(const auto& s : shapes)
    {
        const size_t scales = scaleRegionBytes(s);
        const size_t off    = w4a16::zeroPointOffset(s.m, kGroupsOf(s));
        EXPECT_GE(off, scales) << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
        // ...nor waste a whole boundary.
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
        // No zero-points, no padding.
        EXPECT_EQ(w4a16::scaleBytes(s.m, kGroups, false), scaleRegionBytes(s))
            << "m=" << s.m << " k=" << s.k << " g=" << s.groupSize;
    }
}

TEST(w4a16_encoding, cpp_api_accepts_only_supported_encodings)
{
    hipblaslt_ext::GemmProblemType problem;
    for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                         HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
    {
        problem.setInt4EncodingA(encoding);
        EXPECT_EQ(problem.getInt4EncodingA(), encoding);
    }
    for(int value : {-1, 2, 3, 99})
    {
        EXPECT_THROW(problem.setInt4EncodingA(static_cast<hipblasLtInt4Encoding_t>(value)),
                     std::invalid_argument);
        EXPECT_EQ(problem.getInt4EncodingA(), HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT);
    }
}
