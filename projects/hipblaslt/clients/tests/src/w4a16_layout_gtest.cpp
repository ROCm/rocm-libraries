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
#include <cstring>

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
        {65, 256, 64},      // odd M: zero-point rows pad to a multiple of eight
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
        const size_t  zpBytes = static_cast<size_t>((s.m + 7) / 8) * static_cast<size_t>(kGroups) * 4;
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

TEST(w4a16_layout, eight_row_words_match_dequantized_reference)
{
    for(int64_t m : {1, 7, 8, 9, 65})
        for(int groupSize : {32, 64, 128})
            for(auto dtype : {HIP_R_16F, HIP_R_16BF})
                for(auto encoding : {HIPBLASLT_INT4_ENCODING_SIGNED_EXT,
                                     HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXT})
                {
                    constexpr int64_t k = 384;
                    const int64_t groups = k / groupSize;
                    const size_t offset = (m * groups * 2 + 255) & ~size_t(255);
                    const size_t bytes = offset + ((m + 7) / 8) * groups * 4;
                    std::vector<uint8_t> packed(m * k / 2, 0);
                    std::vector<uint32_t> allocation((bytes + 4) / 4, 0);
                    allocation.back() = 0xdeadbeef;
                    const auto reference = generateW4A16Input(packed.data(), allocation.data(),
                        dtype, m, k, k, groupSize, true, encoding);
                    const auto* scales = reinterpret_cast<const uint8_t*>(allocation.data());
                    for(int64_t row = 0; row < m; ++row)
                        for(int64_t col = 0; col < k; ++col)
                        {
                            uint32_t word;
                            std::memcpy(&word, scales + offset
                                + 4 * ((row / 8) * groups + col / groupSize), 4);
                            int z = (word >> (4 * (row % 8))) & 15;
                            const size_t element = row * k + col;
                            int q = (packed[element / 2] >> (4 * (element % 2))) & 15;
                            if(encoding == HIPBLASLT_INT4_ENCODING_SIGNED_EXT)
                            {
                                q = (q ^ 8) - 8;
                                z = (z ^ 8) - 8;
                            }
                            const float scale = w4a16::loadAs(scales,
                                row * groups + col / groupSize, dtype);
                            const float value = (q - z) * scale;
                            const float rounded = dtype == HIP_R_16F
                                ? float(hipblasLtHalf(value)) : float(hip_bfloat16(value));
                            ASSERT_EQ(reference[element], rounded)
                                << "m=" << m << " row=" << row << " col=" << col
                                << " group=" << groupSize << " dtype=" << dtype
                                << " encoding=" << encoding;
                        }
                    EXPECT_EQ(allocation.back(), 0xdeadbeef);
                }
}
