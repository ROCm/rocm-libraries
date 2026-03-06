// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include <cmath>
#include <cstring>

#include "ck_tile/core.hpp"

using ck_tile::bit_cast;
using ck_tile::float_to_tf32;
using ck_tile::numeric_traits;
using ck_tile::tf32_rounding_mode;
using ck_tile::tf32_t;

static uint32_t to_bits(float x) { return bit_cast<uint32_t>(x); }
static float from_bits(uint32_t i) { return bit_cast<float>(i); }

TEST(Tf32, NumericTraits)
{
    EXPECT_EQ(numeric_traits<tf32_t>::exp, 8);
    EXPECT_EQ(numeric_traits<tf32_t>::mant, 10);
    EXPECT_EQ(numeric_traits<tf32_t>::bias, 127);
    EXPECT_EQ(numeric_traits<tf32_t>::PackedSize, 1);
}

TEST(Tf32, TruncBasic)
{
    // 1.0f = 0x3F800000 -> no low bits, stays the same
    EXPECT_EQ(float_to_tf32(1.0f), 1.0f);
    EXPECT_EQ(float_to_tf32(-1.0f), -1.0f);
    EXPECT_EQ(float_to_tf32(0.0f), 0.0f);
    EXPECT_EQ(float_to_tf32(-0.0f), -0.0f);
    EXPECT_EQ(float_to_tf32(2.0f), 2.0f);
    EXPECT_EQ(float_to_tf32(0.5f), 0.5f);
}

TEST(Tf32, TruncZerosLow13Bits)
{
    // 1.1f = 0x3F8CCCCD -> should become 0x3F8CC000
    float val       = 1.1f;
    uint32_t expect = to_bits(val) & 0xFFFFE000u;
    EXPECT_EQ(to_bits(float_to_tf32(val)), expect);

    // pi = 0x40490FDB -> should become 0x40490000
    val    = 3.14159265358979323846f;
    expect = to_bits(val) & 0xFFFFE000u;
    EXPECT_EQ(to_bits(float_to_tf32(val)), expect);

    // arbitrary value
    val    = 123.456f;
    expect = to_bits(val) & 0xFFFFE000u;
    EXPECT_EQ(to_bits(float_to_tf32(val)), expect);
}

TEST(Tf32, TruncSpecialValues)
{
    // +Inf
    float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ(float_to_tf32(inf), inf);

    // -Inf
    EXPECT_EQ(float_to_tf32(-inf), -inf);

    // NaN stays NaN (mantissa bits may change but exponent stays all-ones)
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    EXPECT_TRUE(std::isnan(float_to_tf32(nan_val)));

    // Subnormals: lower 13 bits get zeroed
    float subnormal = std::numeric_limits<float>::denorm_min();
    // denorm_min = 0x00000001, after masking -> 0x00000000 = 0.0f
    EXPECT_EQ(float_to_tf32(subnormal), 0.0f);
}

TEST(Tf32, RtneBasic)
{
    // Values that are exact in TF32 stay the same
    EXPECT_EQ((float_to_tf32<tf32_rounding_mode::rne>(1.0f)), 1.0f);
    EXPECT_EQ((float_to_tf32<tf32_rounding_mode::rne>(-1.0f)), -1.0f);
    EXPECT_EQ((float_to_tf32<tf32_rounding_mode::rne>(0.0f)), 0.0f);
}

TEST(Tf32, RtneRoundsUp)
{
    // Construct a value where rounding should go up:
    // Start from 1.0 (0x3F800000) and set bit 12 (just above the truncation boundary)
    // plus bit 11 to push past the midpoint
    uint32_t base  = 0x3F800000u; // 1.0f
    uint32_t val_i = base | (1u << 12) | (1u << 11);
    float val      = from_bits(val_i);

    float trunc_result = float_to_tf32<tf32_rounding_mode::trunc>(val);
    float rne_result   = float_to_tf32<tf32_rounding_mode::rne>(val);

    // RTNE should round up (away from truncated value)
    EXPECT_NE(to_bits(rne_result), to_bits(trunc_result));

    // RNE result should have low 13 bits zero
    EXPECT_EQ(to_bits(rne_result) & 0x1FFFu, 0u);
}

TEST(Tf32, RtneSpecialValues)
{
    // Inf/NaN should pass through unchanged
    float inf = std::numeric_limits<float>::infinity();
    EXPECT_EQ((float_to_tf32<tf32_rounding_mode::rne>(inf)), inf);
    EXPECT_EQ((float_to_tf32<tf32_rounding_mode::rne>(-inf)), -inf);
    EXPECT_TRUE(std::isnan(
        float_to_tf32<tf32_rounding_mode::rne>(std::numeric_limits<float>::quiet_NaN())));
}

TEST(Tf32, TypeConvert)
{
    // type_convert<tf32_t> should use truncation by default
    float val = 1.1f;
    EXPECT_EQ(ck_tile::type_convert<tf32_t>(val), float_to_tf32(val));

    val = -3.14f;
    EXPECT_EQ(ck_tile::type_convert<tf32_t>(val), float_to_tf32(val));

    val = 0.0f;
    EXPECT_EQ(ck_tile::type_convert<tf32_t>(val), float_to_tf32(val));
}

TEST(Tf32, Precision10MantissaBits)
{
    // After TF32 truncation, the top 10 mantissa bits are preserved.
    // The lower 13 out of 23 mantissa bits are zeroed.
    for(float val : {1.0f, 1.5f, 2.0f, 0.1f, 100.0f, -42.5f, 1e10f, 1e-10f})
    {
        uint32_t orig = to_bits(val);
        uint32_t tf32 = to_bits(float_to_tf32(val));

        // Top 19 bits (1 sign + 8 exp + 10 mant) must be preserved for normal numbers
        EXPECT_EQ(tf32 & 0xFFFFE000u, tf32) << "val=" << val;
        EXPECT_EQ(orig & 0xFFFFE000u, tf32) << "val=" << val;
    }
}
