/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

#include <gtest/gtest.h>

#include <Reference.hpp>

using TensileLite::Client::AlmostEqual;
using TensileLite::Client::AlmostEqualTolerance_BFloat16;
using TensileLite::Client::AlmostEqualTolerance_BFloat8;
using TensileLite::Client::AlmostEqualTolerance_Double;
using TensileLite::Client::AlmostEqualTolerance_Float;
using TensileLite::Client::AlmostEqualTolerance_Float8;
using TensileLite::Client::AlmostEqualTolerance_Half;

// ============================================================================
// Helper: construct fp8/bf8 values with a specific raw byte
// ============================================================================
template <typename T>
T fromRawByte(uint8_t byte)
{
    static_assert(sizeof(T) == 1, "fromRawByte only works with 1-byte types");
    T val{};
    std::memcpy(&val, &byte, sizeof(byte));
    return val;
}

// ============================================================================
// Float tests  (tolerance = AlmostEqualTolerance_Float = 0.0001)
// ============================================================================
TEST(AlmostEqual_Float, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(1.0f, 1.0f));
    EXPECT_TRUE(AlmostEqual(-1.0f, -1.0f));
    EXPECT_TRUE(AlmostEqual(0.0f, 0.0f));
}

TEST(AlmostEqual_Float, NaN)
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, 1.0f));
    EXPECT_FALSE(AlmostEqual(1.0f, nan));
}

TEST(AlmostEqual_Float, Inf)
{
    float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(AlmostEqual(inf, inf));
    EXPECT_TRUE(AlmostEqual(-inf, -inf));
    EXPECT_FALSE(AlmostEqual(inf, -inf));
    EXPECT_FALSE(AlmostEqual(inf, 1.0f));
}

TEST(AlmostEqual_Float, WithinTolerance)
{
    // tol=0.0001, a=1.0, threshold = 0.0001*(1+1+1) = 0.0003
    EXPECT_TRUE(AlmostEqual(1.0f, 1.0002f));
}

TEST(AlmostEqual_Float, OutsideTolerance)
{
    // tol=0.0001, a=1.0, b=1.001 -> diff=0.001, threshold~0.0003 -> outside
    EXPECT_FALSE(AlmostEqual(1.0f, 1.001f));
}

TEST(AlmostEqual_Float, BoundaryJustBelow)
{
    // For V=10.0, threshold = tol*(10+10+1) = 0.0021
    // delta_boundary = tol*(2V+1)/(1-tol) ~ 0.0021002
    // Use a delta just under threshold
    float V     = 10.0f;
    float tol   = AlmostEqualTolerance_Float;
    float delta = tol * (2.0f * V + 1.0f) / (1.0f - tol) * 0.99f;
    EXPECT_TRUE(AlmostEqual(V, V + delta));
}

TEST(AlmostEqual_Float, BoundaryJustAbove)
{
    float V     = 10.0f;
    float tol   = AlmostEqualTolerance_Float;
    float delta = tol * (2.0f * V + 1.0f) / (1.0f - tol) * 1.01f;
    EXPECT_FALSE(AlmostEqual(V, V + delta));
}

TEST(AlmostEqual_Float, Symmetry)
{
    EXPECT_TRUE(AlmostEqual(1.0f, 1.0002f));
    EXPECT_TRUE(AlmostEqual(1.0002f, 1.0f));
    EXPECT_FALSE(AlmostEqual(1.0f, 1.001f));
    EXPECT_FALSE(AlmostEqual(1.001f, 1.0f));
}

TEST(AlmostEqual_Float, NegativeZero)
{
    EXPECT_TRUE(AlmostEqual(-0.0f, 0.0f));
    EXPECT_TRUE(AlmostEqual(0.0f, -0.0f));
}

TEST(AlmostEqual_Float, ZeroVsSmall)
{
    // threshold = tol*(0 + |small| + 1) ~ tol = 0.0001
    EXPECT_TRUE(AlmostEqual(0.0f, 0.00005f));
    EXPECT_FALSE(AlmostEqual(0.0f, 0.0005f));
}

TEST(AlmostEqual_Float, OppositeSignsNearZero)
{
    // diff=0.0002, threshold~0.0001*(0.0001+0.0001+1)~0.0001 -> outside
    EXPECT_FALSE(AlmostEqual(-0.0001f, 0.0001f));
}

TEST(AlmostEqual_Float, LargeMagnitude)
{
    // threshold = 0.0001*(1e6 + 1e6 + 1) ~ 200
    EXPECT_TRUE(AlmostEqual(1e6f, 1e6f + 100.0f));
    EXPECT_FALSE(AlmostEqual(1e6f, 1e6f + 500.0f));
}

TEST(AlmostEqual_Float, NegativeWithinTolerance)
{
    EXPECT_TRUE(AlmostEqual(-1.0f, -1.0002f));
}

TEST(AlmostEqual_Float, NegativeOutsideTolerance)
{
    EXPECT_FALSE(AlmostEqual(-1.0f, -1.001f));
}

// ============================================================================
// Double tests  (tolerance = AlmostEqualTolerance_Double = 1e-12)
// ============================================================================
TEST(AlmostEqual_Double, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(1.0, 1.0));
    EXPECT_TRUE(AlmostEqual(-1.0, -1.0));
    EXPECT_TRUE(AlmostEqual(0.0, 0.0));
}

TEST(AlmostEqual_Double, NaN)
{
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, 1.0));
    EXPECT_FALSE(AlmostEqual(1.0, nan));
}

TEST(AlmostEqual_Double, Inf)
{
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(AlmostEqual(inf, inf));
    EXPECT_TRUE(AlmostEqual(-inf, -inf));
    EXPECT_FALSE(AlmostEqual(inf, -inf));
    EXPECT_FALSE(AlmostEqual(inf, 1.0));
}

TEST(AlmostEqual_Double, WithinTolerance)
{
    // tol=1e-12, a=1.0, threshold = 1e-12*(1+1+1) = 3e-12
    EXPECT_TRUE(AlmostEqual(1.0, 1.0 + 2e-12));
}

TEST(AlmostEqual_Double, OutsideTolerance)
{
    EXPECT_FALSE(AlmostEqual(1.0, 1.0 + 1e-9));
}

TEST(AlmostEqual_Double, BoundaryJustBelow)
{
    double V     = 10.0;
    double tol   = AlmostEqualTolerance_Double;
    double delta = tol * (2.0 * V + 1.0) / (1.0 - tol) * 0.99;
    EXPECT_TRUE(AlmostEqual(V, V + delta));
}

TEST(AlmostEqual_Double, BoundaryJustAbove)
{
    double V     = 10.0;
    double tol   = AlmostEqualTolerance_Double;
    double delta = tol * (2.0 * V + 1.0) / (1.0 - tol) * 1.01;
    EXPECT_FALSE(AlmostEqual(V, V + delta));
}

TEST(AlmostEqual_Double, Symmetry)
{
    EXPECT_TRUE(AlmostEqual(1.0, 1.0 + 2e-12));
    EXPECT_TRUE(AlmostEqual(1.0 + 2e-12, 1.0));
    EXPECT_FALSE(AlmostEqual(1.0, 1.0 + 1e-9));
    EXPECT_FALSE(AlmostEqual(1.0 + 1e-9, 1.0));
}

TEST(AlmostEqual_Double, NegativeZero)
{
    EXPECT_TRUE(AlmostEqual(-0.0, 0.0));
    EXPECT_TRUE(AlmostEqual(0.0, -0.0));
}

TEST(AlmostEqual_Double, ZeroVsSmall)
{
    // threshold = tol*(0 + |small| + 1) ~ tol = 1e-12
    EXPECT_TRUE(AlmostEqual(0.0, 5e-13));
    EXPECT_FALSE(AlmostEqual(0.0, 5e-9));
}

TEST(AlmostEqual_Double, OppositeSignsNearZero)
{
    // diff=2e-6, threshold~1e-12*(1e-6+1e-6+1)~1e-12 -> outside
    EXPECT_FALSE(AlmostEqual(-1e-6, 1e-6));
}

TEST(AlmostEqual_Double, LargeMagnitude)
{
    // threshold = 1e-12*(1e12 + 1e12 + 1) ~ 2.0
    EXPECT_TRUE(AlmostEqual(1e12, 1e12 + 1.0));
    EXPECT_FALSE(AlmostEqual(1e12, 1e12 + 5.0));
}

TEST(AlmostEqual_Double, NegativeWithinTolerance)
{
    EXPECT_TRUE(AlmostEqual(-1.0, -1.0 - 2e-12));
}

TEST(AlmostEqual_Double, NegativeOutsideTolerance)
{
    EXPECT_FALSE(AlmostEqual(-1.0, -1.0 - 1e-9));
}

// ============================================================================
// Half tests  (tolerance = AlmostEqualTolerance_Half = 0.01)
// ============================================================================
TEST(AlmostEqual_Half, ExactEqual)
{
    using TensileLite::Half;
    EXPECT_TRUE(AlmostEqual(Half(1.0f), Half(1.0f)));
    EXPECT_TRUE(AlmostEqual(Half(-1.0f), Half(-1.0f)));
    EXPECT_TRUE(AlmostEqual(Half(0.0f), Half(0.0f)));
}

TEST(AlmostEqual_Half, NaN)
{
    using TensileLite::Half;
    Half nan = static_cast<Half>(std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, Half(1.0f)));
    EXPECT_FALSE(AlmostEqual(Half(1.0f), nan));
}

TEST(AlmostEqual_Half, Inf)
{
    using TensileLite::Half;
    Half inf     = static_cast<Half>(std::numeric_limits<float>::infinity());
    Half neg_inf = static_cast<Half>(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(AlmostEqual(inf, inf));
    EXPECT_TRUE(AlmostEqual(neg_inf, neg_inf));
    EXPECT_FALSE(AlmostEqual(inf, neg_inf));
    EXPECT_FALSE(AlmostEqual(inf, Half(1.0f)));
}

TEST(AlmostEqual_Half, WithinTolerance)
{
    using TensileLite::Half;
    // tol=0.01, a=1.0, threshold = 0.01*(1+1+1) = 0.03
    EXPECT_TRUE(AlmostEqual(Half(1.0f), Half(1.02f)));
}

TEST(AlmostEqual_Half, OutsideTolerance)
{
    using TensileLite::Half;
    // a=1.0, b=1.1 -> diff=0.1, threshold=0.01*(1+1.1+1)=0.031 -> outside
    EXPECT_FALSE(AlmostEqual(Half(1.0f), Half(1.1f)));
}

TEST(AlmostEqual_Half, BoundaryJustBelow)
{
    using TensileLite::Half;
    // Wider margin (0.9/1.1) because Half's 10-bit mantissa cannot represent
    // exact boundary values. Float/double use tighter 0.99/1.01 margins.
    float V     = 4.0f;
    float tol   = AlmostEqualTolerance_Half;
    float delta = tol * (2.0f * V + 1.0f) / (1.0f - tol) * 0.9f;
    EXPECT_TRUE(AlmostEqual(Half(V), Half(V + delta)));
}

TEST(AlmostEqual_Half, BoundaryJustAbove)
{
    using TensileLite::Half;
    // Wider margin (see BoundaryJustBelow comment).
    float V     = 4.0f;
    float tol   = AlmostEqualTolerance_Half;
    float delta = tol * (2.0f * V + 1.0f) / (1.0f - tol) * 1.1f;
    EXPECT_FALSE(AlmostEqual(Half(V), Half(V + delta)));
}

TEST(AlmostEqual_Half, Symmetry)
{
    using TensileLite::Half;
    EXPECT_TRUE(AlmostEqual(Half(1.0f), Half(1.02f)));
    EXPECT_TRUE(AlmostEqual(Half(1.02f), Half(1.0f)));
    EXPECT_FALSE(AlmostEqual(Half(1.0f), Half(1.1f)));
    EXPECT_FALSE(AlmostEqual(Half(1.1f), Half(1.0f)));
}

TEST(AlmostEqual_Half, ZeroVsSmall)
{
    using TensileLite::Half;
    // threshold = tol*(0 + |small| + 1) ~ tol = 0.01
    EXPECT_TRUE(AlmostEqual(Half(0.0f), Half(0.005f)));
    EXPECT_FALSE(AlmostEqual(Half(0.0f), Half(0.05f)));
}

TEST(AlmostEqual_Half, NegativeWithinTolerance)
{
    using TensileLite::Half;
    EXPECT_TRUE(AlmostEqual(Half(-1.0f), Half(-1.02f)));
}

TEST(AlmostEqual_Half, NegativeOutsideTolerance)
{
    using TensileLite::Half;
    EXPECT_FALSE(AlmostEqual(Half(-1.0f), Half(-1.1f)));
}

// ============================================================================
// BFloat16 tests  (tolerance = AlmostEqualTolerance_BFloat16 = 0.1)
// ============================================================================
TEST(AlmostEqual_BFloat16, ExactEqual)
{
    using TensileLite::BFloat16;
    EXPECT_TRUE(AlmostEqual(BFloat16(1.0f), BFloat16(1.0f)));
    EXPECT_TRUE(AlmostEqual(BFloat16(-1.0f), BFloat16(-1.0f)));
    EXPECT_TRUE(AlmostEqual(BFloat16(0.0f), BFloat16(0.0f)));
}

TEST(AlmostEqual_BFloat16, NaN)
{
    using TensileLite::BFloat16;
    BFloat16 nan(std::numeric_limits<float>::quiet_NaN());
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, BFloat16(1.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat16(1.0f), nan));
}

TEST(AlmostEqual_BFloat16, Inf)
{
    using TensileLite::BFloat16;
    BFloat16 inf(std::numeric_limits<float>::infinity());
    BFloat16 neg_inf(-std::numeric_limits<float>::infinity());
    EXPECT_TRUE(AlmostEqual(inf, inf));
    EXPECT_TRUE(AlmostEqual(neg_inf, neg_inf));
    EXPECT_FALSE(AlmostEqual(inf, neg_inf));
    EXPECT_FALSE(AlmostEqual(inf, BFloat16(1.0f)));
}

TEST(AlmostEqual_BFloat16, WithinTolerance)
{
    using TensileLite::BFloat16;
    // tol=0.1, a=2.0, threshold = 0.1*(2+2+1) = 0.5
    EXPECT_TRUE(AlmostEqual(BFloat16(2.0f), BFloat16(2.25f)));
}

TEST(AlmostEqual_BFloat16, OutsideTolerance)
{
    using TensileLite::BFloat16;
    // a=1.0, b=2.0 -> diff=1.0, threshold=0.1*(1+2+1)=0.4 -> outside
    EXPECT_FALSE(AlmostEqual(BFloat16(1.0f), BFloat16(2.0f)));
}

TEST(AlmostEqual_BFloat16, BoundaryJustBelow)
{
    using TensileLite::BFloat16;
    // Wider margin (0.85/1.15) because BFloat16's 8-bit mantissa cannot
    // represent exact boundary values. Float/double use tighter margins.
    float V     = 4.0f;
    float tol   = AlmostEqualTolerance_BFloat16;
    float delta = tol * (2.0f * V + 1.0f) / (1.0f - tol) * 0.85f;
    EXPECT_TRUE(AlmostEqual(BFloat16(V), BFloat16(V + delta)));
}

TEST(AlmostEqual_BFloat16, BoundaryJustAbove)
{
    using TensileLite::BFloat16;
    // Wider margin (see BoundaryJustBelow comment).
    float V     = 4.0f;
    float tol   = AlmostEqualTolerance_BFloat16;
    float delta = tol * (2.0f * V + 1.0f) / (1.0f - tol) * 1.15f;
    EXPECT_FALSE(AlmostEqual(BFloat16(V), BFloat16(V + delta)));
}

TEST(AlmostEqual_BFloat16, Symmetry)
{
    using TensileLite::BFloat16;
    EXPECT_TRUE(AlmostEqual(BFloat16(2.0f), BFloat16(2.25f)));
    EXPECT_TRUE(AlmostEqual(BFloat16(2.25f), BFloat16(2.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat16(1.0f), BFloat16(2.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat16(2.0f), BFloat16(1.0f)));
}

TEST(AlmostEqual_BFloat16, ZeroVsSmall)
{
    using TensileLite::BFloat16;
    // threshold = tol*(0 + |small| + 1) ~ tol = 0.1
    EXPECT_TRUE(AlmostEqual(BFloat16(0.0f), BFloat16(0.05f)));
    EXPECT_FALSE(AlmostEqual(BFloat16(0.0f), BFloat16(0.5f)));
}

TEST(AlmostEqual_BFloat16, NegativeWithinTolerance)
{
    using TensileLite::BFloat16;
    EXPECT_TRUE(AlmostEqual(BFloat16(-2.0f), BFloat16(-2.25f)));
}

TEST(AlmostEqual_BFloat16, NegativeOutsideTolerance)
{
    using TensileLite::BFloat16;
    EXPECT_FALSE(AlmostEqual(BFloat16(-1.0f), BFloat16(-2.0f)));
}

// ============================================================================
// Float8 (e4m3) tests  (tolerance = AlmostEqualTolerance_Float8 = 0.125, no inf)
// ============================================================================
TEST(AlmostEqual_Float8, ExactEqual)
{
    using TensileLite::Float8;
    EXPECT_TRUE(AlmostEqual(Float8(1.0f), Float8(1.0f)));
    EXPECT_TRUE(AlmostEqual(Float8(-1.0f), Float8(-1.0f)));
    EXPECT_TRUE(AlmostEqual(Float8(0.0f), Float8(0.0f)));
}

TEST(AlmostEqual_Float8, NaN)
{
    using TensileLite::Float8;
    // Float8 e4m3 NaN: byte 0x7f (and 0xff)
    Float8 nan = fromRawByte<Float8>(0x7f);
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, Float8(1.0f)));
    EXPECT_FALSE(AlmostEqual(Float8(1.0f), nan));
}

TEST(AlmostEqual_Float8, WithinTolerance)
{
    using TensileLite::Float8;
    // tol=0.125, a=2.0, threshold = 0.125*(2+2+1) = 0.625
    // Pick b=2.5 -> diff=0.5, within
    EXPECT_TRUE(AlmostEqual(Float8(2.0f), Float8(2.5f)));
}

TEST(AlmostEqual_Float8, OutsideTolerance)
{
    using TensileLite::Float8;
    // a=1.0, b=2.0 -> diff=1.0, threshold = 0.125*(1+2+1) = 0.5 -> outside
    EXPECT_FALSE(AlmostEqual(Float8(1.0f), Float8(2.0f)));
}

TEST(AlmostEqual_Float8, BoundaryJustBelow)
{
    using TensileLite::Float8;
    // Use V=4.0, tol=0.125
    // boundary = 0.125*(8+1)/(1-0.125) = 1.125/0.875 ~= 1.2857
    // Just below: factor 0.85 -> delta ~= 1.09 -> b ~= 5.09 -> fp8 = 5.0
    // With fp8 representable values: 4.0 and 5.0 differ by 1.0
    // threshold = 0.125*(4+5+1)=1.25 > 1.0 -> within
    EXPECT_TRUE(AlmostEqual(Float8(4.0f), Float8(5.0f)));
}

TEST(AlmostEqual_Float8, BoundaryJustAbove)
{
    using TensileLite::Float8;
    // a=2.0, b=4.0 -> diff=2.0, threshold = 0.125*(2+4+1) = 0.875 -> outside
    EXPECT_FALSE(AlmostEqual(Float8(2.0f), Float8(4.0f)));
}

TEST(AlmostEqual_Float8, Symmetry)
{
    using TensileLite::Float8;
    EXPECT_TRUE(AlmostEqual(Float8(2.0f), Float8(2.5f)));
    EXPECT_TRUE(AlmostEqual(Float8(2.5f), Float8(2.0f)));
    EXPECT_FALSE(AlmostEqual(Float8(1.0f), Float8(2.0f)));
    EXPECT_FALSE(AlmostEqual(Float8(2.0f), Float8(1.0f)));
}

TEST(AlmostEqual_Float8, MaxValue)
{
    using TensileLite::Float8;
    Float8 maxVal = fromRawByte<Float8>(0x7e); // max = 448.0
    EXPECT_TRUE(AlmostEqual(maxVal, maxVal));
}

// ============================================================================
// BFloat8 (e5m2) tests  (tolerance = AlmostEqualTolerance_BFloat8 = 0.25, has inf)
// ============================================================================
TEST(AlmostEqual_BFloat8, ExactEqual)
{
    using TensileLite::BFloat8;
    EXPECT_TRUE(AlmostEqual(BFloat8(1.0f), BFloat8(1.0f)));
    EXPECT_TRUE(AlmostEqual(BFloat8(-1.0f), BFloat8(-1.0f)));
    EXPECT_TRUE(AlmostEqual(BFloat8(0.0f), BFloat8(0.0f)));
}

TEST(AlmostEqual_BFloat8, NaN)
{
    using TensileLite::BFloat8;
    // BFloat8 e5m2 NaN: byte > 0x7c (e.g. 0x7f, 0x7d, 0x7e)
    BFloat8 nan = fromRawByte<BFloat8>(0x7f);
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, BFloat8(1.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat8(1.0f), nan));
}

TEST(AlmostEqual_BFloat8, Inf)
{
    using TensileLite::BFloat8;
    // BFloat8 e5m2 inf: byte 0x7c (+inf), 0xfc (-inf)
    BFloat8 inf     = fromRawByte<BFloat8>(0x7c);
    BFloat8 neg_inf = fromRawByte<BFloat8>(0xfc);
    EXPECT_TRUE(AlmostEqual(inf, inf));
    EXPECT_TRUE(AlmostEqual(neg_inf, neg_inf));
    EXPECT_FALSE(AlmostEqual(inf, neg_inf));
    EXPECT_FALSE(AlmostEqual(inf, BFloat8(1.0f)));
}

TEST(AlmostEqual_BFloat8, WithinTolerance)
{
    using TensileLite::BFloat8;
    // tol=0.25, a=1.0, b=1.5 -> diff=0.5, threshold = 0.25*(1+1.5+1) = 0.875 -> within
    EXPECT_TRUE(AlmostEqual(BFloat8(1.0f), BFloat8(1.5f)));
}

TEST(AlmostEqual_BFloat8, OutsideTolerance)
{
    using TensileLite::BFloat8;
    // a=1.0, b=3.0 -> diff=2.0, threshold = 0.25*(1+3+1) = 1.25 -> outside
    EXPECT_FALSE(AlmostEqual(BFloat8(1.0f), BFloat8(3.0f)));
}

TEST(AlmostEqual_BFloat8, BoundaryJustBelow)
{
    using TensileLite::BFloat8;
    // a=2.0, b=3.0 -> diff=1.0, threshold = 0.25*(2+3+1) = 1.5 -> within
    EXPECT_TRUE(AlmostEqual(BFloat8(2.0f), BFloat8(3.0f)));
}

TEST(AlmostEqual_BFloat8, BoundaryJustAbove)
{
    using TensileLite::BFloat8;
    // a=1.0, b=4.0 -> diff=3.0, threshold = 0.25*(1+4+1) = 1.5 -> outside
    EXPECT_FALSE(AlmostEqual(BFloat8(1.0f), BFloat8(4.0f)));
}

TEST(AlmostEqual_BFloat8, Symmetry)
{
    using TensileLite::BFloat8;
    EXPECT_TRUE(AlmostEqual(BFloat8(1.0f), BFloat8(1.5f)));
    EXPECT_TRUE(AlmostEqual(BFloat8(1.5f), BFloat8(1.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat8(1.0f), BFloat8(3.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat8(3.0f), BFloat8(1.0f)));
}

// ============================================================================
// Float8_fnuz (e4m3 fnuz) tests  (tolerance = AlmostEqualTolerance_Float8 = 0.125, no inf)
// ============================================================================
TEST(AlmostEqual_Float8Fnuz, ExactEqual)
{
    using TensileLite::Float8_fnuz;
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(1.0f), Float8_fnuz(1.0f)));
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(-1.0f), Float8_fnuz(-1.0f)));
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(0.0f), Float8_fnuz(0.0f)));
}

TEST(AlmostEqual_Float8Fnuz, NaN)
{
    using TensileLite::Float8_fnuz;
    // Float8_fnuz NaN: byte 0x80
    Float8_fnuz nan = fromRawByte<Float8_fnuz>(0x80);
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, Float8_fnuz(1.0f)));
    EXPECT_FALSE(AlmostEqual(Float8_fnuz(1.0f), nan));
}

TEST(AlmostEqual_Float8Fnuz, WithinTolerance)
{
    using TensileLite::Float8_fnuz;
    // Same as Float8: tol=0.125, a=2.0, b=2.5 -> within
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(2.0f), Float8_fnuz(2.5f)));
}

TEST(AlmostEqual_Float8Fnuz, OutsideTolerance)
{
    using TensileLite::Float8_fnuz;
    // a=1.0, b=2.0 -> diff=1.0, threshold = 0.125*(1+2+1) = 0.5 -> outside
    EXPECT_FALSE(AlmostEqual(Float8_fnuz(1.0f), Float8_fnuz(2.0f)));
}

TEST(AlmostEqual_Float8Fnuz, BoundaryJustBelow)
{
    using TensileLite::Float8_fnuz;
    // a=4.0, b=5.0 -> diff=1.0, threshold = 0.125*(4+5+1) = 1.25 -> within
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(4.0f), Float8_fnuz(5.0f)));
}

TEST(AlmostEqual_Float8Fnuz, BoundaryJustAbove)
{
    using TensileLite::Float8_fnuz;
    // a=2.0, b=4.0 -> diff=2.0, threshold = 0.125*(2+4+1) = 0.875 -> outside
    EXPECT_FALSE(AlmostEqual(Float8_fnuz(2.0f), Float8_fnuz(4.0f)));
}

TEST(AlmostEqual_Float8Fnuz, Symmetry)
{
    using TensileLite::Float8_fnuz;
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(2.0f), Float8_fnuz(2.5f)));
    EXPECT_TRUE(AlmostEqual(Float8_fnuz(2.5f), Float8_fnuz(2.0f)));
    EXPECT_FALSE(AlmostEqual(Float8_fnuz(1.0f), Float8_fnuz(2.0f)));
    EXPECT_FALSE(AlmostEqual(Float8_fnuz(2.0f), Float8_fnuz(1.0f)));
}

TEST(AlmostEqual_Float8Fnuz, MaxValue)
{
    using TensileLite::Float8_fnuz;
    // Float8_fnuz max: byte 0x7f = 240.0
    Float8_fnuz maxVal = fromRawByte<Float8_fnuz>(0x7f);
    EXPECT_TRUE(AlmostEqual(maxVal, maxVal));
}

// ============================================================================
// BFloat8_fnuz (e5m2 fnuz) tests  (tolerance = AlmostEqualTolerance_BFloat8 = 0.25, no inf)
// ============================================================================
TEST(AlmostEqual_BFloat8Fnuz, ExactEqual)
{
    using TensileLite::BFloat8_fnuz;
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(1.0f), BFloat8_fnuz(1.0f)));
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(-1.0f), BFloat8_fnuz(-1.0f)));
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(0.0f), BFloat8_fnuz(0.0f)));
}

TEST(AlmostEqual_BFloat8Fnuz, NaN)
{
    using TensileLite::BFloat8_fnuz;
    // BFloat8_fnuz NaN: byte 0x80
    BFloat8_fnuz nan = fromRawByte<BFloat8_fnuz>(0x80);
    EXPECT_FALSE(AlmostEqual(nan, nan));
    EXPECT_FALSE(AlmostEqual(nan, BFloat8_fnuz(1.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat8_fnuz(1.0f), nan));
}

TEST(AlmostEqual_BFloat8Fnuz, WithinTolerance)
{
    using TensileLite::BFloat8_fnuz;
    // tol=0.25, a=1.0, b=1.5 -> diff=0.5, threshold = 0.25*(1+1.5+1) = 0.875 -> within
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(1.0f), BFloat8_fnuz(1.5f)));
}

TEST(AlmostEqual_BFloat8Fnuz, OutsideTolerance)
{
    using TensileLite::BFloat8_fnuz;
    // a=1.0, b=3.0 -> diff=2.0, threshold = 0.25*(1+3+1) = 1.25 -> outside
    EXPECT_FALSE(AlmostEqual(BFloat8_fnuz(1.0f), BFloat8_fnuz(3.0f)));
}

TEST(AlmostEqual_BFloat8Fnuz, BoundaryJustBelow)
{
    using TensileLite::BFloat8_fnuz;
    // a=2.0, b=3.0 -> diff=1.0, threshold = 0.25*(2+3+1) = 1.5 -> within
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(2.0f), BFloat8_fnuz(3.0f)));
}

TEST(AlmostEqual_BFloat8Fnuz, BoundaryJustAbove)
{
    using TensileLite::BFloat8_fnuz;
    // a=1.0, b=4.0 -> diff=3.0, threshold = 0.25*(1+4+1) = 1.5 -> outside
    EXPECT_FALSE(AlmostEqual(BFloat8_fnuz(1.0f), BFloat8_fnuz(4.0f)));
}

TEST(AlmostEqual_BFloat8Fnuz, Symmetry)
{
    using TensileLite::BFloat8_fnuz;
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(1.0f), BFloat8_fnuz(1.5f)));
    EXPECT_TRUE(AlmostEqual(BFloat8_fnuz(1.5f), BFloat8_fnuz(1.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat8_fnuz(1.0f), BFloat8_fnuz(3.0f)));
    EXPECT_FALSE(AlmostEqual(BFloat8_fnuz(3.0f), BFloat8_fnuz(1.0f)));
}

TEST(AlmostEqual_BFloat8Fnuz, MaxValueNoInf)
{
    using TensileLite::BFloat8_fnuz;
    // BFloat8_fnuz has no infinity — unlike BFloat8 where 0x7c is +inf
    BFloat8_fnuz maxVal = fromRawByte<BFloat8_fnuz>(0x7f);
    EXPECT_TRUE(AlmostEqual(maxVal, maxVal));
}

// ============================================================================
// Int8 tests  (exact equality)
// ============================================================================
TEST(AlmostEqual_Int8, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(int8_t(42), int8_t(42)));
    EXPECT_TRUE(AlmostEqual(int8_t(0), int8_t(0)));
    EXPECT_TRUE(AlmostEqual(int8_t(-1), int8_t(-1)));
}

TEST(AlmostEqual_Int8, NotEqual)
{
    EXPECT_FALSE(AlmostEqual(int8_t(1), int8_t(2)));
    EXPECT_FALSE(AlmostEqual(int8_t(-1), int8_t(1)));
}

TEST(AlmostEqual_Int8, OffByOne)
{
    EXPECT_FALSE(AlmostEqual(int8_t(10), int8_t(11)));
    EXPECT_FALSE(AlmostEqual(int8_t(0), int8_t(1)));
}

TEST(AlmostEqual_Int8, ExtremeValues)
{
    EXPECT_TRUE(AlmostEqual(INT8_MAX, INT8_MAX));
    EXPECT_TRUE(AlmostEqual(INT8_MIN, INT8_MIN));
    EXPECT_FALSE(AlmostEqual(INT8_MIN, INT8_MAX));
}

// ============================================================================
// Int tests  (exact equality)
// ============================================================================
TEST(AlmostEqual_Int, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(42, 42));
    EXPECT_TRUE(AlmostEqual(0, 0));
    EXPECT_TRUE(AlmostEqual(-100, -100));
}

TEST(AlmostEqual_Int, NotEqual)
{
    EXPECT_FALSE(AlmostEqual(1, 2));
    EXPECT_FALSE(AlmostEqual(-1, 1));
}

TEST(AlmostEqual_Int, OffByOne)
{
    EXPECT_FALSE(AlmostEqual(1000, 1001));
    EXPECT_FALSE(AlmostEqual(0, 1));
}

TEST(AlmostEqual_Int, ExtremeValues)
{
    EXPECT_TRUE(AlmostEqual(INT_MAX, INT_MAX));
    EXPECT_TRUE(AlmostEqual(INT_MIN, INT_MIN));
    EXPECT_FALSE(AlmostEqual(INT_MIN, INT_MAX));
}

// ============================================================================
// Unsigned Int tests  (exact equality)
// ============================================================================
TEST(AlmostEqual_UInt, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(42u, 42u));
    EXPECT_TRUE(AlmostEqual(0u, 0u));
}

TEST(AlmostEqual_UInt, NotEqual)
{
    EXPECT_FALSE(AlmostEqual(1u, 2u));
    EXPECT_FALSE(AlmostEqual(100u, 200u));
}

TEST(AlmostEqual_UInt, OffByOne)
{
    EXPECT_FALSE(AlmostEqual(1000u, 1001u));
    EXPECT_FALSE(AlmostEqual(0u, 1u));
}

TEST(AlmostEqual_UInt, ExtremeValues)
{
    EXPECT_TRUE(AlmostEqual(UINT_MAX, UINT_MAX));
    EXPECT_TRUE(AlmostEqual(0u, 0u));
    EXPECT_FALSE(AlmostEqual(0u, UINT_MAX));
}

// ============================================================================
// Complex<float> tests  (delegates to float AlmostEqual per component)
// ============================================================================
using cf = std::complex<float>;

TEST(AlmostEqual_ComplexFloat, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(cf(1.0f, 2.0f), cf(1.0f, 2.0f)));
    EXPECT_TRUE(AlmostEqual(cf(0.0f, 0.0f), cf(0.0f, 0.0f)));
}

TEST(AlmostEqual_ComplexFloat, NaN)
{
    float nan = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(AlmostEqual(cf(nan, 0.0f), cf(1.0f, 0.0f)));
    EXPECT_FALSE(AlmostEqual(cf(0.0f, nan), cf(0.0f, 1.0f)));
    EXPECT_FALSE(AlmostEqual(cf(nan, nan), cf(nan, nan)));
}

TEST(AlmostEqual_ComplexFloat, Inf)
{
    float inf = std::numeric_limits<float>::infinity();
    EXPECT_TRUE(AlmostEqual(cf(inf, 0.0f), cf(inf, 0.0f)));
    EXPECT_TRUE(AlmostEqual(cf(0.0f, inf), cf(0.0f, inf)));
}

TEST(AlmostEqual_ComplexFloat, RealPartDiffers)
{
    // real part outside tolerance, imag identical
    EXPECT_FALSE(AlmostEqual(cf(1.0f, 0.0f), cf(1.001f, 0.0f)));
}

TEST(AlmostEqual_ComplexFloat, ImagPartDiffers)
{
    // real part identical, imag outside tolerance
    EXPECT_FALSE(AlmostEqual(cf(0.0f, 1.0f), cf(0.0f, 1.001f)));
}

TEST(AlmostEqual_ComplexFloat, BothWithinTolerance)
{
    // Both parts within float tolerance (0.0001)
    // threshold at a=1.0: 0.0001*(1+1+1)=0.0003
    EXPECT_TRUE(AlmostEqual(cf(1.0f, 1.0f), cf(1.0002f, 1.0002f)));
}

TEST(AlmostEqual_ComplexFloat, Symmetry)
{
    EXPECT_TRUE(AlmostEqual(cf(1.0f, 1.0f), cf(1.0002f, 1.0002f)));
    EXPECT_TRUE(AlmostEqual(cf(1.0002f, 1.0002f), cf(1.0f, 1.0f)));
    EXPECT_FALSE(AlmostEqual(cf(1.0f, 1.0f), cf(1.001f, 1.001f)));
    EXPECT_FALSE(AlmostEqual(cf(1.001f, 1.001f), cf(1.0f, 1.0f)));
}

TEST(AlmostEqual_ComplexFloat, RealWithinImagOutside)
{
    EXPECT_FALSE(AlmostEqual(cf(1.0f, 1.0f), cf(1.0002f, 1.001f)));
}

TEST(AlmostEqual_ComplexFloat, RealOutsideImagWithin)
{
    EXPECT_FALSE(AlmostEqual(cf(1.0f, 1.0f), cf(1.001f, 1.0002f)));
}

// ============================================================================
// Complex<double> tests  (delegates to double AlmostEqual per component)
// ============================================================================
using cd = std::complex<double>;

TEST(AlmostEqual_ComplexDouble, ExactEqual)
{
    EXPECT_TRUE(AlmostEqual(cd(1.0, 2.0), cd(1.0, 2.0)));
    EXPECT_TRUE(AlmostEqual(cd(0.0, 0.0), cd(0.0, 0.0)));
}

TEST(AlmostEqual_ComplexDouble, NaN)
{
    double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(AlmostEqual(cd(nan, 0.0), cd(1.0, 0.0)));
    EXPECT_FALSE(AlmostEqual(cd(0.0, nan), cd(0.0, 1.0)));
    EXPECT_FALSE(AlmostEqual(cd(nan, nan), cd(nan, nan)));
}

TEST(AlmostEqual_ComplexDouble, Inf)
{
    double inf = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(AlmostEqual(cd(inf, 0.0), cd(inf, 0.0)));
    EXPECT_TRUE(AlmostEqual(cd(0.0, inf), cd(0.0, inf)));
}

TEST(AlmostEqual_ComplexDouble, RealPartDiffers)
{
    EXPECT_FALSE(AlmostEqual(cd(1.0, 0.0), cd(1.0 + 1e-9, 0.0)));
}

TEST(AlmostEqual_ComplexDouble, ImagPartDiffers)
{
    EXPECT_FALSE(AlmostEqual(cd(0.0, 1.0), cd(0.0, 1.0 + 1e-9)));
}

TEST(AlmostEqual_ComplexDouble, BothWithinTolerance)
{
    // threshold at a=1.0: 1e-12*(1+1+1) = 3e-12
    EXPECT_TRUE(AlmostEqual(cd(1.0, 1.0), cd(1.0 + 2e-12, 1.0 + 2e-12)));
}

TEST(AlmostEqual_ComplexDouble, Symmetry)
{
    EXPECT_TRUE(AlmostEqual(cd(1.0, 1.0), cd(1.0 + 2e-12, 1.0 + 2e-12)));
    EXPECT_TRUE(AlmostEqual(cd(1.0 + 2e-12, 1.0 + 2e-12), cd(1.0, 1.0)));
    EXPECT_FALSE(AlmostEqual(cd(1.0, 1.0), cd(1.0 + 1e-9, 1.0 + 1e-9)));
    EXPECT_FALSE(AlmostEqual(cd(1.0 + 1e-9, 1.0 + 1e-9), cd(1.0, 1.0)));
}

TEST(AlmostEqual_ComplexDouble, RealWithinImagOutside)
{
    EXPECT_FALSE(AlmostEqual(cd(1.0, 1.0), cd(1.0 + 2e-12, 1.0 + 1e-9)));
}

TEST(AlmostEqual_ComplexDouble, RealOutsideImagWithin)
{
    EXPECT_FALSE(AlmostEqual(cd(1.0, 1.0), cd(1.0 + 1e-9, 1.0 + 2e-12)));
}
