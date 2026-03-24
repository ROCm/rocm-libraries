// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <limits>
#include <type_traits>

using namespace hipdnn_data_sdk::types;

class TestCrossTypeConversion : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE = 0.2f;

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }
};

// ============================================================================
// bfloat16 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Bfloat16ToHalf)
{
    const bfloat16 a(2.5f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp4E2M1)
{
    const bfloat16 a(2.0f);
    const fp4_e2m1 b(a);
    EXPECT_EQ(static_cast<float>(b), 2.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp6E2M3)
{
    const bfloat16 a(3.5f);
    const fp6_e2m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 3.5f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp6E3M2)
{
    const bfloat16 a(12.0f);
    const fp6_e3m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 12.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E4M3)
{
    const bfloat16 a(4.0f);
    const fp8_e4m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E5M2)
{
    const bfloat16 a(4.0f);
    const fp8_e5m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E8M0)
{
    const bfloat16 a(4.0f);
    const fp8_e8m0 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFloat)
{
    const bfloat16 a(3.14159f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 3.14159f, 0.02f));
}

TEST_F(TestCrossTypeConversion, Bfloat16ToDouble)
{
    const bfloat16 a(2.71828f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.71828f, 0.02f));
}

// ============================================================================
// half -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, HalfToBfloat16)
{
    const half a(2.5f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, HalfToFp4E2M1)
{
    const half a(4.0f);
    const fp4_e2m1 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp6E2M3)
{
    const half a(5.5f);
    const fp6_e2m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 5.5f);
}

TEST_F(TestCrossTypeConversion, HalfToFp6E3M2)
{
    const half a(14.0f);
    const fp6_e3m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 14.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp8E4M3)
{
    const half a(4.0f);
    const fp8_e4m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp8E5M2)
{
    const half a(4.0f);
    const fp8_e5m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp8E8M0)
{
    const half a(8.0f);
    const fp8_e8m0 b(a);
    EXPECT_EQ(static_cast<float>(b), 8.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFloat)
{
    const half a(3.14159f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 3.14159f, 0.002f));
}

TEST_F(TestCrossTypeConversion, HalfToDouble)
{
    const half a(2.71828f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.71828f, 0.002f));
}

// ============================================================================
// fp4_e2m1 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp4E2M1ToBfloat16)
{
    const fp4_e2m1 a(3.0f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 3.0f);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1ToHalf)
{
    const fp4_e2m1 a(4.0f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1ToFloat)
{
    const fp4_e2m1 a(3.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 3.0f);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1ToDouble)
{
    const fp4_e2m1 a(4.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 4.0);
}

// ============================================================================
// fp6_e2m3 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp6E2M3ToBfloat16)
{
    const fp6_e2m3 a(3.5f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 3.5f);
}

TEST_F(TestCrossTypeConversion, Fp6E2M3ToHalf)
{
    const fp6_e2m3 a(5.0f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 5.0f);
}

TEST_F(TestCrossTypeConversion, Fp6E2M3ToFloat)
{
    const fp6_e2m3 a(7.5f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 7.5f);
}

TEST_F(TestCrossTypeConversion, Fp6E2M3ToDouble)
{
    const fp6_e2m3 a(4.5f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 4.5);
}

// ============================================================================
// fp6_e3m2 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp6E3M2ToBfloat16)
{
    const fp6_e3m2 a(4.0f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp6E3M2ToHalf)
{
    const fp6_e3m2 a(8.0f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp6E3M2ToFloat)
{
    const fp6_e3m2 a(28.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 28.0f);
}

TEST_F(TestCrossTypeConversion, Fp6E3M2ToDouble)
{
    const fp6_e3m2 a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 16.0);
}

// ============================================================================
// fp8_e4m3 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E4M3ToBfloat16)
{
    const fp8_e4m3 a(4.0f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToHalf)
{
    const fp8_e4m3 a(4.0f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToFp8E5M2)
{
    const fp8_e4m3 a(4.0f);
    const fp8_e5m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToFloat)
{
    const fp8_e4m3 a(8.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToDouble)
{
    const fp8_e4m3 a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 16.0);
}

// ============================================================================
// fp8_e5m2 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E5M2ToBfloat16)
{
    const fp8_e5m2 a(4.0f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToHalf)
{
    const fp8_e5m2 a(4.0f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToFp8E4M3)
{
    const fp8_e5m2 a(4.0f);
    const fp8_e4m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToFloat)
{
    const fp8_e5m2 a(8.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToDouble)
{
    const fp8_e5m2 a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 16.0);
}

// ============================================================================
// fp8_e8m0 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E8M0ToBfloat16)
{
    const fp8_e8m0 a(4.0f);
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0ToHalf)
{
    const fp8_e8m0 a(8.0f);
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0ToFloat)
{
    const fp8_e8m0 a(16.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 16.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0ToDouble)
{
    const fp8_e8m0 a(32.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 32.0);
}

// ============================================================================
// float -> custom types
// ============================================================================

TEST_F(TestCrossTypeConversion, FloatToBfloat16)
{
    const float a = 2.5f;
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, FloatToHalf)
{
    const float a = 2.5f;
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, FloatToFp4E2M1)
{
    const float a = 6.0f;
    const fp4_e2m1 b(a);
    EXPECT_EQ(static_cast<float>(b), 6.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp6E2M3)
{
    const float a = 6.5f;
    const fp6_e2m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 6.5f);
}

TEST_F(TestCrossTypeConversion, FloatToFp6E3M2)
{
    const float a = 20.0f;
    const fp6_e3m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 20.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp8E4M3)
{
    const float a = 4.0f;
    const fp8_e4m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp8E5M2)
{
    const float a = 4.0f;
    const fp8_e5m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp8E8M0)
{
    const float a = 16.0f;
    const fp8_e8m0 b(a);
    EXPECT_EQ(static_cast<float>(b), 16.0f);
}

// ============================================================================
// double -> custom types
// ============================================================================

TEST_F(TestCrossTypeConversion, DoubleToBfloat16)
{
    const double a = 2.5;
    const bfloat16 b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToHalf)
{
    const double a = 2.5;
    const half b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp4E2M1)
{
    const double a = 0.5;
    const fp4_e2m1 b(a);
    EXPECT_EQ(static_cast<float>(b), 0.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp6E2M3)
{
    const double a = 2.5;
    const fp6_e2m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp6E3M2)
{
    const double a = 24.0;
    const fp6_e3m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 24.0f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E4M3)
{
    const double a = 4.0;
    const fp8_e4m3 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E5M2)
{
    const double a = 4.0;
    const fp8_e5m2 b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E8M0)
{
    const double a = 64.0;
    const fp8_e8m0 b(a);
    EXPECT_EQ(static_cast<float>(b), 64.0f);
}

// ============================================================================
// Roundtrip conversion tests
// ============================================================================

TEST_F(TestCrossTypeConversion, Bfloat16RoundtripViaFloat)
{
    const bfloat16 a(3.5f);
    auto f = static_cast<float>(a);
    const bfloat16 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, HalfRoundtripViaFloat)
{
    const half a(3.5f);
    auto f = static_cast<float>(a);
    const half b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1RoundtripViaFloat)
{
    const fp4_e2m1 a(4.0f);
    auto f = static_cast<float>(a);
    const fp4_e2m1 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp6E2M3RoundtripViaFloat)
{
    const fp6_e2m3 a(5.5f);
    auto f = static_cast<float>(a);
    const fp6_e2m3 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp6E3M2RoundtripViaFloat)
{
    const fp6_e3m2 a(10.0f);
    auto f = static_cast<float>(a);
    const fp6_e3m2 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3RoundtripViaFloat)
{
    const fp8_e4m3 a(4.0f);
    auto f = static_cast<float>(a);
    const fp8_e4m3 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2RoundtripViaFloat)
{
    const fp8_e5m2 a(4.0f);
    auto f = static_cast<float>(a);
    const fp8_e5m2 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0RoundtripViaFloat)
{
    const fp8_e8m0 a(8.0f);
    auto f = static_cast<float>(a);
    const fp8_e8m0 b(f);
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Special values conversion tests
// ============================================================================

TEST_F(TestCrossTypeConversion, InfinityConversion)
{
    // half infinity -> bfloat16
    const half hInf = std::numeric_limits<half>::infinity();
    const bfloat16 bfInf(hInf);
    EXPECT_TRUE(isinf(bfInf));

    // bfloat16 infinity -> half
    const bfloat16 bfInf2 = std::numeric_limits<bfloat16>::infinity();
    const half hInf2(bfInf2);
    EXPECT_TRUE(isinf(hInf2));
}

TEST_F(TestCrossTypeConversion, NaNConversion)
{
    // half NaN -> bfloat16
    const half hNan = std::numeric_limits<half>::quiet_NaN();
    const bfloat16 bfNan(hNan);
    EXPECT_TRUE(isnan(bfNan));

    // bfloat16 NaN -> half
    const bfloat16 bfNan2 = std::numeric_limits<bfloat16>::quiet_NaN();
    const half hNan2(bfNan2);
    EXPECT_TRUE(isnan(hNan2));
}

TEST_F(TestCrossTypeConversion, ZeroConversion)
{
    // Positive zero
    const half hZero(0.0f);
    const bfloat16 bfZero(hZero);
    EXPECT_EQ(static_cast<float>(bfZero), 0.0f);
    EXPECT_FALSE(signbit(bfZero));

    // Negative zero
    const half hNegZero = half::from_bits(0x8000);
    const bfloat16 bfNegZero(hNegZero);
    EXPECT_EQ(static_cast<float>(bfNegZero), -0.0f);
    EXPECT_TRUE(signbit(bfNegZero));
}

// ============================================================================
// Type trait verification
// ============================================================================

TEST_F(TestCrossTypeConversion, TypeTraitsVerification)
{
    // Verify all types are trivially copyable (important for GPU usage)
    EXPECT_TRUE(std::is_trivially_copyable_v<bfloat16>);
    EXPECT_TRUE(std::is_trivially_copyable_v<half>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp4_e2m1>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp6_e2m3>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp6_e3m2>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e8m0>);

    // Verify standard layout
    EXPECT_TRUE(std::is_standard_layout_v<bfloat16>);
    EXPECT_TRUE(std::is_standard_layout_v<half>);
    EXPECT_TRUE(std::is_standard_layout_v<fp4_e2m1>);
    EXPECT_TRUE(std::is_standard_layout_v<fp6_e2m3>);
    EXPECT_TRUE(std::is_standard_layout_v<fp6_e3m2>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e8m0>);
}
