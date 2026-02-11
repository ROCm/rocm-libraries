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
    static constexpr float K_TOLERANCE = 0.2f; // NOLINT(readability-identifier-naming)

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
    bfloat16 a(2.5f);
    half b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f, 0.01f));
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E4M3)
{
    bfloat16 a(4.0f);
    fp8_e4m3 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f));
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E5M2)
{
    bfloat16 a(4.0f);
    fp8_e5m2 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFloat)
{
    bfloat16 a(3.14159f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 3.14159f, 0.02f));
}

TEST_F(TestCrossTypeConversion, Bfloat16ToDouble)
{
    bfloat16 a(2.71828f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.71828f, 0.02f));
}

// ============================================================================
// half -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, HalfToBfloat16)
{
    half a(2.5f);
    bfloat16 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f, 0.01f));
}

TEST_F(TestCrossTypeConversion, HalfToFp8E4M3)
{
    half a(4.0f);
    fp8_e4m3 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f));
}

TEST_F(TestCrossTypeConversion, HalfToFp8E5M2)
{
    half a(4.0f);
    fp8_e5m2 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, HalfToFloat)
{
    half a(3.14159f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 3.14159f, 0.002f));
}

TEST_F(TestCrossTypeConversion, HalfToDouble)
{
    half a(2.71828f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.71828f, 0.002f));
}

// ============================================================================
// fp8_e4m3 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E4M3ToBfloat16)
{
    fp8_e4m3 a(4.0f);
    bfloat16 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.1f));
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToHalf)
{
    fp8_e4m3 a(4.0f);
    half b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.1f));
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToFp8E5M2)
{
    fp8_e4m3 a(4.0f);
    fp8_e5m2 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToFloat)
{
    fp8_e4m3 a(8.0f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 8.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToDouble)
{
    fp8_e4m3 a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 16.0f, 1.0f));
}

// ============================================================================
// fp8_e5m2 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E5M2ToBfloat16)
{
    fp8_e5m2 a(4.0f);
    bfloat16 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToHalf)
{
    fp8_e5m2 a(4.0f);
    half b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToFp8E4M3)
{
    fp8_e5m2 a(4.0f);
    fp8_e4m3 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToFloat)
{
    fp8_e5m2 a(8.0f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 8.0f, 1.0f));
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToDouble)
{
    fp8_e5m2 a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 16.0f, 2.0f));
}

// ============================================================================
// float -> custom types
// ============================================================================

TEST_F(TestCrossTypeConversion, FloatToBfloat16)
{
    float a = 2.5f;
    bfloat16 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f, 0.01f));
}

TEST_F(TestCrossTypeConversion, FloatToHalf)
{
    float a = 2.5f;
    half b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f, 0.001f));
}

TEST_F(TestCrossTypeConversion, FloatToFp8E4M3)
{
    float a = 4.0f;
    fp8_e4m3 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f));
}

TEST_F(TestCrossTypeConversion, FloatToFp8E5M2)
{
    float a = 4.0f;
    fp8_e5m2 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

// ============================================================================
// double -> custom types
// ============================================================================

TEST_F(TestCrossTypeConversion, DoubleToBfloat16)
{
    double a = 2.5;
    bfloat16 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f, 0.01f));
}

TEST_F(TestCrossTypeConversion, DoubleToHalf)
{
    double a = 2.5;
    half b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f, 0.001f));
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E4M3)
{
    double a = 4.0;
    fp8_e4m3 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f));
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E5M2)
{
    double a = 4.0;
    fp8_e5m2 b(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

// ============================================================================
// Roundtrip conversion tests
// ============================================================================

TEST_F(TestCrossTypeConversion, Bfloat16RoundtripViaFloat)
{
    bfloat16 a(3.5f);
    auto f = static_cast<float>(a);
    bfloat16 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, HalfRoundtripViaFloat)
{
    half a(3.5f);
    auto f = static_cast<float>(a);
    half b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3RoundtripViaFloat)
{
    fp8_e4m3 a(4.0f);
    auto f = static_cast<float>(a);
    fp8_e4m3 b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2RoundtripViaFloat)
{
    fp8_e5m2 a(4.0f);
    auto f = static_cast<float>(a);
    fp8_e5m2 b(f);
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Static cast syntax tests
// ============================================================================

TEST_F(TestCrossTypeConversion, StaticCastSyntaxBfloat16ToHalf)
{
    bfloat16 a(2.0f);
    auto b = static_cast<half>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.0f, 0.01f));
}

TEST_F(TestCrossTypeConversion, StaticCastSyntaxHalfToBfloat16)
{
    half a(2.0f);
    auto b = static_cast<bfloat16>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.0f, 0.01f));
}

TEST_F(TestCrossTypeConversion, StaticCastSyntaxFp8E4M3ToHalf)
{
    fp8_e4m3 a(4.0f);
    auto b = static_cast<half>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.1f));
}

TEST_F(TestCrossTypeConversion, StaticCastSyntaxFp8E5M2ToBfloat16)
{
    fp8_e5m2 a(4.0f);
    auto b = static_cast<bfloat16>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));
}

// ============================================================================
// Special values conversion tests
// ============================================================================

TEST_F(TestCrossTypeConversion, InfinityConversion)
{
    // half infinity -> bfloat16
    half hInf = std::numeric_limits<half>::infinity();
    bfloat16 bfInf(hInf);
    EXPECT_TRUE(isinf(bfInf));

    // bfloat16 infinity -> half
    bfloat16 bfInf2 = std::numeric_limits<bfloat16>::infinity();
    half hInf2(bfInf2);
    EXPECT_TRUE(isinf(hInf2));
}

TEST_F(TestCrossTypeConversion, NaNConversion)
{
    // half NaN -> bfloat16
    half hNan = std::numeric_limits<half>::quiet_NaN();
    bfloat16 bfNan(hNan);
    EXPECT_TRUE(isnan(bfNan));

    // bfloat16 NaN -> half
    bfloat16 bfNan2 = std::numeric_limits<bfloat16>::quiet_NaN();
    half hNan2(bfNan2);
    EXPECT_TRUE(isnan(hNan2));
}

TEST_F(TestCrossTypeConversion, ZeroConversion)
{
    // Positive zero
    half hZero(0.0f);
    bfloat16 bfZero(hZero);
    EXPECT_EQ(static_cast<float>(bfZero), 0.0f);
    EXPECT_FALSE(signbit(bfZero));

    // Negative zero
    half hNegZero = half::from_bits(0x8000);
    bfloat16 bfNegZero(hNegZero);
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
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e5m2>);

    // Verify standard layout
    EXPECT_TRUE(std::is_standard_layout_v<bfloat16>);
    EXPECT_TRUE(std::is_standard_layout_v<half>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e5m2>);
}
