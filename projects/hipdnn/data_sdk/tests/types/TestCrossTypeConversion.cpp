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
    bfloat16 const a(2.5f);
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp4E2M1)
{
    bfloat16 const a(2.0f);
    fp4_e2m1 const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E4M3)
{
    bfloat16 const a(4.0f);
    fp8_e4m3 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E5M2)
{
    bfloat16 const a(4.0f);
    fp8_e5m2 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFp8E8M0)
{
    bfloat16 const a(4.0f);
    fp8_e8m0 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Bfloat16ToFloat)
{
    bfloat16 const a(3.14159f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 3.14159f, 0.02f));
}

TEST_F(TestCrossTypeConversion, Bfloat16ToDouble)
{
    bfloat16 const a(2.71828f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.71828f, 0.02f));
}

// ============================================================================
// half -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, HalfToBfloat16)
{
    half const a(2.5f);
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, HalfToFp4E2M1)
{
    half const a(4.0f);
    fp4_e2m1 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp8E4M3)
{
    half const a(4.0f);
    fp8_e4m3 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp8E5M2)
{
    half const a(4.0f);
    fp8_e5m2 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFp8E8M0)
{
    half const a(8.0f);
    fp8_e8m0 const b(a);
    EXPECT_EQ(static_cast<float>(b), 8.0f);
}

TEST_F(TestCrossTypeConversion, HalfToFloat)
{
    half const a(3.14159f);
    auto b = static_cast<float>(a);
    EXPECT_TRUE(nearEqual(b, 3.14159f, 0.002f));
}

TEST_F(TestCrossTypeConversion, HalfToDouble)
{
    half const a(2.71828f);
    auto b = static_cast<double>(a);
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.71828f, 0.002f));
}

// ============================================================================
// fp4_e2m1 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp4E2M1ToBfloat16)
{
    fp4_e2m1 const a(3.0f);
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 3.0f);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1ToHalf)
{
    fp4_e2m1 const a(4.0f);
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1ToFloat)
{
    fp4_e2m1 const a(3.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 3.0f);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1ToDouble)
{
    fp4_e2m1 const a(4.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 4.0);
}

// ============================================================================
// fp8_e4m3 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E4M3ToBfloat16)
{
    fp8_e4m3 const a(4.0f);
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToHalf)
{
    fp8_e4m3 const a(4.0f);
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToFp8E5M2)
{
    fp8_e4m3 const a(4.0f);
    fp8_e5m2 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToFloat)
{
    fp8_e4m3 const a(8.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3ToDouble)
{
    fp8_e4m3 const a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 16.0);
}

// ============================================================================
// fp8_e5m2 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E5M2ToBfloat16)
{
    fp8_e5m2 const a(4.0f);
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToHalf)
{
    fp8_e5m2 const a(4.0f);
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToFp8E4M3)
{
    fp8_e5m2 const a(4.0f);
    fp8_e4m3 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToFloat)
{
    fp8_e5m2 const a(8.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2ToDouble)
{
    fp8_e5m2 const a(16.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 16.0);
}

// ============================================================================
// fp8_e8m0 -> other types
// ============================================================================

TEST_F(TestCrossTypeConversion, Fp8E8M0ToBfloat16)
{
    fp8_e8m0 const a(4.0f);
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0ToHalf)
{
    fp8_e8m0 const a(8.0f);
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 8.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0ToFloat)
{
    fp8_e8m0 const a(16.0f);
    auto b = static_cast<float>(a);
    EXPECT_EQ(b, 16.0f);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0ToDouble)
{
    fp8_e8m0 const a(32.0f);
    auto b = static_cast<double>(a);
    EXPECT_EQ(b, 32.0);
}

// ============================================================================
// float -> custom types
// ============================================================================

TEST_F(TestCrossTypeConversion, FloatToBfloat16)
{
    float const a = 2.5f;
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, FloatToHalf)
{
    float const a = 2.5f;
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, FloatToFp4E2M1)
{
    float const a = 6.0f;
    fp4_e2m1 const b(a);
    EXPECT_EQ(static_cast<float>(b), 6.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp8E4M3)
{
    float const a = 4.0f;
    fp8_e4m3 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp8E5M2)
{
    float const a = 4.0f;
    fp8_e5m2 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, FloatToFp8E8M0)
{
    float const a = 16.0f;
    fp8_e8m0 const b(a);
    EXPECT_EQ(static_cast<float>(b), 16.0f);
}

// ============================================================================
// double -> custom types
// ============================================================================

TEST_F(TestCrossTypeConversion, DoubleToBfloat16)
{
    double const a = 2.5;
    bfloat16 const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToHalf)
{
    double const a = 2.5;
    half const b(a);
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp4E2M1)
{
    double const a = 0.5;
    fp4_e2m1 const b(a);
    EXPECT_EQ(static_cast<float>(b), 0.5f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E4M3)
{
    double const a = 4.0;
    fp8_e4m3 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E5M2)
{
    double const a = 4.0;
    fp8_e5m2 const b(a);
    EXPECT_EQ(static_cast<float>(b), 4.0f);
}

TEST_F(TestCrossTypeConversion, DoubleToFp8E8M0)
{
    double const a = 64.0;
    fp8_e8m0 const b(a);
    EXPECT_EQ(static_cast<float>(b), 64.0f);
}

// ============================================================================
// Roundtrip conversion tests
// ============================================================================

TEST_F(TestCrossTypeConversion, Bfloat16RoundtripViaFloat)
{
    bfloat16 const a(3.5f);
    auto f = static_cast<float>(a);
    bfloat16 const b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, HalfRoundtripViaFloat)
{
    half const a(3.5f);
    auto f = static_cast<float>(a);
    half const b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp4E2M1RoundtripViaFloat)
{
    fp4_e2m1 const a(4.0f);
    auto f = static_cast<float>(a);
    fp4_e2m1 const b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E4M3RoundtripViaFloat)
{
    fp8_e4m3 const a(4.0f);
    auto f = static_cast<float>(a);
    fp8_e4m3 const b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E5M2RoundtripViaFloat)
{
    fp8_e5m2 const a(4.0f);
    auto f = static_cast<float>(a);
    fp8_e5m2 const b(f);
    EXPECT_EQ(a.data, b.data);
}

TEST_F(TestCrossTypeConversion, Fp8E8M0RoundtripViaFloat)
{
    fp8_e8m0 const a(8.0f);
    auto f = static_cast<float>(a);
    fp8_e8m0 const b(f);
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Special values conversion tests
// ============================================================================

TEST_F(TestCrossTypeConversion, InfinityConversion)
{
    // half infinity -> bfloat16
    half const hInf = std::numeric_limits<half>::infinity();
    bfloat16 const bfInf(hInf);
    EXPECT_TRUE(isinf(bfInf));

    // bfloat16 infinity -> half
    bfloat16 const bfInf2 = std::numeric_limits<bfloat16>::infinity();
    half const hInf2(bfInf2);
    EXPECT_TRUE(isinf(hInf2));
}

TEST_F(TestCrossTypeConversion, NaNConversion)
{
    // half NaN -> bfloat16
    half const hNan = std::numeric_limits<half>::quiet_NaN();
    bfloat16 const bfNan(hNan);
    EXPECT_TRUE(isnan(bfNan));

    // bfloat16 NaN -> half
    bfloat16 const bfNan2 = std::numeric_limits<bfloat16>::quiet_NaN();
    half const hNan2(bfNan2);
    EXPECT_TRUE(isnan(hNan2));
}

TEST_F(TestCrossTypeConversion, ZeroConversion)
{
    // Positive zero
    half const hZero(0.0f);
    bfloat16 const bfZero(hZero);
    EXPECT_EQ(static_cast<float>(bfZero), 0.0f);
    EXPECT_FALSE(signbit(bfZero));

    // Negative zero
    half const hNegZero = half::from_bits(0x8000);
    bfloat16 const bfNegZero(hNegZero);
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
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_trivially_copyable_v<fp8_e8m0>);

    // Verify standard layout
    EXPECT_TRUE(std::is_standard_layout_v<bfloat16>);
    EXPECT_TRUE(std::is_standard_layout_v<half>);
    EXPECT_TRUE(std::is_standard_layout_v<fp4_e2m1>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e4m3>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e5m2>);
    EXPECT_TRUE(std::is_standard_layout_v<fp8_e8m0>);
}
