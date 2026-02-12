// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>

#include <cmath>
#include <type_traits>

using namespace hipdnn_data_sdk::types;

class TestUserDefinedLiterals : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE = 0.01f; // NOLINT(readability-identifier-naming)

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE)
    {
        return hipdnn_data_sdk::types::fabs(a - b) <= tol;
    }
};

// ============================================================================
// bfloat16 literal (_bf) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Bfloat16LiteralPositive)
{
    auto a = 1.0_bf;
    EXPECT_TRUE((std::is_same_v<decltype(a), bfloat16>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));
}

TEST_F(TestUserDefinedLiterals, Bfloat16LiteralNegative)
{
    auto a = -3.14_bf;
    EXPECT_TRUE((std::is_same_v<decltype(a), bfloat16>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), -3.14f, 0.02f));
}

TEST_F(TestUserDefinedLiterals, Bfloat16LiteralZero)
{
    auto a = 0.0_bf;
    EXPECT_TRUE((std::is_same_v<decltype(a), bfloat16>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Bfloat16LiteralFractional)
{
    auto a = 0.5_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 0.5f));

    auto b = 2.5_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f));
}

TEST_F(TestUserDefinedLiterals, Bfloat16LiteralLargeValue)
{
    auto a = 100.0_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 100.0f, 1.0f));
}

// ============================================================================
// half literal (_h) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, HalfLiteralPositive)
{
    auto a = 1.0_h;
    EXPECT_TRUE((std::is_same_v<decltype(a), half>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f));
}

TEST_F(TestUserDefinedLiterals, HalfLiteralNegative)
{
    auto a = -3.14_h;
    EXPECT_TRUE((std::is_same_v<decltype(a), half>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), -3.14f, 0.01f));
}

TEST_F(TestUserDefinedLiterals, HalfLiteralZero)
{
    auto a = 0.0_h;
    EXPECT_TRUE((std::is_same_v<decltype(a), half>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, HalfLiteralFractional)
{
    auto a = 0.5_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 0.5f));

    auto b = 2.5_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f));
}

TEST_F(TestUserDefinedLiterals, HalfLiteralLargeValue)
{
    auto a = 1000.0_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1000.0f, 1.0f));
}

// ============================================================================
// fp8_e4m3 literal (_fp8) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralPositive)
{
    auto a = 1.0_fp8;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f, 0.2f));
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralNegative)
{
    auto a = -2.0_fp8;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), -2.0f, 0.2f));
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralZero)
{
    auto a = 0.0_fp8;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralFractional)
{
    auto a = 0.5_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 0.5f, 0.2f));

    auto b = 1.5_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 1.5f, 0.2f));
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralPowerOfTwo)
{
    auto a = 2.0_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 2.0f, 0.1f));

    auto b = 4.0_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.1f));

    auto c = 8.0_fp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 8.0f, 0.2f));
}

// ============================================================================
// fp8_e5m2 literal (_bfp8) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralPositive)
{
    auto a = 1.0_bfp8;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.0f, 0.5f));
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralNegative)
{
    auto a = -2.0_bfp8;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2>));
    EXPECT_TRUE(nearEqual(static_cast<float>(a), -2.0f, 0.5f));
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralZero)
{
    auto a = 0.0_bfp8;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralFractional)
{
    auto a = 0.5_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 0.5f, 0.5f));

    auto b = 1.5_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 1.5f, 0.5f));
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralPowerOfTwo)
{
    auto a = 2.0_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(a), 2.0f, 0.5f));

    auto b = 4.0_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 4.0f, 0.5f));

    auto c = 8.0_bfp8;
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 8.0f, 0.5f));
}

// ============================================================================
// Literal usage in expressions
// ============================================================================

TEST_F(TestUserDefinedLiterals, LiteralInAddition)
{
    bfloat16 sum = 1.0_bf + 2.0_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(sum), 3.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInSubtraction)
{
    half diff = 5.0_h - 3.0_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(diff), 2.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInMultiplication)
{
    bfloat16 product = 3.0_bf * 4.0_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(product), 12.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInDivision)
{
    half quotient = 10.0_h / 2.0_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(quotient), 5.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInComparison)
{
    EXPECT_TRUE(1.0_bf < 2.0_bf);
    EXPECT_TRUE(2.0_h > 1.0_h);
    EXPECT_TRUE(1.0_fp8 == 1.0_fp8);
    EXPECT_TRUE(1.0_bfp8 != 2.0_bfp8);
}

// ============================================================================
// Literal assignment tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, LiteralAssignment)
{
    bfloat16 a = 1.5_bf;
    half b = 2.5_h;
    fp8_e4m3 c = 3.0_fp8;
    fp8_e5m2 d = 4.0_bfp8;

    EXPECT_TRUE(nearEqual(static_cast<float>(a), 1.5f));
    EXPECT_TRUE(nearEqual(static_cast<float>(b), 2.5f));
    EXPECT_TRUE(nearEqual(static_cast<float>(c), 3.0f, 0.2f));
    EXPECT_TRUE(nearEqual(static_cast<float>(d), 4.0f, 0.5f));
}

TEST_F(TestUserDefinedLiterals, LiteralCopyAssignment)
{
    bfloat16 a = 1.0_bf;
    bfloat16 b = 0.0_bf;
    b = a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Literal with math functions
// ============================================================================

TEST_F(TestUserDefinedLiterals, LiteralWithAbs)
{
    EXPECT_TRUE(nearEqual(static_cast<float>(abs(-5.0_bf)), 5.0f, 0.1f));
    EXPECT_TRUE(nearEqual(static_cast<float>(abs(-5.0_h)), 5.0f, 0.01f));
}

TEST_F(TestUserDefinedLiterals, LiteralWithSqrt)
{
    EXPECT_TRUE(nearEqual(static_cast<float>(sqrt(4.0_bf)), 2.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(sqrt(4.0_h)), 2.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralWithMax)
{
    EXPECT_TRUE(nearEqual(static_cast<float>(max(1.0_bf, 2.0_bf)), 2.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(max(1.0_h, 2.0_h)), 2.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralWithMin)
{
    EXPECT_TRUE(nearEqual(static_cast<float>(min(1.0_bf, 2.0_bf)), 1.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(min(1.0_h, 2.0_h)), 1.0f));
}

// ============================================================================
// Type deduction tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, AutoTypeDeduction)
{
    auto bf = 1.0_bf;
    auto h = 1.0_h;
    auto fp8 = 1.0_fp8;
    auto bfp8 = 1.0_bfp8;

    static_assert(std::is_same_v<decltype(bf), bfloat16>);
    static_assert(std::is_same_v<decltype(h), half>);
    static_assert(std::is_same_v<decltype(fp8), fp8_e4m3>);
    static_assert(std::is_same_v<decltype(bfp8), fp8_e5m2>);
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(TestUserDefinedLiterals, SmallPositiveValue)
{
    auto a = 0.001_bf;
    EXPECT_GT(static_cast<float>(a), 0.0f);

    auto b = 0.001_h;
    EXPECT_GT(static_cast<float>(b), 0.0f);
}

TEST_F(TestUserDefinedLiterals, SmallNegativeValue)
{
    auto a = -0.001_bf;
    EXPECT_LT(static_cast<float>(a), 0.0f);

    auto b = -0.001_h;
    EXPECT_LT(static_cast<float>(b), 0.0f);
}
