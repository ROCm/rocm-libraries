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
    static constexpr float K_TOLERANCE = 0.01f;

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
    EXPECT_EQ(static_cast<float>(a), 1.0f);
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
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 2.5_bf;
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestUserDefinedLiterals, Bfloat16LiteralLargeValue)
{
    auto a = 100.0_bf;
    EXPECT_EQ(static_cast<float>(a), 100.0f);
}

// ============================================================================
// half literal (_h) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, HalfLiteralPositive)
{
    auto a = 1.0_h;
    EXPECT_TRUE((std::is_same_v<decltype(a), half>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
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
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 2.5_h;
    EXPECT_EQ(static_cast<float>(b), 2.5f);
}

TEST_F(TestUserDefinedLiterals, HalfLiteralLargeValue)
{
    auto a = 1000.0_h;
    EXPECT_EQ(static_cast<float>(a), 1000.0f);
}

// ============================================================================
// fp4_e2m1 literal (_e2m1) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp4E2M1LiteralPositive)
{
    auto a = 1.0_e2m1;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp4_e2m1>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp4E2M1LiteralNegative)
{
    auto a = -2.0_e2m1;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp4_e2m1>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp4E2M1LiteralZero)
{
    auto a = 0.0_e2m1;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp4_e2m1>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp4E2M1LiteralFractional)
{
    auto a = 0.5_e2m1;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e2m1;
    EXPECT_EQ(static_cast<float>(b), 1.5f);
}

// ============================================================================
// fp6_e2m3 literal (_e2m3) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp6E2M3LiteralPositive)
{
    auto a = 1.0_e2m3;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp6_e2m3>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp6E2M3LiteralNegative)
{
    auto a = -2.0_e2m3;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp6_e2m3>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp6E2M3LiteralZero)
{
    auto a = 0.0_e2m3;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp6_e2m3>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp6E2M3LiteralFractional)
{
    auto a = 0.5_e2m3;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e2m3;
    EXPECT_EQ(static_cast<float>(b), 1.5f);

    auto c = 0.125_e2m3;
    EXPECT_EQ(static_cast<float>(c), 0.125f);
}

// ============================================================================
// fp6_e3m2 literal (_e3m2) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp6E3M2LiteralPositive)
{
    auto a = 1.0_e3m2;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp6_e3m2>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp6E3M2LiteralNegative)
{
    auto a = -2.0_e3m2;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp6_e3m2>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp6E3M2LiteralZero)
{
    auto a = 0.0_e3m2;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp6_e3m2>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp6E3M2LiteralFractional)
{
    auto a = 0.5_e3m2;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e3m2;
    EXPECT_EQ(static_cast<float>(b), 1.5f);
}

// ============================================================================
// fp8_e4m3 literal (_e4m3) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralPositive)
{
    auto a = 1.0_e4m3;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralNegative)
{
    auto a = -2.0_e4m3;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralZero)
{
    auto a = 0.0_e4m3;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralFractional)
{
    auto a = 0.5_e4m3;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e4m3;
    EXPECT_EQ(static_cast<float>(b), 1.5f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3LiteralPowerOfTwo)
{
    auto a = 2.0_e4m3;
    EXPECT_EQ(static_cast<float>(a), 2.0f);

    auto b = 4.0_e4m3;
    EXPECT_EQ(static_cast<float>(b), 4.0f);

    auto c = 8.0_e4m3;
    EXPECT_EQ(static_cast<float>(c), 8.0f);
}

// ============================================================================
// fp8_e4m3_fnuz literal (_e4m3_fnuz) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E4M3FnuzLiteralPositive)
{
    auto a = 1.0_e4m3_fnuz;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3_fnuz>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3FnuzLiteralNegative)
{
    auto a = -2.0_e4m3_fnuz;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3_fnuz>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3FnuzLiteralZero)
{
    auto a = 0.0_e4m3_fnuz;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e4m3_fnuz>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3FnuzLiteralFractional)
{
    auto a = 0.5_e4m3_fnuz;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e4m3_fnuz;
    EXPECT_EQ(static_cast<float>(b), 1.5f);
}

TEST_F(TestUserDefinedLiterals, Fp8E4M3FnuzLiteralPowerOfTwo)
{
    auto a = 2.0_e4m3_fnuz;
    EXPECT_EQ(static_cast<float>(a), 2.0f);

    auto b = 4.0_e4m3_fnuz;
    EXPECT_EQ(static_cast<float>(b), 4.0f);

    auto c = 8.0_e4m3_fnuz;
    EXPECT_EQ(static_cast<float>(c), 8.0f);
}

// ============================================================================
// fp8_e5m2 literal (_e5m2) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralPositive)
{
    auto a = 1.0_e5m2;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralNegative)
{
    auto a = -2.0_e5m2;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralZero)
{
    auto a = 0.0_e5m2;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralFractional)
{
    auto a = 0.5_e5m2;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e5m2;
    EXPECT_EQ(static_cast<float>(b), 1.5f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2LiteralPowerOfTwo)
{
    auto a = 2.0_e5m2;
    EXPECT_EQ(static_cast<float>(a), 2.0f);

    auto b = 4.0_e5m2;
    EXPECT_EQ(static_cast<float>(b), 4.0f);

    auto c = 8.0_e5m2;
    EXPECT_EQ(static_cast<float>(c), 8.0f);
}

// ============================================================================
// fp8_e5m2_fnuz literal (_e5m2_fnuz) tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E5M2FnuzLiteralPositive)
{
    auto a = 1.0_e5m2_fnuz;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2_fnuz>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2FnuzLiteralNegative)
{
    auto a = -2.0_e5m2_fnuz;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2_fnuz>));
    EXPECT_EQ(static_cast<float>(a), -2.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2FnuzLiteralZero)
{
    auto a = 0.0_e5m2_fnuz;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e5m2_fnuz>));
    EXPECT_EQ(static_cast<float>(a), 0.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2FnuzLiteralFractional)
{
    auto a = 0.5_e5m2_fnuz;
    EXPECT_EQ(static_cast<float>(a), 0.5f);

    auto b = 1.5_e5m2_fnuz;
    EXPECT_EQ(static_cast<float>(b), 1.5f);
}

TEST_F(TestUserDefinedLiterals, Fp8E5M2FnuzLiteralPowerOfTwo)
{
    auto a = 2.0_e5m2_fnuz;
    EXPECT_EQ(static_cast<float>(a), 2.0f);

    auto b = 4.0_e5m2_fnuz;
    EXPECT_EQ(static_cast<float>(b), 4.0f);

    auto c = 8.0_e5m2_fnuz;
    EXPECT_EQ(static_cast<float>(c), 8.0f);
}

// ============================================================================
// fp8_e8m0 literal (_e8m0) tests
// Note: E8M0 is an unsigned type (scale type) - only positive powers of 2
// ============================================================================

TEST_F(TestUserDefinedLiterals, Fp8E8M0LiteralPositive)
{
    auto a = 1.0_e8m0;
    EXPECT_TRUE((std::is_same_v<decltype(a), fp8_e8m0>));
    EXPECT_EQ(static_cast<float>(a), 1.0f);
}

TEST_F(TestUserDefinedLiterals, Fp8E8M0LiteralPowerOfTwo)
{
    // E8M0 only represents powers of 2 exactly
    auto a = 2.0_e8m0;
    EXPECT_EQ(static_cast<float>(a), 2.0f);

    auto b = 4.0_e8m0;
    EXPECT_EQ(static_cast<float>(b), 4.0f);

    auto c = 8.0_e8m0;
    EXPECT_EQ(static_cast<float>(c), 8.0f);

    auto d = 0.5_e8m0;
    EXPECT_EQ(static_cast<float>(d), 0.5f);

    auto e = 0.25_e8m0;
    EXPECT_EQ(static_cast<float>(e), 0.25f);
}

// ============================================================================
// Literal usage in expressions
// ============================================================================

TEST_F(TestUserDefinedLiterals, LiteralInAddition)
{
    const bfloat16 sum = 1.0_bf + 2.0_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(sum), 3.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInSubtraction)
{
    const half diff = 5.0_h - 3.0_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(diff), 2.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInMultiplication)
{
    const bfloat16 product = 3.0_bf * 4.0_bf;
    EXPECT_TRUE(nearEqual(static_cast<float>(product), 12.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInDivision)
{
    const half quotient = 10.0_h / 2.0_h;
    EXPECT_TRUE(nearEqual(static_cast<float>(quotient), 5.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralInComparison)
{
    EXPECT_TRUE(1.0_bf < 2.0_bf);
    EXPECT_TRUE(2.0_h > 1.0_h);
}

// ============================================================================
// Literal assignment tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, LiteralAssignment)
{
    const bfloat16 bf = 1.5_bf;
    const half h = 2.5_h;
    const fp4_e2m1 e2m1 = 3.0_e2m1;
    const fp6_e2m3 e2m3 = 1.5_e2m3;
    const fp6_e3m2 e3m2 = 2.0_e3m2;
    const fp8_e4m3 e4m3 = 3.0_e4m3;
    const fp8_e4m3_fnuz e4m3Fnuz = 3.0_e4m3_fnuz;
    const fp8_e5m2 e5m2 = 4.0_e5m2;
    const fp8_e5m2_fnuz e5m2Fnuz = 4.0_e5m2_fnuz;
    const fp8_e8m0 e8m0 = 4.0_e8m0;

    EXPECT_EQ(static_cast<float>(bf), 1.5f);
    EXPECT_EQ(static_cast<float>(h), 2.5f);
    EXPECT_EQ(static_cast<float>(e2m1), 3.0f);
    EXPECT_EQ(static_cast<float>(e2m3), 1.5f);
    EXPECT_EQ(static_cast<float>(e3m2), 2.0f);
    EXPECT_EQ(static_cast<float>(e4m3), 3.0f);
    EXPECT_EQ(static_cast<float>(e4m3Fnuz), 3.0f);
    EXPECT_EQ(static_cast<float>(e5m2), 4.0f);
    EXPECT_EQ(static_cast<float>(e5m2Fnuz), 4.0f);
    EXPECT_EQ(static_cast<float>(e8m0), 4.0f);
}

TEST_F(TestUserDefinedLiterals, LiteralCopyAssignment)
{
    const bfloat16 a = 1.0_bf;
    bfloat16 b = 0.0_bf;
    b = a;
    EXPECT_EQ(a.data, b.data);
}

// ============================================================================
// Literal with math functions
// ============================================================================

TEST_F(TestUserDefinedLiterals, LiteralWithAbs)
{
    EXPECT_EQ(static_cast<float>(abs(-5.0_bf)), 5.0f);
    EXPECT_EQ(static_cast<float>(abs(-5.0_h)), 5.0f);
}

TEST_F(TestUserDefinedLiterals, LiteralWithSqrt)
{
    EXPECT_TRUE(nearEqual(static_cast<float>(sqrt(4.0_bf)), 2.0f));
    EXPECT_TRUE(nearEqual(static_cast<float>(sqrt(4.0_h)), 2.0f));
}

TEST_F(TestUserDefinedLiterals, LiteralWithMax)
{
    EXPECT_EQ(static_cast<float>(max(1.0_bf, 2.0_bf)), 2.0f);
    EXPECT_EQ(static_cast<float>(max(1.0_h, 2.0_h)), 2.0f);
}

TEST_F(TestUserDefinedLiterals, LiteralWithMin)
{
    EXPECT_EQ(static_cast<float>(min(1.0_bf, 2.0_bf)), 1.0f);
    EXPECT_EQ(static_cast<float>(min(1.0_h, 2.0_h)), 1.0f);
}

// ============================================================================
// Type deduction tests
// ============================================================================

TEST_F(TestUserDefinedLiterals, AutoTypeDeduction)
{
    auto bf = 1.0_bf;
    auto h = 1.0_h;
    auto e2m1 = 1.0_e2m1;
    auto e2m3 = 1.0_e2m3;
    auto e3m2 = 1.0_e3m2;
    auto e4m3 = 1.0_e4m3;
    auto e4m3Fnuz = 1.0_e4m3_fnuz;
    auto e5m2 = 1.0_e5m2;
    auto e5m2Fnuz = 1.0_e5m2_fnuz;
    auto e8m0 = 1.0_e8m0;

    static_assert(std::is_same_v<decltype(bf), bfloat16>);
    static_assert(std::is_same_v<decltype(h), half>);
    static_assert(std::is_same_v<decltype(e2m1), fp4_e2m1>);
    static_assert(std::is_same_v<decltype(e2m3), fp6_e2m3>);
    static_assert(std::is_same_v<decltype(e3m2), fp6_e3m2>);
    static_assert(std::is_same_v<decltype(e4m3), fp8_e4m3>);
    static_assert(std::is_same_v<decltype(e4m3Fnuz), fp8_e4m3_fnuz>);
    static_assert(std::is_same_v<decltype(e5m2), fp8_e5m2>);
    static_assert(std::is_same_v<decltype(e5m2Fnuz), fp8_e5m2_fnuz>);
    static_assert(std::is_same_v<decltype(e8m0), fp8_e8m0>);
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
