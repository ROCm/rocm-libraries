/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <gtest/gtest.h>
#include <rocsparse/rocsparse.h>

#include "rocsparse_data.hpp"

#include <cmath>
#include <sstream>

// Fixture that skips tests when --yaml filter is active (non-yaml tests)
class bfloat16_pre_checkin : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(RocSPARSE_TestData::is_yaml_filter_active())
        {
            GTEST_SKIP() << "Skipping non-yaml test when --yaml filter is active";
        }
    }
};

// =============================================================================
// Test Constructors
// =============================================================================
TEST_F(bfloat16_pre_checkin, DefaultConstructor)
{
    rocsparse_bfloat16 bf;
    // Default constructed, value is undefined but should not crash
    (void)bf;
}

TEST_F(bfloat16_pre_checkin, FloatConstructor)
{
    rocsparse_bfloat16 bf(3.14f);
    EXPECT_NEAR(float(bf), 3.14f, 0.01f);
}

TEST_F(bfloat16_pre_checkin, DoubleConstructor)
{
    rocsparse_bfloat16 bf(2.71828);
    EXPECT_NEAR(float(bf), 2.71828f, 0.01f);
}

TEST_F(bfloat16_pre_checkin, Int32Constructor)
{
    rocsparse_bfloat16 bf(42);
    EXPECT_FLOAT_EQ(float(bf), 42.0f);
}

TEST_F(bfloat16_pre_checkin, Int64Constructor)
{
    rocsparse_bfloat16 bf(int64_t(100));
    EXPECT_FLOAT_EQ(float(bf), 100.0f);
}

TEST_F(bfloat16_pre_checkin, RoundingModeConstructors)
{
    float test_val = 3.14159f;

    rocsparse_bfloat16 bf_near_even(test_val, rocsparse_bfloat16::rocsparse_round_near_even);
    rocsparse_bfloat16 bf_near_zero(test_val, rocsparse_bfloat16::rocsparse_round_near_zero);
    rocsparse_bfloat16 bf_truncate(test_val, rocsparse_bfloat16::rocsparse_truncate);

    EXPECT_NEAR(float(bf_near_even), test_val, 0.02f);
    EXPECT_NEAR(float(bf_near_zero), test_val, 0.02f);
    EXPECT_NEAR(float(bf_truncate), test_val, 0.02f);
}

// =============================================================================
// Test Assignment
// =============================================================================
TEST_F(bfloat16_pre_checkin, AssignmentFromFloat)
{
    rocsparse_bfloat16 bf;
    bf = 2.5f;
    EXPECT_FLOAT_EQ(float(bf), 2.5f);

    bf = 100.0f;
    EXPECT_FLOAT_EQ(float(bf), 100.0f);
}

// =============================================================================
// Test Conversions
// =============================================================================
TEST_F(bfloat16_pre_checkin, ConversionToFloat)
{
    rocsparse_bfloat16 bf(42.0f);
    float              f = float(bf);
    EXPECT_FLOAT_EQ(f, 42.0f);
}

TEST_F(bfloat16_pre_checkin, ConversionToDouble)
{
    rocsparse_bfloat16 bf(42.0f);
    double             d = static_cast<double>(bf);
    EXPECT_NEAR(d, 42.0, 1.0);
}

TEST_F(bfloat16_pre_checkin, ConversionToInt32)
{
    rocsparse_bfloat16 bf(42.0f);
    int32_t            i32 = static_cast<int32_t>(bf);
    EXPECT_EQ(i32, 42);
}

TEST_F(bfloat16_pre_checkin, ConversionToInt64)
{
    rocsparse_bfloat16 bf(42.0f);
    int64_t            i64 = static_cast<int64_t>(bf);
    EXPECT_EQ(i64, 42);
}

TEST_F(bfloat16_pre_checkin, ConversionToBool)
{
    rocsparse_bfloat16 bf_zero(0.0f);
    rocsparse_bfloat16 bf_nonzero(1.0f);

    EXPECT_FALSE(static_cast<bool>(bf_zero));
    EXPECT_TRUE(static_cast<bool>(bf_nonzero));
}

// =============================================================================
// Test Arithmetic Operators
// =============================================================================
TEST_F(bfloat16_pre_checkin, UnaryPlus)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 result = +a;
    EXPECT_FLOAT_EQ(float(result), 5.0f);
}

TEST_F(bfloat16_pre_checkin, UnaryMinus)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 result = -a;
    EXPECT_FLOAT_EQ(float(result), -5.0f);
}

TEST_F(bfloat16_pre_checkin, Addition)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 sum = a + b;
    EXPECT_FLOAT_EQ(float(sum), 8.0f);
}

TEST_F(bfloat16_pre_checkin, Subtraction)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 diff = a - b;
    EXPECT_FLOAT_EQ(float(diff), 2.0f);
}

TEST_F(bfloat16_pre_checkin, Multiplication)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 prod = a * b;
    EXPECT_FLOAT_EQ(float(prod), 15.0f);
}

TEST_F(bfloat16_pre_checkin, MultiplicationFloatBFloat16)
{
    float              a = 2.0f;
    rocsparse_bfloat16 b(5.0f);
    float              prod = a * b;
    EXPECT_FLOAT_EQ(prod, 10.0f);
}

TEST_F(bfloat16_pre_checkin, Division)
{
    rocsparse_bfloat16 a(15.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 quot = a / b;
    EXPECT_FLOAT_EQ(float(quot), 5.0f);
}

// =============================================================================
// Test Comparison Operators
// =============================================================================
TEST_F(bfloat16_pre_checkin, LessThan)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);

    EXPECT_TRUE(b < a);
    EXPECT_FALSE(a < b);
    EXPECT_FALSE(a < a);
}

TEST_F(bfloat16_pre_checkin, Equality)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(5.0f);
    rocsparse_bfloat16 c(3.0f);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(bfloat16_pre_checkin, GreaterThan)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);

    EXPECT_TRUE(a > b);
    EXPECT_FALSE(b > a);
    EXPECT_FALSE(a > a);
}

TEST_F(bfloat16_pre_checkin, LessThanOrEqual)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 c(5.0f);

    EXPECT_TRUE(b <= a);
    EXPECT_TRUE(a <= c);
    EXPECT_FALSE(a <= b);
}

TEST_F(bfloat16_pre_checkin, NotEqual)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 c(5.0f);

    EXPECT_TRUE(a != b);
    EXPECT_FALSE(a != c);
}

TEST_F(bfloat16_pre_checkin, NotEqualInt)
{
    rocsparse_bfloat16 a(5.0f);

    EXPECT_TRUE(a != 3);
    EXPECT_FALSE(a != 5);
}

TEST_F(bfloat16_pre_checkin, GreaterThanOrEqual)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b(3.0f);
    rocsparse_bfloat16 c(5.0f);

    EXPECT_TRUE(a >= b);
    EXPECT_TRUE(a >= c);
    EXPECT_FALSE(b >= a);
}

// =============================================================================
// Test Compound Assignment Operators
// =============================================================================
TEST_F(bfloat16_pre_checkin, AddAssignBFloat16)
{
    rocsparse_bfloat16 a(5.0f);
    a += rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(float(a), 8.0f);
}

TEST_F(bfloat16_pre_checkin, AddAssignFloat)
{
    rocsparse_bfloat16 a(5.0f);
    a += 3.0f;
    EXPECT_FLOAT_EQ(float(a), 8.0f);
}

TEST_F(bfloat16_pre_checkin, FloatAddAssignBFloat16)
{
    float a = 5.0f;
    a += rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(a, 8.0f);
}

TEST_F(bfloat16_pre_checkin, SubtractAssignBFloat16)
{
    rocsparse_bfloat16 a(5.0f);
    a -= rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(float(a), 2.0f);
}

TEST_F(bfloat16_pre_checkin, SubtractAssignFloat)
{
    rocsparse_bfloat16 a(5.0f);
    a -= 3.0f;
    EXPECT_FLOAT_EQ(float(a), 2.0f);
}

TEST_F(bfloat16_pre_checkin, FloatSubtractAssignBFloat16)
{
    float a = 5.0f;
    a -= rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(a, 2.0f);
}

TEST_F(bfloat16_pre_checkin, MultiplyAssignBFloat16)
{
    rocsparse_bfloat16 a(5.0f);
    a *= rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(float(a), 15.0f);
}

TEST_F(bfloat16_pre_checkin, MultiplyAssignFloat)
{
    rocsparse_bfloat16 a(5.0f);
    a *= 3.0f;
    EXPECT_FLOAT_EQ(float(a), 15.0f);
}

TEST_F(bfloat16_pre_checkin, FloatMultiplyAssignBFloat16)
{
    float a = 5.0f;
    a *= rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(a, 15.0f);
}

TEST_F(bfloat16_pre_checkin, DivideAssignBFloat16)
{
    rocsparse_bfloat16 a(15.0f);
    a /= rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(float(a), 5.0f);
}

TEST_F(bfloat16_pre_checkin, DivideAssignFloat)
{
    rocsparse_bfloat16 a(15.0f);
    a /= 3.0f;
    EXPECT_FLOAT_EQ(float(a), 5.0f);
}

TEST_F(bfloat16_pre_checkin, FloatDivideAssignBFloat16)
{
    float a = 15.0f;
    a /= rocsparse_bfloat16(3.0f);
    EXPECT_FLOAT_EQ(a, 5.0f);
}

// =============================================================================
// Test Increment/Decrement Operators
// =============================================================================
TEST_F(bfloat16_pre_checkin, PreIncrement)
{
    rocsparse_bfloat16 a(5.0f);
    ++a;
    EXPECT_FLOAT_EQ(float(a), 6.0f);
}

TEST_F(bfloat16_pre_checkin, PreDecrement)
{
    rocsparse_bfloat16 a(5.0f);
    --a;
    EXPECT_FLOAT_EQ(float(a), 4.0f);
}

TEST_F(bfloat16_pre_checkin, PostIncrement)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b = a++;
    EXPECT_FLOAT_EQ(float(b), 5.0f);
    EXPECT_FLOAT_EQ(float(a), 6.0f);
}

TEST_F(bfloat16_pre_checkin, PostDecrement)
{
    rocsparse_bfloat16 a(5.0f);
    rocsparse_bfloat16 b = a--;
    EXPECT_FLOAT_EQ(float(b), 5.0f);
    EXPECT_FLOAT_EQ(float(a), 4.0f);
}

// =============================================================================
// Test std namespace functions
// =============================================================================
TEST_F(bfloat16_pre_checkin, StdIsInf)
{
    rocsparse_bfloat16 inf;
    inf.data = 0x7f80; // Positive infinity
    EXPECT_TRUE(std::isinf(inf));

    rocsparse_bfloat16 finite(5.0f);
    EXPECT_FALSE(std::isinf(finite));
}

TEST_F(bfloat16_pre_checkin, StdIsNan)
{
    rocsparse_bfloat16 nan;
    nan.data = 0x7fc0; // NaN
    EXPECT_TRUE(std::isnan(nan));

    rocsparse_bfloat16 finite(5.0f);
    EXPECT_FALSE(std::isnan(finite));
}

TEST_F(bfloat16_pre_checkin, StdIsZero)
{
    rocsparse_bfloat16 zero(0.0f);
    EXPECT_TRUE(std::iszero(zero));

    rocsparse_bfloat16 nonzero(5.0f);
    EXPECT_FALSE(std::iszero(nonzero));
}

TEST_F(bfloat16_pre_checkin, StdAbs)
{
    rocsparse_bfloat16 neg(-5.0f);
    rocsparse_bfloat16 abs_neg = std::abs(neg);
    EXPECT_FLOAT_EQ(float(abs_neg), 5.0f);

    rocsparse_bfloat16 pos(5.0f);
    rocsparse_bfloat16 abs_pos = std::abs(pos);
    EXPECT_FLOAT_EQ(float(abs_pos), 5.0f);
}

TEST_F(bfloat16_pre_checkin, StdSin)
{
    rocsparse_bfloat16 angle(0.0f);
    rocsparse_bfloat16 sin_val = std::sin(angle);
    EXPECT_NEAR(float(sin_val), 0.0f, 0.01f);
}

TEST_F(bfloat16_pre_checkin, StdCos)
{
    rocsparse_bfloat16 angle(0.0f);
    rocsparse_bfloat16 cos_val = std::cos(angle);
    EXPECT_NEAR(float(cos_val), 1.0f, 0.01f);
}

TEST_F(bfloat16_pre_checkin, StdReal)
{
    rocsparse_bfloat16 val(3.14f);
    rocsparse_bfloat16 real_val = std::real(val);
    EXPECT_FLOAT_EQ(float(real_val), float(val));
}

// =============================================================================
// Test Special Values
// =============================================================================
TEST_F(bfloat16_pre_checkin, Zero)
{
    rocsparse_bfloat16 zero(0.0f);
    EXPECT_FLOAT_EQ(float(zero), 0.0f);
}

TEST_F(bfloat16_pre_checkin, NegativeZero)
{
    rocsparse_bfloat16 neg_zero(-0.0f);
    EXPECT_FLOAT_EQ(float(neg_zero), -0.0f);
}

TEST_F(bfloat16_pre_checkin, SmallValues)
{
    rocsparse_bfloat16 small(0.001f);
    EXPECT_NEAR(float(small), 0.001f, 0.0001f);
}

TEST_F(bfloat16_pre_checkin, LargeValues)
{
    rocsparse_bfloat16 large(30000.0f);
    EXPECT_NEAR(float(large), 30000.0f, 500.0f);
}

// =============================================================================
// Test Stream Output
// =============================================================================
TEST_F(bfloat16_pre_checkin, StreamOutput)
{
    rocsparse_bfloat16 bf(3.14f);
    std::ostringstream oss;
    oss << bf;

    // The output should be approximately "3.14"
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
}
