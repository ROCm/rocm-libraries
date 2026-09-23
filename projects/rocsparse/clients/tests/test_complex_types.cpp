/*! \file */
/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <complex>
#include <sstream>

// Fixture that skips tests when --yaml filter is active (non-yaml tests)
class complex_types_pre_checkin : public ::testing::Test
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
// Float Complex Tests
// =============================================================================

// Test Constructors
TEST_F(complex_types_pre_checkin, FloatDefaultConstructor)
{
    rocsparse_float_complex c;
    // Default constructed, should not crash
    (void)c;
}

TEST_F(complex_types_pre_checkin, FloatRealImaginaryConstructor)
{
    rocsparse_float_complex c(3.0f, 4.0f);
    EXPECT_FLOAT_EQ(std::real(c), 3.0f);
    EXPECT_FLOAT_EQ(std::imag(c), 4.0f);
}

TEST_F(complex_types_pre_checkin, FloatRealOnlyConstructor)
{
    rocsparse_float_complex c(5.0f);
    EXPECT_FLOAT_EQ(std::real(c), 5.0f);
    EXPECT_FLOAT_EQ(std::imag(c), 0.0f);
}

TEST_F(complex_types_pre_checkin, FloatStdComplexConstructor)
{
    std::complex<float>     std_c(2.0f, 3.0f);
    rocsparse_float_complex c(std_c);
    EXPECT_FLOAT_EQ(std::real(c), 2.0f);
    EXPECT_FLOAT_EQ(std::imag(c), 3.0f);
}

TEST_F(complex_types_pre_checkin, FloatConversionToStdComplex)
{
    rocsparse_float_complex c(3.0f, 4.0f);
    std::complex<float>     std_c = static_cast<std::complex<float>>(c);
    EXPECT_FLOAT_EQ(std_c.real(), 3.0f);
    EXPECT_FLOAT_EQ(std_c.imag(), 4.0f);
}

TEST_F(complex_types_pre_checkin, FloatCopyConstructor)
{
    rocsparse_float_complex c1(3.0f, 4.0f);
    rocsparse_float_complex c2(c1);
    EXPECT_FLOAT_EQ(std::real(c2), 3.0f);
    EXPECT_FLOAT_EQ(std::imag(c2), 4.0f);
}

// Test Unary Operators
TEST_F(complex_types_pre_checkin, FloatUnaryMinus)
{
    rocsparse_float_complex c(3.0f, 4.0f);
    rocsparse_float_complex neg = -c;
    EXPECT_FLOAT_EQ(std::real(neg), -3.0f);
    EXPECT_FLOAT_EQ(std::imag(neg), -4.0f);
}

// Test Arithmetic Operations
TEST_F(complex_types_pre_checkin, FloatAddition)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    rocsparse_float_complex sum = a + b;
    EXPECT_FLOAT_EQ(std::real(sum), 4.0f);
    EXPECT_FLOAT_EQ(std::imag(sum), 6.0f);
}

TEST_F(complex_types_pre_checkin, FloatSubtraction)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    rocsparse_float_complex diff = a - b;
    EXPECT_FLOAT_EQ(std::real(diff), 2.0f);
    EXPECT_FLOAT_EQ(std::imag(diff), 2.0f);
}

TEST_F(complex_types_pre_checkin, FloatMultiplication)
{
    // (3+4i) * (1+2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    rocsparse_float_complex prod = a * b;
    EXPECT_FLOAT_EQ(std::real(prod), -5.0f);
    EXPECT_FLOAT_EQ(std::imag(prod), 10.0f);
}

TEST_F(complex_types_pre_checkin, FloatDivision)
{
    // (3+4i) / (1+2i) = 2.2 - 0.4i
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    rocsparse_float_complex quot = a / b;
    EXPECT_NEAR(std::real(quot), 2.2f, 0.001f);
    EXPECT_NEAR(std::imag(quot), -0.4f, 0.001f);
}

// Test Compound Assignment
TEST_F(complex_types_pre_checkin, FloatAddAssign)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    a += b;
    EXPECT_FLOAT_EQ(std::real(a), 4.0f);
    EXPECT_FLOAT_EQ(std::imag(a), 6.0f);
}

TEST_F(complex_types_pre_checkin, FloatSubtractAssign)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    a -= b;
    EXPECT_FLOAT_EQ(std::real(a), 2.0f);
    EXPECT_FLOAT_EQ(std::imag(a), 2.0f);
}

TEST_F(complex_types_pre_checkin, FloatMultiplyAssign)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    a *= b;
    EXPECT_FLOAT_EQ(std::real(a), -5.0f);
    EXPECT_FLOAT_EQ(std::imag(a), 10.0f);
}

TEST_F(complex_types_pre_checkin, FloatDivideAssign)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(1.0f, 2.0f);
    a /= b;
    EXPECT_NEAR(std::real(a), 2.2f, 0.001f);
    EXPECT_NEAR(std::imag(a), -0.4f, 0.001f);
}

// Test Comparison
TEST_F(complex_types_pre_checkin, FloatEquality)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(3.0f, 4.0f);
    rocsparse_float_complex c(1.0f, 2.0f);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(complex_types_pre_checkin, FloatInequality)
{
    rocsparse_float_complex a(3.0f, 4.0f);
    rocsparse_float_complex b(3.0f, 4.0f);
    rocsparse_float_complex c(1.0f, 2.0f);

    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

// Test std namespace functions
TEST_F(complex_types_pre_checkin, FloatStdConj)
{
    // (3+4i)* = 3-4i
    rocsparse_float_complex c(3.0f, 4.0f);
    rocsparse_float_complex conj = std::conj(c);
    EXPECT_FLOAT_EQ(std::real(conj), 3.0f);
    EXPECT_FLOAT_EQ(std::imag(conj), -4.0f);
}

TEST_F(complex_types_pre_checkin, FloatStdAbs)
{
    // |3+4i| = sqrt(9+16) = 5
    rocsparse_float_complex c(3.0f, 4.0f);
    float                   magnitude = std::abs(c);
    EXPECT_NEAR(magnitude, 5.0f, 0.001f);
}

TEST_F(complex_types_pre_checkin, FloatStdAbsPureImaginary)
{
    rocsparse_float_complex c(0.0f, 5.0f);
    float                   magnitude = std::abs(c);
    EXPECT_NEAR(magnitude, 5.0f, 0.001f);
}

TEST_F(complex_types_pre_checkin, FloatStdAbsPureReal)
{
    rocsparse_float_complex c(5.0f, 0.0f);
    float                   magnitude = std::abs(c);
    EXPECT_NEAR(magnitude, 5.0f, 0.001f);
}

TEST_F(complex_types_pre_checkin, FloatStdAbsZero)
{
    rocsparse_float_complex c(0.0f, 0.0f);
    float                   magnitude = std::abs(c);
    EXPECT_FLOAT_EQ(magnitude, 0.0f);
}

TEST_F(complex_types_pre_checkin, FloatStdFma)
{
    // fma(p, q, r) = p*q + r
    // (2+i)(3+2i) = 6 + 4i + 3i + 2i² = 4 + 7i
    // (4+7i) + (1+i) = 5 + 8i
    rocsparse_float_complex p(2.0f, 1.0f);
    rocsparse_float_complex q(3.0f, 2.0f);
    rocsparse_float_complex r(1.0f, 1.0f);
    rocsparse_float_complex result = std::fma(p, q, r);
    EXPECT_FLOAT_EQ(std::real(result), 5.0f);
    EXPECT_FLOAT_EQ(std::imag(result), 8.0f);
}

// Test Special Cases
TEST_F(complex_types_pre_checkin, FloatMultiplicationByConjugate)
{
    // (a+bi)(a-bi) = a² + b²
    rocsparse_float_complex c(3.0f, 4.0f);
    rocsparse_float_complex c_conj = std::conj(c);
    rocsparse_float_complex prod   = c * c_conj;
    EXPECT_FLOAT_EQ(std::real(prod), 25.0f); // 3² + 4² = 25
    EXPECT_NEAR(std::imag(prod), 0.0f, 0.001f);
}

TEST_F(complex_types_pre_checkin, FloatStreamOutput)
{
    rocsparse_float_complex c(3.0f, 4.0f);
    std::ostringstream      oss;
    oss << c;

    // The output should be "(3,4)"
    std::string output = oss.str();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find('('), std::string::npos);
    EXPECT_NE(output.find(','), std::string::npos);
    EXPECT_NE(output.find(')'), std::string::npos);
}

// =============================================================================
// Double Complex Tests
// =============================================================================

TEST_F(complex_types_pre_checkin, DoubleDefaultConstructor)
{
    rocsparse_double_complex c;
    (void)c;
}

TEST_F(complex_types_pre_checkin, DoubleRealImaginaryConstructor)
{
    rocsparse_double_complex c(3.0, 4.0);
    EXPECT_DOUBLE_EQ(std::real(c), 3.0);
    EXPECT_DOUBLE_EQ(std::imag(c), 4.0);
}

TEST_F(complex_types_pre_checkin, DoubleRealOnlyConstructor)
{
    rocsparse_double_complex c(5.0);
    EXPECT_DOUBLE_EQ(std::real(c), 5.0);
    EXPECT_DOUBLE_EQ(std::imag(c), 0.0);
}

TEST_F(complex_types_pre_checkin, DoubleStdComplexConstructor)
{
    std::complex<double>     std_c(2.0, 3.0);
    rocsparse_double_complex c(std_c);
    EXPECT_DOUBLE_EQ(std::real(c), 2.0);
    EXPECT_DOUBLE_EQ(std::imag(c), 3.0);
}

TEST_F(complex_types_pre_checkin, DoubleConversionToStdComplex)
{
    rocsparse_double_complex c(3.0, 4.0);
    std::complex<double>     std_c = static_cast<std::complex<double>>(c);
    EXPECT_DOUBLE_EQ(std_c.real(), 3.0);
    EXPECT_DOUBLE_EQ(std_c.imag(), 4.0);
}

TEST_F(complex_types_pre_checkin, DoubleUnaryMinus)
{
    rocsparse_double_complex c(3.0, 4.0);
    rocsparse_double_complex neg = -c;
    EXPECT_DOUBLE_EQ(std::real(neg), -3.0);
    EXPECT_DOUBLE_EQ(std::imag(neg), -4.0);
}

TEST_F(complex_types_pre_checkin, DoubleAddition)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    rocsparse_double_complex sum = a + b;
    EXPECT_DOUBLE_EQ(std::real(sum), 4.0);
    EXPECT_DOUBLE_EQ(std::imag(sum), 6.0);
}

TEST_F(complex_types_pre_checkin, DoubleSubtraction)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    rocsparse_double_complex diff = a - b;
    EXPECT_DOUBLE_EQ(std::real(diff), 2.0);
    EXPECT_DOUBLE_EQ(std::imag(diff), 2.0);
}

TEST_F(complex_types_pre_checkin, DoubleMultiplication)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    rocsparse_double_complex prod = a * b;
    EXPECT_DOUBLE_EQ(std::real(prod), -5.0);
    EXPECT_DOUBLE_EQ(std::imag(prod), 10.0);
}

TEST_F(complex_types_pre_checkin, DoubleDivision)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    rocsparse_double_complex quot = a / b;
    EXPECT_NEAR(std::real(quot), 2.2, 0.001);
    EXPECT_NEAR(std::imag(quot), -0.4, 0.001);
}

TEST_F(complex_types_pre_checkin, DoubleAddAssign)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    a += b;
    EXPECT_DOUBLE_EQ(std::real(a), 4.0);
    EXPECT_DOUBLE_EQ(std::imag(a), 6.0);
}

TEST_F(complex_types_pre_checkin, DoubleSubtractAssign)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    a -= b;
    EXPECT_DOUBLE_EQ(std::real(a), 2.0);
    EXPECT_DOUBLE_EQ(std::imag(a), 2.0);
}

TEST_F(complex_types_pre_checkin, DoubleMultiplyAssign)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    a *= b;
    EXPECT_DOUBLE_EQ(std::real(a), -5.0);
    EXPECT_DOUBLE_EQ(std::imag(a), 10.0);
}

TEST_F(complex_types_pre_checkin, DoubleDivideAssign)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(1.0, 2.0);
    a /= b;
    EXPECT_NEAR(std::real(a), 2.2, 0.001);
    EXPECT_NEAR(std::imag(a), -0.4, 0.001);
}

TEST_F(complex_types_pre_checkin, DoubleEquality)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(3.0, 4.0);
    rocsparse_double_complex c(1.0, 2.0);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a == c);
}

TEST_F(complex_types_pre_checkin, DoubleInequality)
{
    rocsparse_double_complex a(3.0, 4.0);
    rocsparse_double_complex b(3.0, 4.0);
    rocsparse_double_complex c(1.0, 2.0);

    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
}

TEST_F(complex_types_pre_checkin, DoubleStdConj)
{
    rocsparse_double_complex c(3.0, 4.0);
    rocsparse_double_complex conj = std::conj(c);
    EXPECT_DOUBLE_EQ(std::real(conj), 3.0);
    EXPECT_DOUBLE_EQ(std::imag(conj), -4.0);
}

TEST_F(complex_types_pre_checkin, DoubleStdAbs)
{
    rocsparse_double_complex c(3.0, 4.0);
    double                   magnitude = std::abs(c);
    EXPECT_NEAR(magnitude, 5.0, 0.001);
}

TEST_F(complex_types_pre_checkin, DoubleStdAbsZero)
{
    rocsparse_double_complex c(0.0, 0.0);
    double                   magnitude = std::abs(c);
    EXPECT_DOUBLE_EQ(magnitude, 0.0);
}

TEST_F(complex_types_pre_checkin, DoubleStdFma)
{
    rocsparse_double_complex p(2.0, 1.0);
    rocsparse_double_complex q(3.0, 2.0);
    rocsparse_double_complex r(1.0, 1.0);
    rocsparse_double_complex result = std::fma(p, q, r);
    EXPECT_DOUBLE_EQ(std::real(result), 5.0);
    EXPECT_DOUBLE_EQ(std::imag(result), 8.0);
}

TEST_F(complex_types_pre_checkin, DoubleMultiplicationByConjugate)
{
    rocsparse_double_complex c(3.0, 4.0);
    rocsparse_double_complex c_conj = std::conj(c);
    rocsparse_double_complex prod   = c * c_conj;
    EXPECT_DOUBLE_EQ(std::real(prod), 25.0);
    EXPECT_NEAR(std::imag(prod), 0.0, 0.001);
}

// =============================================================================
// Cross-Precision Tests
// =============================================================================

TEST_F(complex_types_pre_checkin, FloatToDoubleConversion)
{
    rocsparse_float_complex  cf(1.5f, 2.5f);
    rocsparse_double_complex cd(cf);
    EXPECT_DOUBLE_EQ(std::real(cd), 1.5);
    EXPECT_DOUBLE_EQ(std::imag(cd), 2.5);
}

TEST_F(complex_types_pre_checkin, DoubleToFloatConversion)
{
    rocsparse_double_complex cd(1.5, 2.5);
    rocsparse_float_complex  cf(cd);
    EXPECT_FLOAT_EQ(std::real(cf), 1.5f);
    EXPECT_FLOAT_EQ(std::imag(cf), 2.5f);
}
