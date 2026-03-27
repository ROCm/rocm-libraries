// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types/Double.hpp>
#include <hipdnn_data_sdk/types/Float.hpp>
#include <hipdnn_data_sdk/types/Int32.hpp>
#include <hipdnn_data_sdk/types/Int8.hpp>
#include <hipdnn_data_sdk/types/Uint8.hpp>

#include <cmath>
#include <limits>

// These tests verify that the forwarding functions in our namespace work correctly.
// We test by calling functions explicitly with the namespace prefix to avoid ambiguity.

class TestForwardingFunctions : public ::testing::Test
{
protected:
    static constexpr float K_TOLERANCE_F = 0.0001f;
    static constexpr double K_TOLERANCE_D = 0.0001;

    static bool nearEqual(float a, float b, float tol = K_TOLERANCE_F)
    {
        return std::fabs(a - b) <= tol;
    }

    static bool nearEqual(double a, double b, double tol = K_TOLERANCE_D)
    {
        return std::fabs(a - b) <= tol;
    }
};

// ============================================================================
// Float forwarding function tests (using explicit namespace)
// ============================================================================

TEST_F(TestForwardingFunctions, FloatAbs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::abs(-5.0f), std::abs(-5.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::abs(5.0f), std::abs(5.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::abs(0.0f), std::abs(0.0f));
}

TEST_F(TestForwardingFunctions, FloatFabs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::fabs(-5.0f), std::fabs(-5.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::fabs(5.0f), std::fabs(5.0f));
}

TEST_F(TestForwardingFunctions, FloatIsnan)
{
    EXPECT_EQ(hipdnn_data_sdk::types::isnan(std::numeric_limits<float>::quiet_NaN()),
              std::isnan(std::numeric_limits<float>::quiet_NaN()));
    EXPECT_EQ(hipdnn_data_sdk::types::isnan(1.0f), std::isnan(1.0f));
}

TEST_F(TestForwardingFunctions, FloatIsinf)
{
    EXPECT_EQ(hipdnn_data_sdk::types::isinf(std::numeric_limits<float>::infinity()),
              std::isinf(std::numeric_limits<float>::infinity()));
    EXPECT_EQ(hipdnn_data_sdk::types::isinf(1.0f), std::isinf(1.0f));
}

TEST_F(TestForwardingFunctions, FloatSignbit)
{
    EXPECT_EQ(hipdnn_data_sdk::types::signbit(-1.0f), std::signbit(-1.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::signbit(1.0f), std::signbit(1.0f));
}

TEST_F(TestForwardingFunctions, FloatIsfinite)
{
    EXPECT_EQ(hipdnn_data_sdk::types::isfinite(1.0f), std::isfinite(1.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::isfinite(std::numeric_limits<float>::infinity()),
              std::isfinite(std::numeric_limits<float>::infinity()));
}

TEST_F(TestForwardingFunctions, FloatCopysign)
{
    EXPECT_EQ(hipdnn_data_sdk::types::copysign(3.0f, -1.0f), std::copysign(3.0f, -1.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::copysign(-3.0f, 1.0f), std::copysign(-3.0f, 1.0f));
}

TEST_F(TestForwardingFunctions, FloatMax)
{
    EXPECT_EQ(hipdnn_data_sdk::types::max(1.0f, 2.0f), std::fmax(1.0f, 2.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::max(2.0f, 1.0f), std::fmax(2.0f, 1.0f));
}

TEST_F(TestForwardingFunctions, FloatMin)
{
    EXPECT_EQ(hipdnn_data_sdk::types::min(1.0f, 2.0f), std::fmin(1.0f, 2.0f));
    EXPECT_EQ(hipdnn_data_sdk::types::min(2.0f, 1.0f), std::fmin(2.0f, 1.0f));
}

TEST_F(TestForwardingFunctions, FloatFloor)
{
    EXPECT_EQ(hipdnn_data_sdk::types::floor(2.7f), std::floor(2.7f));
    EXPECT_EQ(hipdnn_data_sdk::types::floor(-2.3f), std::floor(-2.3f));
}

TEST_F(TestForwardingFunctions, FloatCeil)
{
    EXPECT_EQ(hipdnn_data_sdk::types::ceil(2.3f), std::ceil(2.3f));
    EXPECT_EQ(hipdnn_data_sdk::types::ceil(-2.7f), std::ceil(-2.7f));
}

TEST_F(TestForwardingFunctions, FloatRound)
{
    EXPECT_EQ(hipdnn_data_sdk::types::round(2.3f), std::round(2.3f));
    EXPECT_EQ(hipdnn_data_sdk::types::round(2.7f), std::round(2.7f));
}

TEST_F(TestForwardingFunctions, FloatTrunc)
{
    EXPECT_EQ(hipdnn_data_sdk::types::trunc(2.7f), std::trunc(2.7f));
    EXPECT_EQ(hipdnn_data_sdk::types::trunc(-2.7f), std::trunc(-2.7f));
}

TEST_F(TestForwardingFunctions, FloatExp)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp(0.0f), std::exp(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp(1.0f), std::exp(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatLog)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log(1.0f), std::log(1.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log(2.71828f), std::log(2.71828f)));
}

TEST_F(TestForwardingFunctions, FloatSqrt)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sqrt(4.0f), std::sqrt(4.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sqrt(9.0f), std::sqrt(9.0f)));
}

TEST_F(TestForwardingFunctions, FloatRsqrt)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::rsqrt(4.0f), 1.0f / std::sqrt(4.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::rsqrt(9.0f), 1.0f / std::sqrt(9.0f)));
}

TEST_F(TestForwardingFunctions, FloatPow)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::pow(2.0f, 3.0f), std::pow(2.0f, 3.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::pow(3.0f, 2.0f), std::pow(3.0f, 2.0f)));
}

TEST_F(TestForwardingFunctions, FloatTanh)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tanh(0.0f), std::tanh(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tanh(1.0f), std::tanh(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatFma)
{
    EXPECT_TRUE(
        nearEqual(hipdnn_data_sdk::types::fma(2.0f, 3.0f, 1.0f), std::fma(2.0f, 3.0f, 1.0f)));
}

TEST_F(TestForwardingFunctions, FloatExp2)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp2(2.0f), std::exp2(2.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp2(0.0f), std::exp2(0.0f)));
}

TEST_F(TestForwardingFunctions, FloatLog2)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log2(4.0f), std::log2(4.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log2(8.0f), std::log2(8.0f)));
}

TEST_F(TestForwardingFunctions, FloatLog10)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log10(10.0f), std::log10(10.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log10(100.0f), std::log10(100.0f)));
}

TEST_F(TestForwardingFunctions, FloatSin)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sin(0.0f), std::sin(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sin(1.0f), std::sin(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatCos)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cos(0.0f), std::cos(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cos(1.0f), std::cos(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatTan)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tan(0.0f), std::tan(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tan(0.5f), std::tan(0.5f)));
}

TEST_F(TestForwardingFunctions, FloatAsin)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::asin(0.0f), std::asin(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::asin(0.5f), std::asin(0.5f)));
}

TEST_F(TestForwardingFunctions, FloatAcos)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::acos(0.0f), std::acos(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::acos(0.5f), std::acos(0.5f)));
}

TEST_F(TestForwardingFunctions, FloatAtan)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::atan(0.0f), std::atan(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::atan(1.0f), std::atan(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatSinh)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sinh(0.0f), std::sinh(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sinh(1.0f), std::sinh(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatCosh)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cosh(0.0f), std::cosh(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cosh(1.0f), std::cosh(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatErf)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::erf(0.0f), std::erf(0.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::erf(1.0f), std::erf(1.0f)));
}

TEST_F(TestForwardingFunctions, FloatFmod)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::fmod(5.0f, 2.0f), std::fmod(5.0f, 2.0f)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::fmod(7.5f, 3.0f), std::fmod(7.5f, 3.0f)));
}

// ============================================================================
// Double forwarding function tests
// ============================================================================

TEST_F(TestForwardingFunctions, DoubleAbs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::abs(-5.0), std::abs(-5.0));
    EXPECT_EQ(hipdnn_data_sdk::types::abs(5.0), std::abs(5.0));
}

TEST_F(TestForwardingFunctions, DoubleFabs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::fabs(-5.0), std::fabs(-5.0));
    EXPECT_EQ(hipdnn_data_sdk::types::fabs(5.0), std::fabs(5.0));
}

TEST_F(TestForwardingFunctions, DoubleIsnan)
{
    EXPECT_EQ(hipdnn_data_sdk::types::isnan(std::numeric_limits<double>::quiet_NaN()),
              std::isnan(std::numeric_limits<double>::quiet_NaN()));
    EXPECT_EQ(hipdnn_data_sdk::types::isnan(1.0), std::isnan(1.0));
}

TEST_F(TestForwardingFunctions, DoubleSqrt)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sqrt(4.0), std::sqrt(4.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sqrt(9.0), std::sqrt(9.0)));
}

TEST_F(TestForwardingFunctions, DoubleTanh)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tanh(0.0), std::tanh(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tanh(1.0), std::tanh(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleIsinf)
{
    EXPECT_EQ(hipdnn_data_sdk::types::isinf(std::numeric_limits<double>::infinity()),
              std::isinf(std::numeric_limits<double>::infinity()));
    EXPECT_EQ(hipdnn_data_sdk::types::isinf(1.0), std::isinf(1.0));
}

TEST_F(TestForwardingFunctions, DoubleSignbit)
{
    EXPECT_EQ(hipdnn_data_sdk::types::signbit(-1.0), std::signbit(-1.0));
    EXPECT_EQ(hipdnn_data_sdk::types::signbit(1.0), std::signbit(1.0));
}

TEST_F(TestForwardingFunctions, DoubleIsfinite)
{
    EXPECT_EQ(hipdnn_data_sdk::types::isfinite(1.0), std::isfinite(1.0));
    EXPECT_EQ(hipdnn_data_sdk::types::isfinite(std::numeric_limits<double>::infinity()),
              std::isfinite(std::numeric_limits<double>::infinity()));
}

TEST_F(TestForwardingFunctions, DoubleCopysign)
{
    EXPECT_EQ(hipdnn_data_sdk::types::copysign(3.0, -1.0), std::copysign(3.0, -1.0));
    EXPECT_EQ(hipdnn_data_sdk::types::copysign(-3.0, 1.0), std::copysign(-3.0, 1.0));
}

TEST_F(TestForwardingFunctions, DoubleMax)
{
    EXPECT_EQ(hipdnn_data_sdk::types::max(1.0, 2.0), std::fmax(1.0, 2.0));
    EXPECT_EQ(hipdnn_data_sdk::types::max(2.0, 1.0), std::fmax(2.0, 1.0));
}

TEST_F(TestForwardingFunctions, DoubleMin)
{
    EXPECT_EQ(hipdnn_data_sdk::types::min(1.0, 2.0), std::fmin(1.0, 2.0));
    EXPECT_EQ(hipdnn_data_sdk::types::min(2.0, 1.0), std::fmin(2.0, 1.0));
}

TEST_F(TestForwardingFunctions, DoubleFloor)
{
    EXPECT_EQ(hipdnn_data_sdk::types::floor(2.7), std::floor(2.7));
    EXPECT_EQ(hipdnn_data_sdk::types::floor(-2.3), std::floor(-2.3));
}

TEST_F(TestForwardingFunctions, DoubleCeil)
{
    EXPECT_EQ(hipdnn_data_sdk::types::ceil(2.3), std::ceil(2.3));
    EXPECT_EQ(hipdnn_data_sdk::types::ceil(-2.7), std::ceil(-2.7));
}

TEST_F(TestForwardingFunctions, DoubleRound)
{
    EXPECT_EQ(hipdnn_data_sdk::types::round(2.3), std::round(2.3));
    EXPECT_EQ(hipdnn_data_sdk::types::round(2.7), std::round(2.7));
}

TEST_F(TestForwardingFunctions, DoubleTrunc)
{
    EXPECT_EQ(hipdnn_data_sdk::types::trunc(2.7), std::trunc(2.7));
    EXPECT_EQ(hipdnn_data_sdk::types::trunc(-2.7), std::trunc(-2.7));
}

TEST_F(TestForwardingFunctions, DoubleExp)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp(0.0), std::exp(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp(1.0), std::exp(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleExp2)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp2(2.0), std::exp2(2.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::exp2(0.0), std::exp2(0.0)));
}

TEST_F(TestForwardingFunctions, DoubleLog)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log(1.0), std::log(1.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log(2.71828), std::log(2.71828)));
}

TEST_F(TestForwardingFunctions, DoubleLog2)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log2(4.0), std::log2(4.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log2(8.0), std::log2(8.0)));
}

TEST_F(TestForwardingFunctions, DoubleLog10)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log10(10.0), std::log10(10.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::log10(100.0), std::log10(100.0)));
}

TEST_F(TestForwardingFunctions, DoubleRsqrt)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::rsqrt(4.0), 1.0 / std::sqrt(4.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::rsqrt(9.0), 1.0 / std::sqrt(9.0)));
}

TEST_F(TestForwardingFunctions, DoublePow)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::pow(2.0, 3.0), std::pow(2.0, 3.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::pow(3.0, 2.0), std::pow(3.0, 2.0)));
}

TEST_F(TestForwardingFunctions, DoubleSin)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sin(0.0), std::sin(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sin(1.0), std::sin(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleCos)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cos(0.0), std::cos(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cos(1.0), std::cos(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleTan)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tan(0.0), std::tan(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::tan(0.5), std::tan(0.5)));
}

TEST_F(TestForwardingFunctions, DoubleAsin)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::asin(0.0), std::asin(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::asin(0.5), std::asin(0.5)));
}

TEST_F(TestForwardingFunctions, DoubleAcos)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::acos(0.0), std::acos(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::acos(0.5), std::acos(0.5)));
}

TEST_F(TestForwardingFunctions, DoubleAtan)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::atan(0.0), std::atan(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::atan(1.0), std::atan(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleSinh)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sinh(0.0), std::sinh(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::sinh(1.0), std::sinh(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleCosh)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cosh(0.0), std::cosh(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::cosh(1.0), std::cosh(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleErf)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::erf(0.0), std::erf(0.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::erf(1.0), std::erf(1.0)));
}

TEST_F(TestForwardingFunctions, DoubleFmod)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::fmod(5.0, 2.0), std::fmod(5.0, 2.0)));
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::fmod(7.5, 3.0), std::fmod(7.5, 3.0)));
}

TEST_F(TestForwardingFunctions, DoubleFma)
{
    EXPECT_TRUE(nearEqual(hipdnn_data_sdk::types::fma(2.0, 3.0, 1.0), std::fma(2.0, 3.0, 1.0)));
}

// ============================================================================
// Integer type forwarding function tests
// ============================================================================

TEST_F(TestForwardingFunctions, Int8Abs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::abs(int8_t{-5}), 5);
    EXPECT_EQ(hipdnn_data_sdk::types::abs(int8_t{5}), 5);
}

TEST_F(TestForwardingFunctions, Int8Max)
{
    EXPECT_EQ(hipdnn_data_sdk::types::max(int8_t{-5}, int8_t{10}), 10);
    EXPECT_EQ(hipdnn_data_sdk::types::max(int8_t{10}, int8_t{-5}), 10);
    EXPECT_EQ(hipdnn_data_sdk::types::max(int8_t{-10}, int8_t{-5}), -5);
}

TEST_F(TestForwardingFunctions, Int8Min)
{
    EXPECT_EQ(hipdnn_data_sdk::types::min(int8_t{-5}, int8_t{10}), -5);
    EXPECT_EQ(hipdnn_data_sdk::types::min(int8_t{10}, int8_t{-5}), -5);
    EXPECT_EQ(hipdnn_data_sdk::types::min(int8_t{-10}, int8_t{-5}), -10);
}

TEST_F(TestForwardingFunctions, Int32Abs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::abs(int32_t{-1000}), 1000);
    EXPECT_EQ(hipdnn_data_sdk::types::abs(int32_t{1000}), 1000);
}

TEST_F(TestForwardingFunctions, Uint8Max)
{
    EXPECT_EQ(hipdnn_data_sdk::types::max(uint8_t{5}, uint8_t{10}), 10);
    EXPECT_EQ(hipdnn_data_sdk::types::max(uint8_t{10}, uint8_t{5}), 10);
}

TEST_F(TestForwardingFunctions, Uint8Min)
{
    EXPECT_EQ(hipdnn_data_sdk::types::min(uint8_t{5}, uint8_t{10}), 5);
    EXPECT_EQ(hipdnn_data_sdk::types::min(uint8_t{10}, uint8_t{5}), 5);
}

TEST_F(TestForwardingFunctions, Int32Max)
{
    EXPECT_EQ(hipdnn_data_sdk::types::max(int32_t{-100}, int32_t{100}), 100);
    EXPECT_EQ(hipdnn_data_sdk::types::max(int32_t{100}, int32_t{-100}), 100);
}

TEST_F(TestForwardingFunctions, Int32Min)
{
    EXPECT_EQ(hipdnn_data_sdk::types::min(int32_t{-100}, int32_t{100}), -100);
    EXPECT_EQ(hipdnn_data_sdk::types::min(int32_t{100}, int32_t{-100}), -100);
}
