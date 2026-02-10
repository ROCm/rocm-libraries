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
    static constexpr float K_TOLERANCE_F = 0.0001f; // NOLINT(readability-identifier-naming)
    static constexpr double K_TOLERANCE_D = 0.0001; // NOLINT(readability-identifier-naming)

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

// ============================================================================
// Integer type forwarding function tests
// ============================================================================

TEST_F(TestForwardingFunctions, Int8Abs)
{
    EXPECT_EQ(hipdnn_data_sdk::types::abs(int8_t{-5}), 5);
    EXPECT_EQ(hipdnn_data_sdk::types::abs(int8_t{5}), 5);
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
