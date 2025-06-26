/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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


#include <dataTypeInfo.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <time.h>

using namespace DGen;
using DT = DGen::f32;

constexpr double NotANumber = DGen::Constants::QNaN;
constexpr double MAXNORM    = std::numeric_limits<float>::max();
constexpr double MAXSUBNORM = constexpr_pow(2, -126) * (1 - constexpr_pow(2, -23));

class f32_test : public ::testing::Test
{
protected:
    // [E5M10] Element Data, stored as pairs of uint8
    const uint8_t data[40] = {
        // clang-format off
        0b00000000, 0b00000000, 0b00000000, 0b00000000, // 0
        0b00000000, 0b00000000, 0b10000000, 0b00111111, // 1
        0b00000000, 0b00000000, 0b10000001, 0b01111111, // Nan example 1
        0b00100000, 0b00000000, 0b10000000, 0b01111111, // Nan example 2
        0b00000000, 0b00100000, 0b10000000, 0b01111111, // Nan example 3
        0b00000000, 0b00000000, 0b10000000, 0b01111111, // inf
        0b00000000, 0b00000000, 0b10000000, 0b00000000, // min norm
        0b11111111, 0b11111111, 0b01111111, 0b01111111, // max norm
        0b00000001, 0b00000000, 0b00000000, 0b00000000, // min subnorm
        0b11111111, 0b11111111, 0b01111111, 0b00000000  // max subnorm
        // clang-format on
    };
    // [E5M10] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[40] = {
        // clang-format off
        0b00000000, 0b00000000, 0b00000000, 0b10000000, // 0
        0b00000000, 0b00000000, 0b10000000, 0b10111111, // 1
        0b00000000, 0b00000000, 0b10000001, 0b11111111, // Nan example 1
        0b00100000, 0b00000000, 0b10000000, 0b11111111, // Nan example 2
        0b00000000, 0b00100000, 0b10000000, 0b11111111, // Nan example 3
        0b00000000, 0b00000000, 0b10000000, 0b11111111, // inf
        0b00000000, 0b00000000, 0b10000000, 0b10000000, // min norm
        0b11111111, 0b11111111, 0b01111111, 0b11111111, // max norm
        0b00000001, 0b00000000, 0b00000000, 0b10000000, // min subnorm
        0b11111111, 0b11111111, 0b01111111, 0b10000000  // max subnorm
        // clang-format on
    };
};

TEST_F(f32_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(true, isOne<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 0)); // -0
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 1)); // -1
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 2)); // -NaN 1
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 3)); // -NaN 2
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 4)); // -NaN 3
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 5)); // -Inf
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 6)); // -min normal
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 7)); // -max normal
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 8)); // -min sub-normal
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 9)); // -max sub-normal
}

TEST_F(f32_test, isOnePacked)
{
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(true, isOnePacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(true, isZero<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isZeroPacked)
{
    EXPECT_EQ(true, isZeroPacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(true, isZeroPacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaN<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaN<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isNaNPacked)
{
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isInf)
{
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInf<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInf<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isInfPacked)
{
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInfPacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInfPacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(f32_test, isSubnorm)
{
    uint8_t temp[4];

    for(int i = 0; i < (1 << 26); i += 10)
    {
        uint data = static_cast<uint>(i);

        uint _1_8   = data & 0xff;
        uint _9_16  = (data >> 8) & 0xff;
        uint _17_24 = (data >> 16) & 0xff;
        uint _25_32 = (data >> 24) & 0xff;

        *(temp)     = _1_8;
        *(temp + 1) = _9_16;
        *(temp + 2) = _17_24;
        *(temp + 3) = _25_32;

        uint exp = (data >> getDataMantissaBits<DT>()) & 0xff;

        double value = toDouble<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_FALSE(isSubnorm<DT>(temp, 0))
                << std::bitset<16>(i) << ", " << (exp != 0b0) << ", " << std::isnan(value) << ", "
                << std::isinf(value);
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 0));
    }
}

TEST_F(f32_test, isSubnormPacked)
{
    uint8_t temp[4];

    for(int i = 0; i < (1 << 26); i += 10)
    {
        uint data = static_cast<uint>(i);

        uint _1_8   = data & 0xff;
        uint _9_16  = (data >> 8) & 0xff;
        uint _17_24 = (data >> 16) & 0xff;
        uint _25_32 = (data >> 24) & 0xff;

        *(temp)     = _1_8;
        *(temp + 1) = _9_16;
        *(temp + 2) = _17_24;
        *(temp + 3) = _25_32;

        uint exp = (data >> getDataMantissaBits<DT>()) & 0xff;

        double value = toDouble<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_FALSE(isSubnorm<DT>(temp, 0))
                << std::bitset<16>(i) << ", " << (exp != 0b0) << ", " << std::isnan(value) << ", "
                << std::isinf(value);
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, 0));
    }
}

// true if XN < val (first arg)
TEST_F(f32_test, isLess)
{
    double values[] = {DGen::Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       DGen::Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLess<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd < values[k], isLess<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

TEST_F(f32_test, isLessPacked)
{
    double values[] = {DGen::Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       DGen::Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLessPacked<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd < values[k],
                          isLessPacked<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

// true if XN > val (first arg)
TEST_F(f32_test, isGreater)
{
    double values[] = {DGen::Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       DGen::Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreater<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreater<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

TEST_F(f32_test, isGreaterPacked)
{
    double values[] = {DGen::Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       DGen::Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreaterPacked<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreaterPacked<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

TEST_F(f32_test, toFloat)
{
    float lb = -MAXNORM;
    float ub = MAXNORM;

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[4];

    for(uint i = 0; i < 100000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);
        cvt t;

        t.num = unif(re);

        uint data = static_cast<uint>(t.bRep);

        uint _1_8   = data & 0xff;
        uint _9_16  = (data >> 8) & 0xff;
        uint _17_24 = (data >> 16) & 0xff;
        uint _25_32 = (data >> 24) & 0xff;

        *(temp)     = _1_8;
        *(temp + 1) = _9_16;
        *(temp + 2) = _17_24;
        *(temp + 3) = _25_32;

        float res = toFloat<DT>(nullptr, temp, 0, 0);

        if(std::isnan(t.num)) // Don't compare NaN to NaN
            EXPECT_TRUE(std::isnan(res));
        else if(std::isinf(t.num))
            EXPECT_TRUE(std::isinf(res));
        else
            EXPECT_EQ(t.num, res);
    }
}

TEST_F(f32_test, toFloatPacked)
{
    float lb = -MAXNORM;
    float ub = MAXNORM;

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[4];

    for(uint i = 0; i < 100000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);
        cvt t;

        t.num = unif(re);

        uint data = static_cast<uint>(t.bRep);

        uint _1_8   = data & 0xff;
        uint _9_16  = (data >> 8) & 0xff;
        uint _17_24 = (data >> 16) & 0xff;
        uint _25_32 = (data >> 24) & 0xff;

        *(temp)     = _1_8;
        *(temp + 1) = _9_16;
        *(temp + 2) = _17_24;
        *(temp + 3) = _25_32;

        float res = toFloatPacked<DT>(nullptr, temp, 0, 0);

        if(std::isnan(t.num)) // Don't compare NaN to NaN
            EXPECT_TRUE(std::isnan(res));
        else if(std::isinf(t.num))
            EXPECT_TRUE(std::isinf(res));
        else
            EXPECT_EQ(t.num, res);
    }
}

TEST_F(f32_test, toDouble)
{
    float lb = -MAXNORM;
    float ub = MAXNORM;

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[4];

    for(uint i = 0; i < 100000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);
        cvt t;

        t.num = unif(re);

        double expected = t.num;

        uint data = static_cast<uint>(t.bRep);

        uint _1_8   = data & 0xff;
        uint _9_16  = (data >> 8) & 0xff;
        uint _17_24 = (data >> 16) & 0xff;
        uint _25_32 = (data >> 24) & 0xff;

        *(temp)     = _1_8;
        *(temp + 1) = _9_16;
        *(temp + 2) = _17_24;
        *(temp + 3) = _25_32;

        float res = toDouble<DT>(nullptr, temp, 0, 0);

        if(std::isnan(expected)) // Don't compare NaN to NaN
            EXPECT_TRUE(std::isnan(res));
        else if(std::isinf(expected))
            EXPECT_TRUE(std::isinf(res));
        else
            EXPECT_EQ(expected, res);
    }
}

TEST_F(f32_test, toDoublePacked)
{
    float lb = -MAXNORM;
    float ub = MAXNORM;

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[4];

    for(uint i = 0; i < 100000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);
        cvt t;

        t.num = unif(re);

        double expected = t.num;

        uint data = static_cast<uint>(t.bRep);

        uint _1_8   = data & 0xff;
        uint _9_16  = (data >> 8) & 0xff;
        uint _17_24 = (data >> 16) & 0xff;
        uint _25_32 = (data >> 24) & 0xff;

        *(temp)     = _1_8;
        *(temp + 1) = _9_16;
        *(temp + 2) = _17_24;
        *(temp + 3) = _25_32;

        float res = toDoublePacked<DT>(nullptr, temp, 0, 0);

        if(std::isnan(expected)) // Don't compare NaN to NaN
            EXPECT_TRUE(std::isnan(res));
        else if(std::isinf(expected))
            EXPECT_TRUE(std::isinf(res));
        else
            EXPECT_EQ(expected, res);
    }
}

TEST_F(f32_test, setOne)
{
    uint8_t data[4];
    setOne<DT>(nullptr, data, 0, 0);
    double dElem = toDouble<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(f32_test, setOnePacked)
{
    uint8_t data[4];
    setOnePacked<DT>(nullptr, data, 0, 0);
    double dElem = toDoublePacked<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(f32_test, setZero)
{
    uint8_t data[4];
    setZero<DT>(nullptr, data, 0, 0);
    double dElem = toDouble<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(f32_test, setZeroPacked)
{
    uint8_t data[4];
    setZeroPacked<DT>(nullptr, data, 0, 0);
    double dElem = toDoublePacked<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(f32_test, setNaN)
{
    uint8_t data[4];
    setNaN<DT>(nullptr, data, 0, 0);
    double dElem = toDouble<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(f32_test, setNaNPacked)
{
    uint8_t data[4];
    setNaNPacked<DT>(nullptr, data, 0, 0);
    double dElem = toDoublePacked<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(f32_test, setDataMax)
{
    uint8_t data[4];

    setDataMax<DT>(data, 0, false, true); // Leave optional params to normal, pos
    EXPECT_EQ(MAXNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, false, false); // Leave optional params to normal, neg
    EXPECT_EQ(-MAXNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, true, true); // Leave optional params to subnormal, pos
    EXPECT_EQ(MAXSUBNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, true, false); // Leave optional params to subnormal, neg
    EXPECT_EQ(-MAXSUBNORM, toDouble<DT>(data, data, 0, 0));
}

TEST_F(f32_test, setDataMaxNormPacked)
{
    uint8_t data[4];

    setDataMaxPacked<DT>(data, 0);
    EXPECT_EQ(MAXNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, false, false);
    EXPECT_EQ(-MAXNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, true, true);
    EXPECT_EQ(MAXSUBNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, true, false);
    EXPECT_EQ(-MAXSUBNORM, toDoublePacked<DT>(data, data, 0, 0));
}

TEST_F(f32_test, getDataMax)
{

    EXPECT_EQ(std::numeric_limits<float>::max(), getDataMax<DT>());
}

TEST_F(f32_test, getDataMin)
{

    EXPECT_EQ(std::numeric_limits<float>::min(), getDataMin<DT>());
}

TEST_F(f32_test, getDataMaxSubnorm)
{

    float subMax = 0;
    for(size_t i = 0; i < f32::dataInfo.mantissaBits; i++)
        subMax += std::pow(2, -int32_t(i + 1));

    subMax *= std::pow(2, -126);

    EXPECT_EQ(subMax, getDataMaxSubnorm<DT>());
}

TEST_F(f32_test, getDataMinSubnorm)
{

    EXPECT_EQ(std::pow(2, -149), getDataMinSubnorm<DT>());
}

// Just runs all tests from this file only
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    auto retval = RUN_ALL_TESTS();
    return retval;
}
