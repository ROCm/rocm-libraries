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

#include "float_16_data_gen.hpp"
#include "scale.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <time.h>
#include <vector>

using namespace DGen;
using DT = DGen::bf16;

constexpr double NotANumber = Constants::QNaN;
constexpr double MAXNORM    = constexpr_pow(2, 127)
                           * (1 + constexpr_pow(2, -1) + constexpr_pow(2, -2) + constexpr_pow(2, -3)
                              + constexpr_pow(2, -4) + constexpr_pow(2, -5) + constexpr_pow(2, -6)
                              + constexpr_pow(2, -7));
constexpr double MAXSUBNORM
    = constexpr_pow(2, 1 - 127)
      * (constexpr_pow(2, -1) + constexpr_pow(2, -2) + constexpr_pow(2, -3) + constexpr_pow(2, -4)
         + constexpr_pow(2, -5) + constexpr_pow(2, -6) + constexpr_pow(2, -7));
constexpr double MINSUBNORM = constexpr_pow(2, 1 - 126) * constexpr_pow(2, -7);

class bf16_test : public ::testing::Test
{
protected:
    int const NUMS = 65536;

    std::vector<float> bf16ValuesSorted;
    std::vector<float> bf16Values;

    // [bf16] Element Data
    // clang-format off
    const uint8_t data[20] = {
        0b00000000, 0b00000000, // 0
        0b10000000, 0b00111111, // 1
        0b11000000, 0b01111111, // NaN 1
        0b11000010, 0b01111111, // NaN 2
        0b11000011, 0b01111111, // NaN 3
        0b10000000, 0b01111111, // inf
        0b10000000, 0b00000000, //  (min norm)
        0b01111111, 0b01111111, //  (max norm)
        0b00000001, 0b00000000, //  (min subnorm)
        0b01111111, 0b00000000  //  (max subnorm)
    };
    // [bf16] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[20] = {
        0b00000000, 0b10000000,  // 0
        0b10000000, 0b10111111,  // 1
        0b11000000, 0b11111111,  // NaN 1
        0b11000010, 0b11111111,  // NaN 2
        0b11000011, 0b11111111,  // NaN 3
        0b10000000, 0b11111111,  // inf
        0b10000000, 0b10000000,  //  (min norm)
        0b01111111, 0b11111111,  //  (max norm)
        0b00000001, 0b10000000,  //  (min subnorm)
        0b01111111, 0b10000000  //  (max subnorm)
    };

    // clang-format on

    // FP8 is already packed as is, so packed/unpacked data and methods are same
    void SetUp() override
    {
        bf16ValuesSorted.resize(NUMS);
        bf16Values.resize(NUMS);

        for(int i = 0; i < NUMS; i++)
        {
            float x = convert(i, 7, 8, 127);
            if(!std::isnan(x))
                bf16ValuesSorted[i] = x;
            bf16Values[i] = x;
        }

        std::sort(bf16ValuesSorted.begin(), bf16ValuesSorted.end());
    }

    double getClosestDiff(double num)
    {
        auto lowerIt  = std::lower_bound(bf16ValuesSorted.begin(), bf16ValuesSorted.end(), num);
        auto higherIt = std::upper_bound(bf16ValuesSorted.begin(), bf16ValuesSorted.end(), num);

        lowerIt--;

        return std::abs(*lowerIt - num) < std::abs(*higherIt - num) ? std::abs(*lowerIt - num)
                                                                    : std::abs(*higherIt - num);
    }
};

TEST_F(bf16_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(true, isOne<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOne<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOne<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isOnePacked)
{
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(true, isOnePacked<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOnePacked<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOnePacked<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZero<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(true, isZero<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZero<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isZeroPacked)
{
    EXPECT_EQ(true, isZeroPacked<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZeroPacked<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(true, isZeroPacked<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZeroPacked<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(true, isNaN<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaN<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaN<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaN<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(true, isNaN<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaN<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaN<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaN<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isNaNPacked)
{
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(true, isNaNPacked<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaNPacked<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaNPacked<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaNPacked<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(true, isNaNPacked<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaNPacked<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaNPacked<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaNPacked<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isInf)
{
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInf<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInf<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInf<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInf<DT>(data, negativeData, 0, 9)); // max sub-normal
}

TEST_F(bf16_test, isInfPacked)
{
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 0)); // 0
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 1)); // 1
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInfPacked<DT>(data, data, 0, 5)); // Inf
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 6)); // min normal
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 7)); // max normal
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInfPacked<DT>(data, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInfPacked<DT>(data, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInfPacked<DT>(data, negativeData, 0, 9)); // max sub-normal
}

// true if X < val (first arg)
TEST_F(bf16_test, isLess)
{
    double values[] = {Constants::NegInf,
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
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(data, data, i, j);
                double negProd = toDouble<DT>(data, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLess<DT>(values[k], data, data, i, j));
                EXPECT_EQ(negProd < values[k], isLess<DT>(values[k], data, negativeData, i, j));
            }
        }
    }
}

TEST_F(bf16_test, isLessPacked)
{
    double values[] = {Constants::NegInf,
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
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(data, data, i, j);
                double negProd = toDouble<DT>(data, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLessPacked<DT>(values[k], data, data, i, j));
                EXPECT_EQ(negProd < values[k],
                          isLessPacked<DT>(values[k], data, negativeData, i, j));
            }
        }
    }
}

// true if X > val (first arg)
TEST_F(bf16_test, isGreater)
{
    double values[] = {Constants::NegInf,
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
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(data, data, i, j);
                double negProd = toDouble<DT>(data, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreater<DT>(values[k], data, data, i, j));
                EXPECT_EQ(negProd > values[k], isGreater<DT>(values[k], data, negativeData, i, j));
            }
        }
    }
}

TEST_F(bf16_test, isGreaterPacked)
{
    double values[] = {Constants::NegInf,
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
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(data, data, i, j);
                double negProd = toDouble<DT>(data, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreaterPacked<DT>(values[k], data, data, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreaterPacked<DT>(values[k], data, negativeData, i, j));
            }
        }
    }
}

TEST_F(bf16_test, toFloatAlldataAllValues)
{
    for(int i = 0; i < NUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            float res      = toFloat<DT>(e8m0Bits, temp, i, 0);
            float expected = bf16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));

            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(bf16_test, toFloatAlldataAllValuesPacked)
{
    for(int i = 0; i < NUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            float res      = toFloatPacked<DT>(e8m0Bits, temp, i, 0);
            float expected = bf16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));

            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(bf16_test, toDoubleAlldataAllValues)
{
    for(int i = 0; i < NUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            double res      = toDouble<DT>(e8m0Bits, temp, i, 0);
            double expected = bf16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));

            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(bf16_test, toDoubleAlldataAllValuesPacked)
{
    for(int i = 0; i < NUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            double res      = toDoublePacked<DT>(e8m0Bits, temp, i, 0);
            double expected = bf16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));

            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(bf16_test, setOne)
{
    uint8_t data[] = {0b0, 0b0};
    setOne<DT>(data, data, 0, 0);
    double dElem = toDouble<DT>(data, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(bf16_test, setOnePacked)
{
    uint8_t data[] = {0b0, 0b0};
    setOnePacked<DT>(data, data, 0, 0);
    double dElem = toDouble<DT>(data, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(bf16_test, setZero)
{
    uint8_t data[] = {0b0, 0b0};
    setZero<DT>(data, data, 0, 0);
    double dElem = toDouble<DT>(data, data, 0, 0);
    EXPECT_EQ(0, dElem);
}

TEST_F(bf16_test, setZeroPacked)
{
    uint8_t data[] = {0b0, 0b0};
    setZeroPacked<DT>(data, data, 0, 0);
    double dElem = toDouble<DT>(data, data, 0, 0);
    EXPECT_EQ(0, dElem);
}

TEST_F(bf16_test, setNaN)
{
    uint8_t data[] = {0b0, 0b0};
    setNaN<DT>(data, data, 0, 0);
    double dElem = toDouble<DT>(data, data, 0, 0);
    EXPECT_TRUE(std::isnan(dElem));
}

TEST_F(bf16_test, setNaNPacked)
{
    uint8_t data[] = {0b0, 0b0};
    setNaNPacked<DT>(data, data, 0, 0);
    double dElem = toDouble<DT>(data, data, 0, 0);
    EXPECT_TRUE(std::isnan(dElem));
}

TEST_F(bf16_test, setDataMaxNorm)
{
    uint8_t data[] = {0b0, 0b0};

    setDataMax<DT>(data, 0);
    EXPECT_EQ(MAXNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, false, false);
    EXPECT_EQ(-MAXNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, true, true);
    EXPECT_EQ(MAXSUBNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, true, false);
    EXPECT_EQ(-MAXSUBNORM, toDouble<DT>(data, data, 0, 0));
}

TEST_F(bf16_test, setDataMaxNormPacked)
{
    uint8_t data[] = {0b0, 0b0};

    setDataMaxPacked<DT>(data, 0);
    EXPECT_EQ(MAXNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, false, false);
    EXPECT_EQ(-MAXNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, true, true);
    EXPECT_EQ(MAXSUBNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, true, false);
    EXPECT_EQ(-MAXSUBNORM, toDoublePacked<DT>(data, data, 0, 0));
}

// Saturated tests
TEST_F(bf16_test, satConvertToTypeRounding)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1.567;
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)))
        << norm << "\n"
        << toDouble<DT>(temp, temp, 0, 0);
}

TEST_F(bf16_test, satConvertToTypeRoundingSmallSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MINSUBNORM + 1e-8;
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MINSUBNORM + 1e-8);
    res  = satConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(bf16_test, satConvertToTypeRoundingLargeSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1e-8;
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MAXSUBNORM - 1e-8);
    res  = satConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(bf16_test, satConvertToTypeLarge)
{
    EXPECT_EQ(0b0111111101111111, satConvertToType<DT>(1e60)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToType<DT>(-1e60)); // Expect max norm
}

TEST_F(bf16_test, satConvertToTypeSRLarge)
{
    EXPECT_EQ(0b0111111101111111, satConvertToTypeSR<DT>(1e60, 0)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToTypeSR<DT>(-1e60, 0)); // Expect max norm
    EXPECT_EQ(0b0111111101111111, satConvertToTypeSR<DT>(1e60, UINT_MAX)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToTypeSR<DT>(-1e60, UINT_MAX)); // Expect max norm
    EXPECT_EQ(0b0111111101111111, satConvertToTypeSR<DT>(1e60, UINT_MAX / 2)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToTypeSR<DT>(-1e60, UINT_MAX / 2)); // Expect max norm
}

TEST_F(bf16_test, satConvertToTypeMax)
{
    EXPECT_EQ(0b0111111101111111, satConvertToType<DT>(MAXNORM)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToType<DT>(-1 * MAXNORM)); // Expect max norm
}

TEST_F(bf16_test, satConvertToTypeSRMax)
{
    EXPECT_EQ(0b0111111101111111, satConvertToTypeSR<DT>(MAXNORM, 0)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToTypeSR<DT>(-MAXNORM, 0)); // Expect max norm
    EXPECT_EQ(0b0111111101111111, satConvertToTypeSR<DT>(MAXNORM, UINT_MAX)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, satConvertToTypeSR<DT>(-MAXNORM, UINT_MAX)); // Expect max norm
    EXPECT_EQ(0b0111111101111111, satConvertToTypeSR<DT>(MAXNORM, UINT_MAX / 2)); // Expect max norm
    EXPECT_EQ(0b1111111101111111,
              satConvertToTypeSR<DT>(-MAXNORM, UINT_MAX / 2)); // Expect max norm
}

TEST_F(bf16_test, satConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, satConvertToType<DT>(zero));
}

TEST_F(bf16_test, satConvertToTypeSRZero)
{
    EXPECT_EQ(0, satConvertToTypeSR<DT>(0, 0));
    EXPECT_EQ(0, satConvertToTypeSR<DT>(0, UINT_MAX));
    EXPECT_EQ(0, satConvertToTypeSR<DT>(0, UINT_MAX / 2));
}

TEST_F(bf16_test, satConvertToTypeNaN)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = std::numeric_limits<float>::quiet_NaN();
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
}

TEST_F(bf16_test, satConvertToTypeSRNaN)
{
    uint8_t temp[] = {0b0, 0b0};

    float norm = std::numeric_limits<float>::quiet_NaN();

    uint16_t res = satConvertToTypeSR<DT>(norm, 0);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));

    res = satConvertToTypeSR<DT>(norm, UINT_MAX);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));

    res = satConvertToTypeSR<DT>(norm, UINT_MAX / 2);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
}

// Generate 10000000 numbers and see if the conversion is good
TEST_F(bf16_test, satConvertToTypeRandom)
{

    double lb = -(MAXNORM + (MAXNORM / 5));
    double ub = MAXNORM + (MAXNORM / 5);

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[] = {0b0, 0b0};

    for(int i = 0; i < 10000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        double closestDiff = getClosestDiff(rNum);

        uint16_t res = satConvertToType<DT>(rNum);

        uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
        uint8_t msb = static_cast<uint16_t>(res) >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        if(std::abs(rNum) < MINSUBNORM)
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }
        else if(std::isnan(rNum))
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }

        EXPECT_NEAR(closestDiff, std::abs(rNum - toDouble<DT>(temp, temp, 0, 0)), 1e-40) << rNum;
    }
}

//NON SATURATED TESTS

TEST_F(bf16_test, nonSatConvertToTypeRounding)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1.567;
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)));
}

TEST_F(bf16_test, nonSatConvertToTypeRoundingSmallSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MINSUBNORM + 1e-8;
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MINSUBNORM + 1e-8);
    res  = nonSatConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(bf16_test, nonSatConvertToTypeRoundingLargeSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1e-8;
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MAXSUBNORM - 1e-8);
    res  = nonSatConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(bf16_test, nonSatConvertToTypeLarge)
{
    EXPECT_EQ(0b0111111110000000, nonSatConvertToType<DT>(1e60)); // Expect pos inf
    EXPECT_EQ(0b1111111110000000, nonSatConvertToType<DT>(-1e60)); // Expect negative inf

    uint8_t temp[] = {0b0, 0b0};

    float    largeNum = 1e60;
    uint16_t res      = nonSatConvertToType<DT>(largeNum);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(toFloat<DT>(temp, temp, 0, 0) > 0);
    EXPECT_TRUE(std::isinf(toFloat<DT>(temp, temp, 0, 0)));

    largeNum = -1e60;
    res      = nonSatConvertToType<DT>(largeNum);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;
    EXPECT_TRUE(toFloat<DT>(temp, temp, 0, 0) < 0);
    EXPECT_TRUE(std::isinf(toFloat<DT>(temp, temp, 0, 0)));
}

TEST_F(bf16_test, nonSatConvertToTypeSRLarge)
{
    EXPECT_EQ(0b0111111110000000, nonSatConvertToTypeSR<DT>(1e60, 0)); // Expect pos inf
    EXPECT_EQ(0b1111111110000000, nonSatConvertToTypeSR<DT>(-1e60, 0)); // Expect negative inf
    EXPECT_EQ(0b0111111110000000, nonSatConvertToTypeSR<DT>(1e60, UINT_MAX)); // Expect pos inf
    EXPECT_EQ(0b1111111110000000,
              nonSatConvertToTypeSR<DT>(-1e60, UINT_MAX)); // Expect negative inf
    EXPECT_EQ(0b0111111110000000, nonSatConvertToTypeSR<DT>(1e60, UINT_MAX / 2)); // Expect pos inf
    EXPECT_EQ(0b1111111110000000,
              nonSatConvertToTypeSR<DT>(-1e60, UINT_MAX / 2)); // Expect negative inf
}

TEST_F(bf16_test, nonSatConvertToTypeMax)
{
    EXPECT_EQ(0b0111111101111111, nonSatConvertToType<DT>(MAXNORM)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, nonSatConvertToType<DT>(-1 * MAXNORM)); // Expect max norm
}

TEST_F(bf16_test, nonSatConvertToTypeSRMax)
{
    EXPECT_EQ(0b0111111101111111, nonSatConvertToTypeSR<DT>(MAXNORM, 0)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, nonSatConvertToTypeSR<DT>(-MAXNORM, 0)); // Expect max norm

    EXPECT_EQ(0b0111111101111111, nonSatConvertToTypeSR<DT>(MAXNORM, UINT_MAX)); // Expect max norm
    EXPECT_EQ(0b1111111101111111, nonSatConvertToTypeSR<DT>(-MAXNORM, UINT_MAX)); // Expect max norm

    EXPECT_EQ(0b0111111101111111,
              nonSatConvertToTypeSR<DT>(MAXNORM, UINT_MAX / 2)); // Expect max norm
    EXPECT_EQ(0b1111111101111111,
              nonSatConvertToTypeSR<DT>(-MAXNORM, UINT_MAX / 2)); // Expect max norm
}

TEST_F(bf16_test, nonSatConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, nonSatConvertToType<DT>(zero));
}

TEST_F(bf16_test, nonSatConvertToTypeSRZero)
{
    EXPECT_EQ(0, nonSatConvertToTypeSR<DT>(0, 0));
    EXPECT_EQ(0, nonSatConvertToTypeSR<DT>(0, UINT_MAX));
    EXPECT_EQ(0, nonSatConvertToTypeSR<DT>(0, UINT_MAX / 2));
}

TEST_F(bf16_test, nonSatConvertToTypeNaN)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = std::numeric_limits<float>::quiet_NaN();
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
}

TEST_F(bf16_test, nonSatConvertToTypeSRNaN)
{
    uint8_t temp[] = {0b0, 0b0};

    float norm = std::numeric_limits<float>::quiet_NaN();

    uint16_t res = nonSatConvertToTypeSR<DT>(norm, 0);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));

    res = nonSatConvertToTypeSR<DT>(norm, UINT_MAX);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));

    res = nonSatConvertToTypeSR<DT>(norm, UINT_MAX / 2);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
}

// Generate 10000000 numbers and see if the conversion is good
TEST_F(bf16_test, nonSatConvertToTypeRandom)
{

    double lb = -(MAXNORM + (MAXNORM / 5));
    double ub = MAXNORM + (MAXNORM / 5);

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[] = {0b0, 0b0};

    for(int i = 0; i < 10000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        double closestDiff = getClosestDiff(rNum);

        uint16_t res = nonSatConvertToType<DT>(rNum);

        uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
        uint8_t msb = static_cast<uint16_t>(res) >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        if(std::abs(rNum) < MINSUBNORM)
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }
        else if(std::isnan(rNum))
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }
        else if(std::abs(rNum) > MAXNORM)
        {
            EXPECT_TRUE(std::isinf(toDouble<DT>(temp, temp, 0, 0)));

            if(rNum < 0)
                EXPECT_TRUE(toDouble<DT>(temp, temp, 0, 0) < 0);
            else
                EXPECT_TRUE(toDouble<DT>(temp, temp, 0, 0) > 0);
            continue;
        }

        EXPECT_NEAR(closestDiff, std::abs(rNum - toDouble<DT>(temp, temp, 0, 0)), 1e-40) << rNum;
    }
}

TEST_F(bf16_test, isSubnormal)
{
    uint8_t temp[] = {0b0, 0b0};

    for(int i = 0; i < NUMS; i++)
    {
        uint16_t data = static_cast<uint16_t>(i);

        uint8_t lsb = data & 0b11111111;
        uint8_t msb = data >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        uint16_t exp = (data >> getDataMantissaBits<DT>()) & 0xff;

        double value = toDouble<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_TRUE(!isSubnorm<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 0));
    }
}

TEST_F(bf16_test, isSubnormalPacked)
{
    uint8_t temp[] = {0b0, 0b0};

    for(int i = 0; i < NUMS; i++)
    {
        uint16_t data = static_cast<uint16_t>(i);

        uint8_t lsb = data & 0b11111111;
        uint8_t msb = data >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        uint16_t exp = (data >> getDataMantissaBits<DT>()) & 0xff;

        double value = toDouble<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_TRUE(!isSubnormPacked<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, 0));
    }
}

TEST_F(bf16_test, getDataMax)
{
    float temp = 1;
    for(int i = 1; i <= 7; i++)
        temp += std::pow(2, -i);

    float maxi = std::pow(2, 127) * temp;
    EXPECT_EQ(maxi, getDataMax<DT>());
}

TEST_F(bf16_test, getDataMin)
{
    EXPECT_EQ(std::pow(2, -126), getDataMin<DT>());
}

TEST_F(bf16_test, getDataMaxSubnorm)
{
    float subMax = 0;
    for(size_t i = 0; i < getDataMantissaBits<DT>(); i++)
        subMax += std::pow(2, -int32_t(i + 1));

    subMax *= std::pow(2, -126);

    EXPECT_EQ(subMax, getDataMaxSubnorm<DT>());
}

TEST_F(bf16_test, getDataMinSubnorm)
{
    EXPECT_EQ(std::pow(2, -133), getDataMinSubnorm<DT>());
}

TEST_F(bf16_test, roundToEvenTest)
{

    uint8_t tData[2];

    for(int i = 0; i < (1 << 16); i += 2)
    {
        float    input = (bf16Values[i] + bf16Values[i + 1]) / 2;
        uint16_t temp  = satConvertToType<DT>(input);

        uint8_t lsb = temp & 0b11111111;
        uint8_t msb = temp >> 8;

        *(tData)     = lsb;
        *(tData + 1) = msb;

        float  fOutput = toFloat<DT>(tData, tData, 0, 0);
        double dOutput = toDouble<DT>(tData, tData, 0, 0);

        if(std::isnan(input))
        {
            EXPECT_TRUE(std::isnan(fOutput) && std::isnan(dOutput));
            continue;
        }

        if(std::isinf(input))
        {
            EXPECT_TRUE(std::abs(fOutput) == std::abs(getDataMax<DT>()));
            EXPECT_TRUE(std::abs(dOutput) == std::abs(getDataMax<DT>()));
            continue;
        }

        EXPECT_EQ(bf16Values[i], fOutput);
        EXPECT_EQ(static_cast<double>(bf16Values[i]), dOutput);
    }
}

TEST_F(bf16_test, roundToZeroTest)
{
    uint8_t tData[2];

    int offset = 1 << 15;
    for(int i = 0; i < (1 << 15) - 128; i++)
    {
        float diff      = std::abs(bf16Values[i] - bf16Values[i + 1]);
        float increment = diff / 6; // 5 increments each

        float negNum = bf16Values[i + offset];
        float posNum = bf16Values[i];

        while(posNum < bf16Values[i + 1])
        {

            uint16_t temp = satConvertToTypeSR<DT>(posNum, 0);

            uint8_t lsb = temp & 0b11111111;
            uint8_t msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i] << " Right Number: " << bf16Values[i + 1]
                << "\n--- Current Input: " << posNum
                << "\n--- Output: " << toFloat<DT>(nullptr, tData, 0, 0)
                << "\n--- Bit Set Left:  " << std::bitset<16>(satConvertToType<DT>(bf16Values[i]))
                << "\n--- Bit Set Right: "
                << std::bitset<16>(satConvertToType<DT>(bf16Values[i + 1]));

            temp = satConvertToTypeSR<DT>(negNum, 0);

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i + offset], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i + offset]
                << "Right Number: " << bf16Values[i + offset + 1]
                << "\n--- Current Input: " << negNum
                << "\n--- Output: " << toFloat<DT>(nullptr, tData, 0, 0)
                << "\n--- Output: " << toFloat<DT>(nullptr, tData, 0, 0) << "\n--- Bit Set Left:  "
                << std::bitset<16>(satConvertToType<DT>(bf16Values[i + offset]))
                << "\n--- Bit Set Right: "
                << std::bitset<16>(satConvertToType<DT>(bf16Values[i + offset + 1]));

            temp = nonSatConvertToTypeSR<DT>(posNum, 0);

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i] << " Right Number: " << bf16Values[i + 1]
                << "\n--- Current Input: " << posNum
                << "\n--- Output: " << toFloat<DT>(nullptr, tData, 0, 0)
                << "\n--- Output: " << toFloat<DT>(nullptr, tData, 0, 0)
                << "\n--- Bit Set Left:  " << std::bitset<16>(satConvertToType<DT>(bf16Values[i]))
                << "\n--- Bit Set Right: "
                << std::bitset<16>(satConvertToType<DT>(bf16Values[i + 1]));

            temp = nonSatConvertToTypeSR<DT>(negNum, 0);

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i + offset], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i + offset]
                << " Right Number: " << bf16Values[i + 1 + offset]
                << "\n--- Current Input: " << negNum
                << "\n--- Output: " << toFloat<DT>(nullptr, tData, 0, 0) << "\n--- Bit Set Left:  "
                << std::bitset<16>(satConvertToType<DT>(bf16Values[i + offset]))
                << "\n--- Bit Set Right: "
                << std::bitset<16>(satConvertToType<DT>(bf16Values[i + offset + 1]));

            negNum -= increment;
            posNum += increment;
        }
    }
}

TEST_F(bf16_test, roundToNextTest)
{
    uint8_t tData[2];

    int offset = 1 << 15;
    for(int i = 0; i < (1 << 15) - 128; i++)
    {
        float diff      = std::abs(bf16Values[i] - bf16Values[i + 1]);
        float increment = diff / 6; // 5 increments each

        float negNum = bf16Values[i + offset] - increment;
        float posNum = bf16Values[i] + increment;

        while(posNum < bf16Values[i + 1])
        {

            uint16_t temp = satConvertToTypeSR<DT>(posNum, UINT_MAX);

            uint8_t lsb = temp & 0b11111111;
            uint8_t msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i + 1], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i] << " Right Number: " << bf16Values[i + 1]
                << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(nullptr, tData, 0, 0);

            temp = satConvertToTypeSR<DT>(negNum, UINT_MAX);

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i + offset + 1], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i + offset]
                << " Right Number: " << bf16Values[i + offset + 1]
                << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(nullptr, tData, 0, 0);

            temp = nonSatConvertToTypeSR<DT>(posNum, UINT_MAX);

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i + 1], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i] << " Right Number: " << bf16Values[i + 1]
                << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(nullptr, tData, 0, 0);

            temp = nonSatConvertToTypeSR<DT>(negNum, UINT_MAX);

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            EXPECT_EQ(bf16Values[i + offset + 1], toFloat<DT>(nullptr, tData, 0, 0))
                << "Left Number: " << bf16Values[i + offset]
                << " Right Number: " << bf16Values[i + 1 + offset]
                << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(nullptr, tData, 0, 0);

            negNum -= increment;
            posNum += increment;
        }
    }
}

TEST_F(bf16_test, midPointTest)
{
    uint8_t tData[2];

    u_int64_t sInc = UINT_MAX / 15;

    uint offset = 1 << 15;
    for(uint i = 0; i < (1 << 15) - 129; i++)
    {
        //cast to double to prevent float overflow
        float pMidPoint = static_cast<float>(
            (static_cast<double>(bf16Values[i]) + static_cast<double>(bf16Values[i + 1])) / 2);
        float nMidPoint = static_cast<float>((static_cast<double>(bf16Values[i + offset])
                                              + static_cast<double>(bf16Values[i + offset + 1]))
                                             / 2);

        int pSatCount = 0, nSatCount = 0, pNSatCount = 0, nNSatCount = 0;

        for(u_int64_t seed = 0; seed <= UINT_MAX; seed += sInc)
        {

            uint16_t temp = satConvertToTypeSR<DT>(pMidPoint, static_cast<uint>(seed));

            uint8_t lsb = temp & 0b11111111;
            uint8_t msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            pSatCount += toFloat<DT>(nullptr, tData, 0, 0) == bf16Values[i] ? 1 : -1;

            temp = satConvertToTypeSR<DT>(nMidPoint, static_cast<uint>(seed));

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            nSatCount += toFloat<DT>(nullptr, tData, 0, 0) == bf16Values[i + offset] ? 1 : -1;

            temp = nonSatConvertToTypeSR<DT>(pMidPoint, static_cast<uint>(seed));

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            pNSatCount += toFloat<DT>(nullptr, tData, 0, 0) == bf16Values[i] ? 1 : -1;

            temp = nonSatConvertToTypeSR<DT>(nMidPoint, static_cast<uint>(seed));

            lsb = temp & 0b11111111;
            msb = temp >> 8;

            *(tData)     = lsb;
            *(tData + 1) = msb;

            nNSatCount += toFloat<DT>(nullptr, tData, 0, 0) == bf16Values[i + offset] ? 1 : -1;
        }

        EXPECT_EQ(0, pSatCount) << "Index: " << i << " Input: " << pMidPoint
                                << " Left Num: " << bf16Values[i]
                                << " Right Num: " << bf16Values[i + 1];
        EXPECT_EQ(0, nSatCount) << "Index: " << i << " Input: " << nMidPoint
                                << " Left Num: " << bf16Values[i + offset]
                                << " Right Num: " << bf16Values[i + offset + 1];
        EXPECT_EQ(0, pNSatCount) << "Index: " << i << " Input: " << pMidPoint
                                 << " Left Num: " << bf16Values[i]
                                 << " Right Num: " << bf16Values[i + 1];
        EXPECT_EQ(0, nNSatCount) << "Index: " << i << " Input: " << nMidPoint
                                 << " Left Num: " << bf16Values[i + offset]
                                 << " Right Num: " << bf16Values[i + offset + 1];
    }
}

TEST_F(bf16_test, preserveSign)
{
    cvt t;

    t.num = std::numeric_limits<float>::quiet_NaN();

    t.bRep |= 1U << 31;

    uint16_t sOutput  = satConvertToType<DT>(t.num);
    uint16_t nSOutput = nonSatConvertToType<DT>(t.num);

    EXPECT_EQ(0b1111111111000000, sOutput);
    EXPECT_EQ(0b1111111111000000, nSOutput);
}
