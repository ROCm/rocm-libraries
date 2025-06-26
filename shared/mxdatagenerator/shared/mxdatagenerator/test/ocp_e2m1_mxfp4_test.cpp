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

#include <iostream>
#include <limits>

#include "scale.hpp"

using namespace DGen;
using DT = DGen::ocp_e2m1_mxfp4;

constexpr double Inf        = std::numeric_limits<double>::infinity();
constexpr double NegInf     = -std::numeric_limits<double>::infinity();
constexpr double NotANumber = Constants::QNaN;
constexpr double EPSILON = 0.5; // 2^-m, for (m)antissa = 1 bit; diff between 1.0 and next fp (1.5)

constexpr std::array<float, 16> e2m1ValuesOCP = {
    // clang-format off
    0.0000000000, 0.5000000000,
    1.0000000000, 1.5000000000,
    2.0000000000, 3.0000000000,
    4.0000000000, 6.0000000000,
    -0.0000000000, -0.5000000000,
    -1.0000000000, -1.5000000000,
    -2.0000000000, -3.0000000000,
    -4.0000000000, -6.0000000000
    // clang-format on
};

constexpr uint8_t e2m1BitsOCP[] = {
    // clang-format off
    0b0000, 0b0001,
    0b0010, 0b0011,
    0b0100, 0b0101,
    0b0110, 0b0111,
    0b1000, 0b1001,
    0b1010, 0b1011,
    0b1100, 0b1101,
    0b1110, 0b1111
    // clang-format on
};

constexpr uint8_t e2m1BitsOCPPacked[] = {
    // clang-format off
    0b00010000, 0b00110010,
    0b01010100, 0b01110110,
    0b10011000, 0b10111010,
    0b11011100, 0b11111110
    // clang-format on
};

class ocp_e2m1_mxfp4_test : public ::testing::Test
{
protected:
    // std::shared_ptr<DGen::DataTypeInfo> info;

    // [E8M0] Scale Data (no zero, inf)
    const uint8_t scales[6] = {
        Constants::E8M0_1, // 1
        Constants::E8M0_NAN, // NaN
        0b01111000, // 0.0078125
        0b10000101, // 64
        Constants::E8M0_MIN, // 2^-127 (min)
        Constants::E8M0_MAX // 2^127 (max)
    };
    // [E2M1] Element Data (no inf, NaN)
    const uint8_t data[6] = {
        0b0000, // 0
        0b0010, // 1 (min norm)
        0b0100, // 2.0
        0b0111, // 6.0 (max norm)
        0b0001, // 0.5 (min subnorm)
        0b0001 // 0.5 (max subnorm)
    };
    // [E2M1] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[6] = {
        0b1000, // -0
        0b1010, // -1 (min norm)
        0b1100, // -2.0
        0b1111, // -6.0 (max norm)
        0b1001, // -0.125 (min subnorm)
        0b1001 // -0.875 (max subnorm)
    };
    // Packed variants of data[], negativeData[]
    const uint8_t packedData[3]         = {0b00100000, 0b01110100, 0b00010001};
    const uint8_t negativePackedData[3] = {0b10101000, 0b11111100, 0b10011001};

    void SetUp() override
    {
        // info = DGen::DataTypeInfo::Get(DGen::OCP_E2M1_MXFP4);
    }

    float getClosest(float num)
    {
        float closestDiff = 500;

        for(size_t i = 0; i < e2m1ValuesOCP.size(); i++)
        {
            if(std::isnan(e2m1ValuesOCP[i]))
                continue;
            closestDiff = std::min(closestDiff, std::abs(num - e2m1ValuesOCP[i]));
        }
        return closestDiff;
    }
};

TEST_F(ocp_e2m1_mxfp4_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(true, isOne<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isOnePacked)
{
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(true, isOnePacked<DT>(scales, packedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 4)); // NaN * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 4)); // 64 * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isZeroPacked)
{
    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 4)); // 1 * 0.5

    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 3)); // NaN * 6
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 3)); // NaN * 6
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isNaNPacked)
{
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 3)); // NaN * 6
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 6
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isInf)
{
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.5
}

TEST_F(ocp_e2m1_mxfp4_test, isInfPacked)
{
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, data, 5, 4)); // 2^127 * 0.5

    // =============================================================================== //

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.5

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 6
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.5
}

// true if XN < val (first arg)
TEST_F(ocp_e2m1_mxfp4_test, isLess)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(scales, data, i, j);
                double negProd = toDouble<DT>(scales, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLess<DT>(values[k], scales, data, i, j));
                EXPECT_EQ(negProd < values[k], isLess<DT>(values[k], scales, negativeData, i, j));
            }
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, isLessPacked)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(scales, data, i, j);
                double negProd = toDouble<DT>(scales, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLessPacked<DT>(values[k], scales, packedData, i, j));
                EXPECT_EQ(negProd < values[k],
                          isLessPacked<DT>(values[k], scales, negativePackedData, i, j));
            }
        }
    }
}

// true if XN > val (first arg)
TEST_F(ocp_e2m1_mxfp4_test, isGreater)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(scales, data, i, j);
                double negProd = toDouble<DT>(scales, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreater<DT>(values[k], scales, data, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreater<DT>(values[k], scales, negativeData, i, j));
            }
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, isGreaterPacked)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 6; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(scales, data, i, j);
                double negProd = toDouble<DT>(scales, negativeData, i, j);

                EXPECT_EQ(prod > values[k],
                          isGreaterPacked<DT>(values[k], scales, packedData, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreaterPacked<DT>(values[k], scales, negativePackedData, i, j));
            }
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, toFloatAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        float ref = e8m0Values[i];
        for(size_t j = 0; j < e2m1ValuesOCP.size(); j++)
        {
            float  res      = toFloat<DT>(e8m0Bits, e2m1BitsOCP, i, j);
            double expected = ref * e2m1ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref));
            else if(expected > FLT_MAX)
                EXPECT_EQ(std::numeric_limits<double>::infinity(), res);
            else if(expected < -FLT_MAX)
                EXPECT_EQ(-std::numeric_limits<double>::infinity(), res);
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, toFloatAllScalesAllValuesPacked)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        float ref = e8m0Values[i];
        for(size_t j = 0; j < e2m1ValuesOCP.size(); j++)
        {
            float  res      = toFloatPacked<DT>(e8m0Bits, e2m1BitsOCPPacked, i, j);
            double expected = ref * e2m1ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref));
            else if(expected > FLT_MAX)
                EXPECT_EQ(std::numeric_limits<double>::infinity(), res);
            else if(expected < -FLT_MAX)
                EXPECT_EQ(-std::numeric_limits<double>::infinity(), res);
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, toDoubleAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        double ref = e8m0Values[i];

        for(size_t j = 0; j < e2m1ValuesOCP.size(); j++)
        {
            double res      = toDouble<DT>(e8m0Bits, e2m1BitsOCP, i, j);
            double expected = ref * e2m1ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref));
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, toDoubleAllScalesAllValuesPacked)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        double ref = e8m0Values[i];

        for(size_t j = 0; j < e2m1ValuesOCP.size(); j++)
        {
            double res      = toDoublePacked<DT>(e8m0Bits, e2m1BitsOCPPacked, i, j);
            double expected = ref * e2m1ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref));
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, setOne)
{
    uint8_t scale[] = {0b0};
    uint8_t data[]  = {0b0};
    setOne<DT>(scale, data, 0, 0);
    double dElem = toDouble<DT>(scale, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(ocp_e2m1_mxfp4_test, setOnePacked)
{
    uint8_t scale[]
        = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};
    uint8_t data[] = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};

    setOnePacked<DT>(scale, data, 0, 0);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 0, 0));
    setOnePacked<DT>(scale, data, 1, 1);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 1, 1));
    setOnePacked<DT>(scale, data, 2, 2);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 2, 2));
    setOnePacked<DT>(scale, data, 3, 3);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 3, 3));
    setOnePacked<DT>(scale, data, 4, 4);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 4, 4));
    setOnePacked<DT>(scale, data, 5, 5);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 5, 5));
    setOnePacked<DT>(scale, data, 6, 6);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 6, 6));
    setOnePacked<DT>(scale, data, 7, 7);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 7, 7));
    setOnePacked<DT>(scale, data, 8, 8);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 8, 8));
    setOnePacked<DT>(scale, data, 9, 9);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 9, 9));
    setOnePacked<DT>(scale, data, 10, 10);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 10, 10));
    setOnePacked<DT>(scale, data, 11, 11);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 11, 11));
    setOnePacked<DT>(scale, data, 12, 12);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 12, 12));
    setOnePacked<DT>(scale, data, 13, 13);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 13, 13));
    setOnePacked<DT>(scale, data, 14, 14);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 14, 14));
    setOnePacked<DT>(scale, data, 15, 15);
    EXPECT_EQ(1.0, toDoublePacked<DT>(scale, data, 15, 15));
}

TEST_F(ocp_e2m1_mxfp4_test, setZero)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setZero<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(ocp_e2m1_mxfp4_test, setZeroPacked)
{
    uint8_t scale[] = {0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110};
    uint8_t data[]  = {0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110,
                       0b11111110};

    setZeroPacked<DT>(scale, data, 0, 0);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 0, 0));
    setZeroPacked<DT>(scale, data, 1, 1);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 1, 1));
    setZeroPacked<DT>(scale, data, 2, 2);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 2, 2));
    setZeroPacked<DT>(scale, data, 3, 3);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 3, 3));
    setZeroPacked<DT>(scale, data, 4, 4);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 4, 4));
    setZeroPacked<DT>(scale, data, 5, 5);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 5, 5));
    setZeroPacked<DT>(scale, data, 6, 6);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 6, 6));
    setZeroPacked<DT>(scale, data, 7, 7);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 7, 7));
    setZeroPacked<DT>(scale, data, 8, 8);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 8, 8));
    setZeroPacked<DT>(scale, data, 9, 9);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 9, 9));
    setZeroPacked<DT>(scale, data, 10, 10);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 10, 10));
    setZeroPacked<DT>(scale, data, 11, 11);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 11, 11));
    setZeroPacked<DT>(scale, data, 12, 12);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 12, 12));
    setZeroPacked<DT>(scale, data, 13, 13);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 13, 13));
    setZeroPacked<DT>(scale, data, 14, 14);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 14, 14));
    setZeroPacked<DT>(scale, data, 15, 15);
    EXPECT_EQ(0.0, toDoublePacked<DT>(scale, data, 15, 15));
}

TEST_F(ocp_e2m1_mxfp4_test, setNaN)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setNaN<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(ocp_e2m1_mxfp4_test, setNaNPacked)
{
    uint8_t scale[]
        = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};
    uint8_t data[] = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};

    setNaNPacked<DT>(scale, data, 0, 0);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 0, 0)));
    setNaNPacked<DT>(scale, data, 1, 1);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 1, 1)));
    setNaNPacked<DT>(scale, data, 2, 2);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 2, 2)));
    setNaNPacked<DT>(scale, data, 3, 3);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 3, 3)));
    setNaNPacked<DT>(scale, data, 4, 4);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 4, 4)));
    setNaNPacked<DT>(scale, data, 5, 5);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 5, 5)));
    setNaNPacked<DT>(scale, data, 6, 6);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 6, 6)));
    setNaNPacked<DT>(scale, data, 7, 7);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 7, 7)));
    setNaNPacked<DT>(scale, data, 8, 8);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 8, 8)));
    setNaNPacked<DT>(scale, data, 9, 9);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 9, 9)));
    setNaNPacked<DT>(scale, data, 10, 10);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 10, 10)));
    setNaNPacked<DT>(scale, data, 11, 11);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 11, 11)));
    setNaNPacked<DT>(scale, data, 12, 12);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 12, 12)));
    setNaNPacked<DT>(scale, data, 13, 13);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 13, 13)));
    setNaNPacked<DT>(scale, data, 14, 14);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 14, 14)));
    setNaNPacked<DT>(scale, data, 15, 15);
    EXPECT_TRUE(std::isnan(toDoublePacked<DT>(scale, data, 15, 15)));
}

TEST_F(ocp_e2m1_mxfp4_test, setDataMaxNorm)
{
    uint8_t scale[] = {Constants::E8M0_1}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0); // Leave optional params to normal, pos
    EXPECT_EQ(0b0111, data[0]);
    EXPECT_EQ(6.0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m1_mxfp4_test, setDataMaxNormPacked)
{
    uint8_t scale[] = {Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1};
    uint8_t data[]  = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};

    for(int i = 0; i < 16; i++)
    {
        setDataMaxPacked<DT>(data, i); // Leave optional params to normal, pos
        EXPECT_EQ(6.0, toDoublePacked<DT>(scale, data, i, i));
    }
}

TEST_F(ocp_e2m1_mxfp4_test, setDataMaxNeg)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, false, false); // Normal, negative
    EXPECT_EQ(0b1111, data[0]);
    EXPECT_EQ(-6.0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m1_mxfp4_test, setDataMaxNegPacked)
{
    uint8_t scale[] = {Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1};
    uint8_t data[]  = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};

    for(int i = 0; i < 16; i++)
    {
        setDataMaxPacked<DT>(data, i, false, false); // Leave optional params to normal, pos
        EXPECT_EQ(-6.0, toDoublePacked<DT>(scale, data, i, i));
    }
}

TEST_F(ocp_e2m1_mxfp4_test, setDataMaxSubnorm)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, 1, 1); // Subnorm, positive
    EXPECT_EQ(0b00001, data[0]);
    EXPECT_EQ(0.5, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m1_mxfp4_test, setDataMaxSubnormPacked)
{
    uint8_t scale[] = {Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1,
                       Constants::E8M0_1};
    uint8_t data[]  = {0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0, 0b0};

    for(int i = 0; i < 16; i++)
    {
        setDataMaxPacked<DT>(data, i, true, true); // Leave optional params to normal, pos
        EXPECT_EQ(0.5, toDoublePacked<DT>(scale, data, i, i));
    }
}

// Saturated tests
TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeRounding)
{
    float   norm    = 5.5341;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(norm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeRoundingSmallSubnorm)
{
    float   subnorm = 0.00000005960;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};
    EXPECT_EQ(0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeRoundingLargeSubnorm)
{
    float   subnorm = 0.127;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeRoundingSmallSubnormNeg)
{
    float   subnorm = -0.00000005960;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};
    EXPECT_EQ(0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeRoundingLargeSubnormNeg)
{
    float   subnorm = -0.627;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b0111, satConvertToType<DT>(largePos)); // Expect +max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeSRLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(largePos, 0)); // Expect +max norm
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(largePos, UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(largePos, UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypePosMax)
{
    float e2m1Max = 7.5f;
    EXPECT_EQ(0b0111, satConvertToType<DT>(e2m1Max)); // Expect +max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeSRPosMax)
{
    float e2m1Max = 7.5f;
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(e2m1Max, 0)); // Expect +max norm
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(e2m1Max, UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(e2m1Max, UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b0, satConvertToType<DT>(zero));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeSRZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(zero, 0));
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(zero, UINT_MAX));
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(zero, UINT_MAX / 2));
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeNegMax)
{
    float e2m1NegMax = -7.5f;
    EXPECT_EQ(0b1111,
              satConvertToType<DT>(e2m1NegMax)); // Expect -max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeSRNegMax)
{
    float e2m1NegMax = -7.5f;
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(e2m1NegMax, 0)); // Expect -max norm
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(e2m1NegMax, UINT_MAX)); // Expect -max norm
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(e2m1NegMax, UINT_MAX / 2)); // Expect -max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeLargeNeg)
{
    float largeNeg = -123456.7891234567f;
    EXPECT_EQ(0b1111, satConvertToType<DT>(largeNeg)); // Expect -max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeSRLargeNeg)
{
    float largeNeg = -123456.7891234567f;
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(largeNeg, 0)); // Expect -max norm
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(largeNeg, UINT_MAX)); // Expect -max norm
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(largeNeg, UINT_MAX / 2)); // Expect -max norm
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeNaN)
{
    uint8_t tData[] = {static_cast<uint8_t>(satConvertToType<DT>(NAN))};
    uint8_t scale[] = {Constants::E8M0_1};
    EXPECT_EQ(toFloat<DT>(scale, tData, 0, 0), getDataMax<DT>());
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeSRNaN)
{
    uint8_t tData[] = {static_cast<uint8_t>(satConvertToTypeSR<DT>(NAN, 0))};
    uint8_t scale[] = {Constants::E8M0_1};
    EXPECT_EQ(toFloat<DT>(scale, tData, 0, 0), getDataMax<DT>());
    *tData = static_cast<uint8_t>(satConvertToTypeSR<DT>(NAN, UINT_MAX));
    EXPECT_EQ(toFloat<DT>(scale, tData, 0, 0), getDataMax<DT>());
    *tData = static_cast<uint8_t>(satConvertToTypeSR<DT>(NAN, UINT_MAX / 2));
    EXPECT_EQ(toFloat<DT>(scale, tData, 0, 0), getDataMax<DT>());
}

TEST_F(ocp_e2m1_mxfp4_test, satConvertToTypeRandom)
{

    double lb = -7;
    double ub = 7;

    srandom(time(NULL));

    uint8_t scale[] = {Constants::E8M0_1};

    std::default_random_engine re;

    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        float closestDiff = getClosest(rNum);

        uint8_t res    = satConvertToType<DT>(rNum);
        uint8_t data[] = {res};

        EXPECT_EQ(closestDiff, std::abs(rNum - toDouble<DT>(scale, data, 0, 0)));
    }
}

// E2M1 only has a specified saturated mode, no unsaturated modes

TEST_F(ocp_e2m1_mxfp4_test, isSubnormal)
{
    uint8_t temp[] = {0b0, Constants::E8M0_1};

    for(size_t i = 0; i < e2m1ValuesOCP.size(); i++)
    {
        uint8_t data = static_cast<uint8_t>(i) & 0xf;

        temp[0] = data;

        uint8_t exp = (data >> getDataMantissaBits<DT>()) & 0b11;

        double value = toDouble<DT>(temp, temp, 1, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_TRUE(!isSubnorm<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 0));
    }
}

TEST_F(ocp_e2m1_mxfp4_test, isSubnormalPacked)
{
    uint8_t temp[] = {0b0, Constants::E8M0_1};

    for(size_t i = 0; i < e2m1ValuesOCP.size(); i += 2)
    {
        uint8_t rData = static_cast<uint8_t>(i) & 0xf;
        uint8_t lData = static_cast<uint8_t>(i + 1) & 0xf;

        temp[0] = (lData << 4) | rData;

        uint8_t rExp = (rData >> 1) & 0b11;
        uint8_t lExp = (lData >> 1) & 0b11;

        double rValue = toDoublePacked<DT>(temp, temp, 1, 0);
        double lValue = toDoublePacked<DT>(temp, temp, 1, 1);

        if(rExp != 0b0 || std::isnan(rValue) || std::isinf(rValue))
            EXPECT_TRUE(!isSubnormPacked<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, 0));

        if(lExp != 0b0 || std::isnan(lValue) || std::isinf(lValue))
            EXPECT_TRUE(!isSubnormPacked<DT>(temp, 1));
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, 1));
    }
}

TEST_F(ocp_e2m1_mxfp4_test, getDataMax)
{
    float mantissa = 1;
    for(int m = 1; m <= 1; m++)
        mantissa += std::pow(2, -m);

    float maxi = std::pow(2, 2) * mantissa; // Multiply max biased exp
    EXPECT_EQ(maxi, getDataMax<DT>());
}

TEST_F(ocp_e2m1_mxfp4_test, getDataMin)
{
    EXPECT_EQ(1.f, getDataMin<DT>()); // Min biased exp
}

TEST_F(ocp_e2m1_mxfp4_test, getDataMaxSubnorm)
{
    float exp      = 1.f; // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * (std::pow(2, mBits) - 1);
    EXPECT_EQ(exp * mantissa, getDataMaxSubnorm<DT>());
}

TEST_F(ocp_e2m1_mxfp4_test, getDataMinSubnorm)
{
    float exp          = 1.f; // Min biased exp
    float mantissaBits = getDataMantissaBits<DT>();
    float mantissa     = std::pow(2, -mantissaBits) * 1;
    EXPECT_EQ(exp * mantissa, getDataMinSubnorm<DT>());
}

TEST_F(ocp_e2m1_mxfp4_test, roundToEvenTest)
{

    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < (1 << 4); i += 2)
    {
        float input = (e2m1ValuesOCP[i] + e2m1ValuesOCP[i + 1]) / 2;
        *tData      = satConvertToType<DT>(input);

        EXPECT_EQ(e2m1ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0));
        EXPECT_EQ(static_cast<double>(e2m1ValuesOCP[i]), toDouble<DT>(tScale, tData, 0, 0));
    }
}

TEST_F(ocp_e2m1_mxfp4_test, roundToZeroTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 7; i++)
    {

        float negNum = e2m1ValuesOCP[i + 8];
        float posNum = e2m1ValuesOCP[i];

        while(posNum < e2m1ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, 0);
            EXPECT_EQ(e2m1ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m1ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, 0);
            EXPECT_EQ(e2m1ValuesOCP[i + 8], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m1ValuesOCP[i + 8] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

TEST_F(ocp_e2m1_mxfp4_test, roundToNextTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 7; i++)
    {

        float negNum = e2m1ValuesOCP[i + 8] - 0.1;
        float posNum = e2m1ValuesOCP[i] + 0.1;

        while(posNum < e2m1ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, UINT_MAX);
            EXPECT_EQ(e2m1ValuesOCP[i + 1], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m1ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, UINT_MAX);
            EXPECT_EQ(e2m1ValuesOCP[i + 9], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m1ValuesOCP[i + 8] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

// SR probablity is defined by the distanec to the next number
// if a number is in the middle it should be converted to the
// two numbers half the time
TEST_F(ocp_e2m1_mxfp4_test, midPointSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};
    for(int i = 0; i < 7; i++)
    {

        float lP = e2m1ValuesOCP[i], rP = e2m1ValuesOCP[i + 1], lN = e2m1ValuesOCP[i + 8],
              rN = e2m1ValuesOCP[i + 9];

        int plc = 0, prc = 0, nlc = 0, nrc = 0;

        float pMid = (lP + rP) / 2;
        float nMid = (lN + rN) / 2;
        for(long seed = 0; seed <= UINT_MAX; seed += 4096)
        {
            *tData = satConvertToTypeSR<DT>(pMid, static_cast<uint>(seed));

            if(toFloat<DT>(tScale, tData, 0, 0) == lP)
                plc++;
            else
                prc++;

            *tData = satConvertToTypeSR<DT>(nMid, static_cast<uint>(seed));

            if(toFloat<DT>(tScale, tData, 0, 0) == lN)
                nlc++;
            else
                nrc++;
        }
        EXPECT_EQ(plc, prc) << "left point: " << lP << " right Point: " << rP
                            << " mid point: " << pMid;
        EXPECT_EQ(nlc, nrc) << "left point: " << lN << " right Point: " << rN
                            << " mid point: " << nMid;
    }
}

TEST_F(ocp_e2m1_mxfp4_test, preserveSign)
{
    cvt t;

    t.num = std::numeric_limits<float>::quiet_NaN();

    t.bRep |= 1u << 31;

    EXPECT_EQ(0b1111, satConvertToType<DT>(t.num));
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(t.num, 0));
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(t.num, UINT_MAX));
    EXPECT_EQ(0b1111, satConvertToTypeSR<DT>(t.num, UINT_MAX / 2));

    t.bRep <<= 1;
    t.bRep >>= 1;

    EXPECT_EQ(0b0111, satConvertToType<DT>(t.num));
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(t.num, 0));
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(t.num, UINT_MAX));
    EXPECT_EQ(0b0111, satConvertToTypeSR<DT>(t.num, UINT_MAX / 2));
}
