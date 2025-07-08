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

#include "scale.hpp"
#include <dataTypeInfo.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

using namespace DGen;
using DT = ocp_e2m3_mxfp6;

constexpr double EPSILON
    = 0.125; // 2^-m, for (m)antissa = 3 bits; diff between 1.0 and next fp (1.125)

constexpr double NotANumber = Constants::QNaN;

constexpr std::array<float, 64> e2m3ValuesOCP = {
    // clang-format off
    0.0000000000, 0.1250000000, 0.2500000000, 0.3750000000, 0.5000000000, 0.6250000000, 0.7500000000, 0.8750000000,
    1.0000000000, 1.1250000000, 1.2500000000, 1.3750000000, 1.5000000000, 1.6250000000, 1.7500000000, 1.8750000000,
    2.0000000000, 2.2500000000, 2.5000000000, 2.7500000000, 3.0000000000, 3.2500000000, 3.5000000000, 3.7500000000,
    4.0000000000, 4.5000000000, 5.0000000000, 5.5000000000, 6.0000000000, 6.5000000000, 7.0000000000, 7.5000000000,
    -0.0000000000, -0.1250000000, -0.2500000000, -0.3750000000, -0.5000000000, -0.6250000000, -0.7500000000, -0.8750000000,
    -1.0000000000, -1.1250000000, -1.2500000000, -1.3750000000, -1.5000000000, -1.6250000000, -1.7500000000, -1.8750000000,
    -2.0000000000, -2.2500000000, -2.5000000000, -2.7500000000, -3.0000000000, -3.2500000000, -3.5000000000, -3.7500000000,
    -4.0000000000, -4.5000000000, -5.0000000000, -5.5000000000, -6.0000000000, -6.5000000000, -7.0000000000, -7.5000000000
    // clang-format on
};

constexpr uint8_t e2m3BitsOCP[] = {
    // clang-format off
    0b000000, 0b000001, 0b000010, 0b000011,
    0b000100, 0b000101, 0b000110, 0b000111,
    0b001000, 0b001001, 0b001010, 0b001011,
    0b001100, 0b001101, 0b001110, 0b001111,
    0b010000, 0b010001, 0b010010, 0b010011,
    0b010100, 0b010101, 0b010110, 0b010111,
    0b011000, 0b011001, 0b011010, 0b011011,
    0b011100, 0b011101, 0b011110, 0b011111,
    0b100000, 0b100001, 0b100010, 0b100011,
    0b100100, 0b100101, 0b100110, 0b100111,
    0b101000, 0b101001, 0b101010, 0b101011,
    0b101100, 0b101101, 0b101110, 0b101111,
    0b110000, 0b110001, 0b110010, 0b110011,
    0b110100, 0b110101, 0b110110, 0b110111,
    0b111000, 0b111001, 0b111010, 0b111011,
    0b111100, 0b111101, 0b111110, 0b111111
    // clang-format on
};

constexpr uint8_t e2m3BitsOCPPacked[] = {
    // clang-format off
    0b01000000, 0b00100000, 0b00001100, 0b01000100, 0b01100001, 0b00011100, 0b01001000, 0b10100010,
    0b00101100, 0b01001100, 0b11100011, 0b00111100, 0b01010000, 0b00100100, 0b01001101, 0b01010100,
    0b01100101, 0b01011101, 0b01011000, 0b10100110, 0b01101101, 0b01011100, 0b11100111, 0b01111101,
    0b01100000, 0b00101000, 0b10001110, 0b01100100, 0b01101001, 0b10011110, 0b01101000, 0b10101010,
    0b10101110, 0b01101100, 0b11101011, 0b10111110, 0b01110000, 0b00101100, 0b11001111, 0b01110100,
    0b01101101, 0b11011111, 0b01111000, 0b10101110, 0b11101111, 0b01111100, 0b11101111, 0b11111111
    // clang-format on
};

class ocp_e2m3_mxfp6_test : public ::testing::Test
{
protected:
    // [E8M0] Scale Data (no zero, inf)
    const uint8_t scales[6] = {
        Constants::E8M0_1, // 1
        Constants::E8M0_NAN, // NaN
        0b01111000, // 0.0078125
        0b10000101, // 64
        Constants::E8M0_MIN, // 2^-127 (min)
        Constants::E8M0_MAX // 2^127 (max)
    };
    // [E2M3] Element Data (no inf, NaN)
    const uint8_t data[6] = {
        0b000000, // 0
        0b001000, // 1 (min norm)
        0b010000, // 2.0
        0b011111, // 7.5 (max norm)
        0b000001, // 0.125 (min subnorm)
        0b000111 // 0.875 (max subnorm)
    };
    // [E2M3] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[6] = {
        0b100000, // -0
        0b101000, // -1 (min norm)
        0b101000, // -2.0
        0b111111, // -7.5 (max norm)
        0b100001, // -0.125 (min subnorm)
        0b100111 // -0.875 (max subnorm)
    };
    // Packed variants of data[], negativeData[]
    const uint8_t packedData[5] = {0b00000000, 0b00000010, 0b01111101, 0b11000001, 0b00000001};
    const uint8_t negativePackedData[5]
        = {0b00100000, 0b00001010, 0b11111111, 0b11100001, 0b00001001};

    float getClosest(float num)
    {
        float closestDiff = 50000;

        for(size_t i = 0; i < e2m3ValuesOCP.size(); i++)
        {
            if(std::isnan(e2m3ValuesOCP[i]))
                continue;
            closestDiff = std::min(closestDiff, std::abs(num - e2m3ValuesOCP[i]));
        }
        return closestDiff;
    }
};

TEST_F(ocp_e2m3_mxfp6_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(true, isOne<DT>(scales, data, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isOnePacked)
{
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(true, isOnePacked<DT>(scales, packedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, packedData, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isOnePacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isOnePacked<DT>(scales, negativePackedData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isZeroPacked)
{
    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, packedData, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(
        false,
        isZeroPacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(
        false,
        isZeroPacked<DT>(scales, negativePackedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(true, isZeroPacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isZeroPacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isZeroPacked<DT>(scales, negativePackedData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isNaNPacked)
{
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(true, isNaNPacked<DT>(scales, packedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, packedData, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(true, isNaNPacked<DT>(scales, negativePackedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isNaNPacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isNaNPacked<DT>(scales, negativePackedData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isInf)
{
    // Neither of E8M0/E2M3 support inf, should never return true
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isInfPacked)
{
    // Neither of E8M0/E2M3 support inf, should never return false
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, packedData, 5, 5)); // 2^127 * 0.875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 1)); // 1 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 2)); // 1 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 3)); // 1 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 4)); // 1 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 0, 5)); // 1 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 1)); // NaN * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 2)); // NaN * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 1, 3)); // NaN * 7.5 (max norm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 1, 4)); // NaN * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 1, 5)); // NaN * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 1)); // 0.0078125 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 2, 2)); // 0.0078125 * 2
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 2, 3)); // 0.0078125 * 7.5 (max norm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 2, 4)); // 0.0078125 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 2, 5)); // 0.0078125 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 1)); // 64 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 2)); // 64 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 3)); // 64 * 7.5 (max norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 4)); // 64 * 0.125 (min subnorm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 3, 5)); // 64 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 1)); // 2^-127 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 2)); // 2^-127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 4, 3)); // 2^-127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 4, 4)); // 2^-127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 4, 5)); // 2^-127 * 0.875 (max subnorm)

    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 1)); // 2^127 * 1 (min norm)
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 2)); // 2^127 * 2
    EXPECT_EQ(false, isInfPacked<DT>(scales, negativePackedData, 5, 3)); // 2^127 * 7.5 (max norm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 5, 4)); // 2^127 * 0.125 (min subnorm)
    EXPECT_EQ(false,
              isInfPacked<DT>(scales, negativePackedData, 5, 5)); // 2^127 * 0.875 (max subnorm)
}

TEST_F(ocp_e2m3_mxfp6_test, isSubnorm)
{
    uint8_t temp[] = {Constants::E8M0_1, 0b0};

    for(size_t i = 0; i < e2m3ValuesOCP.size(); i++)
    {
        uint8_t data = static_cast<uint8_t>(i) & 0x2f;
        temp[1]      = data;

        uint8_t exp = (data >> getDataMantissaBits<DT>()) & 0b011;

        if(exp != 0b0)
            EXPECT_FALSE(isSubnorm<DT>(temp, 1));
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 1));
    }
}

TEST_F(ocp_e2m3_mxfp6_test, isSubnormPacked)
{
    uint8_t temp[] = {0b0, 0b0, 0b0};

    for(size_t i = 0; i < e2m3ValuesOCP.size(); i++)
    {
        size_t  rem = i % 4;
        uint8_t l = 0b0, r = 0b0;

        uint8_t data = static_cast<uint8_t>(i) & 0x3f;

        // Determine where to place packed data
        switch(rem)
        {
        case 0:
            *(temp) &= 0b11000000; // clear last 6 bits to fill
            *(temp) |= data;
            break;
        case 1:
            l = (data & 0b00000011) << 6; // gather first 2 bits of temp[1]
            r = (data & 0b00111100) >> 2; // extract remaining 4 bits to go into temp[2]
            *(temp) &= 0b00111111; // clear space for l
            *(temp + 1) &= 0b11110000; // clear space for r
            *(temp) |= l; //set the buffers
            *(temp + 1) |= r;
            break;
        case 2:
            l = (data & 0b00001111) << 4; // separate bits into position for temp[1]
            r = (data & 0b00110000) >> 4; // and for temp[2]
            *(temp + 1) &= 0b00001111; // clear space for l
            *(temp + 2) &= 0b11111100; // clear space for r
            *(temp + 1) |= l; //set the buffers
            *(temp + 2) |= r;
            break;
        case 3:
            *(temp + 2) &= 0b00000011; // clear first 6 bits to fill
            *(temp + 2) |= (data << 2);
            break;
        }

        uint8_t exp = (data >> getDataMantissaBits<DT>()) & 0b011;
        if(exp != 0b0)
            EXPECT_FALSE(isSubnormPacked<DT>(temp, rem));
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, rem));
    }
}

// true if XN < val (first arg)
TEST_F(ocp_e2m3_mxfp6_test, isLess)
{
    double values[] = {Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.0005,
                       -0,
                       NotANumber,
                       0,
                       0.5,
                       0.0005,
                       1,
                       5,
                       10,
                       Constants::Inf};
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

TEST_F(ocp_e2m3_mxfp6_test, isLessPacked)
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
        for(int j = 0; j < 6; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDoublePacked<DT>(scales, packedData, i, j);
                double negProd = toDoublePacked<DT>(scales, negativePackedData, i, j);

                EXPECT_EQ(prod < values[k], isLessPacked<DT>(values[k], scales, packedData, i, j));
                EXPECT_EQ(negProd < values[k],
                          isLessPacked<DT>(values[k], scales, negativePackedData, i, j));
            }
        }
    }
}

// true if XN > val (first arg)
TEST_F(ocp_e2m3_mxfp6_test, isGreater)
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

TEST_F(ocp_e2m3_mxfp6_test, isGreaterPacked)
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
        for(int j = 0; j < 6; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDoublePacked<DT>(scales, packedData, i, j);
                double negProd = toDoublePacked<DT>(scales, negativePackedData, i, j);

                EXPECT_EQ(prod > values[k],
                          isGreaterPacked<DT>(values[k], scales, packedData, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreaterPacked<DT>(values[k], scales, negativePackedData, i, j));
            }
        }
    }
}

TEST_F(ocp_e2m3_mxfp6_test, toFloatAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        float ref = e8m0Values[i];
        for(size_t j = 0; j < e2m3ValuesOCP.size(); j++)
        {
            float  res      = toFloat<DT>(e8m0Bits, e2m3BitsOCP, i, j);
            double expected = ref * e2m3ValuesOCP[j];
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

TEST_F(ocp_e2m3_mxfp6_test, toFloatAllScalesAllValuesPacked)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        float ref = e8m0Values[i];
        for(size_t j = 0; j < e2m3ValuesOCP.size(); j++)
        {
            float  res      = toFloatPacked<DT>(e8m0Bits, e2m3BitsOCPPacked, i, j);
            double expected = ref * e2m3ValuesOCP[j];
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

TEST_F(ocp_e2m3_mxfp6_test, toDoubleAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        double ref = e8m0Values[i];

        for(size_t j = 0; j < e2m3ValuesOCP.size(); j++)
        {
            double res      = toDouble<DT>(e8m0Bits, e2m3BitsOCP, i, j);
            double expected = ref * e2m3ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref));
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e2m3_mxfp6_test, toDoubleAllScalesAllValuesPacked)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        double ref = e8m0Values[i];

        for(size_t j = 0; j < e2m3ValuesOCP.size(); j++)
        {
            double res      = toDoublePacked<DT>(e8m0Bits, e2m3BitsOCPPacked, i, j);
            double expected = ref * e2m3ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref));
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e2m3_mxfp6_test, setOne)
{
    uint8_t scale[] = {0b0};
    uint8_t data[]  = {0b0};
    setOne<DT>(scale, data, 0, 0);
    double dElem = toDouble<DT>(scale, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(ocp_e2m3_mxfp6_test, setOnePacked)
{
    uint8_t scale[] = {0b0};
    uint8_t data[]  = {0b0};
    setOnePacked<DT>(scale, data, 0, 0);
    double dElem = toDoublePacked<DT>(scale, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(ocp_e2m3_mxfp6_test, setZero)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setZero<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(ocp_e2m3_mxfp6_test, setZeroPacked)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setZeroPacked<DT>(scale, data, 1, 1);
    double dElem = toDoublePacked<DT>(scale, data, 1, 1);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(ocp_e2m3_mxfp6_test, setNaN)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setNaN<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(ocp_e2m3_mxfp6_test, setNaNPacked)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setNaNPacked<DT>(scale, data, 1, 1);
    double dElem = toDoublePacked<DT>(scale, data, 1, 1);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(ocp_e2m3_mxfp6_test, setDataMaxNorm)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0); // Leave optional params to normal, pos
    EXPECT_EQ(0b011111, data[0]);
    EXPECT_EQ(7.5, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m3_mxfp6_test, setDataMaxNormPacked)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMaxPacked<DT>(data, 0); // Leave optional params to normal, pos
    EXPECT_EQ(0b011111, data[0]);
    EXPECT_EQ(7.5, toDoublePacked<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m3_mxfp6_test, setDataMaxNeg)
{
    uint8_t scale[] = {Constants::E8M0_1}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, false, false); // Normal, negative
    EXPECT_EQ(0b111111, data[0]);
    EXPECT_EQ(-7.5, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m3_mxfp6_test, setDataMaxNegPacked)
{
    uint8_t scale[] = {Constants::E8M0_1}; // 1
    uint8_t data[]  = {0b0};
    setDataMaxPacked<DT>(data, 0, false, false); // Normal, negative
    EXPECT_EQ(0b111111, data[0]);
    EXPECT_EQ(-7.5, toDoublePacked<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m3_mxfp6_test, setDataMaxSubnorm)
{
    uint8_t scale[] = {Constants::E8M0_1}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, 1, 1); // Subnorm, positive
    EXPECT_EQ(0b000111, data[0]);
    EXPECT_EQ(0.875, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e2m3_mxfp6_test, setDataMaxSubnormPacked)
{
    uint8_t scale[] = {Constants::E8M0_1}; // 1
    uint8_t data[]  = {0b0};
    setDataMaxPacked<DT>(data, 0, 1, 1); // Subnorm, positive
    EXPECT_EQ(0b000111, data[0]);
    EXPECT_EQ(0.875, toDoublePacked<DT>(scale, data, 0, 0));
}

// Saturated tests
TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeRounding)
{
    float   norm    = 7.2341;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(norm);

    uint8_t data[] = {res};

    double closestDiff = getClosest(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeRoundingSmallSubnorm)
{
    float   subnorm = 0.00000005960;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));

    subnorm  = -0.123;
    scale[0] = {Constants::E8M0_1};
    res      = satConvertToType<DT>(subnorm);
    data[0]  = {res};

    closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeRoundingLargeSubnorm)
{
    float   subnorm = 0.127;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeRoundingSmallSubnormNeg)
{
    float   subnorm = -0.00000005960;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeRoundingLargeSubnormNeg)
{
    float   subnorm = -0.127;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b011111, satConvertToType<DT>(largePos)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeSRLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(largePos, 0)); // Expect +max norm
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(largePos, UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(largePos, UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypePosMax)
{
    float e2m3Max = 7.5f;
    EXPECT_EQ(0b011111, satConvertToType<DT>(e2m3Max)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeSRPosMax)
{
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(getDataMax<DT>(), 0)); // Expect +max norm
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(getDataMax<DT>(), UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(getDataMax<DT>(), UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b0, satConvertToType<DT>(zero));
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeSRZero)
{
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(0.0f, 0)); // Expect +max norm
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(0.0f, UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(0.0f, UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeNegMax)
{
    float e2m3NegMax = -7.5f;
    EXPECT_EQ(0b111111,
              satConvertToType<DT>(e2m3NegMax)); // Expect -max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeSRNegMax)
{
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(-getDataMax<DT>(), 0)); // Expect +max norm
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(-getDataMax<DT>(), UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b111111,
              satConvertToTypeSR<DT>(-getDataMax<DT>(), UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeLargeNeg)
{
    float largeNeg = -123456.7891234567f;
    EXPECT_EQ(0b111111, satConvertToType<DT>(largeNeg)); // Expect -max norm
}

TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeSRLargeNeg)
{
    float largePos = -123456.7891234567f;
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(largePos, 0)); // Expect +max norm
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(largePos, UINT_MAX)); // Expect +max norm
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(largePos, UINT_MAX / 2)); // Expect +max norm
}

TEST_F(ocp_e2m3_mxfp6_test, preserveSignWithNaN)
{
    cvt t;

    t.num = std::numeric_limits<float>::quiet_NaN();

    t.bRep |= 1U << 31;

    EXPECT_EQ(0b111111, satConvertToType<DT>(t.num));
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(t.num, 0));
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(t.num, UINT_MAX));
    EXPECT_EQ(0b111111, satConvertToTypeSR<DT>(t.num, UINT_MAX / 2));

    t.bRep <<= 1;
    t.bRep >>= 1;

    EXPECT_EQ(0b011111, satConvertToType<DT>(t.num));
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(t.num, 0));
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(t.num, UINT_MAX));
    EXPECT_EQ(0b011111, satConvertToTypeSR<DT>(t.num, UINT_MAX / 2));
}

// Generate 1000000 numbers and see if the conversion is good
TEST_F(ocp_e2m3_mxfp6_test, satConvertToTypeRandom)
{
    double lb = -30;
    double ub = 30;

    srandom(time(NULL));

    uint8_t scale[] = {Constants::E8M0_1};

    std::default_random_engine re;

    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        uint8_t res = satConvertToType<DT>(rNum);

        float closestDiff = getClosest(rNum);

        EXPECT_EQ(closestDiff, std::abs(rNum - toDouble<DT>(scale, &res, 0, 0)));
    }
}

// E2M3 only has a specified saturated mode, no unsaturated modes

TEST_F(ocp_e2m3_mxfp6_test, getDataMax)
{
    float mantissa = 1;
    for(int m = 1; m <= 3; m++)
        mantissa += std::pow(2, -m);

    float maxi = std::pow(2, 2) * mantissa; // Multiply max biased exp
    EXPECT_EQ(maxi, getDataMax<DT>());
}

TEST_F(ocp_e2m3_mxfp6_test, getDataMin)
{
    EXPECT_EQ(1.f, getDataMin<DT>()); // Min biased exp
}

TEST_F(ocp_e2m3_mxfp6_test, getDataMaxSubnorm)
{
    float exp      = 1.f; // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * (std::pow(2, mBits) - 1);
    EXPECT_EQ(exp * mantissa, getDataMaxSubnorm<DT>());
}

TEST_F(ocp_e2m3_mxfp6_test, getDataMinSubnorm)
{
    float exp      = 1.f; // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * 1;
    EXPECT_EQ(exp * mantissa, getDataMinSubnorm<DT>());
}

TEST_F(ocp_e2m3_mxfp6_test, roundToEvenTest)
{

    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < (1 << 6); i += 2)
    {
        float input = (e2m3ValuesOCP[i] + e2m3ValuesOCP[i + 1]) / 2;
        *tData      = satConvertToType<DT>(input);

        EXPECT_EQ(e2m3ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0));
        EXPECT_EQ(static_cast<double>(e2m3ValuesOCP[i]), toDouble<DT>(tScale, tData, 0, 0));
    }
}

TEST_F(ocp_e2m3_mxfp6_test, roundToZeroTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 31; i++)
    {

        float negNum = e2m3ValuesOCP[i + 32];
        float posNum = e2m3ValuesOCP[i];

        while(posNum < e2m3ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, 0);
            EXPECT_EQ(e2m3ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m3ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, 0);
            EXPECT_EQ(e2m3ValuesOCP[i + 32], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m3ValuesOCP[i + 32] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

TEST_F(ocp_e2m3_mxfp6_test, roundToNextTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 31; i++)
    {

        float negNum = e2m3ValuesOCP[i + 32] - 0.01;
        float posNum = e2m3ValuesOCP[i] + 0.01;

        while(posNum < e2m3ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, UINT_MAX);
            EXPECT_EQ(e2m3ValuesOCP[i + 1], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m3ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, UINT_MAX);
            EXPECT_EQ(e2m3ValuesOCP[i + 33], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e2m3ValuesOCP[i + 32] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

// SR probablity is defined by the distanec to the next number
// if a number is in the middle it should be converted to the
// two numbers half the time
TEST_F(ocp_e2m3_mxfp6_test, midPointSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};
    for(int i = 0; i < 31; i++)
    {

        float lP = e2m3ValuesOCP[i], rP = e2m3ValuesOCP[i + 1], lN = e2m3ValuesOCP[i + 32],
              rN = e2m3ValuesOCP[i + 33];

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
