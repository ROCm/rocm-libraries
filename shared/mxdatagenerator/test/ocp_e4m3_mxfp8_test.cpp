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
#include <iostream>
#include <limits>

#include "scale.hpp"

using namespace DGen;
using DT = DGen::ocp_e4m3_mxfp8;

constexpr double Inf        = DGen::Constants::Inf;
constexpr double NegInf     = DGen::Constants::NegInf;
constexpr double NotANumber = DGen::Constants::QNaN;
// 2^-m, for (m)antissa = 3 bits; diff between 1.0 and next fp (1.125)
constexpr double EPSILON = 0.125;

constexpr std::array<double, 256> e4m3ValuesOCP = {
    // clang-format off
       0.0000000000,    0.0019531250,    0.0039062500,    0.0058593750,    0.0078125000,    0.0097656250,    0.0117187500,    0.0136718750,
       0.0156250000,    0.0175781250,    0.0195312500,    0.0214843750,    0.0234375000,    0.0253906250,    0.0273437500,    0.0292968750,
       0.0312500000,    0.0351562500,    0.0390625000,    0.0429687500,    0.0468750000,    0.0507812500,    0.0546875000,    0.0585937500,
       0.0625000000,    0.0703125000,    0.0781250000,    0.0859375000,    0.0937500000,    0.1015625000,    0.1093750000,    0.1171875000,
       0.1250000000,    0.1406250000,    0.1562500000,    0.1718750000,    0.1875000000,    0.2031250000,    0.2187500000,    0.2343750000,
       0.2500000000,    0.2812500000,    0.3125000000,    0.3437500000,    0.3750000000,    0.4062500000,    0.4375000000,    0.4687500000,
       0.5000000000,    0.5625000000,    0.6250000000,    0.6875000000,    0.7500000000,    0.8125000000,    0.8750000000,    0.9375000000,
       1.0000000000,    1.1250000000,    1.2500000000,    1.3750000000,    1.5000000000,    1.6250000000,    1.7500000000,    1.8750000000,
       2.0000000000,    2.2500000000,    2.5000000000,    2.7500000000,    3.0000000000,    3.2500000000,    3.5000000000,    3.7500000000,
       4.0000000000,    4.5000000000,    5.0000000000,    5.5000000000,    6.0000000000,    6.5000000000,    7.0000000000,    7.5000000000,
       8.0000000000,    9.0000000000,   10.0000000000,   11.0000000000,   12.0000000000,   13.0000000000,   14.0000000000,   15.0000000000,
      16.0000000000,   18.0000000000,   20.0000000000,   22.0000000000,   24.0000000000,   26.0000000000,   28.0000000000,   30.0000000000,
      32.0000000000,   36.0000000000,   40.0000000000,   44.0000000000,   48.0000000000,   52.0000000000,   56.0000000000,   60.0000000000,
      64.0000000000,   72.0000000000,   80.0000000000,   88.0000000000,   96.0000000000,  104.0000000000,  112.0000000000,  120.0000000000,
     128.0000000000,  144.0000000000,  160.0000000000,  176.0000000000,  192.0000000000,  208.0000000000,  224.0000000000,  240.0000000000,
     256.0000000000,  288.0000000000,  320.0000000000,  352.0000000000,  384.0000000000,  416.0000000000,  448.0000000000,      NotANumber,
      -0.0000000000,   -0.0019531250,   -0.0039062500,   -0.0058593750,   -0.0078125000,   -0.0097656250,   -0.0117187500,   -0.0136718750,
      -0.0156250000,   -0.0175781250,   -0.0195312500,   -0.0214843750,   -0.0234375000,   -0.0253906250,   -0.0273437500,   -0.0292968750,
      -0.0312500000,   -0.0351562500,   -0.0390625000,   -0.0429687500,   -0.0468750000,   -0.0507812500,   -0.0546875000,   -0.0585937500,
      -0.0625000000,   -0.0703125000,   -0.0781250000,   -0.0859375000,   -0.0937500000,   -0.1015625000,   -0.1093750000,   -0.1171875000,
      -0.1250000000,   -0.1406250000,   -0.1562500000,   -0.1718750000,   -0.1875000000,   -0.2031250000,   -0.2187500000,   -0.2343750000,
      -0.2500000000,   -0.2812500000,   -0.3125000000,   -0.3437500000,   -0.3750000000,   -0.4062500000,   -0.4375000000,   -0.4687500000,
      -0.5000000000,   -0.5625000000,   -0.6250000000,   -0.6875000000,   -0.7500000000,   -0.8125000000,   -0.8750000000,   -0.9375000000,
      -1.0000000000,   -1.1250000000,   -1.2500000000,   -1.3750000000,   -1.5000000000,   -1.6250000000,   -1.7500000000,   -1.8750000000,
      -2.0000000000,   -2.2500000000,   -2.5000000000,   -2.7500000000,   -3.0000000000,   -3.2500000000,   -3.5000000000,   -3.7500000000,
      -4.0000000000,   -4.5000000000,   -5.0000000000,   -5.5000000000,   -6.0000000000,   -6.5000000000,   -7.0000000000,   -7.5000000000,
      -8.0000000000,   -9.0000000000,  -10.0000000000,  -11.0000000000,  -12.0000000000,  -13.0000000000,  -14.0000000000,  -15.0000000000,
     -16.0000000000,  -18.0000000000,  -20.0000000000,  -22.0000000000,  -24.0000000000,  -26.0000000000,  -28.0000000000,  -30.0000000000,
     -32.0000000000,  -36.0000000000,  -40.0000000000,  -44.0000000000,  -48.0000000000,  -52.0000000000,  -56.0000000000,  -60.0000000000,
     -64.0000000000,  -72.0000000000,  -80.0000000000,  -88.0000000000,  -96.0000000000, -104.0000000000, -112.0000000000, -120.0000000000,
    -128.0000000000, -144.0000000000, -160.0000000000, -176.0000000000, -192.0000000000, -208.0000000000, -224.0000000000, -240.0000000000,
    -256.0000000000, -288.0000000000, -320.0000000000, -352.0000000000, -384.0000000000, -416.0000000000, -448.0000000000,      NotANumber
    // clang-format on
};

constexpr uint8_t e4m3BitsOCP[] = {
    // clang-format off
    0b00000000, 0b00000001, 0b00000010, 0b00000011, 0b00000100, 0b00000101, 0b00000110, 0b00000111,
    0b00001000, 0b00001001, 0b00001010, 0b00001011, 0b00001100, 0b00001101, 0b00001110, 0b00001111,
    0b00010000, 0b00010001, 0b00010010, 0b00010011, 0b00010100, 0b00010101, 0b00010110, 0b00010111,
    0b00011000, 0b00011001, 0b00011010, 0b00011011, 0b00011100, 0b00011101, 0b00011110, 0b00011111,
    0b00100000, 0b00100001, 0b00100010, 0b00100011, 0b00100100, 0b00100101, 0b00100110, 0b00100111,
    0b00101000, 0b00101001, 0b00101010, 0b00101011, 0b00101100, 0b00101101, 0b00101110, 0b00101111,
    0b00110000, 0b00110001, 0b00110010, 0b00110011, 0b00110100, 0b00110101, 0b00110110, 0b00110111,
    0b00111000, 0b00111001, 0b00111010, 0b00111011, 0b00111100, 0b00111101, 0b00111110, 0b00111111,
    0b01000000, 0b01000001, 0b01000010, 0b01000011, 0b01000100, 0b01000101, 0b01000110, 0b01000111,
    0b01001000, 0b01001001, 0b01001010, 0b01001011, 0b01001100, 0b01001101, 0b01001110, 0b01001111,
    0b01010000, 0b01010001, 0b01010010, 0b01010011, 0b01010100, 0b01010101, 0b01010110, 0b01010111,
    0b01011000, 0b01011001, 0b01011010, 0b01011011, 0b01011100, 0b01011101, 0b01011110, 0b01011111,
    0b01100000, 0b01100001, 0b01100010, 0b01100011, 0b01100100, 0b01100101, 0b01100110, 0b01100111,
    0b01101000, 0b01101001, 0b01101010, 0b01101011, 0b01101100, 0b01101101, 0b01101110, 0b01101111,
    0b01110000, 0b01110001, 0b01110010, 0b01110011, 0b01110100, 0b01110101, 0b01110110, 0b01110111,
    0b01111000, 0b01111001, 0b01111010, 0b01111011, 0b01111100, 0b01111101, 0b01111110, 0b01111111,
    0b10000000, 0b10000001, 0b10000010, 0b10000011, 0b10000100, 0b10000101, 0b10000110, 0b10000111,
    0b10001000, 0b10001001, 0b10001010, 0b10001011, 0b10001100, 0b10001101, 0b10001110, 0b10001111,
    0b10010000, 0b10010001, 0b10010010, 0b10010011, 0b10010100, 0b10010101, 0b10010110, 0b10010111,
    0b10011000, 0b10011001, 0b10011010, 0b10011011, 0b10011100, 0b10011101, 0b10011110, 0b10011111,
    0b10100000, 0b10100001, 0b10100010, 0b10100011, 0b10100100, 0b10100101, 0b10100110, 0b10100111,
    0b10101000, 0b10101001, 0b10101010, 0b10101011, 0b10101100, 0b10101101, 0b10101110, 0b10101111,
    0b10110000, 0b10110001, 0b10110010, 0b10110011, 0b10110100, 0b10110101, 0b10110110, 0b10110111,
    0b10111000, 0b10111001, 0b10111010, 0b10111011, 0b10111100, 0b10111101, 0b10111110, 0b10111111,
    0b11000000, 0b11000001, 0b11000010, 0b11000011, 0b11000100, 0b11000101, 0b11000110, 0b11000111,
    0b11001000, 0b11001001, 0b11001010, 0b11001011, 0b11001100, 0b11001101, 0b11001110, 0b11001111,
    0b11010000, 0b11010001, 0b11010010, 0b11010011, 0b11010100, 0b11010101, 0b11010110, 0b11010111,
    0b11011000, 0b11011001, 0b11011010, 0b11011011, 0b11011100, 0b11011101, 0b11011110, 0b11011111,
    0b11100000, 0b11100001, 0b11100010, 0b11100011, 0b11100100, 0b11100101, 0b11100110, 0b11100111,
    0b11101000, 0b11101001, 0b11101010, 0b11101011, 0b11101100, 0b11101101, 0b11101110, 0b11101111,
    0b11110000, 0b11110001, 0b11110010, 0b11110011, 0b11110100, 0b11110101, 0b11110110, 0b11110111,
    0b11111000, 0b11111001, 0b11111010, 0b11111011, 0b11111100, 0b11111101, 0b11111110, 0b11111111
    // clang-format on
};

class ocp_e4m3_mxfp8_test : public ::testing::Test
{
protected:
    // [E8M0] Scale Data (no zero, inf)
    const uint8_t scales[6] = {
        DGen::Constants::E8M0_1, // 1
        DGen::Constants::E8M0_NAN, // NaN
        0b01111000, // 0.0078125
        0b10000101, // 64
        DGen::Constants::E8M0_MIN, // 2^-127 (min)
        DGen::Constants::E8M0_MAX // 2^127 (max)
    };
    // [E4M3] Element Data (no inf)
    const uint8_t data[7] = {
        0b00000000, // 0
        0b00111000, // 1
        0b01111111, // NaN
        0b00001000, // 0.015625 (min norm)
        0b01111110, // 448 (max norm)
        0b00000001, // 0.001953125 (min subnorm)
        0b00000111 // 0.013671875 (max subnorm)
    };
    // [E4M3] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[7] = {
        0b10000000, // -0
        0b10111000, // -1
        0b11111111, // NaN
        0b10001000, // -0.015625 (-min norm)
        0b11111110, // -448 (-max norm)
        0b10000001, // -0.001953125 (-min subnorm)
        0b10000111 // -0.013671875 (-max subnorm)
    };

    double getClosest(double num)
    {
        double closestDiff = UINT32_MAX;

        for(size_t i = 0; i < e4m3ValuesOCP.size(); i++)
        {
            if(std::isnan(e4m3ValuesOCP[i]))
                continue;
            closestDiff = std::min(closestDiff, std::abs(num - e4m3ValuesOCP[i]));
        }
        return closestDiff;
    }
};

TEST_F(ocp_e4m3_mxfp8_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(true, isOne<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 2)); // NaN * NaN
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 2)); // 64 * NaN
    EXPECT_EQ(true, isOne<DT>(scales, data, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 6)); // 2^127 * 0.013671875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 2)); // NaN * NaN
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false,
              isOne<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false,
              isOne<DT>(scales, negativeData, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 6)); // 2^127 * 0.013671875 (max subnorm)
}

TEST_F(ocp_e4m3_mxfp8_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 2)); // NaN * NaN
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 6)); // 2^127 * 0.013671875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 2)); // NaN * NaN
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false,
              isZero<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false,
              isZero<DT>(scales, negativeData, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 6)); // 2^127 * 0.013671875 (max subnorm)
}

TEST_F(ocp_e4m3_mxfp8_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 2)); // NaN * NaN
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 6)); // 2^127 * 0.013671875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 2)); // NaN * NaN
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false,
              isNaN<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false,
              isNaN<DT>(scales, negativeData, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 6)); // 2^127 * 0.013671875 (max subnorm)
}

TEST_F(ocp_e4m3_mxfp8_test, isInf)
{
    // Neither of E8M0/E4M3 support inf, should never return true
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 2)); // NaN * NaN
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 6)); // 2^127 * 0.013671875 (max subnorm)

    // ========================================================================================= //

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 2)); // 1 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 3)); // 1 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 4)); // 1 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 5)); // 1 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 6)); // 1 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 2)); // NaN * NaN
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 3)); // NaN * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 4)); // NaN * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 5)); // NaN * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 6)); // NaN * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 3)); // 0.0078125 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 4)); // 0.0078125 * 448 (max norm)
    EXPECT_EQ(false,
              isInf<DT>(scales, negativeData, 2, 5)); // 0.0078125 * 0.001953125 (min subnorm)
    EXPECT_EQ(false,
              isInf<DT>(scales, negativeData, 2, 6)); // 0.0078125 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 2)); // 64 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 3)); // 64 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 4)); // 64 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 5)); // 64 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 6)); // 64 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 3)); // 2^-127 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 4)); // 2^-127 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 5)); // 2^-127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 6)); // 2^-127 * 0.013671875 (max subnorm)

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 2)); // 2^127 * NaN
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 3)); // 2^127 * 0.015625 (min norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 4)); // 2^127 * 448 (max norm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 5)); // 2^127 * 0.001953125 (min subnorm)
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 6)); // 2^127 * 0.013671875 (max subnorm)
}

TEST_F(ocp_e4m3_mxfp8_test, isSubnorm)
{
    uint8_t temp[] = {Constants::E8M0_1, 0b0};

    for(size_t i = 0; i < e4m3ValuesOCP.size(); i++)
    {
        uint8_t data = static_cast<uint8_t>(i) & 0xff;
        temp[1]      = data;

        uint8_t exp = (data >> getDataMantissaBits<DT>()) & 0b01111;

        double value = toDouble<DT>(temp, temp, 0, 1);

        if(exp != 0b0 || std::isnan(value))
            EXPECT_FALSE(isSubnorm<DT>(temp, 1));
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 1));
    }
}

// true if XN < val (first arg)
TEST_F(ocp_e4m3_mxfp8_test, isLess)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 7; j++)
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

// true if XN > val (first arg)
TEST_F(ocp_e4m3_mxfp8_test, isGreater)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 7; j++)
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

TEST_F(ocp_e4m3_mxfp8_test, toFloatAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        float ref = e8m0Values[i];
        for(size_t j = 0; j < e4m3ValuesOCP.size(); j++)
        {
            float  res      = toFloat<DT>(e8m0Bits, e4m3BitsOCP, i, j);
            double expected = ref * e4m3ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref) || std::isnan(e4m3ValuesOCP[j]));
            else if(expected > FLT_MAX)
                EXPECT_EQ(std::numeric_limits<double>::infinity(), res);
            else if(expected < -FLT_MAX)
                EXPECT_EQ(-std::numeric_limits<double>::infinity(), res);
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e4m3_mxfp8_test, toDoubleAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        double ref = e8m0Values[i];

        for(size_t j = 0; j < e4m3ValuesOCP.size(); j++)
        {
            double res      = toDouble<DT>(e8m0Bits, e4m3BitsOCP, i, j);
            double expected = ref * e4m3ValuesOCP[j];
            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref) || std::isnan(e4m3ValuesOCP[j]));
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e4m3_mxfp8_test, setOne)
{
    uint8_t scale[] = {0b0};
    uint8_t data[]  = {0b0};
    setOne<DT>(scale, data, 0, 0);
    double dElem = toDouble<DT>(scale, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(ocp_e4m3_mxfp8_test, setZero)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setZero<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(ocp_e4m3_mxfp8_test, setNaN)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setNaN<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(ocp_e4m3_mxfp8_test, setDataMaxNorm)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0); // Leave optional params to normal, pos
    EXPECT_EQ(0b01111110, data[0]);
    EXPECT_EQ(448.0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e4m3_mxfp8_test, setDataMaxNeg)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, 0, 0); // Normal, negative
    EXPECT_EQ(0b11111110, data[0]);
    EXPECT_EQ(-448.0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e4m3_mxfp8_test, setDataMaxSubnorm)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, 1, 1); // Subnorm, positive
    EXPECT_EQ(0b111, data[0]);
    EXPECT_EQ(0.013671875, toDouble<DT>(scale, data, 0, 0));
}

// Saturated tests
TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeRounding)
{
    float   norm    = 446.789123456789f;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(norm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeRoundingSmallSubnorm)
{
    float   subnorm = 0.00000005960;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    EXPECT_EQ(0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeRoundingLargeSubnorm)
{
    float   subnorm = 0.001953157;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeRoundingSmallSubnormNeg)
{
    float   subnorm = -0.00000005960;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    EXPECT_EQ(0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeRoundingLargeSubnormNeg)
{
    float   subnorm = -0.001953157;
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b01111110, satConvertToType<DT>(largePos)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSRLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b01111110, satConvertToTypeSR<DT>(largePos, 0)); // Expect max norm (448)
    EXPECT_EQ(0b01111110, satConvertToTypeSR<DT>(largePos, UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b01111110, satConvertToTypeSR<DT>(largePos, UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypePosMax)
{
    float e4m3Max = 448.f;
    EXPECT_EQ(0b01111110, satConvertToType<DT>(e4m3Max)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSRLPosMax)
{
    EXPECT_EQ(0b01111110, satConvertToTypeSR<DT>(getDataMax<DT>(), 0)); // Expect max norm (448)
    EXPECT_EQ(0b01111110,
              satConvertToTypeSR<DT>(getDataMax<DT>(), UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b01111110,
              satConvertToTypeSR<DT>(getDataMax<DT>(),
                                     UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, satConvertToType<DT>(zero));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSRZero)
{
    EXPECT_EQ(0, satConvertToTypeSR<DT>(0, 0)); // Expect max norm (448)
    EXPECT_EQ(0, satConvertToTypeSR<DT>(0, UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0, satConvertToTypeSR<DT>(0, UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeNegMax)
{
    float e4m3NegMax = -448.f;
    EXPECT_EQ(0b11111110, satConvertToType<DT>(e4m3NegMax)); // Expect -max norm (-448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSRNegMax)
{
    EXPECT_EQ(0b11111110, satConvertToTypeSR<DT>(-getDataMax<DT>(), 0)); // Expect max norm (448)
    EXPECT_EQ(0b11111110,
              satConvertToTypeSR<DT>(-getDataMax<DT>(), UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b11111110,
              satConvertToTypeSR<DT>(-getDataMax<DT>(),
                                     UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeLargeNeg)
{
    float largeNeg = -123456.7891234567f;
    EXPECT_EQ(0b11111110, satConvertToType<DT>(largeNeg)); // Expect -max norm (-448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSRLargeNeg)
{
    float largeNeg = -123456.7891234567f;
    EXPECT_EQ(0b11111110, satConvertToTypeSR<DT>(largeNeg, 0)); // Expect max norm (448)
    EXPECT_EQ(0b11111110, satConvertToTypeSR<DT>(largeNeg, UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b11111110, satConvertToTypeSR<DT>(largeNeg, UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeNaN)
{
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(NAN);
    uint8_t data[]  = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSRNaN)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    *tData = satConvertToTypeSR<DT>(NAN, 0);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = satConvertToTypeSR<DT>(NAN, UINT_MAX);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = satConvertToTypeSR<DT>(NAN, UINT_MAX / 2);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));
}

// Generate 1000000 numbers and see if the conversion is good
TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeRandom)
{

    double lb = -600;
    double ub = 600;

    srandom(time(NULL));

    uint8_t scale[] = {DGen::Constants::E8M0_1};

    std::default_random_engine re;

    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        uint8_t res    = satConvertToType<DT>(rNum);
        uint8_t data[] = {res};

        double closestDiff = getClosest(rNum);

        EXPECT_EQ(closestDiff, std::abs(rNum - toDouble<DT>(scale, data, 0, 0)))
            << "rNum = " << rNum << "\noutput: " << toDouble<DT>(scale, data, 0, 0);
    }
}

TEST_F(ocp_e4m3_mxfp8_test, satConvertToTypeSROutOfRange)
{
    EXPECT_EQ(0b01111110, satConvertToTypeSR<DT>(Inf, 0));

    EXPECT_EQ(0b01111110, satConvertToTypeSR<DT>(69000, 0));

    EXPECT_EQ(0b11111110, satConvertToTypeSR<DT>(NegInf, 0));

    EXPECT_EQ(0b11111110, satConvertToTypeSR<DT>(-69000, 0));
}

// Non-saturated tests

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeLargePos)
{
    float   largePos = 123456.7891234567f;
    uint8_t scale[]  = {DGen::Constants::E8M0_1};
    uint8_t res      = nonSatConvertToType<DT>(largePos);
    uint8_t data[]   = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSRLargePos)
{
    float   largePos = 123456.7891234567f;
    uint8_t tScale[] = {Constants::E8M0_1};
    uint8_t tData[1];

    *tData = nonSatConvertToTypeSR<DT>(largePos, 0);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = nonSatConvertToTypeSR<DT>(largePos, UINT_MAX);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = nonSatConvertToTypeSR<DT>(largePos, UINT_MAX / 2);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypePosMax)
{
    float e4m3Max = 448.f;
    EXPECT_EQ(0b01111110, nonSatConvertToType<DT>(e4m3Max)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSRPosMax)
{
    EXPECT_EQ(0b01111110, nonSatConvertToTypeSR<DT>(getDataMax<DT>(), 0)); // Expect max norm (448)
    EXPECT_EQ(0b01111110,
              nonSatConvertToTypeSR<DT>(getDataMax<DT>(), UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b01111110,
              nonSatConvertToTypeSR<DT>(getDataMax<DT>(),
                                        UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, nonSatConvertToType<DT>(zero));
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSRZero)
{
    EXPECT_EQ(0b0, nonSatConvertToTypeSR<DT>(0, 0)); // Expect max norm (448)
    EXPECT_EQ(0b0, nonSatConvertToTypeSR<DT>(0, UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b0, nonSatConvertToTypeSR<DT>(0, UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeNegMax)
{
    float e4m3NegMax = -448.f;
    EXPECT_EQ(0b11111110, nonSatConvertToType<DT>(e4m3NegMax)); // Expect -max norm (-448)
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSRNegMax)
{
    EXPECT_EQ(0b11111110, nonSatConvertToTypeSR<DT>(-getDataMax<DT>(), 0)); // Expect max norm (448)
    EXPECT_EQ(0b11111110,
              nonSatConvertToTypeSR<DT>(-getDataMax<DT>(),
                                        UINT_MAX)); // Expect max norm (448)
    EXPECT_EQ(0b11111110,
              nonSatConvertToTypeSR<DT>(-getDataMax<DT>(),
                                        UINT_MAX / 2)); // Expect max norm (448)
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeLargeNeg)
{
    float   largeNeg = -123456.7891234567f;
    uint8_t scale[]  = {DGen::Constants::E8M0_1};
    uint8_t res      = nonSatConvertToType<DT>(largeNeg);
    uint8_t data[]   = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSRLargeNeg)
{
    float   largeNeg = -123456.7891234567f;
    uint8_t tScale[] = {Constants::E8M0_1};
    uint8_t tData[1];

    *tData = nonSatConvertToTypeSR<DT>(largeNeg, 0);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = nonSatConvertToTypeSR<DT>(largeNeg, UINT_MAX);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = nonSatConvertToTypeSR<DT>(largeNeg, UINT_MAX / 2);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeNaN)
{
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(NAN);
    uint8_t data[]  = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSRNaN)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    *tData = nonSatConvertToTypeSR<DT>(NAN, 0);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = nonSatConvertToTypeSR<DT>(NAN, UINT_MAX);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));

    *tData = nonSatConvertToTypeSR<DT>(NAN, UINT_MAX / 2);
    EXPECT_TRUE(std::isnan(toDouble<DT>(tScale, tData, 0, 0)));
}

// Generate 1000000 numbers and see if the conversion is good
TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeRandom)
{

    double lb = -600;
    double ub = 600;
    srandom(time(NULL));

    uint8_t scale[] = {DGen::Constants::E8M0_1};

    std::default_random_engine re;

    uint8_t data[1];
    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<double> unif(lb, ub);

        double rNum = unif(re);
        data[0]     = nonSatConvertToType<DT>(rNum);

        if(std::abs(rNum) > 464) // max magnitude
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(scale, data, 0, 0)))
                << "rNum: " << rNum << "\nOutput: " << toDouble<DT>(scale, data, 0, 0);
            continue;
        }

        double closestDiff = getClosest(rNum);

        EXPECT_EQ(closestDiff, std::abs(rNum - toDouble<DT>(scale, data, 0, 0)));
    }
}

TEST_F(ocp_e4m3_mxfp8_test, nonSatConvertToTypeSROutOfRange)
{
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = nonSatConvertToTypeSR<DT>(Inf, 0);
    uint8_t data[]  = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToTypeSR<DT>(69000, 0);
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToTypeSR<DT>(NegInf, 0);
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToTypeSR<DT>(-69000, 0);
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e4m3_mxfp8_test, getDataMax)
{
    float mBits = getDataMantissaBits<DT>();
    float mantissa
        = 1 + std::pow(2, -mBits) * (std::pow(2, mBits) - 2); // Exclude NaN case (mantissa 0b111)
    float maxi = std::pow(2, 8) * mantissa; // Multiply max biased exp
    EXPECT_EQ(maxi, getDataMax<DT>());
}

TEST_F(ocp_e4m3_mxfp8_test, getDataMin)
{
    EXPECT_EQ(std::pow(2, 1 - 7), getDataMin<DT>()); // Min biased exp
}

TEST_F(ocp_e4m3_mxfp8_test, getDataMaxSubnorm)
{
    float exp      = std::pow(2, 1 - 7); // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * (std::pow(2, mBits) - 1);
    EXPECT_EQ(exp * mantissa, getDataMaxSubnorm<DT>());
}

TEST_F(ocp_e4m3_mxfp8_test, getDataMinSubnorm)
{
    float exp      = std::pow(2, 1 - 7); // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * 1;
    EXPECT_EQ(exp * mantissa, getDataMinSubnorm<DT>());
}

TEST_F(ocp_e4m3_mxfp8_test, roundToEvenTest)
{

    uint8_t tData[1];
    uint8_t tScale[] = {DGen::Constants::E8M0_1};

    for(int i = 0; i < (1 << 8); i += 2)
    {
        float input = (e4m3ValuesOCP[i] + e4m3ValuesOCP[i + 1]) / 2;
        *tData      = satConvertToType<DT>(input);

        float  fOutput = toFloat<DT>(tScale, tData, 0, 0);
        double dOutput = toDouble<DT>(tScale, tData, 0, 0);

        if(std::isnan(input))
        {
            EXPECT_TRUE(std::isnan(fOutput) && std::isnan(dOutput));
            continue;
        }

        EXPECT_EQ(e4m3ValuesOCP[i], fOutput);
        EXPECT_EQ(static_cast<double>(e4m3ValuesOCP[i]), dOutput);
    }
}

TEST_F(ocp_e4m3_mxfp8_test, roundToZeroTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 127; i++)
    {

        float negNum = e4m3ValuesOCP[i + 128];
        float posNum = e4m3ValuesOCP[i];

        while(posNum < e4m3ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, 0);
            EXPECT_EQ(e4m3ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, 0);
            EXPECT_EQ(e4m3ValuesOCP[i + 128], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(posNum, 0);
            EXPECT_EQ(e4m3ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(negNum, 0);
            EXPECT_EQ(e4m3ValuesOCP[i + 128], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

TEST_F(ocp_e4m3_mxfp8_test, roundToNextTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 127; i++)
    {

        float negNum = e4m3ValuesOCP[i + 128] - 0.01;
        float posNum = e4m3ValuesOCP[i] + 0.01;

        while(posNum < e4m3ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, UINT_MAX);
            EXPECT_EQ(e4m3ValuesOCP[i + 1], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, UINT_MAX);
            EXPECT_EQ(e4m3ValuesOCP[i + 129], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(posNum, UINT_MAX);
            EXPECT_EQ(e4m3ValuesOCP[i + 1], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(negNum, UINT_MAX);
            EXPECT_EQ(e4m3ValuesOCP[i + 129], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e4m3ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

// SR probablity is defined by the distanec to the next number
// if a number is in the middle it should be converted to the
// two numbers half the time
TEST_F(ocp_e4m3_mxfp8_test, midPointSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};
    for(int i = 0; i < 31; i++)
    {

        float lP = e4m3ValuesOCP[i], rP = e4m3ValuesOCP[i + 1], lN = e4m3ValuesOCP[i + 32],
              rN = e4m3ValuesOCP[i + 33];

        int satPlc = 0, satPrc = 0, satNlc = 0, satNrc = 0;
        int nSatPlc = 0, nSatPrc = 0, nSatNlc = 0, nSatNrc = 0;

        float pMid = (lP + rP) / 2;
        float nMid = (lN + rN) / 2;
        for(long seed = 0; seed <= UINT_MAX; seed += 4096)
        {
            *tData = satConvertToTypeSR<DT>(pMid, static_cast<uint>(seed));

            if(toFloat<DT>(tScale, tData, 0, 0) == lP)
                satPlc++;
            else
                satPrc++;

            *tData = satConvertToTypeSR<DT>(nMid, static_cast<uint>(seed));

            if(toFloat<DT>(tScale, tData, 0, 0) == lN)
                satNlc++;
            else
                satNrc++;

            *tData = nonSatConvertToTypeSR<DT>(pMid, static_cast<uint>(seed));

            if(toFloat<DT>(tScale, tData, 0, 0) == lP)
                nSatPlc++;
            else
                nSatPrc++;

            *tData = nonSatConvertToTypeSR<DT>(nMid, static_cast<uint>(seed));

            if(toFloat<DT>(tScale, tData, 0, 0) == lN)
                nSatNlc++;
            else
                nSatNrc++;
        }
        EXPECT_EQ(satPlc, satPrc) << "left point: " << lP << " right Point: " << rP
                                  << " mid point: " << pMid;
        EXPECT_EQ(satNlc, satNrc) << "left point: " << lN << " right Point: " << rN
                                  << " mid point: " << nMid;
        EXPECT_EQ(nSatPlc, nSatPrc)
            << "left point: " << lP << " right Point: " << rP << " mid point: " << pMid;
        EXPECT_EQ(nSatNlc, nSatNrc)
            << "left point: " << lN << " right Point: " << rN << " mid point: " << nMid;
    }
}

TEST_F(ocp_e4m3_mxfp8_test, greaterThanMaxTest)
{

    float max = getDataMax<DT>();

    uint8_t tData[1];
    uint8_t tScale[] = {DGen::Constants::E8M0_1};

    for(float input = max; input <= 464 + 1000; input += 1.5)
    {
        uint8_t satOutput  = satConvertToType<DT>(input);
        uint8_t nSatOutput = nonSatConvertToType<DT>(input);

        *tData = satOutput;
        EXPECT_EQ(getDataMax<DT>(), toFloat<DT>(tScale, tData, 0, 0));

        *tData        = nSatOutput;
        float nSatVal = toFloat<DT>(tScale, tData, 0, 0);
        if(input <= 464)
            EXPECT_EQ(getDataMax<DT>(), nSatVal);
        else
            EXPECT_TRUE(std::isnan(nSatVal));
    }
}
