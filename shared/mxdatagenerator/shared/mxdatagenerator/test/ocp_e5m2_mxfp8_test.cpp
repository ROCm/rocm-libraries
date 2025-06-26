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
#include <random>
#include <time.h>

#include "scale.hpp"

using namespace DGen;
using DT = DGen::ocp_e5m2_mxfp8;

constexpr double Inf        = DGen::Constants::Inf;
constexpr double NegInf     = DGen::Constants::NegInf;
constexpr double NotANumber = DGen::Constants::QNaN;
// 2^-m, for (m)antissa = 2 bits; diff between 1.0 and next fp (1.25)
constexpr double EPSILON = 0.25;

constexpr std::array<double, 256> e5m2ValuesOCP = {
    // clang-format off
         0.00000000000000000000,      0.00001525878906250000,      0.00003051757812500000,      0.00004577636718750000,
         0.00006103515625000000,      0.00007629394531250000,      0.00009155273437500000,      0.00010681152343750000,
         0.00012207031250000000,      0.00015258789062500000,      0.00018310546875000000,      0.00021362304687500000,
         0.00024414062500000000,      0.00030517578125000000,      0.00036621093750000000,      0.00042724609375000000,
         0.00048828125000000000,      0.00061035156250000000,      0.00073242187500000000,      0.00085449218750000000,
         0.00097656250000000000,      0.00122070312500000000,      0.00146484375000000000,      0.00170898437500000000,
         0.00195312500000000000,      0.00244140625000000000,      0.00292968750000000000,      0.00341796875000000000,
         0.00390625000000000000,      0.00488281250000000000,      0.00585937500000000000,      0.00683593750000000000,
         0.00781250000000000000,      0.00976562500000000000,      0.01171875000000000000,      0.01367187500000000000,
         0.01562500000000000000,      0.01953125000000000000,      0.02343750000000000000,      0.02734375000000000000,
         0.03125000000000000000,      0.03906250000000000000,      0.04687500000000000000,      0.05468750000000000000,
         0.06250000000000000000,      0.07812500000000000000,      0.09375000000000000000,      0.10937500000000000000,
         0.12500000000000000000,      0.15625000000000000000,      0.18750000000000000000,      0.21875000000000000000,
         0.25000000000000000000,      0.31250000000000000000,      0.37500000000000000000,      0.43750000000000000000,
         0.50000000000000000000,      0.62500000000000000000,      0.75000000000000000000,      0.87500000000000000000,
         1.00000000000000000000,      1.25000000000000000000,      1.50000000000000000000,      1.75000000000000000000,
         2.00000000000000000000,      2.50000000000000000000,      3.00000000000000000000,      3.50000000000000000000,
         4.00000000000000000000,      5.00000000000000000000,      6.00000000000000000000,      7.00000000000000000000,
         8.00000000000000000000,     10.00000000000000000000,     12.00000000000000000000,     14.00000000000000000000,
        16.00000000000000000000,     20.00000000000000000000,     24.00000000000000000000,     28.00000000000000000000,
        32.00000000000000000000,     40.00000000000000000000,     48.00000000000000000000,     56.00000000000000000000,
        64.00000000000000000000,     80.00000000000000000000,     96.00000000000000000000,    112.00000000000000000000,
       128.00000000000000000000,    160.00000000000000000000,    192.00000000000000000000,    224.00000000000000000000,
       256.00000000000000000000,    320.00000000000000000000,    384.00000000000000000000,    448.00000000000000000000,
       512.00000000000000000000,    640.00000000000000000000,    768.00000000000000000000,    896.00000000000000000000,
      1024.00000000000000000000,   1280.00000000000000000000,   1536.00000000000000000000,   1792.00000000000000000000,
      2048.00000000000000000000,   2560.00000000000000000000,   3072.00000000000000000000,   3584.00000000000000000000,
      4096.00000000000000000000,   5120.00000000000000000000,   6144.00000000000000000000,   7168.00000000000000000000,
      8192.00000000000000000000,  10240.00000000000000000000,  12288.00000000000000000000,  14336.00000000000000000000,
     16384.00000000000000000000,  20480.00000000000000000000,  24576.00000000000000000000,  28672.00000000000000000000,
     32768.00000000000000000000,  40960.00000000000000000000,  49152.00000000000000000000,  57344.00000000000000000000,
                            Inf,                  NotANumber,                  NotANumber,                  NotANumber,
        -0.00000000000000000000,     -0.00001525878906250000,     -0.00003051757812500000,     -0.00004577636718750000,
        -0.00006103515625000000,     -0.00007629394531250000,     -0.00009155273437500000,     -0.00010681152343750000,
        -0.00012207031250000000,     -0.00015258789062500000,     -0.00018310546875000000,     -0.00021362304687500000,
        -0.00024414062500000000,     -0.00030517578125000000,     -0.00036621093750000000,     -0.00042724609375000000,
        -0.00048828125000000000,     -0.00061035156250000000,     -0.00073242187500000000,     -0.00085449218750000000,
        -0.00097656250000000000,     -0.00122070312500000000,     -0.00146484375000000000,     -0.00170898437500000000,
        -0.00195312500000000000,     -0.00244140625000000000,     -0.00292968750000000000,     -0.00341796875000000000,
        -0.00390625000000000000,     -0.00488281250000000000,     -0.00585937500000000000,     -0.00683593750000000000,
        -0.00781250000000000000,     -0.00976562500000000000,     -0.01171875000000000000,     -0.01367187500000000000,
        -0.01562500000000000000,     -0.01953125000000000000,     -0.02343750000000000000,     -0.02734375000000000000,
        -0.03125000000000000000,     -0.03906250000000000000,     -0.04687500000000000000,     -0.05468750000000000000,
        -0.06250000000000000000,     -0.07812500000000000000,     -0.09375000000000000000,     -0.10937500000000000000,
        -0.12500000000000000000,     -0.15625000000000000000,     -0.18750000000000000000,     -0.21875000000000000000,
        -0.25000000000000000000,     -0.31250000000000000000,     -0.37500000000000000000,     -0.43750000000000000000,
        -0.50000000000000000000,     -0.62500000000000000000,     -0.75000000000000000000,     -0.87500000000000000000,
        -1.00000000000000000000,     -1.25000000000000000000,     -1.50000000000000000000,     -1.75000000000000000000,
        -2.00000000000000000000,     -2.50000000000000000000,     -3.00000000000000000000,     -3.50000000000000000000,
        -4.00000000000000000000,     -5.00000000000000000000,     -6.00000000000000000000,     -7.00000000000000000000,
        -8.00000000000000000000,    -10.00000000000000000000,    -12.00000000000000000000,    -14.00000000000000000000,
       -16.00000000000000000000,    -20.00000000000000000000,    -24.00000000000000000000,    -28.00000000000000000000,
       -32.00000000000000000000,    -40.00000000000000000000,    -48.00000000000000000000,    -56.00000000000000000000,
       -64.00000000000000000000,    -80.00000000000000000000,    -96.00000000000000000000,   -112.00000000000000000000,
      -128.00000000000000000000,   -160.00000000000000000000,   -192.00000000000000000000,   -224.00000000000000000000,
      -256.00000000000000000000,   -320.00000000000000000000,   -384.00000000000000000000,   -448.00000000000000000000,
      -512.00000000000000000000,   -640.00000000000000000000,   -768.00000000000000000000,   -896.00000000000000000000,
     -1024.00000000000000000000,  -1280.00000000000000000000,  -1536.00000000000000000000,  -1792.00000000000000000000,
     -2048.00000000000000000000,  -2560.00000000000000000000,  -3072.00000000000000000000,  -3584.00000000000000000000,
     -4096.00000000000000000000,  -5120.00000000000000000000,  -6144.00000000000000000000,  -7168.00000000000000000000,
     -8192.00000000000000000000, -10240.00000000000000000000, -12288.00000000000000000000, -14336.00000000000000000000,
    -16384.00000000000000000000, -20480.00000000000000000000, -24576.00000000000000000000, -28672.00000000000000000000,
    -32768.00000000000000000000, -40960.00000000000000000000, -49152.00000000000000000000, -57344.00000000000000000000,
                         NegInf,                  NotANumber,                  NotANumber,                  NotANumber
    // clang-format off
};

constexpr uint8_t e5m2BitsOCP[] = 
{
    // clang-format off
    0b00000000, 0b00000001, 0b00000010, 0b00000011,
    0b00000100, 0b00000101, 0b00000110, 0b00000111,
    0b00001000, 0b00001001, 0b00001010, 0b00001011,
    0b00001100, 0b00001101, 0b00001110, 0b00001111,
    0b00010000, 0b00010001, 0b00010010, 0b00010011,
    0b00010100, 0b00010101, 0b00010110, 0b00010111,
    0b00011000, 0b00011001, 0b00011010, 0b00011011,
    0b00011100, 0b00011101, 0b00011110, 0b00011111,
    0b00100000, 0b00100001, 0b00100010, 0b00100011,
    0b00100100, 0b00100101, 0b00100110, 0b00100111,
    0b00101000, 0b00101001, 0b00101010, 0b00101011,
    0b00101100, 0b00101101, 0b00101110, 0b00101111,
    0b00110000, 0b00110001, 0b00110010, 0b00110011,
    0b00110100, 0b00110101, 0b00110110, 0b00110111,
    0b00111000, 0b00111001, 0b00111010, 0b00111011,
    0b00111100, 0b00111101, 0b00111110, 0b00111111,
    0b01000000, 0b01000001, 0b01000010, 0b01000011,
    0b01000100, 0b01000101, 0b01000110, 0b01000111,
    0b01001000, 0b01001001, 0b01001010, 0b01001011,
    0b01001100, 0b01001101, 0b01001110, 0b01001111,
    0b01010000, 0b01010001, 0b01010010, 0b01010011,
    0b01010100, 0b01010101, 0b01010110, 0b01010111,
    0b01011000, 0b01011001, 0b01011010, 0b01011011,
    0b01011100, 0b01011101, 0b01011110, 0b01011111,
    0b01100000, 0b01100001, 0b01100010, 0b01100011,
    0b01100100, 0b01100101, 0b01100110, 0b01100111,
    0b01101000, 0b01101001, 0b01101010, 0b01101011,
    0b01101100, 0b01101101, 0b01101110, 0b01101111,
    0b01110000, 0b01110001, 0b01110010, 0b01110011,
    0b01110100, 0b01110101, 0b01110110, 0b01110111,
    0b01111000, 0b01111001, 0b01111010, 0b01111011,
    0b01111100, 0b01111101, 0b01111110, 0b01111111,
    0b10000000, 0b10000001, 0b10000010, 0b10000011,
    0b10000100, 0b10000101, 0b10000110, 0b10000111,
    0b10001000, 0b10001001, 0b10001010, 0b10001011,
    0b10001100, 0b10001101, 0b10001110, 0b10001111,
    0b10010000, 0b10010001, 0b10010010, 0b10010011,
    0b10010100, 0b10010101, 0b10010110, 0b10010111,
    0b10011000, 0b10011001, 0b10011010, 0b10011011,
    0b10011100, 0b10011101, 0b10011110, 0b10011111,
    0b10100000, 0b10100001, 0b10100010, 0b10100011,
    0b10100100, 0b10100101, 0b10100110, 0b10100111,
    0b10101000, 0b10101001, 0b10101010, 0b10101011,
    0b10101100, 0b10101101, 0b10101110, 0b10101111,
    0b10110000, 0b10110001, 0b10110010, 0b10110011,
    0b10110100, 0b10110101, 0b10110110, 0b10110111,
    0b10111000, 0b10111001, 0b10111010, 0b10111011,
    0b10111100, 0b10111101, 0b10111110, 0b10111111,
    0b11000000, 0b11000001, 0b11000010, 0b11000011,
    0b11000100, 0b11000101, 0b11000110, 0b11000111,
    0b11001000, 0b11001001, 0b11001010, 0b11001011,
    0b11001100, 0b11001101, 0b11001110, 0b11001111,
    0b11010000, 0b11010001, 0b11010010, 0b11010011,
    0b11010100, 0b11010101, 0b11010110, 0b11010111,
    0b11011000, 0b11011001, 0b11011010, 0b11011011,
    0b11011100, 0b11011101, 0b11011110, 0b11011111,
    0b11100000, 0b11100001, 0b11100010, 0b11100011,
    0b11100100, 0b11100101, 0b11100110, 0b11100111,
    0b11101000, 0b11101001, 0b11101010, 0b11101011,
    0b11101100, 0b11101101, 0b11101110, 0b11101111,
    0b11110000, 0b11110001, 0b11110010, 0b11110011,
    0b11110100, 0b11110101, 0b11110110, 0b11110111,
    0b11111000, 0b11111001, 0b11111010, 0b11111011,
    0b11111100, 0b11111101, 0b11111110, 0b11111111
    // clang-format on
};

class ocp_e5m2_mxfp8_test : public ::testing::Test
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
    // [E5M2] Element Data
    const uint8_t data[10] = {
        0b00000000, // 0
        0b00111100, // 1
        0b01111101, // NaN 1
        0b01111110, // NaN 2
        0b01111111, // NaN 3
        0b01111100, // inf
        0b00000100, //  (min norm)
        0b01111011, //  (max norm)
        0b00000001, //  (min subnorm)
        0b00000011 //  (max subnorm)
    };
    // [E5M2] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[10] = {
        0b10000000, // 0
        0b10111100, // 1
        0b11111101, // NaN 1
        0b11111110, // NaN 2
        0b11111111, // NaN 3
        0b11111100, // inf
        0b10000100, //  (min norm)
        0b11111011, //  (max norm)
        0b10000001, //  (min subnorm)
        0b10000011 //  (max subnorm)
    };

    double getClosest(double num)
    {
        double closestDiff = UINT32_MAX;

        for(size_t i = 0; i < e5m2ValuesOCP.size(); i++)
        {
            if(std::isnan(e5m2ValuesOCP[i]))
                continue;
            closestDiff = std::min(closestDiff, std::abs(num - e5m2ValuesOCP[i]));
        }
        return closestDiff;
    }
};

TEST_F(ocp_e5m2_mxfp8_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(true, isOne<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 5)); // NaN * Inf
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 6)); // NaN * min normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 7)); // NaN * max normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 2)); // 2^127 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 3)); // 2^127 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 4)); // 2^127 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 5)); // 2^127 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 6)); // 2^127 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 7)); // 2^127 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 8)); // 2^127 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, data, 5, 9)); // 2^127 * max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 5)); // NaN * Inf
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 6)); // NaN * min normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 7)); // NaN * max normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 0)); // 2^126 * 0
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 1)); // 2^126 * 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 2)); // 2^126 * NaN 1
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 3)); // 2^126 * NaN 2
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 4)); // 2^126 * NaN 3
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 5)); // 2^126 * Inf
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 6)); // 2^126 * min normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 7)); // 2^126 * max normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 8)); // 2^126 * min sub-normal
    EXPECT_EQ(false, isOne<DT>(scales, negativeData, 5, 9)); // 2^126 * max sub-normal
}

TEST_F(ocp_e5m2_mxfp8_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 5)); // NaN * Inf
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 6)); // NaN * min normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 7)); // NaN * max normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 2)); // 2^127 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 3)); // 2^127 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 4)); // 2^127 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 5)); // 2^127 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 6)); // 2^127 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 7)); // 2^127 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 8)); // 2^127 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, data, 5, 9)); // 2^127 * max sub-normal

    // =========================================================================================//

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 5)); // NaN * Inf
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 6)); // NaN * min normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 7)); // NaN * max normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(true, isZero<DT>(scales, negativeData, 5, 0)); // 2^126 * 0
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 1)); // 2^126 * 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 2)); // 2^126 * NaN 1
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 3)); // 2^126 * NaN 2
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 4)); // 2^126 * NaN 3
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 5)); // 2^126 * Inf
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 6)); // 2^126 * min normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 7)); // 2^126 * max normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 8)); // 2^126 * min sub-normal
    EXPECT_EQ(false, isZero<DT>(scales, negativeData, 5, 9)); // 2^126 * max sub-normal
}

TEST_F(ocp_e5m2_mxfp8_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 5)); // NaN * Inf
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 6)); // NaN * min normal
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 7)); // NaN * max normal
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(true, isNaN<DT>(scales, data, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 5, 2)); // 2^127 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, data, 5, 3)); // 2^127 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, data, 5, 4)); // 2^127 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 5)); // 2^127 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 6)); // 2^127 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 7)); // 2^127 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 8)); // 2^127 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, data, 5, 9)); // 2^127 * max sub-normal

    // =========================================================================================//

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 5)); // NaN * Inf
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 6)); // NaN * min normal
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 7)); // NaN * max normal
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 0)); // 2^126 * 0
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 1)); // 2^126 * 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 5, 2)); // 2^126 * NaN 1
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 5, 3)); // 2^126 * NaN 2
    EXPECT_EQ(true, isNaN<DT>(scales, negativeData, 5, 4)); // 2^126 * NaN 3
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 5)); // 2^126 * Inf
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 6)); // 2^126 * min normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 7)); // 2^126 * max normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 8)); // 2^126 * min sub-normal
    EXPECT_EQ(false, isNaN<DT>(scales, negativeData, 5, 9)); // 2^126 * max sub-normal
}

TEST_F(ocp_e5m2_mxfp8_test, isInf)
{
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, data, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, data, 1, 5)); // NaN * Inf
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 6)); // NaN * min normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 7)); // NaN * max normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, data, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, data, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, data, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 0)); // 2^127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 1)); // 2^127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 2)); // 2^127 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 3)); // 2^127 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 4)); // 2^127 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, data, 5, 5)); // 2^127 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 6)); // 2^127 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 7)); // 2^127 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 8)); // 2^127 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, data, 5, 9)); // 2^127 * max sub-normal

    // =========================================================================================//

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 0)); // 1 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 1)); // 1 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 2)); // 1 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 3)); // 1 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 4)); // 1 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, negativeData, 0, 5)); // 1 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 6)); // 1 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 7)); // 1 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 8)); // 1 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 0, 9)); // 1 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 0)); // NaN * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 1)); // NaN * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 2)); // NaN * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 3)); // NaN * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 4)); // NaN * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, negativeData, 1, 5)); // NaN * Inf
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 6)); // NaN * min normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 7)); // NaN * max normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 8)); // NaN * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 1, 9)); // NaN * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 0)); // 0.0078125 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 1)); // 0.0078125 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 2)); // 0.0078125 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 3)); // 0.0078125 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 4)); // 0.0078125 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, negativeData, 2, 5)); // 0.0078125 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 6)); // 0.0078125 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 7)); // 0.0078125 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 8)); // 0.0078125 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 2, 9)); // 0.0078125 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 0)); // 64 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 1)); // 64 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 2)); // 64 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 3)); // 64 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 4)); // 64 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, negativeData, 3, 5)); // 64 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 6)); // 64 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 7)); // 64 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 8)); // 64 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 3, 9)); // 64 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 0)); // 2^-127 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 1)); // 2^-127 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 2)); // 2^-127 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 3)); // 2^-127 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 4)); // 2^-127 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, negativeData, 4, 5)); // 2^-127 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 6)); // 2^-127 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 7)); // 2^-127 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 8)); // 2^-127 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 4, 9)); // 2^-127 * max sub-normal

    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 0)); // 2^126 * 0
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 1)); // 2^126 * 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 2)); // 2^126 * NaN 1
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 3)); // 2^126 * NaN 2
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 4)); // 2^126 * NaN 3
    EXPECT_EQ(true, isInf<DT>(scales, negativeData, 5, 5)); // 2^126 * Inf
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 6)); // 2^126 * min normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 7)); // 2^126 * max normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 8)); // 2^126 * min sub-normal
    EXPECT_EQ(false, isInf<DT>(scales, negativeData, 5, 9)); // 2^126 * max sub-normal
}

TEST_F(ocp_e5m2_mxfp8_test, isSubnorm)
{
    uint8_t temp[] = {DGen::Constants::E8M0_1, 0b0};

    for(size_t i = 0; i < e5m2ValuesOCP.size(); i++)
    {
        uint8_t data = static_cast<uint8_t>(i) & 0xff;
        temp[1]      = data;

        uint8_t exp = (data >> getDataMantissaBits<DT>()) & 0b011111;

        double value = toDouble<DT>(temp, temp, 0, 1);

        if(exp != 0b0 || std::isnan(value))
            EXPECT_FALSE(isSubnorm<DT>(temp, 1));
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 1));
    }
}

// true if XN < val (first arg)
TEST_F(ocp_e5m2_mxfp8_test, isLess)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
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
TEST_F(ocp_e5m2_mxfp8_test, isGreater)
{
    double values[]
        = {NegInf, -10, -5, -1, -0.5, -0.000005, -0, NotANumber, 0, 0.000005, 0.5, 1, 5, 10, Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
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

TEST_F(ocp_e5m2_mxfp8_test, toFloatAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        float ref = e8m0Values[i];
        for(size_t j = 0; j < e5m2ValuesOCP.size(); j++)
        {
            float  res      = toFloat<DT>(e8m0Bits, e5m2BitsOCP, i, j);
            double expected = ref * e5m2ValuesOCP[j];

            if(ref == 0 && std::isinf(e5m2ValuesOCP[j]))
                expected = e5m2ValuesOCP[j];

            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref) || std::isnan(e5m2ValuesOCP[j]));
            else if(expected > FLT_MAX)
                EXPECT_EQ(std::numeric_limits<float>::infinity(), res);
            else if(expected < -FLT_MAX)
                EXPECT_EQ(-std::numeric_limits<float>::infinity(), res);
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e5m2_mxfp8_test, toDoubleAllScalesAllValues)
{
    for(size_t i = 0; i < e8m0Values.size(); i++)
    {
        double ref = e8m0Values[i];

        for(size_t j = 0; j < e5m2ValuesOCP.size(); j++)
        {
            double res      = toDouble<DT>(e8m0Bits, e5m2BitsOCP, i, j);
            double expected = ref * e5m2ValuesOCP[j];

            if(ref == 0 && std::isinf(e5m2ValuesOCP[j]))
                expected = e5m2ValuesOCP[j];

            if(std::isnan(res)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(ref) || std::isnan(e5m2ValuesOCP[j]));
            else if(std::isinf(res))
                EXPECT_TRUE(std::isinf(expected));
            else
                EXPECT_NEAR(expected, res, EPSILON);
        }
    }
}

TEST_F(ocp_e5m2_mxfp8_test, setOne)
{
    uint8_t scale[] = {0b0};
    uint8_t data[]  = {0b0};
    setOne<DT>(scale, data, 0, 0);
    double dElem = toDouble<DT>(scale, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(ocp_e5m2_mxfp8_test, setZero)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setZero<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(ocp_e5m2_mxfp8_test, setNaN)
{
    uint8_t scale[] = {0b0, 0b0};
    uint8_t data[]  = {0b0, 0b0};
    setNaN<DT>(scale, data, 1, 1);
    double dElem = toDouble<DT>(scale, data, 1, 1);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(ocp_e5m2_mxfp8_test, setDataMaxNorm)
{
    uint8_t scale[] = {DGen::Constants::E8M0_1}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0); // Leave optional params to normal, pos
    EXPECT_EQ(0b01111011, data[0]);
    EXPECT_EQ(57344.0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e5m2_mxfp8_test, setDataMaxNeg)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, 0, 0); // Normal, negative
    EXPECT_EQ(0b11111011, data[0]);
    EXPECT_EQ(-57344.0, toDouble<DT>(scale, data, 0, 0));
}

TEST_F(ocp_e5m2_mxfp8_test, setDataMaxSubnorm)
{
    uint8_t scale[] = {0b01111111}; // 1
    uint8_t data[]  = {0b0};
    setDataMax<DT>(data, 0, 1, 1); // Subnorm, positive
    EXPECT_EQ(0b00000011, data[0]);
    EXPECT_EQ(std::pow(2, -14) * 0.75, toDouble<DT>(scale, data, 0, 0));
}

// Saturated tests
TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeRounding)
{
    float   norm    = 446.789123456789f;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(norm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(norm);

    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeRoundingSmallSubnorm)
{
    float   subnorm = 0.00000005960;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);

    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeRoundingLargeSubnorm)
{
    float   subnorm = 0.001952157;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeRoundingSmallSubnormNeg)
{
    float   subnorm = -0.00000005960;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};
    EXPECT_NEAR(subnorm, toDouble<DT>(scale, data, 0, 0),
                EPSILON); // Just becomes 0 in this case
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeRoundingLargeSubnormNeg)
{
    float   subnorm = -0.001952157;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};
    EXPECT_NEAR(subnorm, toDouble<DT>(scale, data, 0, 0), EPSILON);
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b01111011, satConvertToType<DT>(largePos)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSRLargePos)
{
    float largePos = 123456.7891234567f;
    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(largePos, 0)); // Expect max norm (57344)
    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(largePos, UINT_MAX)); // Expect max norm (57344)
    EXPECT_EQ(0b01111011,
              satConvertToTypeSR<DT>(largePos, UINT_MAX / 2)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypePosMax)
{
    float e5m2Max = 57344.f;
    EXPECT_EQ(0b01111011, satConvertToType<DT>(e5m2Max)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSRPosMax)
{
    float e5m2Max = getDataMax<DT>();
    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(e5m2Max, 0)); // Expect max norm (57344)
    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(e5m2Max, UINT_MAX)); // Expect max norm (57344)
    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(e5m2Max, UINT_MAX / 2)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, satConvertToType<DT>(zero));
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSRZero)
{
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(0, 0)); // Expect max norm (57344)
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(0, UINT_MAX)); // Expect max norm (57344)
    EXPECT_EQ(0b0, satConvertToTypeSR<DT>(0, UINT_MAX / 2)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeNegMax)
{
    float e5m2NegMax = -57344.f;
    EXPECT_EQ(0b11111011, satConvertToType<DT>(e5m2NegMax)); // Expect -max norm (-57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSRNegMax)
{
    EXPECT_EQ(0b11111011,
              satConvertToTypeSR<DT>(-getDataMax<DT>(), 0)); // Expect -max norm (-57344)
    EXPECT_EQ(0b11111011,
              satConvertToTypeSR<DT>(-getDataMax<DT>(),
                                     UINT_MAX)); // Expect -max norm (-57344)
    EXPECT_EQ(0b11111011,
              satConvertToTypeSR<DT>(-getDataMax<DT>(),
                                     UINT_MAX / 2)); // Expect -max norm (-57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeLargeNeg)
{
    float largeNeg = -123456.7891234567f;
    EXPECT_EQ(0b11111011, satConvertToType<DT>(largeNeg)); // Expect -max norm (-57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSRLargeNeg)
{
    EXPECT_EQ(0b11111011,
              satConvertToTypeSR<DT>(-123456.789123456f, 0)); // Expect -max norm (-57344)
    EXPECT_EQ(0b11111011,
              satConvertToTypeSR<DT>(-123456.789123456f, UINT_MAX)); // Expect -max norm (-57344)
    EXPECT_EQ(0b11111011,
              satConvertToTypeSR<DT>(-123456.789123456f,
                                     UINT_MAX / 2)); // Expect -max norm (-57344)
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeNaN)
{
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = satConvertToType<DT>(NAN);
    uint8_t data[]  = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSRNaN)
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
TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeRandom)
{

    double lb = -60000;
    double ub = 60000;

    srandom(time(NULL));

    uint8_t scale[] = {DGen::Constants::E8M0_1};

    std::default_random_engine re;

    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        double closestDiff = getClosest(rNum);

        uint8_t res    = satConvertToType<DT>(rNum);
        uint8_t data[] = {res};

        double output = toDouble<DT>(scale, data, 0, 0);

        EXPECT_EQ(closestDiff, std::abs(rNum - output))
            << "rNum: " << rNum << "\noutput: " << output;
    }
}

TEST_F(ocp_e5m2_mxfp8_test, satConvertToTypeSROutOfRange)
{
    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(Inf, 0));

    EXPECT_EQ(0b01111011, satConvertToTypeSR<DT>(69000, 0));

    EXPECT_EQ(0b11111011, satConvertToTypeSR<DT>(NegInf, 0));

    EXPECT_EQ(0b11111011, satConvertToTypeSR<DT>(-69000, 0));
}

// NON SAT TESTS

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeRounding)
{
    float   norm    = 446.789123456789f;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(norm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(norm);

    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeRoundingSmallSubnorm)
{
    float   subnorm = 0.00000005960;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);

    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeRoundingLargeSubnorm)
{
    float   subnorm = 0.001952157;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeRoundingSmallSubnormNeg)
{
    float   subnorm = -0.00000005960;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};
    EXPECT_NEAR(subnorm, toDouble<DT>(scale, data, 0, 0),
                EPSILON); // Just becomes 0 in this case
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeRoundingLargeSubnormNeg)
{
    float   subnorm = -0.001952157;
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(subnorm);
    uint8_t data[]  = {res};

    double closestDiff = getClosest(subnorm);
    EXPECT_EQ(closestDiff, std::abs(subnorm - toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypePosMax)
{
    float e5m2Max = 57344.f;
    EXPECT_EQ(0b01111011, nonSatConvertToType<DT>(e5m2Max)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeSRPosMax)
{
    float e5m2Max = 57344.f;
    EXPECT_EQ(0b01111011, nonSatConvertToTypeSR<DT>(e5m2Max, 0)); // Expect max norm (57344)
    EXPECT_EQ(0b01111011, nonSatConvertToTypeSR<DT>(e5m2Max, UINT_MAX)); // Expect max norm (57344)
    EXPECT_EQ(0b01111011,
              nonSatConvertToTypeSR<DT>(e5m2Max, UINT_MAX / 2)); // Expect max norm (57344)
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, nonSatConvertToType<DT>(zero));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeSRZero)
{
    EXPECT_EQ(0b00000000, nonSatConvertToTypeSR<DT>(0, 0));
    EXPECT_EQ(0b00000000, nonSatConvertToTypeSR<DT>(0, UINT_MAX));
    EXPECT_EQ(0b00000000, nonSatConvertToTypeSR<DT>(0, UINT_MAX / 2));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeNegMax)
{
    float e5m2NegMax = -57344.f;
    EXPECT_EQ(0b11111011, nonSatConvertToType<DT>(e5m2NegMax)); // Expect -max norm (-57344)
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeSRNegMax)
{
    float e5m2NegMax = -57344.f;
    EXPECT_EQ(0b11111011, nonSatConvertToTypeSR<DT>(e5m2NegMax, 0)); // Expect -max norm (-57344)
    EXPECT_EQ(0b11111011,
              nonSatConvertToTypeSR<DT>(e5m2NegMax, UINT_MAX)); // Expect -max norm (-57344)
    EXPECT_EQ(0b11111011,
              nonSatConvertToTypeSR<DT>(e5m2NegMax, UINT_MAX / 2)); // Expect -max norm (-57344)
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeNaN)
{
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(NAN);

    uint8_t data[] = {res};
    EXPECT_EQ(true, std::isnan(toDouble<DT>(scale, data, 0, 0)));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeSRNaN)
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

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeOutOfRange)
{
    uint8_t scale[] = {DGen::Constants::E8M0_1};
    uint8_t res     = nonSatConvertToType<DT>(Inf);
    uint8_t data[]  = {res};
    EXPECT_TRUE(std::isinf(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToType<DT>(69000);
    EXPECT_TRUE(std::isinf(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToType<DT>(NegInf);
    EXPECT_TRUE(std::isinf(toDouble<DT>(scale, data, 0, 0)));
    EXPECT_TRUE((toDouble<DT>(scale, data, 0, 0) < 0));

    data[0] = nonSatConvertToType<DT>(-69000);
    EXPECT_TRUE(std::isinf(toDouble<DT>(scale, data, 0, 0)));
    EXPECT_TRUE((toDouble<DT>(scale, data, 0, 0) < 0));
}

TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeSROutOfRange)
{
    uint8_t scale[] = {Constants::E8M0_1};
    uint8_t res     = nonSatConvertToTypeSR<DT>(Inf, 0);
    uint8_t data[]  = {res};
    EXPECT_EQ(true, std::isinf(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToTypeSR<DT>(69000, 0);
    EXPECT_EQ(true, std::isinf(toDouble<DT>(scale, data, 0, 0)));

    data[0] = nonSatConvertToTypeSR<DT>(NegInf, 0);
    EXPECT_EQ(true, std::isinf(toDouble<DT>(scale, data, 0, 0)));
    EXPECT_EQ(true, (toDouble<DT>(scale, data, 0, 0) < 0));

    data[0] = nonSatConvertToTypeSR<DT>(-69000, 0);
    EXPECT_EQ(true, std::isinf(toDouble<DT>(scale, data, 0, 0)));
    EXPECT_EQ(true, (toDouble<DT>(scale, data, 0, 0) < 0));
}

// Generate 1000000 numbers and see if the conversion is good
TEST_F(ocp_e5m2_mxfp8_test, nonSatConvertToTypeRandom)
{

    double lb   = -60000;
    double ub   = 60000;
    float  maxi = getDataMax<DT>();

    float prevVal = std::pow(2, 15) * std::pow(2, -1);

    maxi += (maxi - prevVal) / 2;

    srandom(time(NULL));

    uint8_t scale[] = {DGen::Constants::E8M0_1};

    std::default_random_engine re;

    uint8_t data[1];
    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<double> unif(lb, ub);

        double rNum = unif(re);

        double closestDiff = getClosest(rNum);

        data[0] = nonSatConvertToType<DT>(rNum);
        if(std::abs(rNum) > maxi)
        {

            EXPECT_TRUE(std::isinf(toDouble<DT>(scale, data, 0, 0)));

            bool isCorrectSign = rNum < 0 ? toDouble<DT>(scale, data, 0, 0) < 0
                                          : toDouble<DT>(scale, data, 0, 0) > 0;

            EXPECT_TRUE(isCorrectSign);
            continue;
        }

        EXPECT_EQ(closestDiff, std::abs(rNum - toDouble<DT>(scale, data, 0, 0)));
    }
}

TEST_F(ocp_e5m2_mxfp8_test, getDataMax)
{
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = 1 + std::pow(2, -mBits) * (std::pow(2, mBits) - 1);
    float maxi     = std::pow(2, 15) * mantissa; // Multiply max biased exp
    EXPECT_EQ(maxi, getDataMax<DT>());
}

TEST_F(ocp_e5m2_mxfp8_test, getDataMin)
{
    EXPECT_EQ(std::pow(2, 1 - 15), getDataMin<DT>()); // Min biased exp
}

TEST_F(ocp_e5m2_mxfp8_test, getDataMaxSubnorm)
{
    float exp      = std::pow(2, 1 - 15); // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * (std::pow(2, mBits) - 1);
    EXPECT_EQ(exp * mantissa, getDataMaxSubnorm<DT>());
}

TEST_F(ocp_e5m2_mxfp8_test, getDataMinSubnorm)
{
    float exp      = std::pow(2, 1 - 15); // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * 1;
    EXPECT_EQ(exp * mantissa, getDataMinSubnorm<DT>());
}

TEST_F(ocp_e5m2_mxfp8_test, roundToEvenTest)
{

    uint8_t tData[1];
    uint8_t tScale[] = {DGen::Constants::E8M0_1};

    for(int i = 0; i < (1 << 8); i += 2)
    {
        float input = (e5m2ValuesOCP[i] + e5m2ValuesOCP[i + 1]) / 2;
        *tData      = satConvertToType<DT>(input);

        float  fOutput = toFloat<DT>(tScale, tData, 0, 0);
        double dOutput = toDouble<DT>(tScale, tData, 0, 0);

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

        EXPECT_EQ(e5m2ValuesOCP[i], fOutput);
        EXPECT_EQ(static_cast<double>(e5m2ValuesOCP[i]), dOutput);
    }
}

TEST_F(ocp_e5m2_mxfp8_test, roundToZeroTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 123; i++)
    {

        float negNum = e5m2ValuesOCP[i + 128];
        float posNum = e5m2ValuesOCP[i];

        while(posNum < e5m2ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, 0);
            EXPECT_EQ(e5m2ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, 0);
            EXPECT_EQ(e5m2ValuesOCP[i + 128], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(posNum, 0);
            EXPECT_EQ(e5m2ValuesOCP[i], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(negNum, 0);
            EXPECT_EQ(e5m2ValuesOCP[i + 128], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

TEST_F(ocp_e5m2_mxfp8_test, roundToNextTestSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};

    for(int i = 0; i < 123; i++)
    {

        float negNum = e5m2ValuesOCP[i + 128] - 0.01;
        float posNum = e5m2ValuesOCP[i] + 0.01;

        while(posNum < e5m2ValuesOCP[i + 1])
        {
            *tData = satConvertToTypeSR<DT>(posNum, UINT_MAX);
            EXPECT_EQ(e5m2ValuesOCP[i + 1], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = satConvertToTypeSR<DT>(negNum, UINT_MAX);
            EXPECT_EQ(e5m2ValuesOCP[i + 129], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(posNum, UINT_MAX);
            EXPECT_EQ(e5m2ValuesOCP[i + 1], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i] << " --- Current Input: " << posNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            *tData = nonSatConvertToTypeSR<DT>(negNum, UINT_MAX);
            EXPECT_EQ(e5m2ValuesOCP[i + 129], toFloat<DT>(tScale, tData, 0, 0))
                << "Original Number: " << e5m2ValuesOCP[i + 128] << " --- Current Input: " << negNum
                << " --- Output: " << toFloat<DT>(tScale, tData, 0, 0);

            negNum -= 0.01;
            posNum += 0.01;
        }
    }
}

// SR probablity is defined by the distanec to the next number
// if a number is in the middle it should be converted to the
// two numbers half the time
TEST_F(ocp_e5m2_mxfp8_test, midPointSR)
{
    uint8_t tData[1];
    uint8_t tScale[] = {Constants::E8M0_1};
    for(int i = 0; i < 31; i++)
    {

        float lP = e5m2ValuesOCP[i], rP = e5m2ValuesOCP[i + 1], lN = e5m2ValuesOCP[i + 32],
              rN = e5m2ValuesOCP[i + 33];

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

TEST_F(ocp_e5m2_mxfp8_test, greaterThanMaxTest)
{

    float max     = getDataMax<DT>();
    float prevVal = std::pow(2, 15) * (1 + std::pow(2, -1));

    float diff = max - prevVal;

    float actualMax = max + (diff / 2);

    uint8_t tData[1];
    uint8_t tScale[] = {DGen::Constants::E8M0_1};

    for(float input = max; input <= actualMax + 1000; input += 1.5)
    {
        uint8_t satOutput  = satConvertToType<DT>(input);
        uint8_t nSatOutput = nonSatConvertToType<DT>(input);

        *tData = satOutput;
        EXPECT_EQ(getDataMax<DT>(), toFloat<DT>(tScale, tData, 0, 0));

        *tData        = nSatOutput;
        float nSatVal = toFloat<DT>(tScale, tData, 0, 0);

        if(input <= actualMax)
            EXPECT_EQ(getDataMax<DT>(), nSatVal);
        else
            EXPECT_TRUE(std::isinf(nSatVal));
    }
}

TEST_F(ocp_e5m2_mxfp8_test, infInput)
{
    uint8_t output = nonSatConvertToType<DT>(std::numeric_limits<float>::infinity());
    EXPECT_EQ(output, 0b01111100);

    uint8_t tData[]  = {static_cast<uint8_t>(nonSatConvertToTypeSR<DT>(getDataMax<DT>() + 10, 0))};
    uint8_t tScale[] = {Constants::E8M0_1};

    EXPECT_EQ(getDataMax<DT>(), toFloat<DT>(tScale, tData, 0, 0));

    *tData = nonSatConvertToTypeSR<DT>(getDataMax<DT>() + 10, UINT_MAX);
    EXPECT_TRUE(std::isinf(toFloat<DT>(tScale, tData, 0, 0)))
        << "Input: " << getDataMax<DT>() + 10 << " seed: " << UINT_MAX;
    *tData = nonSatConvertToTypeSR<DT>(getDataMax<DT>() + 10, UINT_MAX / 2);
    EXPECT_FALSE(std::isinf(toFloat<DT>(tScale, tData, 0, 0)))
        << "Input: " << getDataMax<DT>() + 10 << " seed: " << UINT_MAX / 2;
    *tData = nonSatConvertToTypeSR<DT>(std::numeric_limits<float>::infinity(), UINT_MAX / 2);
    EXPECT_TRUE(std::isinf(toFloat<DT>(tScale, tData, 0, 0)))
        << "Input: " << std::numeric_limits<float>::infinity() << " seed: " << UINT_MAX / 2;
    *tData = nonSatConvertToTypeSR<DT>(std::numeric_limits<float>::infinity(), UINT_MAX);
    EXPECT_TRUE(std::isinf(toFloat<DT>(tScale, tData, 0, 0)))
        << "Input: " << std::numeric_limits<float>::infinity() << " seed: " << UINT_MAX;
    *tData = nonSatConvertToTypeSR<DT>(std::numeric_limits<float>::infinity(), 0);
    EXPECT_TRUE(std::isinf(toFloat<DT>(tScale, tData, 0, 0)))
        << "Input: " << std::numeric_limits<float>::infinity() << " seed: " << 0;
}
