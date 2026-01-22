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

#pragma once
#include <algorithm>
#include <array>

namespace DGen
{
    struct OCP_E5M2_MXFP8_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 5;
        static constexpr uint mantissaBits = 2;
        static constexpr uint bias         = 15;
        static constexpr uint srShift      = 11;

        static constexpr int unBiasedEMin = -14;
        static constexpr int unBiasedEMax = 16;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 31;

        static constexpr bool hasInf  = true;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    struct ocp_e5m2_mxfp8
    {
        static constexpr OCP_E5M2_MXFP8_DATA dataInfo{};
        static constexpr E8M0_SCALE_INFO     scaleInfo{};

        static constexpr uint8_t oneMask         = 0b00111100;
        static constexpr uint8_t signBitMask     = 0b10000000;
        static constexpr uint8_t positiveInfMask = 0b01111100;
        static constexpr uint8_t negativeInfMask = 0b11111100;

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b01111011;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b11111011;

        static constexpr uint8_t dataSubNormalOneMask = 0b00000010;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b00000011;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b10000011;

        static constexpr float dataMaxNormalNumber    = 57344;
        static constexpr float dataMinSubNormalNumber = 0.00001525878906250000;

        static constexpr uint8_t positiveZeroMask = 0b00000000;
        static constexpr uint8_t negativeZeroMask = 0b10000000;

        static constexpr float dataMaxRoundedRange = 65536; // 2^15 * 2

        static constexpr std::array<uint8_t, 6> dataNaNMasks{
            0b11111101, 0b11111110, 0b11111111, 0b01111101, 0b01111110, 0b01111111};
    };

#include "ocp_e5m2_mxfp8_impl.hpp"
}
