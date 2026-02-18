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
#include "dataTypeInfo.hpp"

namespace DGen
{
    struct FP16_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 5;
        static constexpr uint mantissaBits = 10;
        static constexpr uint bias         = 15;
        static constexpr uint srShift      = 19;

        static constexpr int unBiasedEMin = -14;
        static constexpr int unBiasedEMax = 15;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 31;

        static constexpr bool hasInf  = true;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    // fp16 does not deal with scales

    struct fp16
    {
        static constexpr FP16_DATA dataInfo{};

        static constexpr uint16_t oneMask     = 0b0011110000000000;
        static constexpr uint16_t setSignMask = 0b0111111111111111;

        static constexpr uint16_t dataNanMask         = 0b0111111000000000;
        static constexpr uint16_t dataInfMask         = 0b0111110000000000;
        static constexpr uint16_t dataNegativeInfMask = 0b1111110000000000;
        static constexpr uint16_t dataMantissaMask    = 0b0000001111111111;

        static constexpr uint16_t dataMaxPositiveNormalMask = 0b0111101111111111;
        static constexpr uint16_t dataMaxNegativeNormalMask = 0b1111101111111111;

        static constexpr uint32_t dataMaxPositiveSubNormalMask = 0b0000001111111111;
        static constexpr uint16_t dataMaxNegativeSubNormalMask = 0b1000001111111111;

        static constexpr float dataMaxNormalNumber    = 65504;
        static constexpr float dataMinSubNormalNumber = 0.00000005960464;

        static constexpr uint16_t positiveZeroMask = 0b0000000000000000;
        static constexpr uint16_t negativeZeroMask = 0b1000000000000000;
    };

#include "fp16_impl.hpp"
}
