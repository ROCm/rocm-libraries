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
    struct BF16_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 8;
        static constexpr uint mantissaBits = 7;
        static constexpr uint bias         = 127;
        static constexpr uint srShift      = 16;

        static constexpr int unBiasedEMin = -126;
        static constexpr int unBiasedEMax = 127;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 254;

        static constexpr bool hasInf  = true;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    //** NOTE THAT ALL SCALES WILL BE IGNORED IN FP16*/
    struct bf16
    {
        static constexpr BF16_DATA dataInfo{};

        static constexpr uint16_t oneMask             = 0b0011111110000000;
        static constexpr uint16_t setSignMask         = 0b0111111111111111;
        static constexpr uint16_t dataNanMask         = 0b0111111111000000;
        static constexpr uint16_t dataInfMask         = 0b0111111110000000;
        static constexpr uint16_t dataNegativeInfMask = 0b1111111110000000;
        static constexpr uint16_t dataMantissaMask    = 0b0000000001111111;

        static constexpr uint16_t dataMaxPositiveNormalMask = 0b0111111101111111;
        static constexpr uint16_t dataMaxNegativeNormalMask = 0b1111111101111111;

        static constexpr uint16_t dataMaxPositiveSubNormalMask = 0b0000000001111111;
        static constexpr uint16_t dataMaxNegativeSubNormalMask = 0b1000000001111111;

        static constexpr float dataMaxNormalNumber
            = constexpr_pow(2, 127)
              * (1 + constexpr_pow(2, -1) + constexpr_pow(2, -2) + constexpr_pow(2, -3)
                 + constexpr_pow(2, -4) + constexpr_pow(2, -5) + constexpr_pow(2, -6)
                 + constexpr_pow(2, -7));
        static constexpr float dataMinSubNormalNumber
            = constexpr_pow(2, -126) * constexpr_pow(2, -7);

        static constexpr uint16_t positiveZeroMask = 0b0000000000000000;
        static constexpr uint16_t negativeZeroMask = 0b1000000000000000;
    };

#include "bf16_impl.hpp"
}
