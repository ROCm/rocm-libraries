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

namespace DGen
{
    struct OCP_E2M1_MXFP4_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 2;
        static constexpr uint mantissaBits = 1;
        static constexpr uint bias         = 1;
        static constexpr uint srShift      = 10;

        static constexpr int unBiasedEMin = 0;
        static constexpr int unBiasedEMax = 2;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 3;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = false;
        static constexpr bool hasZero = true;
    };

    struct ocp_e2m1_mxfp4
    {
        static constexpr OCP_E2M1_MXFP4_DATA dataInfo{};
        static constexpr E8M0_SCALE_INFO     scaleInfo{};

        static constexpr uint8_t oneMask     = 0b0010;
        static constexpr uint8_t setSignMask = 0b0111;

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b0111;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b1111;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b0001;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b1001;

        static constexpr uint8_t dataSubNormalOneMask = 0b0001;

        static constexpr float dataMaxNormalNumber    = 6;
        static constexpr float dataMinSubNormalNumber = 0.5;

        static constexpr uint8_t positiveZeroMask = 0b0000;
        static constexpr uint8_t negativeZeroMask = 0b1000;
    };

#include "ocp_e2m1_mxfp4_impl.hpp"
}
