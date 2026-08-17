// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
