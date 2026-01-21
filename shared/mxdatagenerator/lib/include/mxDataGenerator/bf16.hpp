// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
