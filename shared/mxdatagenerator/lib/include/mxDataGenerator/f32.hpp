// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "dataTypeInfo.hpp"

namespace DGen
{
    struct F32_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 8;
        static constexpr uint mantissaBits = 23;
        static constexpr uint bias         = 127;

        static constexpr int unBiasedEMin = -126;
        static constexpr int unBiasedEMax = 127;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 254;

        static constexpr bool hasInf  = true;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    /* NOTE THAT ALL SCALES WILL BE IGNORED IN F32 */
    struct f32
    {
        static constexpr F32_DATA dataInfo{};

        static constexpr uint oneMask     = 0x3f800000; // exp = 127, mantissa = 0
        static constexpr uint setSignMask = 0x7fffffff; // all 1s except for sign;
        static constexpr uint dataNanMask
            = 0x7fc00000; // exponent is all 1s and mantissa has 1 bit set
        static constexpr uint dataInfMask = 0x7f800000; // exponent is all 1s and mantissa is all 0s
        static constexpr uint dataNegativeInfMask = 0xff800000;
        static constexpr uint dataMantissaMask
            = 0x007fffff; // exponent and sign are 0s, mantissa is all 1s

        static constexpr uint dataMaxPositiveNormalMask = 0x7f7fffff;
        static constexpr uint dataMaxNegativeNormalMask = 0xff7fffff;

        static constexpr uint32_t dataMaxPositiveSubNormalMask = 0x007fffff;
        static constexpr uint     dataMaxNegativeSubNormalMask = 0x807fffff;

        static constexpr float dataMaxNormalNumber    = std::numeric_limits<float>::max();
        static constexpr float dataMinSubNormalNumber = constexpr_pow(2, -149);

        static constexpr uint positiveZeroMask = 0;
        static constexpr uint negativeZeroMask = 1 << 23;

    };

#include "f32_impl.hpp"
}
