// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <array>

namespace DGen
{
    struct OCP_E4M3_MXFP8_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 4;
        static constexpr uint mantissaBits = 3;
        static constexpr uint bias         = 7;
        static constexpr uint srShift      = 12;

        static constexpr int unBiasedEMin = -6;
        static constexpr int unBiasedEMax = 8;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 15;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = true;
        static constexpr bool hasZero = true;
    };

    struct ocp_e4m3_mxfp8
    {
        static constexpr OCP_E4M3_MXFP8_DATA        dataInfo{};
        static constexpr ScaleInfo<ScaleType::E8M0> scaleInfo{};

        static constexpr uint8_t oneMask     = 0b00111000;
        static constexpr uint8_t signBitMask = 0b10000000;

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b01111110;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b11111110;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b00000111;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b10000111;

        static constexpr uint8_t dataSubNormalOneMask = 0b00000010;

        static constexpr float dataMaxNormalNumber    = 448;
        static constexpr float dataMinSubnormalNumber = 0.0019531250;

        static constexpr uint8_t positiveZeroMask = 0b00000000;
        static constexpr uint8_t negativeZeroMask = 0b10000000;

        static constexpr uint8_t scaleNanMask = 0b11111111;

        static constexpr float dataMaxRoundedRange = 480; // 2^8 * 1.875

        static constexpr std::array<uint8_t, 2> dataNaNMasks{0b01111111, 0b11111111};
    };

#include "ocp_e4m3_mxfp8_impl.hpp"
}
