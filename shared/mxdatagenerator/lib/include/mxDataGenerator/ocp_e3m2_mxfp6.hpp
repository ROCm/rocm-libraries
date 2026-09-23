// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace DGen
{
    struct OCP_E3M2_MXFP6_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 3;
        static constexpr uint mantissaBits = 2;
        static constexpr uint bias         = 3;
        static constexpr uint srShift      = 11;

        static constexpr int unBiasedEMin = -2;
        static constexpr int unBiasedEMax = 4;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 7;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = false;
        static constexpr bool hasZero = true;
    };

    struct ocp_e3m2_mxfp6
    {
        static constexpr OCP_E3M2_MXFP6_DATA        dataInfo{};
        static constexpr ScaleInfo<ScaleType::E8M0> scaleInfo{};

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b011111;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b111111;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b000011;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b100011;

        static constexpr uint8_t dataSubNormalOneMask = 0b000010;

        static constexpr uint8_t dataMaxNormalNumber    = 28;
        static constexpr float   dataMinSubNormalNumber = 0.0625;

        static constexpr uint8_t positiveZeroMask = 0b000000;
        static constexpr uint8_t negativeZeroMask = 0b100000;
    };

#include "ocp_e3m2_mxfp6_impl.hpp"
}
