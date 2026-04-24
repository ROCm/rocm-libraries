// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace DGen
{
    struct OCP_E2M3_MXFP6_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 2;
        static constexpr uint mantissaBits = 3;
        static constexpr uint bias         = 1;
        static constexpr uint srShift      = 12;

        static constexpr int unBiasedEMin = 0;
        static constexpr int unBiasedEMax = 2;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 3;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = false;
        static constexpr bool hasZero = true;
    };

    struct ocp_e2m3_mxfp6
    {
        static constexpr OCP_E2M3_MXFP6_DATA        dataInfo{};
        static constexpr ScaleInfo<ScaleType::E8M0> scaleInfo{};

        static constexpr uint8_t dataMaxPositiveNormalMask = 0b011111;
        static constexpr uint8_t dataMaxNegativeNormalMask = 0b111111;

        static constexpr uint8_t dataMaxPositiveSubNormalMask = 0b000111;
        static constexpr uint8_t dataMaxNegativeSubNormalMask = 0b100111;

        static constexpr float dataMaxNormalNumber = 7.5;
        static constexpr float dataMinNormalNumber = 1;

        static constexpr float dataMinSubNormalNumber = 0.125;

        static constexpr uint8_t positiveZeroMask = 0b000000;
        static constexpr uint8_t negativeZeroMask = 0b100000;
    };

#include "ocp_e2m3_mxfp6_impl.hpp"
}
