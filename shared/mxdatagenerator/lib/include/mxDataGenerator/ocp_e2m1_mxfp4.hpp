// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    struct ocp_e2m1_mxfp4_base
    {
        static constexpr OCP_E2M1_MXFP4_DATA dataInfo{};

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

    struct ocp_e2m1_mxfp4 : ocp_e2m1_mxfp4_base
    {
        static constexpr ScaleInfo<ScaleType::E8M0> scaleInfo{};
    };

    struct ocp_e2m1_mxfp4_e5m3 : ocp_e2m1_mxfp4_base
    {
        static constexpr ScaleInfo<ScaleType::E5M3> scaleInfo{};
    };

    struct ocp_e2m1_mxfp4_e4m3 : ocp_e2m1_mxfp4_base
    {
        static constexpr ScaleInfo<ScaleType::E4M3> scaleInfo{};
    };

#include "ocp_e2m1_mxfp4_impl.hpp"
}
