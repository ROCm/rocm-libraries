// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace TensileLite
{
    // Solution modes: 0=host, 1/2/3=M/N/MN vector, 5/6/7=vector plus device scalar.
    // A runtime problem requests exactly one mode: 0, 1, 2, or the scalar bit 4.
    constexpr int ScaleAlphaVecVectorMask      = 0x3;
    constexpr int ScaleAlphaVecDeviceScalarBit = 0x4;

    constexpr int scaleAlphaVectorMode(int modes)
    {
        return modes & ScaleAlphaVecVectorMask;
    }

    constexpr bool scaleAlphaSupportsDeviceScalar(int modes)
    {
        return (modes & ScaleAlphaVecDeviceScalarBit) != 0;
    }

    constexpr bool validScaleAlphaVecMode(int modes)
    {
        return modes >= 0 && modes <= 7 && modes != ScaleAlphaVecDeviceScalarBit;
    }
}
