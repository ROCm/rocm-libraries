// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <bitset>
#include <cmath>
#include <iostream>
#include <limits>

inline float convert(int num, int mBits, int eBits, int eBias)
{
    if(num == 0b0)
        return 0.0f;

    int expMask = eBits == 8 ? 0xff : 0b11111;

    int expVal = ((expMask << mBits) & num) >> mBits;

    float sign = (num >> (mBits + eBits)) ? -1 : 1;

    float mantissa = expVal == 0 ? 0 : 1;

    for(int i = 0; i < mBits; i++)
        mantissa += std::pow(2, -(i + 1)) * ((num >> ((mBits - 1) - i)) & 0b1);

    if(expVal == expMask)
    {
        if(mantissa != 1) //since mantissa will be at least one
            return std::numeric_limits<float>::quiet_NaN();
        else
            return sign < 0 ? -std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::infinity();
    }

    expVal = expVal == 0 ? 1 - eBias : expVal - eBias;

    float exp = std::pow(2, expVal);

    return sign * exp * mantissa;
}
