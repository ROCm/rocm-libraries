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
