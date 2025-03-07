/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 Advanced Micro Devices, Inc.
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
j* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cmath>
#include <cstdint>
#include <rocRoller/DataTypes/DataTypes_Half.hpp>

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace rocRoller
{
#pragma once
    inline constexpr uint16_t BFLOAT16_Q_NAN_VALUE = 0xFFC1;
    inline constexpr uint16_t BFLOAT16_ZERO_VALUE  = 0x00;

    namespace DataTypes
    {
        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        template <typename T>
        T cast_from_bf16(uint16_t v)
        {
            union
            {
                T        fp32 = 0;
                uint16_t q[2];
            };

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            q[0] = v;
#else
            q[1] = v;
#endif
            return fp32;
        }

        // truncate lower 16 bits of IEEE float to convert to bfloat16
        template <typename T>
        uint16_t cast_to_bf16(T v)
        {
            uint16_t bf16;
            if(std::isnan(v))
            {
                bf16 = BFLOAT16_Q_NAN_VALUE;
                return bf16;
            }
            union
            {
                T        fp32;
                uint16_t p[2];
            };
            fp32 = v;

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            bf16 = p[0];
#else
            bf16 = p[1];
#endif
            return bf16;
        }

    }

    struct BFloat16;
    float    bf16_to_float(const BFloat16 v);
    BFloat16 float_to_bf16(const float v);
}
