/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#ifndef HIPSPARSE_BFLOAT16_H
#define HIPSPARSE_BFLOAT16_H

#if __cplusplus < 201103L

// If this is a C or C++ compiler below C++11,
// we only include a minimal definition of hipsparseBfloat16
typedef struct hipsparseBfloat16
{
    uint16_t data;
} hipsparseBfloat16;

#else

#include <iostream>
#include <ostream>

class hipsparseBfloat16
{
public:
    uint16_t data;

    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    static float bfloat16_to_float(hipsparseBfloat16 val)
    {
        union
        {
            uint32_t int32;
            float    fp32;
        } u = {uint32_t(val.data) << 16};
        return u.fp32;
    }

    static uint16_t float_to_bfloat16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
        }
        else if(u.int32 & 0xffff)
        {
            u.int32 |= 0x10000; // Preserve signaling NaN
        }
        return uint16_t(u.int32 >> 16);
    }

    hipsparseBfloat16() = default;

    // round upper 16 bits of IEEE float to convert to bfloat16
    explicit hipsparseBfloat16(float f)
        : data(float_to_bfloat16(f))
    {
    }

    hipsparseBfloat16 operator=(float a)
    {
        return hipsparseBfloat16(a);
    }

    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    operator float() const
    {
        union
        {
            uint32_t int32;
            float    fp32;
        } u = {uint32_t(data) << 16};
        return u.fp32;
    }

    explicit operator bool() const
    {
        return data & 0x7fff;
    }
};

inline std::ostream& operator<<(std::ostream& os, const hipsparseBfloat16& bf16)
{
    return os << float(bf16);
}

typedef struct
{
    uint16_t data;
} hipsparseBfloat16_public;

static_assert(std::is_standard_layout<hipsparseBfloat16>{},
              "hipsparseBfloat16 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<hipsparseBfloat16>{},
              "hipsparseBfloat16 is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(hipsparseBfloat16) == sizeof(hipsparseBfloat16_public)
                  && offsetof(hipsparseBfloat16, data) == offsetof(hipsparseBfloat16_public, data),
              "internal hipsparseBfloat16 does not match public hipsparseBfloat16_public");

inline hipsparseBfloat16 operator+=(hipsparseBfloat16 a, hipsparseBfloat16 b)
{
    return a = a + b;
}
inline hipsparseBfloat16 operator+=(hipsparseBfloat16 a, float b)
{
    return a = hipsparseBfloat16(float(a) + b);
}
inline float operator+=(float a, hipsparseBfloat16 b)
{
    return a = a + float(b);
}
inline hipsparseBfloat16 operator-=(hipsparseBfloat16 a, hipsparseBfloat16 b)
{
    return a = a - b;
}
inline hipsparseBfloat16 operator-=(hipsparseBfloat16 a, float b)
{
    return a = hipsparseBfloat16(float(a) - b);
}
inline float operator-=(float a, hipsparseBfloat16 b)
{
    return a = a - float(b);
}
inline hipsparseBfloat16 operator*=(hipsparseBfloat16 a, hipsparseBfloat16 b)
{
    return a = a * b;
}
inline float operator*=(hipsparseBfloat16 a, float b)
{
    return a = float(a) * b;
}
inline float operator*=(float a, hipsparseBfloat16 b)
{
    return a = a * float(b);
}
inline hipsparseBfloat16 operator/=(hipsparseBfloat16 a, hipsparseBfloat16 b)
{
    return a = a / b;
}
inline hipsparseBfloat16 operator/=(hipsparseBfloat16 a, float b)
{
    return a = hipsparseBfloat16(float(a) / b);
}
inline float operator/=(float a, hipsparseBfloat16 b)
{
    return a = a / float(b);
}

#endif

#endif // HIPSPARSE_BFLOAT16_H
