/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

/*!\file
 * \brief tensile_bfloat16.h provides struct for tensile_bfloat16 typedef
 */

#pragma once
#ifndef _TENSILE_BFLOAT16_H_
#define _TENSILE_BFLOAT16_H_

// If this is a C compiler, C++ compiler below C++11, or a host-only compiler,
// we only include a minimal definition of tensile_bfloat16
#if __cplusplus < 201103L || !defined(__HIPCC__)

#include <stdint.h>
typedef struct
{
    uint16_t data;
} tensile_bfloat16;

#else // __cplusplus < 201103L || !defined(__HIPCC__)

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <ostream>
#include <type_traits>

struct tensile_bfloat16
{
    uint16_t data;

    // Don't initialize `data` in purpose so that it could be used with
    // `__shared__`, which forbid any initializer.
    __host__ __device__ tensile_bfloat16() {}

    // round upper 16 bits of IEEE float to convert to bfloat16
    explicit __host__ __device__ tensile_bfloat16(float f)
        : data(float_to_bfloat16(f))
    {
    }

    // zero extend lower 16 bits of bfloat16 to convert to IEEE float
    __host__ __device__ operator float() const
    {
        union
        {
            uint32_t int32;
            float    fp32;
        } u = {uint32_t(data) << 16};
        return u.fp32;
    }

private:
    static __host__ __device__ uint16_t float_to_bfloat16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the bfloat16 mantissa up by adding 0x7FFF, plus
            // 1 if the least significant bit of the bfloat16 mantissa is 1 (odd).
            // This causes the bfloat16's mantissa to be incremented by 1 if the 16
            // least significant bits of the float mantissa are greater than 0x8000,
            // or if they are equal to 0x8000 and the least significant bit of the
            // bfloat16 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 16 bits are exactly 0x8000. If the bfloat16 mantissa already
            // has the value 0x7f, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded bfloat16 value. When the bfloat16 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x7F, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the bfloat16 value has an exponent of 0xFE and a mantissa of 0x7F,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.
            u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
        }
        else if(u.int32 & 0xffff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 16 bits of the mantissa are 1, we set the least significant bit
            // of the bfloat16 mantissa, in order to preserve signaling NaN in case
            // the bloat16's mantissa bits are all 0.
            u.int32 |= 0x10000; // Preserve signaling NaN
        }
        return uint16_t(u.int32 >> 16);
    }
};

static_assert(std::is_standard_layout<tensile_bfloat16>{},
              "tensile_bfloat16 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivially_copyable<tensile_bfloat16>{},
              "tensile_bfloat16 is not trivially copyable, and thus is "
              "incompatible with C.");

inline std::ostream& operator<<(std::ostream& os, const tensile_bfloat16& bf16)
{
    return os << float(bf16);
}
inline __host__ __device__ tensile_bfloat16 operator+(tensile_bfloat16 a)
{
    return a;
}
inline __host__ __device__ tensile_bfloat16 operator-(tensile_bfloat16 a)
{
    a.data ^= 0x8000;
    return a;
}
inline __host__ __device__ tensile_bfloat16 operator+(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return tensile_bfloat16(float(a) + float(b));
}
inline __host__ __device__ tensile_bfloat16 operator+(int a, tensile_bfloat16 b)
{
    return static_cast<tensile_bfloat16>(static_cast<float>(a) + static_cast<float>(b));
}
inline __host__ __device__ tensile_bfloat16 operator+(tensile_bfloat16 a, int b)
{
    return static_cast<tensile_bfloat16>(static_cast<float>(a) + static_cast<float>(b));
}
inline __host__ __device__ tensile_bfloat16 operator-(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return tensile_bfloat16(float(a) - float(b));
}
inline __host__ __device__ tensile_bfloat16 operator*(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return tensile_bfloat16(float(a) * float(b));
}
inline __host__ __device__ tensile_bfloat16 operator/(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return tensile_bfloat16(float(a) / float(b));
}
inline __host__ __device__ bool operator<(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return float(a) < float(b);
}
inline __host__ __device__ bool operator==(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return float(a) == float(b);
}
inline __host__ __device__ bool operator>(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return b < a;
}
inline __host__ __device__ bool operator<=(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return !(a > b);
}
inline __host__ __device__ bool operator!=(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return !(a == b);
}
inline __host__ __device__ bool operator>=(tensile_bfloat16 a, tensile_bfloat16 b)
{
    return !(a < b);
}
inline __host__ __device__ tensile_bfloat16& operator+=(tensile_bfloat16& a, tensile_bfloat16 b)
{
    return a = a + b;
}
inline __host__ __device__ tensile_bfloat16& operator-=(tensile_bfloat16& a, tensile_bfloat16 b)
{
    return a = a - b;
}
inline __host__ __device__ tensile_bfloat16& operator*=(tensile_bfloat16& a, tensile_bfloat16 b)
{
    return a = a * b;
}
inline __host__ __device__ tensile_bfloat16& operator/=(tensile_bfloat16& a, tensile_bfloat16 b)
{
    return a = a / b;
}
inline __host__ __device__ tensile_bfloat16& operator++(tensile_bfloat16& a)
{
    return a += tensile_bfloat16(1.0f);
}
inline __host__ __device__ tensile_bfloat16& operator--(tensile_bfloat16& a)
{
    return a -= tensile_bfloat16(1.0f);
}
inline __host__ __device__ tensile_bfloat16 operator++(tensile_bfloat16& a, int)
{
    tensile_bfloat16 orig = a;
    ++a;
    return orig;
}
inline __host__ __device__ tensile_bfloat16 operator--(tensile_bfloat16& a, int)
{
    tensile_bfloat16 orig = a;
    --a;
    return orig;
}
inline __host__ __device__ bool isinf(tensile_bfloat16 a)
{
    return !(~a.data & 0x7f80) && !(a.data & 0x7f);
}
inline __host__ __device__ bool isnan(tensile_bfloat16 a)
{
    return !(~a.data & 0x7f80) && +(a.data & 0x7f);
}
inline __host__ __device__ bool iszero(tensile_bfloat16 a)
{
    return !(a.data & 0x7fff);
}
inline tensile_bfloat16 sin(tensile_bfloat16 a)
{
    return tensile_bfloat16(sinf(float(a)));
}
inline tensile_bfloat16 cos(tensile_bfloat16 a)
{
    return tensile_bfloat16(cosf(float(a)));
}

#endif // __cplusplus < 201103L || !defined(__HIPCC__)

#endif // _TENSILE_BFLOAT16_H_
