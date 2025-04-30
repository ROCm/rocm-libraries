/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

/*!\file
 * \brief rocblas_xfloat32.h provides struct for rocblas_xfloat32 typedef
 */

#ifndef ROCBLAS_XFLOAT32_H
#define ROCBLAS_XFLOAT32_H

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

// If this is a C compiler, C++ compiler below C++11, or a host-only compiler, we only
// include a minimal definition of rocblas_xfloat32

#include <stdint.h>
typedef struct
{
    float data;
} rocblas_xfloat32;

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <ostream>
#include <type_traits>

struct ROCBLAS_EXPORT rocblas_xfloat32
{
    float data;

    enum round_t
    {
        round_up
    };

    __host__ __device__ rocblas_xfloat32() = default;

    // round upper 19 bits of IEEE float to convert to xfloat32
    explicit __host__ __device__ rocblas_xfloat32(float f, round_t)
        : data(float_to_xfloat32(f))
    {
    }

    explicit __host__ __device__ rocblas_xfloat32(float f)
        : data(truncate_float_to_xfloat32(f))
    {
    }

    // zero extend lower 13 bits of xfloat32 to convert to IEEE float
    __host__ __device__ operator float() const
    {
        return data;
    }

    explicit __host__ __device__ operator bool() const
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {data};
        return u.int32 & 0x7fffe000;
    }

private:
    static __host__ __device__ float float_to_xfloat32(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the xfloat32 mantissa up by adding 0xFFF, plus
            // 1 if the least significant bit of the xfloat32 mantissa is 1 (odd).
            // This causes the xfloat32's mantissa to be incremented by 1 if the 13
            // least significant bits of the float mantissa are greater than 0x1000,
            // or if they are equal to 0x1000 and the least significant bit of the
            // xfloat32 mantissa is 1 (odd). This causes it to be rounded to even when
            // the lower 13 bits are exactly 0x1000. If the xfloat32 mantissa already
            // has the value 0x3ff, then incrementing it causes it to become 0x00 and
            // the exponent is incremented by one, which is the next higher FP value
            // to the unrounded xfloat32 value. When the xfloat32 value is subnormal
            // with an exponent of 0x00 and a mantissa of 0x3FF, it may be rounded up
            // to a normal value with an exponent of 0x01 and a mantissa of 0x00.
            // When the xfloat32 value has an exponent of 0xFE and a mantissa of 0x3FF,
            // incrementing it causes it to become an exponent of 0xFF and a mantissa
            // of 0x00, which is Inf, the next higher value to the unrounded value.

            u.int32 += 0xfff + ((u.int32 >> 13) & 1); // Round to nearest, round to even
        }
        else if(u.int32 & 0x1fff)
        {
            // When all of the exponent bits are 1, the value is Inf or NaN.
            // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
            // mantissa bit. Quiet NaN is indicated by the most significant mantissa
            // bit being 1. Signaling NaN is indicated by the most significant
            // mantissa bit being 0 but some other bit(s) being 1. If any of the
            // lower 13 bits of the mantissa are 1, we set the least significant bit
            // of the xfloat32 mantissa, in order to preserve signaling NaN in case
            // the xfloat32's mantissa bits are all 0.
            u.int32 |= 0x2000; // Preserve signaling NaN
        }

        u.int32 &= 0xffffe000;
        return u.fp32;
    }

    // Truncate instead of rounding
    static __host__ __device__ float truncate_float_to_xfloat32(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};

        u.int32 = u.int32 & 0xffffe000;
        return u.fp32;
    }
};

typedef struct
{
    float data;
} rocblas_xfloat32_public;

static_assert(std::is_standard_layout<rocblas_xfloat32>{},
              "rocblas_xfloat32 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<rocblas_xfloat32>{},
              "rocblas_xfloat32 is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(rocblas_xfloat32) == sizeof(rocblas_xfloat32_public)
                  && offsetof(rocblas_xfloat32, data) == offsetof(rocblas_xfloat32_public, data),
              "internal rocblas_xfloat32 does not match public rocblas_xfloat32");

inline std::ostream& operator<<(std::ostream& os, const rocblas_xfloat32& xf32)
{
    return os << float(xf32);
}
inline __host__ __device__ rocblas_xfloat32 operator+(rocblas_xfloat32 a)
{
    return a;
}
inline __host__ __device__ rocblas_xfloat32 operator-(rocblas_xfloat32 a)
{
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {a.data};
    u.int32 ^= 0x80000000;
    return rocblas_xfloat32(u.fp32);
}
inline __host__ __device__ rocblas_xfloat32 operator+(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return rocblas_xfloat32(float(a) + float(b));
}
inline __host__ __device__ rocblas_xfloat32 operator-(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return rocblas_xfloat32(float(a) - float(b));
}
inline __host__ __device__ rocblas_xfloat32 operator*(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return rocblas_xfloat32(float(a) * float(b));
}
inline __host__ __device__ rocblas_xfloat32 operator/(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return rocblas_xfloat32(float(a) / float(b));
}
inline __host__ __device__ bool operator<(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return float(a) < float(b);
}
inline __host__ __device__ bool operator==(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return float(a) == float(b);
}
inline __host__ __device__ bool operator>(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return b < a;
}
inline __host__ __device__ bool operator<=(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return !(a > b);
}
inline __host__ __device__ bool operator!=(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return !(a == b);
}
inline __host__ __device__ bool operator>=(rocblas_xfloat32 a, rocblas_xfloat32 b)
{
    return !(a < b);
}
inline __host__ __device__ rocblas_xfloat32& operator+=(rocblas_xfloat32& a, rocblas_xfloat32 b)
{
    return a = a + b;
}
inline __host__ __device__ rocblas_xfloat32& operator-=(rocblas_xfloat32& a, rocblas_xfloat32 b)
{
    return a = a - b;
}
inline __host__ __device__ rocblas_xfloat32& operator*=(rocblas_xfloat32& a, rocblas_xfloat32 b)
{
    return a = a * b;
}
inline __host__ __device__ rocblas_xfloat32& operator/=(rocblas_xfloat32& a, rocblas_xfloat32 b)
{
    return a = a / b;
}
inline __host__ __device__ rocblas_xfloat32& operator++(rocblas_xfloat32& a)
{
    return a += rocblas_xfloat32(1.0f);
}
inline __host__ __device__ rocblas_xfloat32& operator--(rocblas_xfloat32& a)
{
    return a -= rocblas_xfloat32(1.0f);
}
inline __host__ __device__ rocblas_xfloat32 operator++(rocblas_xfloat32& a, int)
{
    rocblas_xfloat32 orig = a;
    ++a;
    return orig;
}
inline __host__ __device__ rocblas_xfloat32 operator--(rocblas_xfloat32& a, int)
{
    rocblas_xfloat32 orig = a;
    --a;
    return orig;
}

namespace std
{
    constexpr __host__ __device__ bool isinf(rocblas_xfloat32 a)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a.data};
        return !(~u.int32 & 0x7f800000) && !(u.int32 & 0x7fe000);
    }
    constexpr __host__ __device__ bool isnan(rocblas_xfloat32 a)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a.data};
        return !(~u.int32 & 0x7f800000) && +(u.int32 & 0x7fe000);
    }
    constexpr __host__ __device__ bool iszero(rocblas_xfloat32 a)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a.data};
        return (u.fp32 == 0.0f);
    }
    inline rocblas_xfloat32 sin(rocblas_xfloat32 a)
    {
        return rocblas_xfloat32(sinf(float(a)));
    }
    inline rocblas_xfloat32 cos(rocblas_xfloat32 a)
    {
        return rocblas_xfloat32(cosf(float(a)));
    }
    __device__ __host__ constexpr rocblas_xfloat32 real(const rocblas_xfloat32& a)
    {
        return a;
    }
}

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // ROCBLAS_XFLOAT32_H
