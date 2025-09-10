/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*! \file
 *  \brief rocsparse_bfloat16.h provides struct for rocsparse_bfloat16 typedef
 */

#ifndef ROCSPARSE_BFLOAT16_H
#define ROCSPARSE_BFLOAT16_H

#include "rocsparse/rocsparse-export.h"

#if __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

/* If this is a C compiler, C++ compiler below C++11, or a host-only compiler, we only
   include a minimal definition of rocsparse_bfloat16 */

#include <stdint.h>
/*! \brief Struct to represent a 16 bit brain floating-point number. */
typedef struct
{
    uint16_t data; /**< brain float storage. */
} rocsparse_bfloat16;

#else // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <ostream>
#include <type_traits>

/*! \brief Struct to represent a 16 bit brain floating-point number. */
struct ROCSPARSE_EXPORT rocsparse_bfloat16
{
    uint16_t data;

    enum rocsparse_truncate_t
    {
        rocsparse_truncate,
        rocsparse_round_near_zero,
        rocsparse_round_near_even
    };

    __host__ __device__ rocsparse_bfloat16() = default;

    // round upper 16 bits of IEEE float to convert to bfloat16
    explicit __host__ __device__ rocsparse_bfloat16(float f)
        : data(float_to_bfloat16(f))
    {
    }

    __host__ __device__ rocsparse_bfloat16(double f)
        : data(float_to_bfloat16((float)f))
    {
    }

    __host__ __device__ rocsparse_bfloat16(int32_t i)
        : data(float_to_bfloat16((float)i))
    {
    }

    __host__ __device__ rocsparse_bfloat16(int64_t i)
        : data(float_to_bfloat16((float)i))
    {
    }

    explicit __host__ __device__ rocsparse_bfloat16(float f, rocsparse_truncate_t round)
    {
        switch(round)
        {
        case rocsparse_round_near_even:
            data = float_to_bfloat16(f);
            break;
        case rocsparse_round_near_zero:
            data = rnz_float_to_bfloat16(f);
            break;
        case rocsparse_truncate:
            data = truncate_float_to_bfloat16(f);
            break;
        }
    }

    __host__ __device__ rocsparse_bfloat16& operator=(float a)
    {

        union
        {
            float    fp32;
            uint32_t int32;
        } u = {a};

        // u.fp32 = a;
        data = static_cast<uint16_t>(u.int32 >> 16);
        return *this;
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

    explicit __host__ __device__ operator double() const
    {
        union
        {
            uint64_t int64;
            double   fp64;
        } u = {uint64_t(data) << 48};
        return u.fp64;
    }

    explicit __host__ __device__ operator int32_t() const
    {
        return static_cast<int32_t>(static_cast<float>(data));
    }

    explicit __host__ __device__ operator int64_t() const
    {
        return static_cast<int64_t>(static_cast<float>(data));
    }

    explicit __host__ __device__ operator bool() const
    {
        return data & 0x7fff;
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

    static __host__ __device__ uint16_t rnz_float_to_bfloat16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        if(~u.int32 & 0x7f800000)
        {
            // When the exponent bits are not all 1s, then the value is zero, normal,
            // or subnormal. We round the bfloat16 mantissa by adding 0x7FFF
            // This causes it to be rounded to zero when
            // the lower 16 bits are exactly 0x8000 or less. If the lower 16th bit is one and
            // and any of the lower 15 bits are one, then the addition causes a rounding upward
            u.int32 += 0x7fff; // Round to nearest, round to zero
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

    // Truncate instead of rounding, preserving SNaN
    static __host__ __device__ uint16_t truncate_float_to_bfloat16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};
        return uint16_t(u.int32 >> 16) | (!(~u.int32 & 0x7f800000) && (u.int32 & 0xffff));
    }
};

typedef struct
{
    uint16_t data;
} rocsparse_bfloat16_public;

static_assert(std::is_standard_layout<rocsparse_bfloat16>{},
              "rocsparse_bfloat16 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<rocsparse_bfloat16>{},
              "rocsparse_bfloat16 is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(rocsparse_bfloat16) == sizeof(rocsparse_bfloat16_public)
                  && offsetof(rocsparse_bfloat16, data)
                         == offsetof(rocsparse_bfloat16_public, data),
              "internal rocsparse_bfloat16 does not match public rocsparse_bfloat16");

inline std::ostream& operator<<(std::ostream& os, const rocsparse_bfloat16& bf16)
{
    return os << float(bf16);
}

inline __host__ __device__ rocsparse_bfloat16 operator+(rocsparse_bfloat16 a)
{
    return a;
}
inline __host__ __device__ rocsparse_bfloat16 operator-(rocsparse_bfloat16 a)
{
    a.data ^= 0x8000;
    return a;
}
inline __host__ __device__ rocsparse_bfloat16 operator+(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return rocsparse_bfloat16(float(a) + float(b));
}
inline __host__ __device__ rocsparse_bfloat16 operator-(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return rocsparse_bfloat16(float(a) - float(b));
}
inline __host__ __device__ rocsparse_bfloat16 operator*(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return rocsparse_bfloat16(float(a) * float(b));
}
inline __host__ __device__ float operator*(float a, rocsparse_bfloat16 b)
{
    return a * float(b);
}
inline __host__ __device__ rocsparse_bfloat16 operator/(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return rocsparse_bfloat16(float(a) / float(b));
}
inline __host__ __device__ bool operator<(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return float(a) < float(b);
}
inline __host__ __device__ bool operator==(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return float(a) == float(b);
}
inline __host__ __device__ bool operator>(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return b < a;
}
inline __host__ __device__ bool operator<=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return !(a > b);
}
inline __host__ __device__ bool operator!=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return !(a == b);
}
inline __host__ __device__ bool operator!=(rocsparse_bfloat16 a, int b)
{
    return !(a == rocsparse_bfloat16(b));
}
inline __host__ __device__ bool operator>=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return !(a < b);
}
inline __host__ __device__ rocsparse_bfloat16 operator+=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return a = a + b;
}
inline __host__ __device__ rocsparse_bfloat16 operator+=(rocsparse_bfloat16 a, float b)
{
    return a = rocsparse_bfloat16(float(a) + b);
}
inline __host__ __device__ float operator+=(float a, rocsparse_bfloat16 b)
{
    return a = a + float(b);
}
inline __host__ __device__ rocsparse_bfloat16 operator-=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return a = a - b;
}
inline __host__ __device__ rocsparse_bfloat16 operator-=(rocsparse_bfloat16 a, float b)
{
    return a = rocsparse_bfloat16(float(a) - b);
}
inline __host__ __device__ float operator-=(float a, rocsparse_bfloat16 b)
{
    return a = a - float(b);
}
inline __host__ __device__ rocsparse_bfloat16 operator*=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return a = a * b;
}
inline __host__ __device__ float operator*=(rocsparse_bfloat16 a, float b)
{
    return a = float(a) * b;
}
inline __host__ __device__ float operator*=(float a, rocsparse_bfloat16 b)
{
    return a = a * float(b);
}
inline __host__ __device__ rocsparse_bfloat16 operator/=(rocsparse_bfloat16 a, rocsparse_bfloat16 b)
{
    return a = a / b;
}
inline __host__ __device__ rocsparse_bfloat16 operator/=(rocsparse_bfloat16 a, float b)
{
    return a = rocsparse_bfloat16(float(a) / b);
}
inline __host__ __device__ float operator/=(float a, rocsparse_bfloat16 b)
{
    return a = a / float(b);
}
inline __host__ __device__ rocsparse_bfloat16 operator++(rocsparse_bfloat16 a)
{
    return a += rocsparse_bfloat16(1.0f);
}
inline __host__ __device__ rocsparse_bfloat16 operator--(rocsparse_bfloat16 a)
{
    return a -= rocsparse_bfloat16(1.0f);
}
inline __host__ __device__ rocsparse_bfloat16 operator++(rocsparse_bfloat16 a, int)
{
    rocsparse_bfloat16 orig = a;
    ++a;
    return orig;
}
inline __host__ __device__ rocsparse_bfloat16 operator--(rocsparse_bfloat16 a, int)
{
    rocsparse_bfloat16 orig = a;
    --a;
    return orig;
}

namespace std
{
    constexpr __host__ __device__ bool isinf(rocsparse_bfloat16 a)
    {
        return !(~a.data & 0x7f80) && !(a.data & 0x7f);
    }
    constexpr __host__ __device__ bool isnan(rocsparse_bfloat16 a)
    {
        return !(~a.data & 0x7f80) && +(a.data & 0x7f);
    }
    constexpr __host__ __device__ bool iszero(rocsparse_bfloat16 a)
    {
        return !(a.data & 0x7fff);
    }
    inline __host__ __device__ rocsparse_bfloat16 abs(rocsparse_bfloat16 a)
    {
        return a < rocsparse_bfloat16(0) ? -a : a;
    }
    inline rocsparse_bfloat16 sin(rocsparse_bfloat16 a)
    {
        return rocsparse_bfloat16(sinf(float(a)));
    }
    inline rocsparse_bfloat16 cos(rocsparse_bfloat16 a)
    {
        return rocsparse_bfloat16(cosf(float(a)));
    }
    __device__ __host__ constexpr rocsparse_bfloat16 real(const rocsparse_bfloat16& a)
    {
        return a;
    }
}

#endif // __cplusplus < 201103L || (!defined(__HCC__) && !defined(__HIPCC__))

#endif // ROCSPARSE_BFLOAT16_H
