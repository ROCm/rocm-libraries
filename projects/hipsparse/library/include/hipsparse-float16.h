/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once

#if __cplusplus < 201103L

// If this is a C or C++ compiler below C++11,
// we only include a minimal definition of hipsparseFloat16
typedef struct hipsparseFloat16
{
    uint16_t data;
} hipsparseFloat16;

#else

#include <cmath>
#include <cstdint>
#include <iostream>
#include <ostream>

class hipsparseFloat16
{
public:
    uint16_t data;

    // Convert IEEE float16 to IEEE float32
    static float float16_to_float(hipsparseFloat16 val)
    {
        uint16_t h = val.data;

        // Extract components
        uint32_t sign     = (h >> 15) & 0x1;
        uint32_t exponent = (h >> 10) & 0x1f;
        uint32_t mantissa = h & 0x3ff;

        uint32_t f;

        if(exponent == 0)
        {
            if(mantissa == 0)
            {
                // Zero (preserve sign)
                f = sign << 31;
            }
            else
            {
                // Subnormal float16 -> normalized float32
                exponent = 1;
                while((mantissa & 0x400) == 0)
                {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3ff;
                f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
            }
        }
        else if(exponent == 31)
        {
            // Inf or NaN
            f = (sign << 31) | 0x7f800000 | (mantissa << 13);
        }
        else
        {
            // Normalized number
            f = (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
        }

        union
        {
            uint32_t int32;
            float    fp32;
        } u = {f};
        return u.fp32;
    }

    // Convert IEEE float32 to IEEE float16
    static uint16_t float_to_float16(float f)
    {
        union
        {
            float    fp32;
            uint32_t int32;
        } u = {f};

        uint32_t x = u.int32;

        // Extract components
        uint32_t sign     = (x >> 31) & 0x1;
        int32_t  exponent = ((x >> 23) & 0xff) - 127 + 15;
        uint32_t mantissa = x & 0x7fffff;

        uint16_t h;

        if(exponent <= 0)
        {
            if(exponent < -10)
            {
                // Too small, return signed zero
                h = static_cast<uint16_t>(sign << 15);
            }
            else
            {
                // Subnormal
                mantissa |= 0x800000; // Add implicit leading 1
                int shift = 14 - exponent;
                // Round to nearest, round to even
                uint32_t round_bit  = 1u << (shift - 1);
                uint32_t round_mask = (1u << shift) - 1;
                uint32_t m_shifted  = mantissa >> shift;
                if((mantissa & round_mask) > round_bit
                   || ((mantissa & round_mask) == round_bit && (m_shifted & 1)))
                {
                    m_shifted++;
                }
                h = static_cast<uint16_t>((sign << 15) | m_shifted);
            }
        }
        else if(exponent >= 31)
        {
            // Overflow to infinity or preserve NaN
            if(exponent == 31 && mantissa != 0)
            {
                // NaN - preserve signaling bit
                h = static_cast<uint16_t>((sign << 15) | 0x7c00 | (mantissa >> 13));
                if((h & 0x3ff) == 0)
                {
                    h |= 1; // Ensure NaN mantissa is non-zero
                }
            }
            else
            {
                // Infinity
                h = static_cast<uint16_t>((sign << 15) | 0x7c00);
            }
        }
        else
        {
            // Normalized number - round to nearest, round to even
            uint32_t round_bit = 0x1000; // bit 12
            uint32_t m         = mantissa + round_bit;
            if(m & 0x800000)
            {
                // Mantissa overflow
                m = 0;
                exponent++;
                if(exponent >= 31)
                {
                    // Overflow to infinity
                    h = static_cast<uint16_t>((sign << 15) | 0x7c00);
                    return h;
                }
            }
            h = static_cast<uint16_t>((sign << 15) | (exponent << 10) | (m >> 13));
        }

        return h;
    }

    hipsparseFloat16() = default;

    // Convert float32 to float16
    explicit hipsparseFloat16(float f)
        : data(float_to_float16(f))
    {
    }

    hipsparseFloat16 operator=(float a)
    {
        return hipsparseFloat16(a);
    }

    // Convert float16 to float32
    operator float() const
    {
        return float16_to_float(*this);
    }

    explicit operator bool() const
    {
        return data & 0x7fff;
    }
};

inline std::ostream& operator<<(std::ostream& os, const hipsparseFloat16& f16)
{
    return os << float(f16);
}

typedef struct
{
    uint16_t data;
} hipsparseFloat16_public;

static_assert(std::is_standard_layout<hipsparseFloat16>{},
              "hipsparseFloat16 is not a standard layout type, and thus is "
              "incompatible with C.");

static_assert(std::is_trivial<hipsparseFloat16>{},
              "hipsparseFloat16 is not a trivial type, and thus is "
              "incompatible with C.");

static_assert(sizeof(hipsparseFloat16) == sizeof(hipsparseFloat16_public)
                  && offsetof(hipsparseFloat16, data) == offsetof(hipsparseFloat16_public, data),
              "internal hipsparseFloat16 does not match public hipsparseFloat16_public");

inline hipsparseFloat16 operator+=(hipsparseFloat16 a, hipsparseFloat16 b)
{
    return a = a + b;
}
inline hipsparseFloat16 operator+=(hipsparseFloat16 a, float b)
{
    return a = hipsparseFloat16(float(a) + b);
}
inline float operator+=(float a, hipsparseFloat16 b)
{
    return a = a + float(b);
}
inline hipsparseFloat16 operator-=(hipsparseFloat16 a, hipsparseFloat16 b)
{
    return a = a - b;
}
inline hipsparseFloat16 operator-=(hipsparseFloat16 a, float b)
{
    return a = hipsparseFloat16(float(a) - b);
}
inline float operator-=(float a, hipsparseFloat16 b)
{
    return a = a - float(b);
}
inline hipsparseFloat16 operator*=(hipsparseFloat16 a, hipsparseFloat16 b)
{
    return a = a * b;
}
inline float operator*=(hipsparseFloat16 a, float b)
{
    return a = float(a) * b;
}
inline float operator*=(float a, hipsparseFloat16 b)
{
    return a = a * float(b);
}
inline hipsparseFloat16 operator/=(hipsparseFloat16 a, hipsparseFloat16 b)
{
    return a = a / b;
}
inline hipsparseFloat16 operator/=(hipsparseFloat16 a, float b)
{
    return a = hipsparseFloat16(float(a) / b);
}
inline float operator/=(float a, hipsparseFloat16 b)
{
    return a = a / float(b);
}

#endif
