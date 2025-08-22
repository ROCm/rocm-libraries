/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cinttypes>
#include <cmath>
#include <iostream>

#define TENSILE_USE_BF16

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define BFLOAT16_Q_NAN_VALUE 0xFFC1

namespace TensileLite
{
    /**
 * \ingroup DataTypes
 * @{
 */

    struct BFloat16
    {
        BFloat16()
            : data(BFLOAT16_ZERO_VALUE)
        {
        }

        BFloat16(BFloat16 const& other) = default;

        template <typename T,
                  typename = typename std::enable_if<(!std::is_same<T, BFloat16>::value)
                                                     && std::is_convertible<T, float>::value>::type>
        explicit BFloat16(T const& value)
            : data(float_to_bfloat16(static_cast<float>(value)).data)
        {
        }

        explicit operator _Float16() const
        {
            return _Float16(bfloat16_to_float(*this));
        }

        explicit operator float() const
        {
            return bfloat16_to_float(*this);
        }

        explicit operator double() const
        {
            return static_cast<double>(float(*this));
        }

        explicit operator int() const
        {
            return static_cast<int>(float(*this));
        }

        explicit operator uint32_t() const
        {
            return static_cast<uint32_t>(float(*this));
        }

        explicit operator int8_t() const
        {
            return static_cast<int8_t>(float(*this));
        }

        uint16_t data;

    private:
        static const int16_t BFLOAT16_ZERO_VALUE = 0x00;

        // zero extend lower 16 bits of bfloat16 to convert to IEEE float
        static float bfloat16_to_float(const BFloat16 v)
        {
            union
            {
                float    fp32 = 0;
                uint16_t q[2];
            };

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            q[0] = v.data;
#else
            q[1]      = v.data;
#endif
            return fp32;
        }

        // truncate lower 16 bits of IEEE float to convert to bfloat16
        static BFloat16 float_to_bfloat16_truncate(const float v)
        {
            BFloat16 bf16;
            if(std::isnan(v))
            {
                bf16.data = BFLOAT16_Q_NAN_VALUE;
                return bf16;
            }
            union
            {
                float    fp32;
                uint16_t p[2];
            };
            fp32 = v;

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            bf16.data = p[0];
#else
            bf16.data = p[1];
#endif
            return bf16;
        }

        static BFloat16 float_to_bfloat16(const float v)
        {
            BFloat16 bf16;
            if(std::isnan(v))
            {
                bf16.data = BFLOAT16_Q_NAN_VALUE;
                return bf16;
            }
            union
            {
                float    fp32;
                uint32_t int32;
            };
            fp32 = v;

            if(~int32 & 0x7f800000)
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
                int32 += 0x7fff + ((int32 >> 16) & 1); // Round to nearest, round to even
            }
            else if(int32 & 0xffff)
            {
                // When all of the exponent bits are 1, the value is Inf or NaN.
                // Inf is indicated by a zero mantissa. NaN is indicated by any nonzero
                // mantissa bit. Quiet NaN is indicated by the most significant mantissa
                // bit being 1. Signaling NaN is indicated by the most significant
                // mantissa bit being 0 but some other bit(s) being 1. If any of the
                // lower 16 bits of the mantissa are 1, we set the least significant bit
                // of the bfloat16 mantissa, in order to preserve signaling NaN in case
                // the bloat16's mantissa bits are all 0.
                int32 |= 0x10000; // Preserve signaling NaN
            }
            bf16.data = int32 >> 16;
            return bf16;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const BFloat16& bf16)
    {
        os << static_cast<float>(bf16);
        return os;
    }

    inline BFloat16 operator+(BFloat16 a, BFloat16 b)
    {
        return static_cast<BFloat16>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline BFloat16 operator+(int a, BFloat16 b)
    {
        return static_cast<BFloat16>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline BFloat16 operator+(BFloat16 a, int b)
    {
        return static_cast<BFloat16>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline BFloat16 operator-(BFloat16 a, BFloat16 b)
    {
        return static_cast<BFloat16>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline BFloat16 operator*(BFloat16 a, BFloat16 b)
    {
        return static_cast<BFloat16>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline BFloat16 operator/(BFloat16 a, BFloat16 b)
    {
        return static_cast<BFloat16>(static_cast<float>(a) / static_cast<float>(b));
    }

    inline bool operator<(BFloat16 a, BFloat16 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(BFloat16 a, BFloat16 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(BFloat16 a, BFloat16 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(BFloat16 a, BFloat16 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(BFloat16 a, BFloat16 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(BFloat16 a, BFloat16 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    inline BFloat16& operator+=(BFloat16& a, BFloat16 b)
    {
        a = a + b;
        return a;
    }
    inline BFloat16& operator-=(BFloat16& a, BFloat16 b)
    {
        a = a - b;
        return a;
    }
    inline BFloat16& operator*=(BFloat16& a, BFloat16 b)
    {
        a = a * b;
        return a;
    }
    inline BFloat16& operator/=(BFloat16& a, BFloat16 b)
    {
        a = a / b;
        return a;
    }

    inline BFloat16 operator++(BFloat16& a)
    {
        a += BFloat16(1);
        return a;
    }
    inline BFloat16 operator++(BFloat16& a, int)
    {
        BFloat16 original_value = a;
        ++a;
        return original_value;
    }

    /**
 * @}
 */
} // namespace TensileLite

namespace std
{
    inline bool isinf(const TensileLite::BFloat16& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const TensileLite::BFloat16& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const TensileLite::BFloat16& a)
    {
        return (a.data & 0x7FFF) == 0;
    }

    inline TensileLite::BFloat16 abs(const TensileLite::BFloat16& a)
    {
        return static_cast<TensileLite::BFloat16>(std::abs(static_cast<float>(a)));
    }
    inline TensileLite::BFloat16 sin(const TensileLite::BFloat16& a)
    {
        return static_cast<TensileLite::BFloat16>(std::sin(static_cast<float>(a)));
    }
    inline TensileLite::BFloat16 cos(const TensileLite::BFloat16& a)
    {
        return static_cast<TensileLite::BFloat16>(std::cos(static_cast<float>(a)));
    }

    inline std::string to_string(const TensileLite::BFloat16& a)
    {
        return std::to_string(static_cast<float>(a));
    }
} // namespace std
