/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

#define TENSILE_USE_XF32

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

#define XFloat32_Q_NAN_VALUE 0x7FC00000

namespace Tensile
{
    /**
 * \ingroup DataTypes
 * @{
 */

    struct XFloat32
    {
        XFloat32()
            : data(XFloat32_ZERO_VALUE)
        {
        }

        XFloat32(XFloat32 const& other) = default;

        template <typename T,
                  typename = typename std::enable_if<(!std::is_same<T, XFloat32>::value)
                                                     && std::is_convertible<T, float>::value>::type>
        explicit XFloat32(T const& value)
            : data(float_to_XFloat32(static_cast<float>(value)))
        {
        }

        explicit operator float() const
        {
            return this->data;
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

        float data;

    private:
        static constexpr const float XFloat32_ZERO_VALUE = 0.0f;

        // zero extend lower 13 bits of XFloat32 to convert to IEEE float
        static float XFloat32_to_float(const XFloat32 v)
        {
            return v.data;
        }

        // truncate lower 13 bits of IEEE float to convert to XFloat32
        // not reserved the signaling NaN.
        static float float_to_XFloat32(const float v)
        {
            union
            {
                float    fp32;
                uint32_t p;
            } u = {v};

            u.p &= 0xFFFFE000; //1bit sign, 8 bits exp, 10 bits mantissa
            return u.fp32;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, const XFloat32& xf32)
    {
        os << static_cast<float>(xf32);
        return os;
    }

    inline XFloat32 operator+(XFloat32 a, XFloat32 b)
    {
        return static_cast<XFloat32>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline XFloat32 operator+(int a, XFloat32 b)
    {
        return static_cast<XFloat32>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline XFloat32 operator+(XFloat32 a, int b)
    {
        return static_cast<XFloat32>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline XFloat32 operator-(XFloat32 a, XFloat32 b)
    {
        return static_cast<XFloat32>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline XFloat32 operator*(XFloat32 a, XFloat32 b)
    {
        return static_cast<XFloat32>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline XFloat32 operator/(XFloat32 a, XFloat32 b)
    {
        return static_cast<XFloat32>(static_cast<float>(a) / static_cast<float>(b));
    }
    inline bool operator<(XFloat32 a, XFloat32 b)
    {
        return static_cast<float>(a) < static_cast<float>(b);
    }
    inline bool operator<=(XFloat32 a, XFloat32 b)
    {
        return static_cast<float>(a) <= static_cast<float>(b);
    }
    inline bool operator==(XFloat32 a, XFloat32 b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }
    inline bool operator!=(XFloat32 a, XFloat32 b)
    {
        return static_cast<float>(a) != static_cast<float>(b);
    }
    inline bool operator>(XFloat32 a, XFloat32 b)
    {
        return static_cast<float>(a) > static_cast<float>(b);
    }
    inline bool operator>=(XFloat32 a, XFloat32 b)
    {
        return static_cast<float>(a) >= static_cast<float>(b);
    }

    inline XFloat32& operator+=(XFloat32& a, XFloat32 b)
    {
        a = a + b;
        return a;
    }
    inline XFloat32& operator-=(XFloat32& a, XFloat32 b)
    {
        a = a - b;
        return a;
    }
    inline XFloat32& operator*=(XFloat32& a, XFloat32 b)
    {
        a = a * b;
        return a;
    }
    inline XFloat32& operator/=(XFloat32& a, XFloat32 b)
    {
        a = a / b;
        return a;
    }

    inline XFloat32 operator++(XFloat32& a)
    {
        a += XFloat32(1);
        return a;
    }
    inline XFloat32 operator++(XFloat32& a, int)
    {
        XFloat32 original_value = a;
        ++a;
        return original_value;
    }

    /**
 * @}
 */
} // namespace Tensile

namespace std
{
    inline bool isinf(const Tensile::XFloat32& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const Tensile::XFloat32& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const Tensile::XFloat32& a)
    {
        union
        {
            float    fp32;
            uint32_t p;
        } u = {static_cast<float>(a)};
        return !(u.p & 0x7FFFE000);
    }

    inline Tensile::XFloat32 abs(const Tensile::XFloat32& a)
    {
        return static_cast<Tensile::XFloat32>(std::abs(static_cast<float>(a)));
    }
    inline Tensile::XFloat32 sin(const Tensile::XFloat32& a)
    {
        return static_cast<Tensile::XFloat32>(std::sin(static_cast<float>(a)));
    }
    inline Tensile::XFloat32 cos(const Tensile::XFloat32& a)
    {
        return static_cast<Tensile::XFloat32>(std::cos(static_cast<float>(a)));
    }
} // namespace std
