/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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

#include <rocRoller/DataTypes/DataTypes_F8_Utils.hpp>

#define ROCROLLER_USE_BF8

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     * @{
     */
    /**
    *  @brief Floating point 8-bit type in E5M2 format
    *
    */
    struct BF8
    {
        constexpr BF8()
            : data(F8_ZERO_VALUE)
        {
        }

        BF8(BF8 const& other) = default;

        template <typename T>
        requires(!std::is_same_v<T, BF8> && std::is_convertible_v<T, float>) explicit BF8(
            T const& value)
            : data(float_to_bf8(static_cast<double>(value)).data)
        {
        }

        template <typename T>
        requires(std::is_convertible_v<T, float>) void operator=(T const& value)
        {
            data = float_to_bf8(static_cast<float>(value)).data;
        }

        explicit operator float() const
        {
            return bf8_to_float(*this);
        }

        operator double() const
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

        explicit operator uint64_t() const
        {
            return static_cast<uint64_t>(float(*this));
        }

        uint8_t data;
    };

    inline std::ostream& operator<<(std::ostream& os, const BF8& obj)
    {
        os << static_cast<float>(obj);
        return os;
    }

    inline BF8 operator+(BF8 a, BF8 b)
    {
        return static_cast<BF8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline BF8 operator+(int a, BF8 b)
    {
        return static_cast<BF8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline BF8 operator+(BF8 a, int b)
    {
        return static_cast<BF8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline BF8 operator-(BF8 a, BF8 b)
    {
        return static_cast<BF8>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline BF8 operator*(BF8 a, BF8 b)
    {
        return static_cast<BF8>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline BF8 operator/(BF8 a, BF8 b)
    {
        return static_cast<BF8>(static_cast<float>(a) / static_cast<float>(b));
    }

    inline BF8 operator-(BF8 const& a)
    {
        return static_cast<BF8>(-static_cast<float>(a));
    }

    inline bool operator!(BF8 const& a)
    {
        return !static_cast<float>(a);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline auto operator<=>(BF8 const& a, T const& b)
    {
        return static_cast<float>(a) <=> static_cast<float>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline bool operator==(BF8 const& a, T const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline bool operator==(BF8 const& a, BF8 const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline BF8& operator+=(BF8& a, BF8 b)
    {
        a = a + b;
        return a;
    }
    inline BF8& operator-=(BF8& a, BF8 b)
    {
        a = a - b;
        return a;
    }
    inline BF8& operator*=(BF8& a, BF8 b)
    {
        a = a * b;
        return a;
    }
    inline BF8& operator/=(BF8& a, BF8 b)
    {
        a = a / b;
        return a;
    }

    inline BF8 operator++(BF8& a)
    {
        a += BF8(1);
        return a;
    }
    inline BF8 operator++(BF8& a, int)
    {
        BF8 original_value = a;
        ++a;
        return original_value;
    }

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    inline bool isinf(const rocRoller::BF8& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const rocRoller::BF8& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const rocRoller::BF8& a)
    {
        return (a.data & 0x7FFF) == 0;
    }

    inline rocRoller::BF8 abs(const rocRoller::BF8& a)
    {
        return static_cast<rocRoller::BF8>(std::abs(static_cast<float>(a)));
    }
    inline rocRoller::BF8 sin(const rocRoller::BF8& a)
    {
        return static_cast<rocRoller::BF8>(std::sin(static_cast<float>(a)));
    }
    inline rocRoller::BF8 cos(const rocRoller::BF8& a)
    {
        return static_cast<rocRoller::BF8>(std::cos(static_cast<float>(a)));
    }

    template <>
    struct is_floating_point<rocRoller::BF8> : true_type
    {
    };

    template <>
    struct hash<rocRoller::BF8>
    {
        size_t operator()(const rocRoller::BF8& a) const
        {
            return hash<uint8_t>()(a.data);
        }
    };
} // namespace std
