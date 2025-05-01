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

#include "DataTypes_F6_Utils.hpp"
#include <cassert>
#include <cinttypes>
#include <cmath>

#include <iostream>

#define ROCROLLER_USE_FP6

#ifndef __BYTE_ORDER__
#define __BYTE_ORDER__ __ORDER_LITTLE_ENDIAN__
#endif

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     * @{
     */
    struct FP6
    {
        constexpr FP6()
            : data(F6_ZERO_VALUE)
        {
        }

        FP6(FP6 const& other) = default;

        template <typename T>
        requires(!std::is_same_v<T, FP6> && std::is_convertible_v<T, float>) explicit FP6(
            T const& value)
            : data(float_to_fp6(static_cast<float>(value)).data)
        {
        }

        template <typename T>
        requires(std::is_convertible_v<T, float>) void operator=(T const& value)
        {
            data = float_to_fp6(static_cast<float>(value)).data;
        }

        explicit operator float() const
        {
            return fp6_to_float(*this);
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

    inline std::ostream& operator<<(std::ostream& os, const FP6& obj)
    {
        os << static_cast<float>(obj);
        return os;
    }

    inline FP6 operator+(FP6 a, FP6 b)
    {
        return static_cast<FP6>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP6 operator+(int a, FP6 b)
    {
        return static_cast<FP6>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP6 operator+(FP6 a, int b)
    {
        return static_cast<FP6>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP6 operator-(FP6 a, FP6 b)
    {
        return static_cast<FP6>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline FP6 operator*(FP6 a, FP6 b)
    {
        return static_cast<FP6>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline FP6 operator/(FP6 a, FP6 b)
    {
        return static_cast<FP6>(static_cast<float>(a) / static_cast<float>(b));
    }

    inline FP6 operator-(FP6 const& a)
    {
        return static_cast<FP6>(-static_cast<float>(a));
    }

    inline bool operator!(FP6 const& a)
    {
        return !static_cast<float>(a);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline auto operator<=>(FP6 const& a, T const& b)
    {
        return static_cast<float>(a) <=> static_cast<float>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline bool operator==(FP6 const& a, T const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline bool operator==(FP6 const& a, FP6 const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline FP6& operator+=(FP6& a, FP6 b)
    {
        a = a + b;
        return a;
    }
    inline FP6& operator-=(FP6& a, FP6 b)
    {
        a = a - b;
        return a;
    }
    inline FP6& operator*=(FP6& a, FP6 b)
    {
        a = a * b;
        return a;
    }
    inline FP6& operator/=(FP6& a, FP6 b)
    {
        a = a / b;
        return a;
    }

    inline FP6 operator++(FP6& a)
    {
        a += FP6(1);
        return a;
    }
    inline FP6 operator++(FP6& a, int)
    {
        FP6 original_value = a;
        ++a;
        return original_value;
    }

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    inline bool isinf(const rocRoller::FP6& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const rocRoller::FP6& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const rocRoller::FP6& a)
    {
        return (a.data & 0x1f) == 0x0;
    }

    inline rocRoller::FP6 abs(const rocRoller::FP6& a)
    {
        return static_cast<rocRoller::FP6>(std::abs(static_cast<float>(a)));
    }
    inline rocRoller::FP6 sin(const rocRoller::FP6& a)
    {
        return static_cast<rocRoller::FP6>(std::sin(static_cast<float>(a)));
    }
    inline rocRoller::FP6 cos(const rocRoller::FP6& a)
    {
        return static_cast<rocRoller::FP6>(std::cos(static_cast<float>(a)));
    }
    inline rocRoller::FP6 exp2(const rocRoller::FP6& a)
    {
        return static_cast<rocRoller::FP6>(std::exp2(static_cast<float>(a)));
    }
    inline rocRoller::FP6 exp(const rocRoller::FP6& a)
    {
        return static_cast<rocRoller::FP6>(std::exp(static_cast<float>(a)));
    }

    template <>
    struct is_floating_point<rocRoller::FP6> : true_type
    {
    };

    template <>
    struct hash<rocRoller::FP6>
    {
        size_t operator()(const rocRoller::FP6& a) const
        {
            return hash<uint8_t>()(a.data);
        }
    };
} // namespace std
