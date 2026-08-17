// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/DataTypes/DataTypes_F8_Utils.hpp>

#include <cinttypes>
#include <cmath>
#include <iostream>

#define ROCROLLER_USE_FP8

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
    *  @brief Floating point 8-bit type in E4M3 format
    *
    */
    struct FP8
    {
        constexpr FP8()
            : data(F8_ZERO_VALUE)
        {
        }

        FP8(FP8 const& other) = default;

        template <typename T>
        requires(!std::is_same_v<T, FP8> && std::is_convertible_v<T, float>) explicit FP8(
            T const& value)
            : data(float_to_fp8(static_cast<float>(value)).data)
        {
        }

        template <typename T>
        requires(std::is_convertible_v<T, float>) void operator=(T const& value)
        {
            data = float_to_fp8(static_cast<float>(value)).data;
        }

        explicit operator float() const
        {
            return fp8_to_float(*this);
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

    inline std::ostream& operator<<(std::ostream& os, const FP8& obj)
    {
        os << static_cast<float>(obj);
        return os;
    }

    inline FP8 operator+(FP8 a, FP8 b)
    {
        return static_cast<FP8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP8 operator+(int a, FP8 b)
    {
        return static_cast<FP8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP8 operator+(FP8 a, int b)
    {
        return static_cast<FP8>(static_cast<float>(a) + static_cast<float>(b));
    }
    inline FP8 operator-(FP8 a, FP8 b)
    {
        return static_cast<FP8>(static_cast<float>(a) - static_cast<float>(b));
    }
    inline FP8 operator*(FP8 a, FP8 b)
    {
        return static_cast<FP8>(static_cast<float>(a) * static_cast<float>(b));
    }
    inline FP8 operator/(FP8 a, FP8 b)
    {
        return static_cast<FP8>(static_cast<float>(a) / static_cast<float>(b));
    }

    inline FP8 operator-(FP8 const& a)
    {
        return static_cast<FP8>(-static_cast<float>(a));
    }

    inline bool operator!(FP8 const& a)
    {
        return !static_cast<float>(a);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline auto operator<=>(FP8 const& a, T const& b)
    {
        return static_cast<float>(a) <=> static_cast<float>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, float>) inline bool operator==(FP8 const& a, T const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline bool operator==(FP8 const& a, FP8 const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline FP8& operator+=(FP8& a, FP8 b)
    {
        a = a + b;
        return a;
    }
    inline FP8& operator-=(FP8& a, FP8 b)
    {
        a = a - b;
        return a;
    }
    inline FP8& operator*=(FP8& a, FP8 b)
    {
        a = a * b;
        return a;
    }
    inline FP8& operator/=(FP8& a, FP8 b)
    {
        a = a / b;
        return a;
    }

    inline FP8 operator++(FP8& a)
    {
        a += FP8(1);
        return a;
    }
    inline FP8 operator++(FP8& a, int)
    {
        FP8 original_value = a;
        ++a;
        return original_value;
    }

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    inline bool isinf(const rocRoller::FP8& a)
    {
        return std::isinf(static_cast<float>(a));
    }
    inline bool isnan(const rocRoller::FP8& a)
    {
        return std::isnan(static_cast<float>(a));
    }
    inline bool iszero(const rocRoller::FP8& a)
    {
        return (a.data & 0x7FFF) == 0;
    }

    inline rocRoller::FP8 abs(const rocRoller::FP8& a)
    {
        return static_cast<rocRoller::FP8>(std::abs(static_cast<float>(a)));
    }
    inline rocRoller::FP8 sin(const rocRoller::FP8& a)
    {
        return static_cast<rocRoller::FP8>(std::sin(static_cast<float>(a)));
    }
    inline rocRoller::FP8 cos(const rocRoller::FP8& a)
    {
        return static_cast<rocRoller::FP8>(std::cos(static_cast<float>(a)));
    }

    template <>
    struct is_floating_point<rocRoller::FP8> : true_type
    {
    };

    template <>
    struct hash<rocRoller::FP8>
    {
        size_t operator()(const rocRoller::FP8& a) const
        {
            return hash<uint8_t>()(a.data);
        }
    };
} // namespace std
