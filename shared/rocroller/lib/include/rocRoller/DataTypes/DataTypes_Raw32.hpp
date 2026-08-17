// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     */
    struct Raw32
    {
        Raw32() = default;

        explicit Raw32(uint32_t v)
            : value(v)
        {
        }

        template <std::integral T>
        explicit operator T() const
        {
            return static_cast<T>(value);
        }

        Raw32 operator|(const Raw32& other) const
        {
            return Raw32(value | other.value);
        }

        template <std::integral T>
        T operator|(const T& other) const
        {
            return value | other;
        }

        template <std::integral T>
        T operator&(const T& other) const
        {
            return value & other;
        }

        Raw32 operator~() const
        {
            return Raw32(~value);
        }

        Raw32 operator&(const Raw32& other) const
        {
            return Raw32(value & other.value);
        }

        auto operator<=>(const Raw32&) const = default;
        bool operator==(const Raw32&) const  = default;

        uint32_t value = 0u;
    };

    inline std::ostream& operator<<(std::ostream& os, const Raw32& obj)
    {
        os << static_cast<uint32_t>(obj);
        return os;
    }

    template <std::integral T>
    T operator|(T const& lhs, rocRoller::Raw32 const& rhs)
    {
        return lhs | rhs.value;
    }

    template <std::integral T>
    T operator&(T const& lhs, rocRoller::Raw32 const& rhs)
    {
        return lhs & rhs.value;
    }

    template <std::integral T>
    Raw32 operator<<(rocRoller::Raw32 const& lhs, T const& rhs)
    {
        return Raw32(lhs.value << rhs);
    }

    template <std::integral T>
    Raw32 operator>>(rocRoller::Raw32 const& lhs, T const& rhs)
    {
        return Raw32(lhs.value >> rhs);
    }
} // namespace rocRoller

namespace std
{
    template <typename T>
    bool operator==(rocRoller::Raw32 const& a, T const& b)
    {
        if constexpr(std::is_same_v<T, rocRoller::Raw32>)
            return a == b;
        else
            return false;
    }

    template <>
    struct hash<rocRoller::Raw32>
    {
        size_t operator()(const rocRoller::Raw32& a) const
        {
            return hash<uint32_t>()(a.value);
        }
    };
} // namespace std
