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
