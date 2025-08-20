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

#include <limits>
#ifdef ROCROLLER_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include <rocRoller/DataTypes/DistinctType.hpp>

namespace rocRoller
{
    /**
     * \ingroup DataTypes
     */
    struct E8M0
    {
        E8M0()
            : scale(0)
        {
        }

        explicit E8M0(uint8_t scale)
            : scale(scale)
        {
        }

        explicit E8M0(float scale)
        {
            struct
            {
                uint mantissa : 23;
                uint exponent : 8;
                bool sign : 1;
            } parts;

            static_assert(sizeof(parts) == 4);

            memcpy(&parts, &scale, sizeof(parts));

            this->scale = parts.exponent;
        }

        uint8_t scale;

        inline operator float() const
        {
            if(scale == 0xFF)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            return std::pow(2.0f, int(this->scale) - 127);
        }

        explicit inline operator uint8_t() const
        {
            return this->scale;
        }
    };
    static_assert(sizeof(E8M0) == 1, "E8M0 must be 1 byte.");

    inline E8M0 operator-(E8M0 const& a)
    {
        return static_cast<E8M0>(static_cast<uint8_t>(~(a.scale + uint8_t(1))));
    }

    inline std::ostream& operator<<(std::ostream& os, const E8M0 val)
    {
        os << val.scale;
        return os;
    }
} // namespace rocRoller

namespace std
{

    template <typename T>
    requires(std::is_convertible_v<T, uint8_t>&& std::is_integral_v<T>) inline bool
        operator==(rocRoller::E8M0 const& a, T const& b)
    {
        return a.scale == static_cast<uint8_t>(b);
    }

    template <typename T>
    requires(std::is_convertible_v<T, uint8_t>&& std::is_integral_v<T>) inline bool
        operator!=(rocRoller::E8M0 const& a, T const& b)
    {
        return a.scale != static_cast<uint8_t>(b);
    }

    template <>
    struct is_floating_point<rocRoller::E8M0> : true_type
    {
    };

    template <>
    struct hash<rocRoller::E8M0>
    {
        size_t operator()(const rocRoller::E8M0& a) const
        {
            return hash<uint8_t>()(a.scale);
        }
    };
} // namespace std
