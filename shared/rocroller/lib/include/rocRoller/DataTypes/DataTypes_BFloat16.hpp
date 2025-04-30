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

#include <rocRoller/DataTypes/DataTypes_BF16_Utils.hpp>

#define ROCROLLER_USE_BFloat16

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
    struct BFloat16
    {
        constexpr BFloat16()
            : data(BFLOAT16_ZERO_VALUE)
        {
        }

        BFloat16(BFloat16 const& other) = default;

        template <typename T,
                  typename
                  = typename std::enable_if<(!std::is_same<T, BFloat16>::value)
                                            && std::is_convertible<T, double>::value>::type>
        explicit BFloat16(T const& value)
            : data(float_to_bf16(static_cast<double>(value)).data)
        {
        }

        explicit operator float() const
        {
            return bf16_to_float(*this);
        }

        operator double() const
        {
            return static_cast<double>(float(*this));
        }

        uint16_t data;
    };

    inline std::ostream& operator<<(std::ostream& os, const BFloat16& obj)
    {
        os << static_cast<float>(obj);
        return os;
    }

    inline BFloat16 operator-(BFloat16 const& a)
    {
        return static_cast<BFloat16>(-static_cast<float>(a));
    }

    template <typename T, typename = typename std::enable_if_t<std::is_convertible_v<T, float>>>
    inline bool operator==(BFloat16 const& a, T const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    inline bool operator==(BFloat16 const& a, BFloat16 const& b)
    {
        return static_cast<float>(a) == static_cast<float>(b);
    }

    /**
     * @}
     */
} // namespace rocRoller

namespace std
{
    template <>
    struct is_floating_point<rocRoller::BFloat16> : true_type
    {
    };

    template <>
    struct hash<rocRoller::BFloat16>
    {
        size_t operator()(const rocRoller::BFloat16& a) const
        {
            return hash<uint16_t>()(a.data);
        }
    };
} // namespace std
