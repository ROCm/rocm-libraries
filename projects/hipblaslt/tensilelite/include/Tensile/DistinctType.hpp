/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Tensile/Comparison.hpp>

namespace TensileLite
{
    /**
 * \ingroup DataTypes
 * @{
 */

    /**
 * Allows the creation of a new type which is not automatically convertible
 * to its underlying data type.
 */
    template <typename T, typename Subclass>
    struct DistinctType
    {
        using Value = T;

        DistinctType()                          = default;
        DistinctType(DistinctType const& other) = default;

        DistinctType(T const& v)
            : value(v)
        {
        }

        DistinctType& operator=(DistinctType const& other) = default;
        DistinctType& operator                             =(T const& other)
        {
            value = other;
            return *this;
        }

        operator const T&() const
        {
            return value;
        }

        T value;
    };

    template <typename T, typename Subclass>
    struct Comparison<DistinctType<T, Subclass>>
    {
        enum
        {
            implemented = true
        };

        static int compare(DistinctType<T, Subclass> const& lhs,
                           DistinctType<T, Subclass> const& rhs)
        {
            return LexicographicCompare(lhs.value, rhs.value);
        }
    };

    template <typename T, typename Subclass>
    struct Comparison<DistinctType<T, Subclass>, T>
    {
        enum
        {
            implemented = true
        };

        static int compare(DistinctType<T, Subclass> const& lhs, T const& rhs)
        {
            return LexicographicCompare(lhs.value, rhs);
        }
    };

    /**
 * @}
 */
} // namespace TensileLite
