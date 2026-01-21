// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <rocRoller/Utilities/Comparison.hpp>

namespace rocRoller
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

        explicit DistinctType(T const& v)
            : value(v)
        {
        }

        DistinctType& operator=(DistinctType const& other) = default;

        DistinctType& operator=(T const& other)
        {
            value = other;
            return *this;
        }

        explicit operator const T&() const
        {
            return value;
        }

        T value = static_cast<T>(0);

        auto operator<=>(DistinctType<T, Subclass> const&) const = default;
    };

    /**
     * @}
     */
}
