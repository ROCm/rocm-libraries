// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "EnumArray.hpp"

namespace rocRoller
{
    template <typename T, CCountedEnum Enum>
    constexpr T& EnumArray<T, Enum>::operator[](Enum val)
    {
        return (*this)[static_cast<size_t>(val)];
    }

    template <typename T, CCountedEnum Enum>
    constexpr T const& EnumArray<T, Enum>::operator[](Enum val) const
    {
        return (*this)[static_cast<size_t>(val)];
    }

    template <typename T, CCountedEnum Enum>
    constexpr T& EnumArray<T, Enum>::at(Enum val)
    {
        return this->at(static_cast<size_t>(val));
    }

    template <typename T, CCountedEnum Enum>
    constexpr T const& EnumArray<T, Enum>::at(Enum val) const
    {
        return this->at(static_cast<size_t>(val));
    }
}
