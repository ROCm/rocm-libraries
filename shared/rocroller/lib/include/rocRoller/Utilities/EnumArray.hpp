// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>

#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{
    /**
     * Array which uses an enum (class) as an indexer. Has the interface of std::array, and can be
     * indexed by the enum.
     *
     * The enum must declare a Count entry.
     */
    template <typename T, CCountedEnum Enum>
    class EnumArray : public std::array<T, static_cast<size_t>(Enum::Count)>
    {
    public:
        static constexpr size_t Size = static_cast<size_t>(Enum::Count);

        using Base = std::array<T, Size>;
        using Base::Base;

        using Base::operator[];
        using Base::at;

        constexpr T&       operator[](Enum val);
        constexpr T const& operator[](Enum val) const;

        constexpr T&       at(Enum val);
        constexpr T const& at(Enum val) const;
    };
}

#include <rocRoller/Utilities/EnumArray_impl.hpp>
