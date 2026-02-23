// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <bitset>

#include <rocRoller/Utilities/Concepts.hpp>

namespace rocRoller
{
    /**
     * Bitset which uses an enum (class) as an indexer. Has the interface of std::bitset, and can be
     * indexed by the enum.
     *
     * All new interfaces are constexpr so it can be used in concepts and to limit template instantiation.
     *
     * - The enum must declare a Count entry.
     * - All other entries must be >= 0 and < Count.
     * - Count must be <= 64.
     */
    template <CCountedEnum Enum>
    class EnumBitset : public std::bitset<static_cast<size_t>(Enum::Count)>
    {
    public:
        static constexpr size_t Size = static_cast<size_t>(Enum::Count);
        static_assert(Size <= 64);

        using Base = std::bitset<Size>;
        using Base::Base;
        using Base::operator[];

        constexpr EnumBitset(std::initializer_list<Enum> items);
        // cppcheck-suppress noExplicitConstructor
        constexpr EnumBitset(Base const& val);

        /**
         * Returns an object with all bits set.
         */
        constexpr static EnumBitset All();

        constexpr bool operator[](Enum val) const;

        /**
         * @brief Sets the taget enum to a value
         */
        void set(Enum target, bool value = true);

    private:
        static constexpr size_t initialValue(std::initializer_list<Enum> items);
    };

    template <CCountedEnum Enum>
    std::string toString(EnumBitset<Enum> const& bs);

    template <CCountedEnum Enum>
    std::string shortString(EnumBitset<Enum> const& bs);

    template <CCountedEnum Enum>
    std::ostream& operator<<(std::ostream& stream, EnumBitset<Enum> const& bs);
}

#include <rocRoller/Utilities/EnumBitset_impl.hpp>
