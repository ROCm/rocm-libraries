// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>

#include <rocRoller/Utilities/EnumBitset.hpp>

namespace rocRoller
{
    template <CCountedEnum Enum>
    inline constexpr size_t EnumBitset<Enum>::initialValue(std::initializer_list<Enum> items)
    {
        size_t rv = 0;

        for(auto const& entry : items)
        {
            // cppcheck-suppress useStlAlgorithm
            rv |= (1UL << static_cast<size_t>(entry));
        }

        return rv;
    }

    template <CCountedEnum Enum>
    inline constexpr EnumBitset<Enum>::EnumBitset(std::initializer_list<Enum> items)
        : Base(initialValue(items))
    {
    }

    template <CCountedEnum Enum>
    inline constexpr EnumBitset<Enum>::EnumBitset(Base const& val)
        : Base(val)
    {
    }

    template <CCountedEnum Enum>
    inline constexpr EnumBitset<Enum> EnumBitset<Enum>::All()
    {
        size_t rv = 0;

        for(size_t i = 0; i < Size; i++)
            rv |= (1UL << static_cast<size_t>(i));

        return EnumBitset<Enum>(rv);
    }

    template <CCountedEnum Enum>
    inline constexpr bool EnumBitset<Enum>::operator[](Enum val) const
    {
        return (*this)[static_cast<size_t>(val)];
    }

    template <CCountedEnum Enum>
    void EnumBitset<Enum>::set(Enum target, bool value)
    {
        std::bitset<static_cast<size_t>(Enum::Count)>::set(static_cast<size_t>(target), value);
    }

    template <CCountedEnum Enum>
    inline std::string toString(EnumBitset<Enum> const& bs)
    {
        using myInt = std::underlying_type_t<Enum>;

        std::ostringstream msg;

        for(myInt i = 0; i < static_cast<myInt>(Enum::Count); ++i)
        {
            auto val = static_cast<Enum>(i);
            bool set = bs[val];
            msg << toString(val) << ": " << std::boolalpha << set << std::endl;
        }

        return msg.str();
    }

    template <CCountedEnum Enum>
    std::string shortString(EnumBitset<Enum> const& set)
    {
        std::string rv = "{";

        bool first = true;
        for(int i = 0; i < static_cast<int>(Enum::Count); i++)
        {
            auto enumValue = static_cast<Enum>(i);

            if(set[enumValue])
            {
                if(!first)
                    rv += ", ";

                rv += toString(enumValue);
                first = false;
            }
        }

        return rv + "}";
    }

    template <CCountedEnum Enum>
    std::ostream& operator<<(std::ostream& stream, EnumBitset<Enum> const& bs)
    {
        return stream << toString(bs);
    }

}
