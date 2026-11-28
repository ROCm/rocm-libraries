// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace TensileLite
{
    template <typename Enum>
    struct EnumStringTraits;

    template <typename Enum>
    concept HasEnumStrings
        = requires { EnumStringTraits<Enum>::typeName; EnumStringTraits<Enum>::entries; };

    template <typename Enum>
    struct EnumStringEntry
    {
        Enum             value;
        std::string_view name;
    };

    template <HasEnumStrings Enum>
    class EnumStrings
    {
        using Traits = EnumStringTraits<Enum>;

    public:
        EnumStrings() = delete;

        static constexpr std::size_t size()
        {
            return Traits::entries.size();
        }
        
        static std::string_view name(Enum value)
        {
            auto const underlying = static_cast<std::underlying_type_t<Enum>>(value);
            auto const ii         = static_cast<std::size_t>(underlying);
            if(ii >= size())
                throw std::runtime_error("Invalid " + std::string(Traits::typeName)
                                         + " value: " + std::to_string(static_cast<int>(value)));
            return Traits::entries[ii].name;
        }

        static std::string toString(Enum value)
        {
            return std::string(name(value));
        }

        static Enum fromString(std::string_view name)
        {
            for(auto const& entry : Traits::entries)
            {
                if(entry.name == name)
                    return entry.value;
            }
            throw std::runtime_error("Invalid " + std::string(Traits::typeName)
                                     + " value: " + std::string(name));
        }
    };

    template <HasEnumStrings Enum>
    std::string toString(Enum value)
    {
        return EnumStrings<Enum>::toString(value);
    }

    template <HasEnumStrings Enum>
    Enum enumFromString(std::string_view name)
    {
        return EnumStrings<Enum>::fromString(name);
    }

    template <HasEnumStrings Enum>
    std::ostream& operator<<(std::ostream& stream, Enum const& value)
    {
        return stream << toString(value);
    }

    template <HasEnumStrings Enum>
    std::istream& operator>>(std::istream& stream, Enum& value)
    {
        std::string name;
        stream >> name;
        value = enumFromString<Enum>(name);
        return stream;
    }
} // namespace TensileLite
