// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Serialization/Base.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        /**
         * Provides serialization for any enumeration type that adheres to the CCountedEnum concept;
         * i.e. provides a Count value and toString (fromString() uses toString).
         */
        template <CCountedEnum Enum>
        struct ScalarTraits<Enum>
        {
            static std::string output(const Enum& value)
            {
                return toString(value);
            }

            static void input(std::string const& scalar, Enum& value)
            {
                value = fromString<Enum>(scalar);
            }
        };

    } // namespace Serialization
} // namespace rocRoller
