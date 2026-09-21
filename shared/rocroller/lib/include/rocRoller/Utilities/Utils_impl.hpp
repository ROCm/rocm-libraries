// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/RTTI.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    template <CCountedEnum T>
    T fromString(std::string const& str)
    {
        using myInt   = std::underlying_type_t<T>;
        auto maxValue = static_cast<myInt>(T::Count);
        for(myInt i = 0; i < maxValue; ++i)
        {
            auto val = static_cast<T>(i);
            if(toString(val) == str)
                return val;
        }
        Throw<FatalError>(
            "Invalid fromString: type name: ", typeName<T>(), ", string input: ", str);
    }

    template <CCountedEnum T>
    std::set<std::string> enumStrings()
    {
        std::set<std::string> result;
        using myInt   = std::underlying_type_t<T>;
        auto maxValue = static_cast<myInt>(T::Count);
        for(myInt i = 0; i < maxValue; ++i)
        {
            result.insert(toString(static_cast<T>(i)));
        }
        return result;
    }
}
