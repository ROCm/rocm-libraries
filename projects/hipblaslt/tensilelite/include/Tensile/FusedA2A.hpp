// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

namespace TensileLite
{
    enum class FusedA2A : int
    {
        Disabled = 0,
        GemmA2A,
        A2AGemm,
        Count
    };

    inline std::string ToString(FusedA2A mode)
    {
        switch(mode)
        {
        case FusedA2A::Disabled:
            return "Disabled";
        case FusedA2A::GemmA2A:
            return "GemmA2A";
        case FusedA2A::A2AGemm:
            return "A2AGemm";
        default:
            return "Invalid";
        }
    }

    inline std::ostream& operator<<(std::ostream& stream, FusedA2A mode)
    {
        return stream << ToString(mode);
    }

    inline std::istream& operator>>(std::istream& stream, FusedA2A& mode)
    {
        std::string name;
        stream >> name;
        for(int i = 0; i < static_cast<int>(FusedA2A::Count); i++)
            if(ToString(static_cast<FusedA2A>(i)) == name)
            {
                mode = static_cast<FusedA2A>(i);
                return stream;
            }
        throw std::invalid_argument("FusedA2A '" + name
                                    + "' is not one of Disabled, GemmA2A, A2AGemm");
    }
} // namespace TensileLite
