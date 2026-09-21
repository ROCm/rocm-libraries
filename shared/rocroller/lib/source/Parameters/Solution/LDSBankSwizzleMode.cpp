// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Parameters/Solution/LDSBankSwizzleMode.hpp>

#include <string>

namespace rocRoller
{
    std::string toString(LDSBankSwizzleMode mode)
    {
        switch(mode)
        {
        case LDSBankSwizzleMode::None:
            return "None";
        case LDSBankSwizzleMode::Swizzle:
            return "Swizzle";
        default:
            break;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, LDSBankSwizzleMode const& mode)
    {
        return stream << toString(mode);
    }
} // namespace rocRoller
