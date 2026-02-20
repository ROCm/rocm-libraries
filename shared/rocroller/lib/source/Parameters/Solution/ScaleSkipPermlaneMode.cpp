// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/Parameters/Solution/ScaleSkipPermlaneMode.hpp>

#include <string>

namespace rocRoller
{
    std::string toString(ScaleSkipPermlaneMode mode)
    {
        switch(mode)
        {
        case ScaleSkipPermlaneMode::None:
            return "None";
        case ScaleSkipPermlaneMode::PreSwizzleScale:
            return "PreSwizzleScale";
        case ScaleSkipPermlaneMode::PreSwizzleScaleGFX950:
            return "PreSwizzleScaleGFX950";
        default:
            break;
        }
        return "Invalid";
    }

    std::ostream& operator<<(std::ostream& stream, ScaleSkipPermlaneMode const& mode)
    {
        return stream << toString(mode);
    }
} // namespace rocRoller
