// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>

namespace rocRoller
{
    enum class ScaleSkipPermlaneMode
    {
        None,
        PreSwizzleScale,
        PreSwizzleScaleGFX950,
        Count,
    };

    std::string   toString(ScaleSkipPermlaneMode mode);
    std::ostream& operator<<(std::ostream& stream, ScaleSkipPermlaneMode const& mode);
} // namespace rocRoller
