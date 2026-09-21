// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <string>

namespace rocRoller
{
    enum class LDSBankSwizzleMode
    {
        None,
        Swizzle,
        Count,
    };

    std::string   toString(LDSBankSwizzleMode mode);
    std::ostream& operator<<(std::ostream& stream, LDSBankSwizzleMode const& mode);
} // namespace rocRoller
