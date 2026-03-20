// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Scratch space command.
 */

#pragma once

#include <string>

namespace rocRoller
{
    namespace Operations
    {
        /**
         * A scratch space operation that is used to set policy guarantees for scratch space content.
        */
        class Scratch;

        enum class ScratchPolicy
        {
            None = 0, //< No guarantees about scratch space content
            ZeroedBeforeAndAfter, //< Scratch space is zeroed before and after kernel launch
            Count
        };

        std::string   toString(ScratchPolicy const& policy);
        std::ostream& operator<<(std::ostream& stream, ScratchPolicy const& policy);

    }
}
