
#pragma once

#include "Register_fwd.hpp"

#include "../CodeGen/Instruction.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    namespace Register
    {
        /**
         * yields code to allocate *reg* if it requires allocation.
         */
        Generator<Instruction> AllocateIfNeeded(Register::ValuePtr reg);
    }
}

#include "RegisterUtils_impl.hpp"
