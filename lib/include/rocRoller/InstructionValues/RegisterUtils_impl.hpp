
#pragma once

#include "RegisterUtils.hpp"

namespace rocRoller
{
    namespace Register
    {
        inline Generator<Instruction> AllocateIfNeeded(Register::ValuePtr reg)
        {
            if(reg->allocationState() == AllocationState::Unallocated)
                co_yield reg->allocate();
        }
    }
}
