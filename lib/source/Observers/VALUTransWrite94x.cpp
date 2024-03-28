#include <rocRoller/Scheduling/Observers/WaitState/VALUTransWrite94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUTransWrite94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUTransWrite94x::trigger(Instruction const& inst) const
        {
            return InstructionRef::isVALUTrans(inst.getOpCode());
        };

        bool VALUTransWrite94x::writeTrigger() const
        {
            return true;
        }

        int VALUTransWrite94x::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isVALU(inst.getOpCode())
               && !InstructionRef::isVALUTrans(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
