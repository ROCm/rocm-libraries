#include <rocRoller/Scheduling/Observers/WaitState/MFMA/VALUWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWrite::trigger(Instruction const& inst) const
        {
            return InstructionRef::isVALU(inst.getOpCode())
                   && !InstructionRef::isMFMA(inst.getOpCode())
                   && !InstructionRef::isDLOP(inst.getOpCode());
        };

        bool VALUWrite::writeTrigger() const
        {
            return true;
        }

        int VALUWrite::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isMFMA(inst.getOpCode())
               || (m_checkACCVGPR && InstructionRef::isACCVGPRWrite(inst.getOpCode())))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
