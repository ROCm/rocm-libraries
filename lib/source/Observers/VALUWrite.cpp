#include <rocRoller/Scheduling/Observers/WaitState/VALUWrite.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWrite::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool VALUWrite::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVALU() && !inst->isMFMA() && !inst->isDLOP();
        };

        bool VALUWrite::writeTrigger() const
        {
            return true;
        }

        int VALUWrite::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isMFMA() || (m_checkACCVGPR && instRef.isACCVGPRWrite()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
