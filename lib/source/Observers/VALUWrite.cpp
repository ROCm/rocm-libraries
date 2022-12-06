#include <rocRoller/Scheduling/Observers/WaitState/VALUWrite.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        InstructionStatus VALUWrite::observe(Instruction const& inst)
        {
            return observe_base(inst);
        }

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
            if(instRef.isMFMA() || (checkACCVGPR && instRef.isACCVGPRWrite()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
