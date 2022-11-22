#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteFollowedByMFMARead.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteFollowedByMFMARead::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return 2;
        }

        bool VALUWriteFollowedByMFMARead::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVALU() && !inst->isXDLOP();
        };

        bool VALUWriteFollowedByMFMARead::writeTrigger() const
        {
            return true;
        }

        int VALUWriteFollowedByMFMARead::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isMFMA())
            {
                auto regs   = inst.getRegisters();
                auto regMap = m_context.lock()->getRegisterHazardMap();

                for(auto const& src : std::get<0>(regs))
                {
                    for(auto const& srcId : src->getRegisterIds())
                    {
                        if(regMap->contains(srcId))
                        {
                            for(auto const& hazard : regMap->at(srcId))
                            {
                                if(hazard.regWasWritten() && trigger(hazard.getInstructionRef()))
                                {
                                    return hazard.getRequiredNops();
                                }
                            }
                        }
                    }
                }
            }
            return 0;
        }
    }
}
