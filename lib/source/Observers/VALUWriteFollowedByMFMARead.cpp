#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteFollowedByMFMARead.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        InstructionStatus VALUWriteFollowedByMFMARead::peek(Instruction const& inst) const
        {
            return InstructionStatus::Nops(getNops(inst));
        };

        void VALUWriteFollowedByMFMARead::modify(Instruction& inst) const
        {
            inst.setNopMin(getNops(inst));
        }

        InstructionStatus VALUWriteFollowedByMFMARead::observe(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(instRef->isVALU() && !instRef->isXDLOP())
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();
                auto regs   = inst.getRegisters();
                for(auto const& dst : std::get<1>(regs))
                {
                    for(auto const& dstId : dst->getRegisterIds())
                    {
                        if(!regMap->contains(dstId))
                        {
                            (*regMap)[dstId] = {};
                        }
                        (*regMap)[dstId].push_back(WaitStateHazardCounter(nops, instRef, true));
                    }
                }
            }

            return InstructionStatus::Nops(inst.getNopCount());
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
                            auto hazards = regMap->at(srcId);
                            for(auto const& hazard : hazards)
                            {
                                if(hazard.regWasWritten() && hazard.getInstructionRef()->isVALU()
                                   && !hazard.getInstructionRef()->isXDLOP())
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
