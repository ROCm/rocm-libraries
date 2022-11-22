#include <rocRoller/Scheduling/Observers/WaitState/GenericWaitStateObserver.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        InstructionStatus GenericWaitStateObserver::peek(Instruction const& inst) const
        {
            return InstructionStatus::Nops(getNops(inst));
        }

        void GenericWaitStateObserver::modify(Instruction& inst) const
        {
            inst.setNopMin(getNops(inst));
        }

        InstructionStatus GenericWaitStateObserver::observe(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(trigger(instRef))
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();
                auto regs   = inst.getRegisters();
                for(auto const& reg : (writeTrigger() ? std::get<1>(regs) : std::get<0>(regs)))
                {

                    for(auto const& regId : reg->getRegisterIds())
                    {
                        if(!regMap->contains(regId))
                        {
                            (*regMap)[regId] = {};
                        }
                        (*regMap)[regId].push_back(
                            WaitStateHazardCounter(getMaxNops(instRef), instRef, writeTrigger()));
                    }
                }
            }

            return InstructionStatus::Nops(inst.getNopCount());
        }
    }
}
