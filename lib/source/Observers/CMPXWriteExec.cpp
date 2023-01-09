#include <rocRoller/Scheduling/Observers/WaitState/CMPXWriteExec.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void CMPXWriteExec::observe(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(trigger(instRef))
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();

                for(auto const& regId : m_context.lock()->getExec()->getRegisterIds())
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

        int CMPXWriteExec::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool CMPXWriteExec::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isCMPX();
        };

        bool CMPXWriteExec::writeTrigger() const
        {
            return true;
        }

        int CMPXWriteExec::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isMFMA() || (checkACCVGPR && instRef.isACCVGPRWrite()))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }
            return 0;
        }
    }
}
