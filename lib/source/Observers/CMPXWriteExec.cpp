#include <rocRoller/Scheduling/Observers/WaitState/MFMA/CMPXWriteExec.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void CMPXWriteExec::observeHazard(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(trigger(instRef))
            {
                for(auto const& regId : m_context.lock()->getExec()->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
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
            return inst->isVCMPX();
        };

        bool CMPXWriteExec::writeTrigger() const
        {
            return true;
        }

        int CMPXWriteExec::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isMFMA() || (m_checkACCVGPR && instRef.isACCVGPRWrite()))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }
            return 0;
        }
    }
}
