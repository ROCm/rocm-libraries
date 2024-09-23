#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/CMPXWriteExec.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void CMPXWriteExec::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                for(auto const& regId : m_context.lock()->getExec()->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                }
            }
        }

        int CMPXWriteExec::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool CMPXWriteExec::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVCMPX(inst.getOpCode());
        };

        int CMPXWriteExec::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isMFMA(inst.getOpCode())
               || (m_checkACCVGPR && GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode())))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }
            return 0;
        }
    }
}
