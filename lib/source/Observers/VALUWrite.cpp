#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/VALUWrite.hpp>

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
            return GPUInstructionInfo::isVALU(inst.getOpCode())
                   && !GPUInstructionInfo::isMFMA(inst.getOpCode())
                   && !GPUInstructionInfo::isDLOP(inst.getOpCode());
        };

        bool VALUWrite::writeTrigger() const
        {
            return true;
        }

        int VALUWrite::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isMFMA(inst.getOpCode())
               || (m_checkACCVGPR && GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode())))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
