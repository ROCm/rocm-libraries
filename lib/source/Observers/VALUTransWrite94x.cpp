#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUTransWrite94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUTransWrite94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUTransWrite94x::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVALUTrans(inst.getOpCode());
        };

        bool VALUTransWrite94x::writeTrigger() const
        {
            return true;
        }

        int VALUTransWrite94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode())
               && !GPUInstructionInfo::isVALUTrans(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
