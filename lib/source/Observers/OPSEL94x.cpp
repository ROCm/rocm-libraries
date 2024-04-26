#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/OPSEL94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int OPSEL94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool OPSEL94x::trigger(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode()))
            {
                if(GPUInstructionInfo::isSDWA(inst.getOpCode()))
                {
                    return true;
                }
                for(auto const& mod : inst.getModifiers())
                {
                    if(mod.rfind("op_sel", 0) == 0)
                        return true;
                }
            }
            return false;
        };

        bool OPSEL94x::writeTrigger() const
        {
            return true;
        }

        int OPSEL94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
