#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteReadlane94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteReadlane94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteReadlane94x::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVALU(inst.getOpCode())
                   && !GPUInstructionInfo::isMFMA(inst.getOpCode())
                   && !GPUInstructionInfo::isDLOP(inst.getOpCode());
        };

        bool VALUWriteReadlane94x::writeTrigger() const
        {
            return true;
        }

        int VALUWriteReadlane94x::getNops(Instruction const& inst) const
        {
            if(inst.getOpCode().rfind("v_readlane", 0) == 0)
            {
                AssertFatal(inst.getSrcs().size() > 0, "Bad readlane sources");
                return checkRegister(inst.getSrcs()[0]).value_or(0);
            }
            return 0;
        }
    }
}
