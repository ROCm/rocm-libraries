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
            return GPUInstructionInfo::isVALU(inst.getOpCode());
        }

        int VALUWriteReadlane94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVReadlane(inst.getOpCode()))
            {
                AssertFatal(inst.getSrcs().size() > 0, "Bad readlane sources");
                return checkRegister(inst.getSrcs()[0]).value_or(0);
            }
            return 0;
        }
    }
}
