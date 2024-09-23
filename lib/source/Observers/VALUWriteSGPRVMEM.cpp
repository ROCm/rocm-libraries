#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVMEM.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteSGPRVMEM::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteSGPRVMEM::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVALU(inst.getOpCode());
        }

        int VALUWriteSGPRVMEM::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVMEM(inst.getOpCode()))
            {
                for(auto const& src : inst.getSrcs())
                {
                    auto val = checkRegister(src);
                    if(val.has_value() && src->regType() == Register::Type::Scalar)
                    {
                        return val.value();
                    }
                }
            }
            return 0;
        }
    }
}
