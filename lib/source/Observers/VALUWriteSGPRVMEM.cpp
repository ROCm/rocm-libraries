#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVMEM.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

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
            return InstructionRef::isVALU(inst.getOpCode())
                   && !InstructionRef::isMFMA(inst.getOpCode())
                   && !InstructionRef::isDLOP(inst.getOpCode());
        };

        bool VALUWriteSGPRVMEM::writeTrigger() const
        {
            return true;
        }

        int VALUWriteSGPRVMEM::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isVMEM(inst.getOpCode()))
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
