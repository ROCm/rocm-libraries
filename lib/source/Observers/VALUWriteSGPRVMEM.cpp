#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVMEM.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteSGPRVMEM::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteSGPRVMEM::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVALU() && !inst->isMFMA() && !inst->isDLOP();
        };

        bool VALUWriteSGPRVMEM::writeTrigger() const
        {
            return true;
        }

        int VALUWriteSGPRVMEM::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isVMEM())
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
