#include <rocRoller/Scheduling/Observers/WaitState/VALUTransWrite94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUTransWrite94x::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool VALUTransWrite94x::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVALUTrans();
        };

        bool VALUTransWrite94x::writeTrigger() const
        {
            return true;
        }

        int VALUTransWrite94x::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isVALU() && !instRef.isVALUTrans())
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
