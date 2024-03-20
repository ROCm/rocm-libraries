#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteReadlane94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteReadlane94x::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteReadlane94x::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVALU() && !inst->isMFMA() && !inst->isDLOP();
        };

        bool VALUWriteReadlane94x::writeTrigger() const
        {
            return true;
        }

        int VALUWriteReadlane94x::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(inst.getOpCode().rfind("v_readlane", 0) == 0)
            {
                AssertFatal(inst.getSrcs().size() > 0, "Bad readlane sources");
                return checkRegister(inst.getSrcs()[0]).value_or(0);
            }
            return 0;
        }
    }
}
