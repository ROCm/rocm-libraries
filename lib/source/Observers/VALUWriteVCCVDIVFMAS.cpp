#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteVCCVDIVFMAS.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteVCCVDIVFMAS::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteVCCVDIVFMAS::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVALU() && !inst->isMFMA() && !inst->isDLOP();
        };

        bool VALUWriteVCCVDIVFMAS::writeTrigger() const
        {
            return true;
        }

        int VALUWriteVCCVDIVFMAS::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isVDivFmas())
            {
                for(auto const& src : inst.getSrcs())
                {
                    auto val = checkRegister(src);
                    if(val.has_value()
                       && (src->regType() == Register::Type::Scalar
                           || src->regType() == Register::Type::VCC))
                    {
                        return val.value();
                    }
                }
            }

            return 0;
        }
    }
}
