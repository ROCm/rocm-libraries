#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteVCCVDIVFMAS.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteVCCVDIVFMAS::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteVCCVDIVFMAS::trigger(Instruction const& inst) const
        {
            return InstructionRef::isVALU(inst.getOpCode())
                   && !InstructionRef::isMFMA(inst.getOpCode())
                   && !InstructionRef::isDLOP(inst.getOpCode());
        };

        bool VALUWriteVCCVDIVFMAS::writeTrigger() const
        {
            return true;
        }

        int VALUWriteVCCVDIVFMAS::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isVDivFmas(inst.getOpCode()))
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
