#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteSGPRVCC94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteSGPRVCC94x::trigger(Instruction const& inst) const
        {
            return InstructionRef::isVCMP(inst.getOpCode())
                   || InstructionRef::isVReadlane(inst.getOpCode())
                   || InstructionRef::isVDivScale(inst.getOpCode())
                   || (InstructionRef::isVAddInst(inst.getOpCode())
                       && (InstructionRef::isIntInst(inst.getOpCode())
                           || InstructionRef::isUIntInst(inst.getOpCode())))
                   || (InstructionRef::isVSubInst(inst.getOpCode())
                       && (InstructionRef::isIntInst(inst.getOpCode())
                           || InstructionRef::isUIntInst(inst.getOpCode())));
        };

        bool VALUWriteSGPRVCC94x::writeTrigger() const
        {
            return true;
        }

        int VALUWriteSGPRVCC94x::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isVReadlane(inst.getOpCode())
               || InstructionRef::isVWritelane(inst.getOpCode()))
            {
                AssertFatal(inst.getSrcs().size() >= 2, "Unexpected instruction", inst.getOpCode());
                auto const& laneSelect = inst.getSrcs()[1];
                auto        val        = checkRegister(laneSelect);
                if(val.has_value()
                   && (laneSelect->regType() == Register::Type::Scalar
                       || laneSelect->regType() == Register::Type::VCC))
                {
                    return val.value();
                }
            }
            else
            {
                for(auto const& src : inst.getSrcs())
                {
                    auto val = checkRegister(src);
                    if(val.has_value()
                       && (src->regType() == Register::Type::Scalar
                           || src->regType() == Register::Type::VCC))
                    {
                        return val.value() - 2;
                    }
                }
            }

            return 0;
        }
    }
}
