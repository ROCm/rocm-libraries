#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteSGPRVCC94x::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteSGPRVCC94x::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isVCMP() || inst->isVReadlane() || inst->isVDivScale()
                   || (inst->isVAddInst() && (inst->isIntInst() || inst->isUIntInst()))
                   || (inst->isVSubInst() && (inst->isIntInst() || inst->isUIntInst()));
        };

        bool VALUWriteSGPRVCC94x::writeTrigger() const
        {
            return true;
        }

        int VALUWriteSGPRVCC94x::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isVReadlane() || instRef.isVWritelane())
            {
                AssertFatal(
                    inst.getSrcs().size() >= 2, "Unexpected instruction", instRef.getOpCode());
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
