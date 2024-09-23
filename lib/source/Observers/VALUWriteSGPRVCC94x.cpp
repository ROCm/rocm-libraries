#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC94x.hpp>

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
            return GPUInstructionInfo::isVCMP(inst.getOpCode())
                   || GPUInstructionInfo::isVReadlane(inst.getOpCode())
                   || GPUInstructionInfo::isVDivScale(inst.getOpCode())
                   || (GPUInstructionInfo::isVAddInst(inst.getOpCode())
                       && (GPUInstructionInfo::isIntInst(inst.getOpCode())
                           || GPUInstructionInfo::isUIntInst(inst.getOpCode())))
                   || (GPUInstructionInfo::isVSubInst(inst.getOpCode())
                       && (GPUInstructionInfo::isIntInst(inst.getOpCode())
                           || GPUInstructionInfo::isUIntInst(inst.getOpCode())));
        };

        int VALUWriteSGPRVCC94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVReadlane(inst.getOpCode())
               || GPUInstructionInfo::isVWritelane(inst.getOpCode()))
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
