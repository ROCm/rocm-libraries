// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteSGPRVCC::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteSGPRVCC::trigger(Instruction const& inst) const
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

        int VALUWriteSGPRVCC::getNops(Instruction const& inst) const
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
                int pos = 0;
                for(auto const& src : inst.getSrcs())
                {
                    auto val = checkRegister(src);
                    if(val.has_value()
                       && (src->regType() == Register::Type::VCC
                           || src->regType() == Register::Type::Scalar))
                    {
                        // Not a hazard if reading VCC or SGPR as carry
                        if((GPUInstructionInfo::isVAddCarryInst(inst.getOpCode())
                            || GPUInstructionInfo::isVSubCarryInst(inst.getOpCode()))
                           && pos == 2)
                        {
                            pos++;
                            continue;
                        }

                        if(!m_isCDNA1orCDNA2 && src->regType() == Register::Type::Scalar)
                        {
                            return val.value() - 2;
                        }
                        else
                        {
                            return val.value() - 3;
                        }
                    }
                    pos++;
                }
            }

            return 0;
        }
    }
}
