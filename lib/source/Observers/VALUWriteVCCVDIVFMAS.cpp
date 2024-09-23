#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteVCCVDIVFMAS.hpp>

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
            return GPUInstructionInfo::isVALU(inst.getOpCode());
        }

        int VALUWriteVCCVDIVFMAS::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVDivFmas(inst.getOpCode()))
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
