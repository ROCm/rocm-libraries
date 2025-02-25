#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRReadWrite.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int ACCVGPRReadWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool ACCVGPRReadWrite::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isACCVGPRRead(inst.getOpCode());
        };

        int ACCVGPRReadWrite::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isMFMA(inst.getOpCode()))
            {
                auto const& srcs = inst.getSrcs();

                std::optional<int> value;

                // SrcA
                AssertFatal(srcs[0] != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs[0])))
                {
                    return *value;
                }

                // ScrB
                AssertFatal(srcs[1] != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs[1])))
                {
                    return *value;
                }
            }
            else if(GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(GPUInstructionInfo::isVMEM(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
