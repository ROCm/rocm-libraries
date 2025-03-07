#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAWrite.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int WMMAWrite::getMaxNops(const Instruction& inst) const
        {
            return m_maxNops;
        }

        bool WMMAWrite::trigger(const Instruction& inst) const
        {
            return GPUInstructionInfo::isWMMA(inst.getOpCode());
        };

        int WMMAWrite::getNops(const Instruction& inst) const
        {
            int requiredNops = 0;

            if(GPUInstructionInfo::isWMMA(inst.getOpCode()))
            {
                const auto&        srcs = inst.getSrcs();
                std::optional<int> value;
                // SrcA RAW
                AssertFatal(nullptr != srcs.at(0), "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    requiredNops = *value;
                }

                // SrcB RAW
                AssertFatal(nullptr != srcs.at(1), "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    requiredNops = *value;
                }
            }

            return requiredNops;
        }
    }
}
