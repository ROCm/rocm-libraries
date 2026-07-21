// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAReadSrcD.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void WMMAReadSrcD::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                auto dstD = inst.getDsts().at(0);
                AssertFatal(dstD != nullptr, "Empty DstD");

                for(auto const& regId : dstD->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                }
            }
        }

        int WMMAReadSrcD::getMaxNops(Instruction const& inst) const
        {
            return getNopFromLatency(inst.getOpCode(), m_latencyAndNops);
        }

        bool WMMAReadSrcD::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isWMMA(inst.getOpCode());
        };

        int WMMAReadSrcD::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode())
               && !GPUInstructionInfo::isWMMA(inst.getOpCode()))
            {
                std::optional<int> value;
                // RAW
                if((value = checkSrcs(inst)))
                {
                    return *value;
                }
            }
            return 0;
        }
    }
}
