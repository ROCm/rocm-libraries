#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void XDLReadSrcC94x::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                auto srcC = inst.getSrcs().at(2);
                AssertFatal(srcC != nullptr, "Empty SrcC");

                for(auto const& regId : srcC->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                }
            }
        }

        int XDLReadSrcC94x::getMaxNops(Instruction const& inst) const
        {
            return getNopFromLatency(inst.getOpCode(), m_latencyAndNops);
        }

        bool XDLReadSrcC94x::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isMFMA(inst.getOpCode());
        };

        int XDLReadSrcC94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode())
               && !GPUInstructionInfo::isMFMA(inst.getOpCode()))
            {
                // WAR
                return checkDsts(inst).value_or(0);
            }
            return 0;
        }
    }
}
