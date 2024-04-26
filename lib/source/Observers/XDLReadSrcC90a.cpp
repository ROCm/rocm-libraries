#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC90a.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void XDLReadSrcC90a::observeHazard(Instruction const& inst)
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

        int XDLReadSrcC90a::getMaxNops(Instruction const& inst) const
        {
            return getNopFromLatency(inst.getOpCode(), m_latencyAndNops);
        }

        bool XDLReadSrcC90a::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isMFMA(inst.getOpCode());
        };

        bool XDLReadSrcC90a::writeTrigger() const
        {
            return false;
        }

        int XDLReadSrcC90a::getNops(Instruction const& inst) const
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
