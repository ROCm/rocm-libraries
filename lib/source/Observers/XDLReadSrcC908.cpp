#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC908.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void XDLReadSrcC908::observeHazard(Instruction const& inst)
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

        int XDLReadSrcC908::getMaxNops(Instruction const& inst) const
        {
            return getNopFromLatency(inst.getOpCode(), m_latencyAndNops);
        }

        bool XDLReadSrcC908::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isMFMA(inst.getOpCode());
        };

        bool XDLReadSrcC908::writeTrigger() const
        {
            return false;
        }

        int XDLReadSrcC908::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode()))
            {
                // WAR
                return checkDsts(inst).value_or(0);
            }
            return 0;
        }
    }
}
