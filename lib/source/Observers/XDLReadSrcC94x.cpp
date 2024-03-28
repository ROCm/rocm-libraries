#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

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
            return InstructionRef::isMFMA(inst.getOpCode());
        };

        bool XDLReadSrcC94x::writeTrigger() const
        {
            return false;
        }

        int XDLReadSrcC94x::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isVALU(inst.getOpCode())
               && !InstructionRef::isMFMA(inst.getOpCode()))
            {
                // WAR
                return checkDsts(inst).value_or(0);
            }
            return 0;
        }
    }
}
