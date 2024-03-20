#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLReadSrcC90a.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void XDLReadSrcC90a::observeHazard(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(trigger(instRef))
            {
                auto srcC = inst.getSrcs().at(2);
                AssertFatal(srcC != nullptr, "Empty SrcC");

                for(auto const& regId : srcC->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(instRef), instRef, writeTrigger()));
                }
            }
        }

        int XDLReadSrcC90a::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return getNopFromLatency(inst->getOpCode(), m_latencyAndNops);
        }

        bool XDLReadSrcC90a::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isMFMA();
        };

        bool XDLReadSrcC90a::writeTrigger() const
        {
            return false;
        }

        int XDLReadSrcC90a::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isVALU() && !instRef.isMFMA())
            {
                // WAR
                return checkDsts(inst).value_or(0);
            }
            return 0;
        }
    }
}
