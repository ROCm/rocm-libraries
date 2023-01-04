#include <rocRoller/Scheduling/Observers/WaitState/XDLReadSrcC908.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void XDLReadSrcC908::observe(Instruction const& inst)
        {
            auto instRef = std::make_shared<InstructionRef>(inst);
            if(trigger(instRef))
            {
                auto regMap = m_context.lock()->getRegisterHazardMap();

                auto srcC = inst.getSrcs().at(2);
                AssertFatal(srcC != nullptr, "Empty SrcC");

                for(auto const& regId : srcC->getRegisterIds())
                {
                    if(!regMap->contains(regId))
                    {
                        (*regMap)[regId] = {};
                    }
                    (*regMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(instRef), instRef, writeTrigger()));
                }
            }
        }

        int XDLReadSrcC908::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return getNopFromLatency(inst->getOpCode(), m_latencyAndNops);
        }

        bool XDLReadSrcC908::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isMFMA();
        };

        bool XDLReadSrcC908::writeTrigger() const
        {
            return false;
        }

        int XDLReadSrcC908::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            if(instRef.isACCVGPRWrite())
            {
                // WAR
                return checkDsts(inst).value_or(0);
            }
            return 0;
        }
    }
}
