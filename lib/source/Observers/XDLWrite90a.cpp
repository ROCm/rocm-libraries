#include <rocRoller/Scheduling/Observers/WaitState/XDLWrite90a.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int XDLWrite90a::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return getNopFromLatency(inst->getOpCode(), m_latencyAndNops);
        }

        bool XDLWrite90a::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            bool excluded
                = std::find(m_excludedOpCodes.begin(), m_excludedOpCodes.end(), inst->getOpCode())
                  != m_excludedOpCodes.end();
            return inst->isMFMA() && !excluded;
        };

        bool XDLWrite90a::writeTrigger() const
        {
            return true;
        }

        int XDLWrite90a::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);

            if(instRef.isMFMA())
            {
                std::optional<int> value;

                auto        regMap = m_context.lock()->getRegisterHazardMap();
                auto const& srcs   = inst.getSrcs();

                // SrcC RAW
                {
                    bool mismatched   = false;
                    bool overlap      = false;
                    int  requiredNops = 0;

                    AssertFatal(srcs.at(2) != nullptr, "Empty SrcC");
                    for(auto const& srcId : srcs.at(2)->getRegisterIds())
                    {
                        if(regMap->contains(srcId))
                        {
                            for(auto const& hazard : regMap->at(srcId))
                            {
                                if(hazard.regWasWritten() && trigger(hazard.getInstructionRef()))
                                {
                                    int decrement = instRef.isDGEMM() ? 2 : 3;
                                    overlap       = true;
                                    requiredNops  = hazard.getRequiredNops() - decrement;
                                }
                            }
                        }
                        else
                        {
                            mismatched = true;
                        }
                    }
                    if(overlap)
                    {
                        return mismatched ? requiredNops : 0;
                    }
                }

                // SrcA RAW
                AssertFatal(srcs.at(0) != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    return *value;
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value;
                }
            }
            else if(instRef.isVMEM() || instRef.isLDS() || instRef.isFlat())
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(instRef.isVALU())
            {
                std::optional<int> value;

                // VALU RAW
                if((value = checkSrcs(inst)))
                {
                    return *value;
                }

                // VALU WAW
                if((value = checkDsts(inst)))
                {
                    return *value;
                }
            }
            return 0;
        }
    }
}
