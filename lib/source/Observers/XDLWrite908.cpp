#include <rocRoller/Scheduling/Observers/WaitState/XDLWrite908.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int XDLWrite908::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return getNopFromLatency(inst->getOpCode(), m_maxLatencyAndNops);
        }

        bool XDLWrite908::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->isMFMA();
        };

        bool XDLWrite908::writeTrigger() const
        {
            return true;
        }

        int XDLWrite908::getNops(Instruction const& inst) const
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
                                    overlap      = true;
                                    requiredNops = hazard.getRequiredNops()
                                                   - getNopFromLatency(
                                                       hazard.getInstructionRef()->getOpCode(),
                                                       m_maxLatencyAndNops)
                                                   + 2;
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
                for(auto const& srcId : srcs.at(0)->getRegisterIds())
                {
                    if(regMap->contains(srcId))
                    {
                        for(auto const& hazard : regMap->at(srcId))
                        {
                            if(hazard.regWasWritten() && trigger(hazard.getInstructionRef()))
                            {
                                return hazard.getRequiredNops()
                                       - (getNopFromLatency(hazard.getInstructionRef()->getOpCode(),
                                                            m_maxLatencyAndNops)
                                          - 4);
                            }
                        }
                    }
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                for(auto const& srcId : srcs.at(1)->getRegisterIds())
                {
                    if(regMap->contains(srcId))
                    {
                        for(auto const& hazard : regMap->at(srcId))
                        {
                            if(hazard.regWasWritten() && trigger(hazard.getInstructionRef()))
                            {
                                return hazard.getRequiredNops()
                                       - (getNopFromLatency(hazard.getInstructionRef()->getOpCode(),
                                                            m_maxLatencyAndNops)
                                          - 4);
                            }
                        }
                    }
                }
            }
            else if(instRef.isACCVGPRRead())
            {
                // ACCVGPR RAW
                return checkSrcs(inst).value_or(0);
            }
            else if(instRef.isACCVGPRWrite())
            {
                // ACCVGPR WAW
                return checkDsts(inst).value_or(0) - 3;
            }
            return 0;
        }
    }
}
