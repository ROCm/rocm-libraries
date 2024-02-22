#include <rocRoller/Scheduling/Observers/WaitState/XDLWrite94x.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>
namespace rocRoller
{
    namespace Scheduling
    {
        int XDLWrite94x::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return getNopFromLatency(inst->getOpCode(), m_latencyAndNops);
        }

        bool XDLWrite94x::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            bool excluded
                = std::find(m_excludedOpCodes.begin(), m_excludedOpCodes.end(), inst->getOpCode())
                  != m_excludedOpCodes.end();
            return inst->isMFMA() && !excluded;
        };

        bool XDLWrite94x::writeTrigger() const
        {
            return true;
        }

        int XDLWrite94x::getNops(Instruction const& inst) const
        {
            InstructionRef instRef(inst);
            int            decrement = 0;

            if(instRef.isSGEMM())
            {
                decrement = 1;
            }

            if(instRef.isMFMA())
            {
                std::optional<int> value;

                auto const& srcs = inst.getSrcs();

                // SrcC RAW
                {
                    bool mismatched   = false;
                    bool overlap      = false;
                    int  requiredNops = 0;

                    AssertFatal(srcs.at(2) != nullptr, "Empty SrcC");
                    for(auto const& srcId : srcs.at(2)->getRegisterIds())
                    {
                        if(m_hazardMap->contains(srcId))
                        {
                            for(auto const& hazard : m_hazardMap->at(srcId))
                            {
                                if(hazard.regWasWritten() && trigger(hazard.getInstructionRef()))
                                {
                                    decrement = 2;
                                    if(instRef.isDGEMM())
                                    {
                                        decrement = 2;
                                    }
                                    else if(instRef.isSGEMM())
                                    {
                                        decrement = 3;
                                    }
                                    overlap      = true;
                                    requiredNops = hazard.getRequiredNops() - decrement;
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
                    return *value - decrement;
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value - decrement;
                }
            }
            else if(instRef.isVMEM() || instRef.isLDS() || instRef.isFlat())
            {
                return checkSrcs(inst).value_or(0) - decrement;
            }
            else if(instRef.isVALU())
            {
                std::optional<int> value;

                // VALU RAW
                if((value = checkSrcs(inst)))
                {
                    return *value - decrement;
                }

                // VALU WAW
                if((value = checkDsts(inst)))
                {
                    return *value - decrement;
                }
            }
            return 0;
        }
    }
}
