#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite90a.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>
namespace rocRoller
{
    namespace Scheduling
    {
        int XDLWrite90a::getMaxNops(Instruction const& inst) const
        {
            return getNopFromLatency(inst.getOpCode(), m_latencyAndNops);
        }

        bool XDLWrite90a::trigger(Instruction const& inst) const
        {
            bool excluded
                = std::find(m_excludedOpCodes.begin(), m_excludedOpCodes.end(), inst.getOpCode())
                  != m_excludedOpCodes.end();
            return InstructionRef::isMFMA(inst.getOpCode()) && !excluded;
        };

        bool XDLWrite90a::writeTrigger() const
        {
            return true;
        }

        int XDLWrite90a::getNops(Instruction const& inst) const
        {
            if(InstructionRef::isMFMA(inst.getOpCode()))
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
                                if(hazard.regWasWritten())
                                {
                                    int decrement
                                        = InstructionRef::isDGEMM(inst.getOpCode()) ? 2 : 3;
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
                    return *value;
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value;
                }
            }
            else if(InstructionRef::isVMEM(inst.getOpCode())
                    || InstructionRef::isLDS(inst.getOpCode())
                    || InstructionRef::isFlat(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(InstructionRef::isVALU(inst.getOpCode()))
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
