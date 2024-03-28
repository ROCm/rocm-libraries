#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM16x16x4Write.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int DGEMM16x16x4Write::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool DGEMM16x16x4Write::trigger(Instruction const& inst) const
        {
            for(auto targetOpCode : m_targetOpCodes)
            {
                if(inst.getOpCode() == targetOpCode)
                    return true;
            }
            return false;
        };

        bool DGEMM16x16x4Write::writeTrigger() const
        {
            return true;
        }

        int DGEMM16x16x4Write::getNops(Instruction const& inst) const
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

                    for(auto const& srcId : srcs[2]->getRegisterIds())
                    {
                        if(m_hazardMap->contains(srcId))
                        {
                            for(auto const& hazard : m_hazardMap->at(srcId))
                            {
                                if(hazard.regWasWritten())
                                {
                                    overlap      = true;
                                    requiredNops = hazard.getRequiredNops() - (m_maxNops - 9);
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
                        if(mismatched && InstructionRef::isDGEMM(inst.getOpCode()))
                        {
                            return requiredNops;
                        }
                        else
                        {
                            return 0;
                        }
                    }
                }

                // SrcA RAW
                AssertFatal(srcs.at(0) != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    return *value - (m_maxNops - 11);
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value - (m_maxNops - 11);
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
                    return *value - (m_maxNops - 11);
                }

                // VALU WAW
                if((value = checkDsts(inst)))
                {
                    return *value - (m_maxNops - 11);
                }
            }
            return 0;
        }
    }
}
