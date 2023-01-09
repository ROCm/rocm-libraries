#include <rocRoller/Scheduling/Observers/WaitState/DGEMM4x4x4Write.hpp>

#include <rocRoller/CodeGen/InstructionRef.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int DGEMM4x4x4Write::getMaxNops(std::shared_ptr<InstructionRef> inst) const
        {
            return m_maxNops;
        }

        bool DGEMM4x4x4Write::trigger(std::shared_ptr<InstructionRef> inst) const
        {
            return inst->getOpCode() == m_targetOpCode;
        };

        bool DGEMM4x4x4Write::writeTrigger() const
        {
            return true;
        }

        int DGEMM4x4x4Write::getNops(Instruction const& inst) const
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
                                    requiredNops = hazard.getRequiredNops() - (m_maxNops - 4);
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
                        if(!mismatched || (mismatched && instRef.isDGEMM()))
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
                    return *value - (m_maxNops - 6);
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value - (m_maxNops - 6);
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
                    return *value - (m_maxNops - 6);
                }

                // VALU WAW
                if((value = checkDsts(inst)))
                {
                    return *value - (m_maxNops - 6);
                }
            }
            return 0;
        }
    }
}
