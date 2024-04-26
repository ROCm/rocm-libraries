#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM4x4x4Write.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int DGEMM4x4x4Write::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool DGEMM4x4x4Write::trigger(Instruction const& inst) const
        {
            return inst.getOpCode() == m_targetOpCode;
        };

        bool DGEMM4x4x4Write::writeTrigger() const
        {
            return true;
        }

        int DGEMM4x4x4Write::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isMFMA(inst.getOpCode()))
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
                        if(!mismatched
                           || (mismatched && GPUInstructionInfo::isDGEMM(inst.getOpCode())))
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
            else if(GPUInstructionInfo::isVMEM(inst.getOpCode())
                    || GPUInstructionInfo::isLDS(inst.getOpCode())
                    || GPUInstructionInfo::isFlat(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(GPUInstructionInfo::isVALU(inst.getOpCode()))
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
