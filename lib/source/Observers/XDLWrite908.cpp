#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite908.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int XDLWrite908::getMaxNops(Instruction const& inst) const
        {
            return getNopFromLatency(inst.getOpCode(), m_maxLatencyAndNops);
        }

        bool XDLWrite908::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isMFMA(inst.getOpCode());
        };

        int XDLWrite908::getNops(Instruction const& inst) const
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
                                    overlap = true;
                                    requiredNops
                                        = hazard.getRequiredNops() - hazard.getMaxNops() + 2;
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
                    if(m_hazardMap->contains(srcId))
                    {
                        for(auto const& hazard : m_hazardMap->at(srcId))
                        {
                            if(hazard.regWasWritten())
                            {
                                return hazard.getRequiredNops() - (hazard.getMaxNops() - 4);
                            }
                        }
                    }
                }

                // SrcB RAW
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                for(auto const& srcId : srcs.at(1)->getRegisterIds())
                {
                    if(m_hazardMap->contains(srcId))
                    {
                        for(auto const& hazard : m_hazardMap->at(srcId))
                        {
                            if(hazard.regWasWritten())
                            {
                                return hazard.getRequiredNops() - (hazard.getMaxNops() - 4);
                            }
                        }
                    }
                }
            }
            else if(GPUInstructionInfo::isACCVGPRRead(inst.getOpCode()))
            {
                // ACCVGPR RAW
                return checkSrcs(inst).value_or(0);
            }
            else if(GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode()))
            {
                // ACCVGPR WAW
                return checkDsts(inst).value_or(0) - 3;
            }
            return 0;
        }
    }
}
