#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/XDLWrite94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int XDLWrite94x::getMaxNops(Instruction const& inst) const
        {
            auto const& architecture = m_context.lock()->targetArchitecture();
            int         passes = architecture.GetInstructionInfo(inst.getOpCode()).getLatency();
            bool        is950  = architecture.target().is950GPU();

            AssertFatal(m_latencyAndNops.contains(passes),
                        "Unexpected number of passes",
                        ShowValue(architecture.target().toString()),
                        ShowValue(inst.getOpCode()),
                        ShowValue(passes));

            int adjustFor950 = (is950 && passes > 2) ? 1 : 0;

            return m_latencyAndNops.at(passes) + adjustFor950;
        }

        bool XDLWrite94x::trigger(Instruction const& inst) const
        {
            bool excluded
                = std::find(m_excludedOpCodes.begin(), m_excludedOpCodes.end(), inst.getOpCode())
                  != m_excludedOpCodes.end();
            return GPUInstructionInfo::isMFMA(inst.getOpCode()) && !excluded;
        };

        int XDLWrite94x::getNops(Instruction const& inst) const
        {
            int decrement = 0;

            if(GPUInstructionInfo::isSGEMM(inst.getOpCode()))
            {
                decrement = 1;
            }

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
                                    decrement = 2;
                                    if(GPUInstructionInfo::isDGEMM(inst.getOpCode()))
                                    {
                                        decrement = 2;
                                    }
                                    else if(GPUInstructionInfo::isSGEMM(inst.getOpCode()))
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
                        auto const& architecture = m_context.lock()->targetArchitecture();
                        int passes = architecture.GetInstructionInfo(inst.getOpCode()).getLatency();
                        bool is950 = architecture.target().is950GPU();

                        if(mismatched)
                        {
                            int adjustFor950 = (is950 && passes == 2) ? 1 : 0;
                            return requiredNops + adjustFor950;
                        }
                        else
                        {
                            // Assumes MFMA trigger instruction and this current instruction have same number of passes
                            return passes > 2 ? 0 : 2;
                        }
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
            else if(GPUInstructionInfo::isVMEM(inst.getOpCode())
                    || GPUInstructionInfo::isLDS(inst.getOpCode())
                    || GPUInstructionInfo::isFlat(inst.getOpCode()))
            {
                std::optional<int> value = checkSrcs(inst);
                if(value)
                    return *value - decrement;
            }
            else if(GPUInstructionInfo::isVALU(inst.getOpCode()))
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
