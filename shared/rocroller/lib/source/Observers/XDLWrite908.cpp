/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
