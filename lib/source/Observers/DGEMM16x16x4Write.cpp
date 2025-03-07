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
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DGEMM16x16x4Write.hpp>

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

        int DGEMM16x16x4Write::getNops(Instruction const& inst) const
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
                        if(mismatched && GPUInstructionInfo::isDGEMM(inst.getOpCode()))
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
