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
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/DLWrite.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void DLWrite::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                m_prevOpCode = inst.getOpCode();
                for(auto iter = (writeTrigger() ? inst.getDsts().begin() : inst.getSrcs().begin());
                    iter != (writeTrigger() ? inst.getDsts().end() : inst.getSrcs().end());
                    iter++)
                {
                    auto reg = *iter;
                    if(reg)
                    {
                        for(auto const& regId : reg->getRegisterIds())
                        {
                            (*m_hazardMap)[regId].push_back(
                                WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                        }
                    }
                }
            }
        }

        int DLWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool DLWrite::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isDLOP(inst.getOpCode());
        };

        int DLWrite::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isDLOP(inst.getOpCode()))
            {
                std::optional<int> value;

                auto const& srcs = inst.getSrcs();

                // SrcC
                AssertFatal(srcs.at(2) != nullptr, "Empty SrcC");
                for(auto const& srcId : srcs.at(2)->getRegisterIds())
                {
                    if(m_hazardMap->contains(srcId))
                    {
                        for(auto const& hazard : m_hazardMap->at(srcId))
                        {
                            if(hazard.regWasWritten() && inst.getOpCode() == m_prevOpCode)
                            {
                                // Supports same opcode of DLops back-to-back SrcC forwarding which is used for accumulation
                                return 0;
                            }
                        }
                    }
                }

                // SrcA
                AssertFatal(srcs.at(0) != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    return *value;
                }

                // SrcB
                AssertFatal(srcs.at(1) != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value;
                }
            }

            // If the opcode is different
            {
                std::optional<int> value;

                // RAW
                if((value = checkSrcs(inst)))
                {
                    return *value;
                }

                // WAW
                if((value = checkDsts(inst)))
                {
                    return *value;
                }
            }
            return 0;
        }
    }
}
