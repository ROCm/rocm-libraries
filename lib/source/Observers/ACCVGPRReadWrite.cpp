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
#include <rocRoller/Scheduling/Observers/WaitState/MFMA/ACCVGPRReadWrite.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int ACCVGPRReadWrite::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool ACCVGPRReadWrite::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isACCVGPRRead(inst.getOpCode());
        };

        int ACCVGPRReadWrite::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isMFMA(inst.getOpCode()))
            {
                auto const& srcs = inst.getSrcs();

                std::optional<int> value;

                // SrcA
                AssertFatal(srcs[0] != nullptr, "Empty SrcA");
                if((value = checkRegister(srcs[0])))
                {
                    return *value;
                }

                // ScrB
                AssertFatal(srcs[1] != nullptr, "Empty SrcB");
                if((value = checkRegister(srcs[1])))
                {
                    return *value;
                }
            }
            else if(GPUInstructionInfo::isACCVGPRWrite(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            else if(GPUInstructionInfo::isVMEM(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0);
            }
            return 0;
        }
    }
}
