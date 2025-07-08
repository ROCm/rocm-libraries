/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/WMMAWrite.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int WMMAWrite::getMaxNops(const Instruction& inst) const
        {
            return m_maxNops;
        }

        bool WMMAWrite::trigger(const Instruction& inst) const
        {
            return GPUInstructionInfo::isWMMA(inst.getOpCode());
        };

        int WMMAWrite::getNops(const Instruction& inst) const
        {
            if(GPUInstructionInfo::isWMMA(inst.getOpCode()))
            {
                const auto&        srcs = inst.getSrcs();
                std::optional<int> value;
                // SrcA RAW
                AssertFatal(nullptr != srcs.at(0), "Empty SrcA");
                if((value = checkRegister(srcs.at(0))))
                {
                    return *value;
                }

                // SrcB RAW
                AssertFatal(nullptr != srcs.at(1), "Empty SrcB");
                if((value = checkRegister(srcs.at(1))))
                {
                    return *value;
                }
            }

            return 0;
        }
    }
}
