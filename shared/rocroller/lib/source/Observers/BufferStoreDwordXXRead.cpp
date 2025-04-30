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
#include <rocRoller/Scheduling/Observers/WaitState/BufferStoreDwordXXRead.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int BufferStoreDwordXXRead::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool BufferStoreDwordXXRead::trigger(Instruction const& inst) const
        {
            if(inst.getOpCode().rfind("buffer_store_dwordx3", 0) == 0
               || inst.getOpCode().rfind("buffer_store_dwordx4", 0) == 0)
            {
                AssertFatal(inst.getSrcs().size() != 4,
                            "Unexpected arguments",
                            ShowValue(inst.toString(LogLevel::Debug)));
                return inst.getSrcs()[3]->regType() != Register::Type::Scalar;
            }
            return false;
        };

        int BufferStoreDwordXXRead::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode()) && m_isCDNA1orCDNA2)
            {
                return checkDsts(inst).value_or(0) - 1;
            }
            return checkDsts(inst).value_or(0);
        }
    }
}
