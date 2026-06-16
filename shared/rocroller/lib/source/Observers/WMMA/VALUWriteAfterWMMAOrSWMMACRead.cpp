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
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Scheduling/Observers/WaitState/WMMA/VALUWriteAfterWMMAOrSWMMACRead.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteAfterWMMAOrSWMMACRead::getMaxNops(const Instruction& inst) const
        {
            const auto& opCode = inst.getOpCode();
            AssertFatal(
                GPUInstructionInfo::isWMMA(opCode), "Expected WMMA instruction but found ", opCode);

            const auto isF8
                = [](auto type) { return type == DataType::FP8x4 || type == DataType::BF8x4; };
            const auto& srcAType = inst.getSrcs().at(0)->variableType();
            const auto& srcBType = inst.getSrcs().at(1)->variableType();

            const auto& ctx         = m_context.lock();
            auto        coexecSlots = 0;
            if(ctx->kernelOptions()->coexecutionEnabled)
            {
                coexecSlots = 4;
                if(opCode.ends_with("_iu8") || opCode.ends_with("_iu4")
                   || (opCode.ends_with("_f8f6f4") && (isF8(srcAType) || isF8(srcBType)))
                   || opCode.ends_with("_f4"))
                {
                    coexecSlots = 8;
                }
            }
            return m_maxNops + coexecSlots;
        }

        bool VALUWriteAfterWMMAOrSWMMACRead::trigger(const Instruction& inst) const
        {
            const auto& opCode = inst.getOpCode();
            if(GPUInstructionInfo::isWMMA(opCode))
            {
                return opCode.ends_with("_f16") || opCode.ends_with("_bf16")
                       || opCode.ends_with("_fp8_fp8") || opCode.ends_with("_fp8_bf8")
                       || opCode.ends_with("_bf8_fp8") || opCode.ends_with("_bf8_bf8")
                       || opCode.ends_with("_f8f6f4") || opCode.ends_with("_iu8")
                       || opCode.ends_with("_iu4") || opCode.ends_with("_f4");
            }
            return false;
        };

        int VALUWriteAfterWMMAOrSWMMACRead::getNops(const Instruction& inst) const
        {
            if(GPUInstructionInfo::isVALU(inst.getOpCode())
               && !GPUInstructionInfo::isWMMA(inst.getOpCode())
               && !GPUInstructionInfo::isSWMMAC(inst.getOpCode()))
            {
                std::optional<int> value;

                // WAR
                if((value = checkDsts(inst)))
                {
                    return *value;
                }
            }
            return 0;
        }
    }
}
