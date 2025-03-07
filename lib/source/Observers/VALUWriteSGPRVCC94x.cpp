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
#include <rocRoller/Scheduling/Observers/WaitState/VALUWriteSGPRVCC94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        int VALUWriteSGPRVCC94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VALUWriteSGPRVCC94x::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVCMP(inst.getOpCode())
                   || GPUInstructionInfo::isVReadlane(inst.getOpCode())
                   || GPUInstructionInfo::isVDivScale(inst.getOpCode())
                   || (GPUInstructionInfo::isVAddInst(inst.getOpCode())
                       && (GPUInstructionInfo::isIntInst(inst.getOpCode())
                           || GPUInstructionInfo::isUIntInst(inst.getOpCode())))
                   || (GPUInstructionInfo::isVSubInst(inst.getOpCode())
                       && (GPUInstructionInfo::isIntInst(inst.getOpCode())
                           || GPUInstructionInfo::isUIntInst(inst.getOpCode())));
        };

        int VALUWriteSGPRVCC94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVReadlane(inst.getOpCode())
               || GPUInstructionInfo::isVWritelane(inst.getOpCode()))
            {
                AssertFatal(inst.getSrcs().size() >= 2, "Unexpected instruction", inst.getOpCode());
                auto const& laneSelect = inst.getSrcs()[1];
                auto        val        = checkRegister(laneSelect);
                if(val.has_value()
                   && (laneSelect->regType() == Register::Type::Scalar
                       || laneSelect->regType() == Register::Type::VCC))
                {
                    return val.value();
                }
            }
            else
            {
                for(auto const& src : inst.getSrcs())
                {
                    auto val = checkRegister(src);
                    if(val.has_value()
                       && (src->regType() == Register::Type::Scalar
                           || src->regType() == Register::Type::VCC))
                    {
                        return val.value() - 2;
                    }
                }
            }

            return 0;
        }
    }
}
