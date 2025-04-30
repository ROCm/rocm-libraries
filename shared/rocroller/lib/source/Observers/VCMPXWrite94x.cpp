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
#include <rocRoller/Scheduling/Observers/WaitState/VCMPXWrite94x.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        void VCMPXWrite94x::observeHazard(Instruction const& inst)
        {
            if(trigger(inst))
            {
                for(auto const& regId : m_context.lock()->getExec()->getRegisterIds())
                {
                    (*m_hazardMap)[regId].push_back(
                        WaitStateHazardCounter(getMaxNops(inst), writeTrigger()));
                }
            }
        }

        int VCMPXWrite94x::getMaxNops(Instruction const& inst) const
        {
            return m_maxNops;
        }

        bool VCMPXWrite94x::trigger(Instruction const& inst) const
        {
            return GPUInstructionInfo::isVCMPX(inst.getOpCode());
        };

        int VCMPXWrite94x::getNops(Instruction const& inst) const
        {
            if(GPUInstructionInfo::isVReadlane(inst.getOpCode())
               || GPUInstructionInfo::isVWritelane(inst.getOpCode()))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }

            if(GPUInstructionInfo::isVPermlane(inst.getOpCode()))
            {
                return checkRegister(m_context.lock()->getExec()).value_or(0);
            }

            // Check if VALU reads EXEC as constant
            if(GPUInstructionInfo::isVALU(inst.getOpCode()))
            {
                return checkSrcs(inst).value_or(0) - 2;
            }
            return 0;
        }
    }
}
