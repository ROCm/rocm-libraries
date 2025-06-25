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

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitecture.hpp>
#include <rocRoller/GPUArchitecture/GPUInstructionInfo.hpp>
#include <rocRoller/Scheduling/Observers/FunctionalUnit/WMMAObserver.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        WMMAObserver::WMMAObserver() {}

        WMMAObserver::WMMAObserver(ContextPtr ctx)
            : m_context(ctx)
        {
        }

        bool WMMAObserver::isWMMAInstruction(const Instruction& inst) const
        {
            return GPUInstructionInfo::isWMMA(inst.getOpCode());
        }

        InstructionStatus WMMAObserver::peek(const Instruction& inst) const
        {
            InstructionStatus rv;
            if(isWMMAInstruction(inst))
                rv.stallCycles = m_remainingCycles;
            return rv;
        }

        void WMMAObserver::modify(Instruction& inst) const
        {
            if(m_remainingCycles > 0 && !inst.isCommentOnly()
               && Settings::Get(Settings::LogLvl) >= LogLevel::Debug)
                inst.addComment(concatenate("WMMA remaining: ", m_remainingCycles));
        }

        void WMMAObserver::observe(const Instruction& inst)
        {
            if(isWMMAInstruction(inst))
            {
                auto info
                    = m_context.lock()->targetArchitecture().GetInstructionInfo(inst.getOpCode());

                m_remainingCycles = info.getLatency();
            }
            else
            {
                m_remainingCycles = std::max(0, m_remainingCycles - inst.numExecutedInstructions());
            }
        }

    }

}
