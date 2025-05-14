/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2025 AMD ROCm(TM) Software
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

#pragma once

#include <rocRoller/Scheduling/Scheduling.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        inline InstructionStatus::InstructionStatus()
        {
            waitLengths.fill(0);
            allocatedRegisters.fill(0);
            remainingRegisters.fill(-1);
            highWaterMarkRegistersDelta.fill(0);
            outOfRegisters.reset();
        }

        inline InstructionStatus InstructionStatus::StallCycles(unsigned int const value)
        {
            InstructionStatus rv;
            rv.stallCycles = value;
            return rv;
        }

        inline InstructionStatus InstructionStatus::Wait(WaitCount const& value)
        {
            InstructionStatus rv;
            rv.waitCount = value;
            return rv;
        }

        inline InstructionStatus InstructionStatus::Nops(unsigned int const value)
        {
            InstructionStatus rv;
            rv.nops = value;
            return rv;
        }

        inline InstructionStatus InstructionStatus::Error(std::string const& msg)
        {
            InstructionStatus rv;
            rv.errors = {msg};
            return rv;
        }

        inline std::string InstructionStatus::toString() const
        {
            return concatenate("Status: {wait ",
                               waitCount.toString(LogLevel::Terse),
                               ", nop ",
                               nops,
                               ", reused ",
                               reusedOperands,
                               ", q {",
                               waitLengths,
                               "}, a {",
                               allocatedRegisters,
                               "}, h {",
                               highWaterMarkRegistersDelta,
                               "}}");
        }

        inline void InstructionStatus::combine(InstructionStatus const& other)
        {
            stallCycles = std::max(stallCycles, other.stallCycles);
            waitCount.combine(other.waitCount);
            nops = std::max(nops, other.nops);

            reusedOperands = std::max(reusedOperands, other.reusedOperands);

            for(int i = 0; i < waitLengths.size(); i++)
            {
                waitLengths[i] = std::max(waitLengths[i], other.waitLengths[i]);
            }

            for(int i = 0; i < allocatedRegisters.size(); i++)
            {
                allocatedRegisters[i]
                    = std::max(allocatedRegisters[i], other.allocatedRegisters[i]);
            }

            for(int i = 0; i < remainingRegisters.size(); i++)
            {
                if(remainingRegisters[i] < 0)
                    remainingRegisters[i] = other.remainingRegisters[i];
                else if(other.remainingRegisters[i] >= 0)
                    remainingRegisters[i]
                        = std::min(remainingRegisters[i], other.remainingRegisters[i]);
            }

            for(int i = 0; i < highWaterMarkRegistersDelta.size(); i++)
            {
                highWaterMarkRegistersDelta[i] = std::max(highWaterMarkRegistersDelta[i],
                                                          other.highWaterMarkRegistersDelta[i]);
            }

            outOfRegisters |= other.outOfRegisters;

            errors.insert(errors.end(), other.errors.begin(), other.errors.end());
        }

        inline IObserver::~IObserver() = default;

    }
}
