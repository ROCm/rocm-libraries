/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

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
