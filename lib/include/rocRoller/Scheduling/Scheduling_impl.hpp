/**
 * @brief
 * @copyright Copyright 2021 Advanced Micro Devices, Inc.
 */

#pragma once

#include "Scheduling.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        inline InstructionStatus InstructionStatus::StallCycles(unsigned int value)
        {
            InstructionStatus rv;
            rv.stallCycles = value;
            return rv;
        }

        inline InstructionStatus InstructionStatus::UnscheduledDependencies(unsigned int value)
        {
            InstructionStatus rv;
            rv.unscheduledDependencies = value;
            return rv;
        }

        inline InstructionStatus InstructionStatus::Wait(WaitCount const& value)
        {
            InstructionStatus rv;
            rv.waitCount = value;
            return rv;
        }

        inline InstructionStatus InstructionStatus::Error(std::string const& msg)
        {
            InstructionStatus rv;
            rv.errors = {msg};
            return rv;
        }

        inline void InstructionStatus::combine(InstructionStatus const& other)
        {
            stallCycles = std::max(stallCycles, other.stallCycles);

            unscheduledDependencies += other.unscheduledDependencies;
            waitCount.combine(other.waitCount);

            errors.insert(errors.end(), other.errors.begin(), other.errors.end());
        }

        inline IObserver::~IObserver() = default;

    }
}
