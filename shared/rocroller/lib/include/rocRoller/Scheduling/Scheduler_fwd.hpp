// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <string>

namespace rocRoller
{
    namespace Scheduling
    {
        enum class SchedulerProcedure : int
        {
            Sequential = 0,
            RoundRobin,
            Random,
            Cooperative,
            Priority,
            Count
        };

        enum class Dependency : int
        {
            None = 0, //< Temporary. Should only be used for unlocking.
            Branch, //< Non-preemptible: Loops and ConditionalOp
            M0, //< Preemptible: The M0 special-purpose register
            VCC, //< Preemptible: The VCC special-purpose register
            SCC, //< Non-preemptible: The SCC special-purpose register, which is
            //  implicitly written by many instructions.
            Count
        };

        enum class LockOperation : int
        {
            None = 0,
            Lock,
            Unlock,
            Count
        };

        class Scheduler;
        class LockState;

        using SchedulerPtr = std::shared_ptr<Scheduler>;

        std::string toString(SchedulerProcedure);
        std::string toString(Dependency);
        std::string toString(LockOperation);
    }
}
