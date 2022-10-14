
#pragma once

namespace rocRoller
{
    namespace Scheduling
    {
        enum class SchedulerProcedure : int
        {
            Sequential = 0,
            RoundRobin,
            Cooperative,
            Priority,
            Count
        };

        enum Dependency
        {
            None = 0,
            SCC,
            VCC,
            Branch,
            Unlock,
            Count
        };

        class Scheduler;

        class SequentialScheduler;
        class RoundRobinScheduler;
    }
}
