
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

        class Scheduler;

        class SequentialScheduler;
        class RoundRobinScheduler;
    }
}
