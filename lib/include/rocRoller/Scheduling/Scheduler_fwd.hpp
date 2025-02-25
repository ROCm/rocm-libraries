
#pragma once

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

        enum class Dependency
        {
            None = 0,
            SCC,
            VCC,
            Branch,
            Unlock,
            M0,
            Count
        };

        class Scheduler;
        class LockState;

        std::string toString(SchedulerProcedure const&);
        std::string toString(Dependency const&);
    }
}
