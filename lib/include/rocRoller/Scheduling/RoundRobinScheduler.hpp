
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "RoundRobinScheduler_fwd.hpp"
#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * A subclass for round robin scheduling
         */
        class RoundRobinScheduler : public Scheduler
        {
        public:
            RoundRobinScheduler(std::shared_ptr<Context>);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `SchedulerProcedure` is RoundRobin
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `RoundRobinScheduler` built from context
             */
            static std::shared_ptr<Scheduler> Build(Argument arg);

            /**
             * Return Name of `RoundRobinScheduler`, used for debugging purposes currently
             */
            std::string name() override;

            /**
             * Call operator schedules instructions based on the round robin priority
             */
            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
