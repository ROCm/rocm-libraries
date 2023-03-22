
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Costs/Cost_fwd.hpp"
#include "PriorityScheduler_fwd.hpp"
#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * A subclass for Priority scheduling
         *
         * This scheduler works by repeatedly yielding the next
         * instruction from the the lowest index stream with the
         * least cost.
         */
        class PriorityScheduler : public Scheduler
        {
        public:
            PriorityScheduler(std::shared_ptr<Context>, CostFunction);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `SchedulerProcedure` is Priority
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `PriorityScheduler` built from context
             */
            static std::shared_ptr<Scheduler> Build(Argument arg);

            /**
             * Return Name of `PriorityScheduler`, used for debugging purposes currently
             */
            std::string name() override;

            /**
             * Call operator schedules instructions based on the Priority mechanism
             */
            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
