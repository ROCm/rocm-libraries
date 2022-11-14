
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Scheduler.hpp"
#include "SequentialScheduler_fwd.hpp"

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * A subclass for Sequential scheduling
         */
        class SequentialScheduler : public Scheduler
        {
        public:
            SequentialScheduler(std::shared_ptr<Context>);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `SchedulerProcedure` is Sequential
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `SequentialScheduler` built from context
             */
            static std::shared_ptr<Scheduler> Build(Argument arg);

            /**
             * Return Name of `SequentialScheduler`, used for debugging purposes currently
             */
            virtual std::string name() override;

            /**
             * Call operator schedules instructions based on Sequential priority
             */
            virtual Generator<Instruction>
                operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
