
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/RandomScheduler_fwd.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * Random Scheduler: Randomly picks the next instruction from one of the sequences until
         * all sequences are done.  Uses the random number generator stored in the context.
         *
         * The same random seed should produce the same program, regardless of addition or
         * removal of comment-only instructions. Respects the locking rules.
         */
        class RandomScheduler : public Scheduler
        {
        public:
            RandomScheduler(ContextPtr);

            using Base = Scheduler;

            static const std::string Name;

            /**
             * Returns true if `SchedulerProcedure` is Random
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `RandomScheduler` built from context
             */
            static std::shared_ptr<Scheduler> Build(Argument arg);

            /**
             * Return Name of `RandomScheduler`, used for debugging purposes currently
             */
            std::string name() const override;

            bool supportsAddingStreams() const override;

            /**
             * Call operator schedules instructions based on Sequential priority
             */
            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
