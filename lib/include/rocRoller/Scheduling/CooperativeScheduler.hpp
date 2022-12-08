
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "CooperativeScheduler_fwd.hpp"
#include "Costs/Cost_fwd.hpp"
#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        /**
         * A subclass for cooperative scheduling
         *
         * This scheduler works by repeatedly choosing the next stream with
         * the minimum cost, and yielding from that stream until an
         * instruction with a non-zero cost is encountered.
         */
        class CooperativeScheduler : public Scheduler
        {
        public:
            CooperativeScheduler(std::shared_ptr<Context>, CostProcedure);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            /**
             * Returns true if `SchedulerProcedure` is Cooperative
             */
            static bool Match(Argument arg);

            /**
             * Return shared pointer of `CooperativeScheduler` built from context
             */
            static std::shared_ptr<Scheduler> Build(Argument arg);

            /**
             * Return Name of `CooperativeScheduler`, used for debugging purposes currently
             */
            std::string name() override;

            /**
             * Call operator schedules instructions based on the cooperative mechanism
             */
            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
