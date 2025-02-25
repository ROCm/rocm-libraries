
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/CooperativeScheduler_fwd.hpp>
#include <rocRoller/Scheduling/Costs/Cost_fwd.hpp>
#include <rocRoller/Scheduling/Scheduler.hpp>

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
            CooperativeScheduler(ContextPtr, CostFunction);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            static bool Match(Argument arg);

            static std::shared_ptr<Scheduler> Build(Argument arg);

            std::string name() const override;

            bool supportsAddingStreams() const override;

            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
