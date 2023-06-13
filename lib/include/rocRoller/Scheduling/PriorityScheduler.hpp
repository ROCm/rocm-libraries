
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
            PriorityScheduler(ContextPtr, CostFunction);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            static bool Match(Argument arg);

            static std::shared_ptr<Scheduler> Build(Argument arg);

            std::string name() const override;

            virtual bool supportsAddingStreams() const override;

            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
