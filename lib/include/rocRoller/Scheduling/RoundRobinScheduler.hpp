
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
         * Round-robin scheduler: Takes the first instruction from each stream, then the
         * second from each stream, and so on.
         *
         * Must also follow the locking rules.
         */
        class RoundRobinScheduler : public Scheduler
        {
        public:
            RoundRobinScheduler(std::shared_ptr<Context>);

            using Base = Scheduler;

            static const std::string Name;

            static bool Match(Argument arg);

            static std::shared_ptr<Scheduler> Build(Argument arg);

            std::string name() const override;

            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
