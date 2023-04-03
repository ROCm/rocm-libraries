
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
         * Takes every instruction from the first stream, then every instruction from the second stream, and so on.
         */
        class SequentialScheduler : public Scheduler
        {
        public:
            SequentialScheduler(std::shared_ptr<Context>);

            using Base = Scheduler;

            static const std::string Name;

            static bool Match(Argument arg);

            static std::shared_ptr<Scheduler> Build(Argument arg);

            std::string name() override;

            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
