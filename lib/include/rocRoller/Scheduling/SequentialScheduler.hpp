
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Scheduling/SequentialScheduler_fwd.hpp>

namespace rocRoller
{
    namespace Scheduling
    {

        /**
         * A subclass for Sequential scheduling
         *
         * Takes every instruction from the first stream, then every instruction
         * from the second stream, and so on.
         *
         * Supports adding streams.  This scheduler will also preemptively take
         * comments from the beginning of new streams, for the purpose of finding
         * and running Deallocate nodes as soon as they are available.
         */
        class SequentialScheduler : public Scheduler
        {
        public:
            SequentialScheduler(ContextPtr);

            using Base = Scheduler;

            static const std::string Name;

            static bool Match(Argument arg);

            static std::shared_ptr<Scheduler> Build(Argument arg);

            /**
             * Return Name of `SequentialScheduler`, used for debugging purposes currently
             */
            std::string name() const override;

            bool supportsAddingStreams() const override;

            Generator<Instruction> operator()(std::vector<Generator<Instruction>>& seqs) override;
        };
    }
}
