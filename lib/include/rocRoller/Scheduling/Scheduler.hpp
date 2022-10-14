
#pragma once

#include <concepts>
#include <string>
#include <vector>

#include "Scheduler_fwd.hpp"

#include "../CodeGen/Instruction.hpp"
#include "../Context_fwd.hpp"
#include "../Utilities/Component.hpp"
#include "../Utilities/Generator.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        class LockState
        {
        public:
            LockState();
            LockState(Dependency dependency);

            void add(Instruction const& instr);
            bool isLocked() const;
            void isValid(bool locked = false) const;

            Dependency getDependency() const;
            int        getLockDepth() const;

        private:
            int        m_lockdepth;
            Dependency m_dependency;
        };

        /**
         * Yields from the beginning of the range [begin, end) any comment-only instruction(s).
         */
        template <typename Begin, typename End>
        Generator<Instruction> consumeComments(Begin& begin, End const& end);

        /**
         * A `Scheduler` is a base class for the different types of schedulers
         *
         * - This class should be able to be made into `ComponentBase` class
         */
        class Scheduler
        {
        public:
            using Argument = std::tuple<SchedulerProcedure, std::shared_ptr<rocRoller::Context>>;

            static const std::string Name;

            virtual std::string            name()                                           = 0;
            virtual Generator<Instruction> operator()(std::vector<Generator<Instruction>>&) = 0;

            LockState getLockState() const;

        protected:
            LockState m_lockstate;
        };

        /**
         * A subclass for Sequential scheduling
         *
         * - This class should be able to be made into `Component` class
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

        private:
            std::weak_ptr<rocRoller::Context> m_ctx;
        };

        /**
         * A subclass for round robin scheduling
         *
         * - This class should be able to be made into `Component` class
         */
        class RoundRobinScheduler : public Scheduler
        {
        public:
            RoundRobinScheduler(std::shared_ptr<Context>);

            using Base = Scheduler;

            static const std::string Basename;
            static const std::string Name;

            /*
                * Returns true if `SchedulerProcedure` is RoundRobin
                */
            static bool Match(Argument arg);

            /*
                * Return shared pointer of `RoundRobinScheduler` built from context
                */
            static std::shared_ptr<Scheduler> Build(Argument arg);

            /*
                * Return Name of `RoundRobinScheduler`, used for debugging purposes currently
                */
            virtual std::string name() override;

            /*
                * Call operator schedules instructions based on the round robin priority
                */
            virtual Generator<Instruction>
                operator()(std::vector<Generator<Instruction>>& seqs) override;

        private:
            std::weak_ptr<rocRoller::Context> m_ctx;
        };

        std::string   toString(SchedulerProcedure);
        std::ostream& operator<<(std::ostream&, SchedulerProcedure);
    }
}

#include "Scheduler_impl.hpp"
