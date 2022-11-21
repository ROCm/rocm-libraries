
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
            LockState(ContextPtr ctx);
            LockState(ContextPtr ctx, Dependency dependency);

            void add(Instruction const& instr);
            bool isLocked() const;
            void isValid(bool locked = false) const;

            /**
             * @brief Extra checks to verify lock state integrity.
             *
             * Note: disabled in Release mode.
             *
             * @param instr The instruction to verify
             */
            void lockCheck(Instruction const& instr);

            Dependency getDependency() const;
            int        getLockDepth() const;

        private:
            int                               m_lockdepth;
            Dependency                        m_dependency;
            std::weak_ptr<rocRoller::Context> m_ctx;
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

            Scheduler(ContextPtr);

            static const std::string Name;
            static const bool        SingleUse = true;

            virtual std::string            name()                                           = 0;
            virtual Generator<Instruction> operator()(std::vector<Generator<Instruction>>&) = 0;

            LockState getLockState() const;

        protected:
            LockState                         m_lockstate;
            std::weak_ptr<rocRoller::Context> m_ctx;
        };

        std::ostream& operator<<(std::ostream&, SchedulerProcedure);
    }
}

#include "Scheduler_impl.hpp"
