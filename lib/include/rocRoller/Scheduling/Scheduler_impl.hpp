
#pragma once

#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::ComponentBase<Scheduler>);

        template <typename Begin, typename End>
        Generator<Instruction> consumeComments(Begin& begin, End const& end)
        {
            while(begin != end && begin->isCommentOnly())
            {
                co_yield *begin;
                ++begin;
            }
        }

        inline LockState::LockState()
            : m_dependency(Dependency::None)
            , m_lockdepth(0)
        {
        }

        inline LockState::LockState(Dependency dependency)
            : m_dependency(dependency)
        {
            AssertFatal(m_dependency != Scheduling::Dependency::Count
                            && m_dependency != Scheduling::Dependency::Unlock,
                        "Can not instantiate LockState with Count or Unlock dependency");

            m_lockdepth = 1;
        }

        inline void LockState::add(Instruction const& instruction)
        {
            int inst_lockvalue = instruction.getLockValue();

            // Instruction does not lock or unlock, do nothing
            if(inst_lockvalue == 0)
            {
                return;
            }

            // Instruction can only lock (1) or unlock (-1)
            if(inst_lockvalue != -1 && inst_lockvalue != 1)
            {
                Throw<FatalError>("Invalid instruction lockstate: ", ShowValue(inst_lockvalue));
            }

            // Instruction trying to unlock when there is no lock
            if(m_lockdepth == 0 && inst_lockvalue == -1)
            {
                Throw<FatalError>("Trying to unlock when not locked");
            }

            // Instruction initializes the lockstate
            if(m_lockdepth == 0)
            {
                m_dependency = instruction.getDependency();
            }

            m_lockdepth += inst_lockvalue;

            // Instruction releases lock
            if(m_lockdepth == 0)
            {
                m_dependency = Scheduling::Dependency::None;
            }
        }

        inline bool LockState::isLocked() const
        {
            return m_lockdepth > 0;
        }

        // Will grow into a function that accepts args and checks the lock is in a valid state against those args
        inline void LockState::isValid(bool locked) const
        {
            AssertFatal(isLocked() == locked, "Lock in invalid state");
        }

        inline Dependency LockState::getDependency() const
        {
            return m_dependency;
        }

        inline int LockState::getLockDepth() const
        {
            return m_lockdepth;
        }

        inline LockState Scheduler::getLockState() const
        {
            return m_lockstate;
        }
    }
}
