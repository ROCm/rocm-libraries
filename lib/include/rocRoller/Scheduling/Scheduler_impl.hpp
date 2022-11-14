
#pragma once

#include "Scheduler.hpp"

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::ComponentBase<Scheduler>);

        inline Scheduler::Scheduler(ContextPtr ctx)
            : m_ctx{ctx}
            , m_lockstate{ctx}
        {
        }

        template <typename Begin, typename End>
        Generator<Instruction> consumeComments(Begin& begin, End const& end)
        {
            while(begin != end && begin->isCommentOnly())
            {
                co_yield *begin;
                ++begin;
            }
        }

        inline LockState Scheduler::getLockState() const
        {
            return m_lockstate;
        }
    }
}
