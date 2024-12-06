
#pragma once

#include <rocRoller/Scheduling/Costs/Cost.hpp>

namespace rocRoller
{
    namespace Scheduling
    {
        static_assert(Component::ComponentBase<Cost>);

        inline Cost::Cost(ContextPtr ctx)
            : m_ctx{ctx}
        {
        }
    }
}
