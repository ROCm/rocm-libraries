#pragma once

#include <string>

#include "ControlEdge_fwd.hpp"

#include <rocRoller/KernelGraph/StructUtils.hpp>

namespace rocRoller
{

    namespace KernelGraph::ControlGraph
    {
        /**
         * Control graph edge types
         */

        /**
         * Sequence edges indicate sequential dependencies.  A node is considered schedulable
         * if all of its incoming Sequence edges have been scheduled.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Sequence);

        /**
         * Body edges indicate code nesting.  A Body node could indicate the body of a kernel,
         * a for loop, an unrolled section, an if statement (for the true branch), or any other
         * control block.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Body);

        /**
         * Else edge indicates the code that should be emmitted given a false condition. 
         */
        RR_EMPTY_STRUCT_WITH_NAME(Else);

        /**
         * Indicates code that should come before a Body edge.  Currently only applicable to
         * for loops.
         */
        RR_EMPTY_STRUCT_WITH_NAME(Initialize);

        /**
         * Indicates the increment node(s) of a for loop.
         */
        RR_EMPTY_STRUCT_WITH_NAME(ForLoopIncrement);

        inline std::string toString(ControlEdge const& e)
        {
            return std::visit([](const auto& a) { return a.toString(); }, e);
        }

        template <CConcreteControlEdge Edge>
        inline std::string toString(Edge const& e)
        {
            return e.toString();
        }

        inline std::string name(ControlEdge const& e)
        {
            return toString(e);
        }
    }
}
