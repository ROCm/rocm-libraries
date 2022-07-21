#pragma once

#include <string>

#include "ControlEdge_fwd.hpp"

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
        struct Sequence
        {
        };

        /**
         * Body edges indicate code nesting.  A Body node could indicate the body of a kernel,
         * a for loop, an unrolled section, an if statement (potentially), or any other
         * control block.
         */
        struct Body
        {
        };

        /**
         * Indicates code that should come before a Body edge.  Currently only applicable to
         * for loops.
         */
        struct Initialize
        {
        };

        /**
         * Indicates the increment node(s) of a for loop.
         */
        struct ForLoopIncrement
        {
        };

        std::string toString(ControlEdge const& e);
    }
}
