#pragma once

#include <string>

#include "ControlEdge_fwd.hpp"

namespace rocRoller
{

    namespace KernelGraph::ControlHypergraph
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
            std::string toString() const
            {
                return "Sequence";
            }
        };

        /**
         * Body edges indicate code nesting.  A Body node could indicate the body of a kernel,
         * a for loop, an unrolled section, an if statement (potentially), or any other
         * control block.
         */
        struct Body
        {
            std::string toString() const
            {
                return "Body";
            }
        };

        /**
         * Indicates code that should come before a Body edge.  Currently only applicable to
         * for loops.
         */
        struct Initialize
        {
            std::string toString() const
            {
                return "Initialize";
            }
        };

        /**
         * Indicates the increment node(s) of a for loop.
         */
        struct ForLoopIncrement
        {
            std::string toString() const
            {
                return "ForLoopIncrement";
            }
        };

        inline std::string toString(ControlEdge const& e)
        {
            return std::visit([](const auto& a) { return a.toString(); }, e);
        }
    }
}
