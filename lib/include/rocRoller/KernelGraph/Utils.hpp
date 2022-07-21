
#include <rocRoller/Expression.hpp>

#include "KernelGraph/KernelGraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Create a range-based for loop.
         */
        std::pair<CoordinateTransform::ForLoop, ControlGraph::ForLoopOp>
            rangeFor(CoordinateTransform::HyperGraph& coordGraph,
                     ControlGraph::ControlGraph&      controlGraph,
                     Expression::ExpressionPtr        size);
    }
}
