
#include <rocRoller/Expression.hpp>

#include "KernelGraph/KernelGraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Create a range-based for loop.
         */
        std::pair<int, int> rangeFor(KernelHypergraph& graph, Expression::ExpressionPtr size);
    }
}
