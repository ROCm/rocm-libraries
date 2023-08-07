
#pragma once

#include <map>

#include "Expression_fwd.hpp"
#include "KernelGraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        class GraphReindexer
        {
        public:
            std::map<int, int> coordinates;
            std::map<int, int> control;
        };

        /**
         * Transducer for reindexing expressions via GraphReindexer.
         *
         * Usage:
         *
         *   GraphReindexer reindexer;
         *   ...
         *   auto newExpr = reindexExpression(oldExpr, reindexer);
         *
         */
        Expression::ExpressionPtr reindexExpression(Expression::ExpressionPtr expr,
                                                    GraphReindexer const&     reindexer);

        void reindexExpressions(KernelGraph& graph, int tag, GraphReindexer const& reindexer);
    }
}
