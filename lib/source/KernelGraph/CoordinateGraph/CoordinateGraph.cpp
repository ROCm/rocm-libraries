
#include <vector>

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdgeVisitor.hpp>

namespace rocRoller
{

    namespace KernelGraph::CoordinateGraph
    {

        std::vector<Expression::ExpressionPtr>
            CoordinateGraph::forward(std::vector<Expression::ExpressionPtr> sdims,
                                     std::vector<int> const&                srcs,
                                     std::vector<int> const&                dsts,
                                     Expression::ExpressionTransducer       transducer)
        {
            AssertFatal(sdims.size() == srcs.size(), ShowValue(sdims));
            auto visitor = ForwardEdgeVisitor();
            return traverse<Graph::Direction::Downstream>(sdims, srcs, dsts, visitor, transducer);
        }

        std::vector<Expression::ExpressionPtr>
            CoordinateGraph::reverse(std::vector<Expression::ExpressionPtr> sdims,
                                     std::vector<int> const&                srcs,
                                     std::vector<int> const&                dsts,
                                     Expression::ExpressionTransducer       transducer)
        {
            AssertFatal(sdims.size() == dsts.size(), ShowValue(sdims));
            auto visitor = ReverseEdgeVisitor();
            return traverse<Graph::Direction::Upstream>(sdims, srcs, dsts, visitor, transducer);
        }

    }
}
