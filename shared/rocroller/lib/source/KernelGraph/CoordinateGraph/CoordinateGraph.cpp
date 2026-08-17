// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <vector>

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>

#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateEdgeVisitor.hpp>

#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{

    namespace KernelGraph::CoordinateGraph
    {

        std::vector<Expression::ExpressionPtr>
            CoordinateGraph::forward(std::vector<Expression::ExpressionPtr> sdims,
                                     std::vector<int> const&                srcs,
                                     std::vector<int> const&                dsts) const
        {
            AssertFatal(sdims.size() == srcs.size(), ShowValue(sdims));
            auto visitor = ForwardEdgeVisitor();
            return traverse<Graph::Direction::Downstream>(sdims, srcs, dsts, visitor);
        }

        std::vector<Expression::ExpressionPtr>
            CoordinateGraph::reverse(std::vector<Expression::ExpressionPtr> sdims,
                                     std::vector<int> const&                srcs,
                                     std::vector<int> const&                dsts) const
        {
            AssertFatal(sdims.size() == dsts.size(), ShowValue(sdims));
            auto visitor = ReverseEdgeVisitor();
            return traverse<Graph::Direction::Upstream>(sdims, srcs, dsts, visitor);
        }

        bool CoordinateGraph::isModificationAllowed(int index) const
        {
            if(not Settings::getInstance()->get(Settings::EnforceGraphConstraints))
                return true;

            if(not m_changesRestricted)
                return true;

            auto const& el = getElement(index);

            if(std::holds_alternative<Edge>(el))
            {
                return std::visit(
                    [](auto&& arg) {
                        using EdgeType = std::decay_t<decltype(arg)>;
                        return !(std::is_same_v<EdgeType, CoordinateTransformEdge>);
                    },
                    std::get<Edge>(el));
            }
            return true;
        }
    }
}
