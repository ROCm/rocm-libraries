/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
