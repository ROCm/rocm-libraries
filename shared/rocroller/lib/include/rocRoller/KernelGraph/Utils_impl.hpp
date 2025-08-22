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

#pragma once

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
    /**
     * @brief Return DataFlowTag of LHS of binary expression in Assign node.
     */
    template <Expression::CBinary T>
    std::tuple<int, Expression::ExpressionPtr> getBinaryLHS(KernelGraph const& kgraph, int assign)
    {
        auto op = kgraph.control.get<ControlGraph::Assign>(assign);

        auto expr = std::get<T>(*op->expression);
        if(!std::holds_alternative<Expression::DataFlowTag>(*expr.lhs))
            return {-1, nullptr};
        auto tag = std::get<Expression::DataFlowTag>(*expr.lhs).tag;
        return {tag, expr.lhs};
    }

    /**
     * @brief Return DataFlowTag of RHS of binary expression in Assign node.
     */
    template <Expression::CBinary T>
    std::tuple<int, Expression::ExpressionPtr> getBinaryRHS(KernelGraph const& kgraph, int assign)
    {
        auto op = kgraph.control.get<ControlGraph::Assign>(assign);

        auto expr = std::get<T>(*op->expression);
        if(!std::holds_alternative<Expression::DataFlowTag>(*expr.rhs))
            return {-1, nullptr};
        auto tag = std::get<Expression::DataFlowTag>(*expr.rhs).tag;
        return {tag, expr.rhs};
    }

    template <std::ranges::forward_range Range>
    void purgeNodes(KernelGraph& kgraph, Range nodes)
    {
        for(int tag : nodes)
        {
            for(auto reap :
                kgraph.control.getNeighbours<Graph::Direction::Upstream>(tag).to<std::vector>())
            {
                kgraph.control.deleteElement(reap);
            }
            for(auto reap :
                kgraph.control.getNeighbours<Graph::Direction::Downstream>(tag).to<std::vector>())
            {
                kgraph.control.deleteElement(reap);
            }

            kgraph.control.deleteElement(tag);
            kgraph.mapper.purge(tag);
        }
    }

    template <CForwardRangeOf<int> Range>
    std::optional<int>
        getForLoopCoord(std::optional<int> forLoopOp, KernelGraph const& kgraph, Range within)
    {
        if(!forLoopOp)
            return {};

        auto forLoopCoord = kgraph.mapper.get<CoordinateGraph::ForLoop>(*forLoopOp);

        if(std::find(within.cbegin(), within.cend(), forLoopCoord) != within.cend())
            return forLoopCoord;
        else
            return {};
    }

    template <typename T>
    std::unordered_set<int> filterCoordinates(auto const& candidates, KernelGraph const& kgraph)
    {
        std::unordered_set<int> rv;
        for(auto candidate : candidates)
            if(kgraph.coordinates.get<T>(candidate))
                rv.insert(candidate);
        return rv;
    }

    template <typename T>
    std::optional<int> findContainingOperation(int candidate, KernelGraph const& kgraph)
    {
        namespace CF = rocRoller::KernelGraph::ControlGraph;

        int lastTag = -1;
        for(auto parent : kgraph.control.depthFirstVisit(candidate, Graph::Direction::Upstream))
        {
            bool containing = lastTag != -1 && kgraph.control.get<CF::Body>(lastTag);
            lastTag         = parent;

            auto forLoop = kgraph.control.get<T>(parent);
            if(forLoop && containing)
                return parent;
        }
        return {};
    }

    template <Graph::Direction direction>
    void reconnect(KernelGraph& graph, int newop, int op)
    {
        auto neighbours = graph.control.getNeighbours<direction>(op).template to<std::vector>();
        for(auto const& tag : neighbours)
        {
            auto edge = graph.control.getElement(tag);
            int  node = *graph.control.getNeighbours<direction>(tag).begin();
            graph.control.deleteElement(tag);
            if(newop != -1)
            {
                if constexpr(direction == Graph::Direction::Upstream)
                {
                    graph.control.addElement(edge, {node}, {newop});
                }
                else
                {
                    graph.control.addElement(edge, {newop}, {node});
                }
            }
        }
    }

    template <typename T>
    std::optional<int> findTopOfContainingOperation(int candidate, KernelGraph const& kgraph)
    {
        namespace CF = rocRoller::KernelGraph::ControlGraph;

        int lastTag       = -1;
        int lastOperation = candidate;
        for(auto parent : kgraph.control.depthFirstVisit(candidate, Graph::Direction::Upstream))
        {
            bool containing = lastTag != -1 && kgraph.control.get<CF::Body>(lastTag);
            lastTag         = parent;

            auto maybeT = kgraph.control.get<T>(parent);
            if(maybeT && containing)
                return lastOperation;

            if(kgraph.control.getElementType(parent) == Graph::ElementType::Node)
                lastOperation = parent;
        }
        return {};
    }

    template <std::predicate<int> Predicate>
    std::vector<int> duplicateControlNodes(KernelGraph&                    graph,
                                           std::shared_ptr<GraphReindexer> reindexer,
                                           std::vector<int> const&         startNodes,
                                           Predicate                       dontDuplicate)
    {
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        std::vector<int> newStartNodes;

        if(reindexer == nullptr)
            reindexer = std::make_shared<GraphReindexer>();

        // Create duplicates of all of the nodes downstream of the startNodes
        for(auto const& node :
            graph.control.depthFirstVisit(startNodes, Graph::Direction::Downstream))
        {
            // Only do this step if element is a node
            if(graph.control.getElementType(node) == Graph::ElementType::Node)
            {
                auto op                  = graph.control.addElement(graph.control.getElement(node));
                reindexer->control[node] = op;
            }
        }

        for(auto const& reindex : reindexer->control)
        {
            // Create all edges within new sub-graph
            auto location = graph.control.getLocation(reindex.first);
            for(auto const& output : location.outgoing)
            {
                int child
                    = *graph.control.getNeighbours<Graph::Direction::Downstream>(output).begin();
                graph.control.addElement(graph.control.getElement(output),
                                         {reindex.second},
                                         {reindexer->control[child]});
            }

            // Use the same coordinate graph mappings
            for(auto const& c : graph.mapper.getConnections(reindex.first))
            {
                auto coord = c.coordinate;

                if(dontDuplicate(coord))
                {
                    graph.mapper.connect(reindex.second, coord, c.connection);
                    reindexer->coordinates.insert_or_assign(coord, coord);
                    continue;
                }

                // If one of the mappings represents storage, duplicate it in the CoordinateGraph.
                // This will allow the RegisterTagManager to recognize that the nodes are pointing
                // to different data and to use different registers.
                // Note: A PassThrough edge is added from any of the duplicate nodes to the original
                // node.
                auto maybeMacroTile
                    = graph.coordinates.get<CoordinateGraph::MacroTile>(c.coordinate);
                auto maybeLDS = graph.coordinates.get<CoordinateGraph::LDS>(c.coordinate);
                if(maybeMacroTile || maybeLDS)
                {
                    if(reindexer->coordinates.count(c.coordinate) == 0)
                    {
                        auto dim = graph.coordinates.addElement(
                            graph.coordinates.getElement(c.coordinate));
                        reindexer->coordinates[c.coordinate] = dim;
                        auto duplicate                       = graph.coordinates
                                             .getOutputNodeIndices(
                                                 coord, CT::isEdge<CoordinateGraph::Duplicate>)
                                             .to<std::vector>();

                        auto origCoord = duplicate.empty() ? coord : duplicate[0];
                        graph.coordinates.addElement(
                            CoordinateGraph::Duplicate(), {dim}, {origCoord});

                        // Add View edges for new duplicated coordinates
                        auto outgoingViews
                            = graph.coordinates
                                  .getOutputNodeIndices(coord, CT::isEdge<CoordinateGraph::View>)
                                  .to<std::vector>();
                        if(!outgoingViews.empty() && reindexer->coordinates.count(outgoingViews[0]))
                        {
                            graph.coordinates.addElement(
                                CoordinateGraph::View(),
                                {dim},
                                {reindexer->coordinates[outgoingViews[0]]});
                        }

                        auto incomingViews
                            = graph.coordinates
                                  .getInputNodeIndices(origCoord, CT::isEdge<CoordinateGraph::View>)
                                  .to<std::vector>();
                        if(!incomingViews.empty() && reindexer->coordinates.count(incomingViews[0]))
                        {
                            graph.coordinates.addElement(CoordinateGraph::View(),
                                                         {reindexer->coordinates[incomingViews[0]]},
                                                         {dim});
                        }
                    }
                    coord = reindexer->coordinates[c.coordinate];
                }
                graph.mapper.connect(reindex.second, coord, c.connection);
            }
        }

        // Change coordinate values in Expressions
        for(auto const& pair : reindexer->control)
        {
            reindexExpressions(graph, pair.second, *reindexer);
        }

        // Return the new start nodes
        for(auto const& startNode : startNodes)
        {
            newStartNodes.push_back(reindexer->control[startNode]);
        }

        return newStartNodes;
    }

    /**
     * @brief Get the first and last nodes from a set of nodes that are totally ordered
     */
    template <typename T>
    std::pair<int, int> getFirstAndLastNodes(KernelGraph const& graph, T const& nodes)
    {
        AssertFatal(not nodes.empty());

        auto firstNode = *nodes.begin();
        auto lastNode  = *nodes.begin();

        for(auto const& n : nodes)
        {
            using namespace rocRoller::KernelGraph::ControlGraph;

            if(firstNode != n)
            {
                auto order = graph.control.compareNodes(rocRoller::IgnoreCache, firstNode, n);
                // If this assertion fails, that means the nodes are not totally ordered
                AssertFatal(order != NodeOrdering::Undefined, "nodes are not totally ordered");
                if(order != NodeOrdering::LeftFirst and order != NodeOrdering::LeftInBodyOfRight)
                {
                    firstNode = n;
                }
            }

            if(lastNode != n)
            {
                auto order = graph.control.compareNodes(rocRoller::IgnoreCache, lastNode, n);
                // If this assertion fails, that means the nodes are not totally ordered
                AssertFatal(order != NodeOrdering::Undefined, "nodes are not totally ordered");
                if(order == NodeOrdering::LeftFirst or order == NodeOrdering::LeftInBodyOfRight)
                {
                    lastNode = n;
                }
            }
        }
        return {firstNode, lastNode};
    }

    template <typename EdgeType>
    void connectAllPairs(std::vector<int> const& A, std::vector<int> const& B, KernelGraph& kg)
    {
        if(A.empty() or B.empty())
            return;

        for(auto const a : A)
            for(auto const b : B)
                kg.control.addElement(EdgeType(), {a}, {b});
    }
}
