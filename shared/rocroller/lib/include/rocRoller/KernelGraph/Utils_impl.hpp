// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    template <Graph::Direction Direction, typename EdgeType>
    std::optional<int> GetEdgeTag(KernelGraph const& graph, int tag)
    {
        for(auto elem : graph.coordinates.getNeighbours<Direction>(tag))
        {
            auto maybeEdge = graph.coordinates.get<EdgeType>(elem);
            if(maybeEdge)
                return elem;
        }
        return {};
    }

    template <std::ranges::forward_range Range>
    void purgeNodes(KernelGraph& kgraph, Range nodes)
    {
        for(int tag : nodes)
        {
            for(auto reap : kgraph.control.getNeighbours<Graph::Direction::Upstream>(tag))
            {
                kgraph.control.deleteElement(reap);
            }
            for(auto reap : kgraph.control.getNeighbours<Graph::Direction::Downstream>(tag))
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
        auto neighbours = graph.control.getNeighbours<direction>(op);
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

    template <ControlGraph::COperation SrcOpType, ControlGraph::COperation DstOpType>
    requires(
        (std::is_same_v<
             SrcOpType,
             ControlGraph::LoadTiled> && std::is_same_v<DstOpType, ControlGraph::StoreLDSTile>)
        || (std::is_same_v<
                SrcOpType,
                ControlGraph::StoreLDSTile> && std::is_same_v<DstOpType, ControlGraph::LoadTiled>))
        std::vector<int> getAssociatedOps(KernelGraph const& kgraph, int srcOpTag)
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        const auto element = kgraph.control.getElement(srcOpTag);
        AssertFatal(std::holds_alternative<Operation>(element),
                    concatenate("Expected Operation but got Edge", ShowValue(srcOpTag)));

        const auto op = std::get<Operation>(element);
        AssertFatal(std::holds_alternative<SrcOpType>(op),
                    fmt::format("Expected {} but got {}", typeName<SrcOpType>(), toString(op)));

        auto macroTileTag = kgraph.mapper.get<MacroTile>(srcOpTag);

        std::vector<int> rv{};

        for(auto conn : kgraph.mapper.getCoordinateConnections(macroTileTag))
        {
            const auto dstOpTag = conn.control;
            const auto element  = kgraph.control.getElement(dstOpTag);
            AssertFatal(std::holds_alternative<Operation>(element),
                        concatenate("Expected Operation but got Edge", ShowValue(dstOpTag)));

            const auto op = std::get<Operation>(element);
            if(std::holds_alternative<DstOpType>(op))
            {
                rv.push_back(dstOpTag);
            }
        }
        return rv;
    }

    template <std::predicate<int> Predicate>
    std::vector<std::pair<int, int>> getLoadTiledStoreLDSTilePairs(KernelGraph const& kgraph,
                                                                   Predicate          predicate)
    {
        using namespace ControlGraph;
        using namespace CoordinateGraph;

        std::vector<std::pair<int, int>> rv;

        for(auto loadTiledTag : kgraph.control.findElements(predicate))
        {
            const auto storeLDSTags{
                getAssociatedOps<LoadTiled, StoreLDSTile>(kgraph, loadTiledTag)};

            if(storeLDSTags.size() == 1)
            {
                rv.push_back({loadTiledTag, storeLDSTags[0]});
            }
            else
            {
                AssertFatal(storeLDSTags.size() <= 2,
                            "getLoadTiledStoreLDSTilePairs: More than 2 ComputeIndex operation "
                            "required for StoreLDSTile.",
                            ShowValue(loadTiledTag),
                            ShowValue(storeLDSTags.size()));
                for(const auto& storeLDS : storeLDSTags)
                {
                    auto maybeForLoopOfLoad
                        = findContainingOperation<ForLoopOp>(loadTiledTag, kgraph);
                    auto maybeForLoopOfStore = findContainingOperation<ForLoopOp>(storeLDS, kgraph);

                    const auto isLoadInLoop  = maybeForLoopOfLoad.has_value();
                    const auto isStoreInLoop = maybeForLoopOfStore.has_value();

                    const auto bothInSameLoop
                        = isLoadInLoop && isStoreInLoop
                          && maybeForLoopOfLoad.value() == maybeForLoopOfStore.value();

                    const auto bothNotInLoop = not isLoadInLoop && not isStoreInLoop;

                    if(bothInSameLoop || bothNotInLoop)
                    {
                        rv.push_back({loadTiledTag, storeLDS});
                    }
                }
            }
        }
        return rv;
    }
}
