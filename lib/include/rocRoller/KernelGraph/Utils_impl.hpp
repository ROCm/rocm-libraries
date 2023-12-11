
#pragma once

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{
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

        auto forLoopCoord = kgraph.mapper.get<ForLoop>(*forLoopOp);

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

    template <std::predicate<int> Predicate>
    std::vector<int> duplicateControlNodes(KernelGraph&                    graph,
                                           std::shared_ptr<GraphReindexer> reindexer,
                                           std::vector<int> const&         startNodes,
                                           Predicate                       dontDuplicate)
    {
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
                auto maybeMacroTile = graph.coordinates.get<MacroTile>(c.coordinate);
                auto maybeLDS       = graph.coordinates.get<LDS>(c.coordinate);
                if(maybeMacroTile || maybeLDS)
                {
                    if(reindexer->coordinates.count(c.coordinate) == 0)
                    {
                        auto dim = graph.coordinates.addElement(
                            graph.coordinates.getElement(c.coordinate));
                        reindexer->coordinates[c.coordinate] = dim;
                        auto duplicate
                            = graph.coordinates.getOutputNodeIndices(coord, CT::isEdge<PassThrough>)
                                  .to<std::vector>();
                        if(duplicate.empty())
                            graph.coordinates.addElement(PassThrough(), {dim}, {coord});
                        else
                            graph.coordinates.addElement(PassThrough(), {dim}, {duplicate[0]});
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

}
