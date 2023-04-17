
#pragma once

#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller::KernelGraph
{

    template <typename T>
    std::optional<T> only(std::vector<T> v)
    {
        if(v.size() == 1)
            return v[0];
        return {};
    }

    std::optional<int> only(Generator<int> g)
    {
        auto it = g.begin();

        if(it == g.end())
            return {};

        auto first = *it;

        it = std::next(it);
        if(it == g.end())
            return first;

        return {};
    }

    template <typename T>
    std::unordered_set<int> filterCoordinates(std::vector<int>   candidates,
                                              KernelGraph const& kgraph)
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
    std::vector<int> duplicateControlNodes(KernelGraph&            graph,
                                           GraphReindexer&         reindexer,
                                           std::vector<int> const& startNodes,
                                           Predicate               dontDuplicate)
    {
        std::vector<int> newStartNodes;

        // Create duplicates of all of the nodes downstream of the startNodes
        for(auto const& node :
            graph.control.depthFirstVisit(startNodes, Graph::Direction::Downstream))
        {
            // Only do this step if element is a node
            if(graph.control.getElementType(node) == Graph::ElementType::Node)
            {
                auto op                 = graph.control.addElement(graph.control.getElement(node));
                reindexer.control[node] = op;
            }
        }

        for(auto const& reindex : reindexer.control)
        {
            // Create all edges within new sub-graph
            auto location = graph.control.getLocation(reindex.first);
            for(auto const& output : location.outgoing)
            {
                int child
                    = *graph.control.getNeighbours<Graph::Direction::Downstream>(output).begin();
                graph.control.addElement(
                    graph.control.getElement(output), {reindex.second}, {reindexer.control[child]});
            }

            // Use the same coordinate graph mappings
            for(auto const& c : graph.mapper.getConnections(reindex.first))
            {
                auto coord = c.coordinate;

                if(dontDuplicate(coord))
                {
                    graph.mapper.connect(reindex.second, coord, c.connection);
                    reindexer.coordinates.insert_or_assign(coord, coord);
                    continue;
                }

                // If one of the mappings represents storage, duplicate it in the CoordinateGraph.
                // This will allow the RegisterTagManager to recognize that the nodes are pointing
                // to different data and to use different registers.
                // Note: A PassThrough edge is added from any of the duplicate nodes to the original
                // node.
                auto mt = graph.coordinates.get<MacroTile>(c.coordinate);
                if(mt)
                {
                    if(reindexer.coordinates.count(c.coordinate) == 0)
                    {
                        auto dim = graph.coordinates.addElement(
                            graph.coordinates.getElement(c.coordinate));
                        reindexer.coordinates[c.coordinate] = dim;
                        auto duplicate
                            = graph.coordinates.getOutputNodeIndices(coord, CT::isEdge<PassThrough>)
                                  .to<std::vector>();
                        if(duplicate.empty())
                            graph.coordinates.addElement(PassThrough(), {dim}, {coord});
                        else
                            graph.coordinates.addElement(PassThrough(), {dim}, {duplicate[0]});
                    }
                    coord = reindexer.coordinates[c.coordinate];
                }
                graph.mapper.connect(reindex.second, coord, c.connection);
            }
        }

        // Change coordinate values in Expressions
        for(auto const& reindex : reindexer.control)
        {
            auto elem = graph.control.getElement(reindex.first);
            if(isOperation<Assign>(elem))
            {
                auto                     new_assign = graph.control.getNode<Assign>(reindex.second);
                ReindexExpressionVisitor visitor(reindexer);
                new_assign.expression = visitor.call(new_assign.expression);
                graph.control.setElement(reindex.second, new_assign);
            }
        }

        // Return the new start nodes
        for(auto const& startNode : startNodes)
        {
            newStartNodes.push_back(reindexer.control[startNode]);
        }

        return newStartNodes;
    }

}
