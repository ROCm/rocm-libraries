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

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>
#include <rocRoller/KernelGraph/Visitors.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        using namespace ControlGraph;
        namespace CT = rocRoller::KernelGraph::CoordinateGraph;

        /**
         * @brief Fuse Loops Transformation
         *
         * This transformation looks for the following pattern:
         *
         *     ForLoop
         * |            |
         * Set Coord    Set Coord
         * |            |
         * For Loop     For Loop
         *
         * If it finds it, it will fuse the lower loops into a single
         * loop, as long as they are the same size.
         */
        namespace FuseLoopsNS
        {
            using GD = rocRoller::Graph::Direction;

            /**
             * @brief Gather all for loops to fuse below the starting node
             *
             * @param graph
             * @param start
             * @return std::vector<std::unordered_set<int>>
             */
            std::vector<std::unordered_set<int>> gatherForLoops(KernelGraph& graph, int start)
            {
                auto bodies      = graph.control.getOutputNodeIndices<Body>(start).to<std::set>();
                auto isForLoopOp = graph.control.isElemType<ForLoopOp>();
                auto isSetCoordinate = graph.control.isElemType<SetCoordinate>();
                auto isSequence      = graph.control.isElemType<Sequence>();
                auto isBody          = graph.control.isElemType<Body>();

                // Get a set of all ForLoops and SetCoordinates which contain (tail) ForLoops
                auto maybeForLoops
                    = graph.control.depthFirstVisit(bodies, isSequence, GD::Downstream)
                          .filter([&](int tag) -> bool {
                              if(isForLoopOp(tag))
                              {
                                  return true;
                              }
                              if(isSetCoordinate(tag))
                              {
                                  return graph.mapper.get<CoordinateGraph::ForLoop>(tag) > 0;
                              }
                              return false;
                          })
                          .to<std::unordered_set>();

                // Filter the previous set of nodes to only the ForLoops under consideration
                std::vector<int> forLoops;
                for(auto const& maybeForLoop : maybeForLoops)
                {
                    if(isForLoopOp(maybeForLoop))
                    {
                        forLoops.insert(forLoops.begin(), maybeForLoop);
                    }
                    else
                    {
                        // The filter means that this is now a SetCoordinate connected to an Unroll dimension
                        // Tail loops are always downstream of these
                        std::optional<int> tag = maybeForLoop;
                        while(tag && isSetCoordinate(*tag))
                        {
                            tag = only(graph.control.getOutputNodeIndices<Body>(*tag));
                        }
                        if(tag && isForLoopOp(*tag))
                        {
                            forLoops.push_back(*tag);
                        }
                    }
                }

                // Determine sets of loops to be fused together
                // Currently loops should only be fused if they're the "same" loop (from unrolling)
                std::vector<std::unordered_set<int>> loopGroupsToFuse;
                while(!forLoops.empty())
                {
                    std::unordered_set<int>   loopGroup;
                    Expression::ExpressionPtr loopIncrement;
                    Expression::ExpressionPtr loopLength;
                    std::map<int, int>        baseLoopContents;
                    for(auto const& loop : forLoops)
                    {
                        if(loopGroup.count(loop) != 0)
                            continue;

                        auto loopDim = getSize(std::get<CT::Dimension>(graph.coordinates.getElement(
                            graph.mapper.get(loop, NaryArgument::DEST))));

                        // Loops to be fused must have the same length
                        if(loopLength)
                        {
                            if(!identical(loopDim, loopLength))
                                continue;
                        }
                        else
                        {
                            loopLength = loopDim;
                        }

                        // Loops to be fused must have the same increment value
                        auto [dataTag, increment] = getForLoopIncrement(graph, loop);
                        if(loopIncrement)
                        {
                            if(!identical(loopIncrement, increment))
                                continue;
                        }
                        else
                        {
                            loopIncrement = increment;
                        }

                        // Loop similarity heuristic
                        // We only want to fuse loops which are "the same" or similar.
                        // Currently this is just counting the number of each type of node in the body.
                        // We may want to replace this with a better heuristic in the future.
                        auto               loopContentsGenerator = graph.control.nodesInBody(loop);
                        std::map<int, int> loopContents;
                        std::for_each(loopContentsGenerator.begin(),
                                      loopContentsGenerator.end(),
                                      [&](int tag) -> void {
                                          auto type = graph.control.get<Operation>(tag)->index();
                                          if(loopContents.contains(type))
                                          {
                                              loopContents[type]++;
                                          }
                                          else
                                          {
                                              loopContents[type] = 1;
                                          }
                                      });

                        if(!baseLoopContents.empty())
                        {
                            if(baseLoopContents != loopContents)
                                continue;
                        }
                        else
                        {
                            baseLoopContents = loopContents;
                        }

                        loopGroup.insert(loop);
                    }

                    for(auto loop : loopGroup)
                    {
                        std::erase(forLoops, loop);
                    }

                    if(loopGroup.size() > 1)
                    {
                        loopGroupsToFuse.push_back(loopGroup);
                    }
                }

                return loopGroupsToFuse;
            }

            /**
             * @brief Find the first and last nodes in `nodes` which are totally ordered
             *
             * Returns the first and last nodes as a pair
             *
             */
            static std::pair<int, int>
                getFirstAndLastNodes(rocRoller::KernelGraph::KernelGraph const& graph,
                                     std::vector<int> const&                    nodes)
            {
                AssertFatal(not nodes.empty());

                using namespace rocRoller::KernelGraph::ControlGraph;

                int firstNode = nodes[0];
                int lastNode  = nodes[0];
                for(int i = 1; i < nodes.size(); i++)
                {
                    auto order
                        = graph.control.compareNodes(rocRoller::IgnoreCache, firstNode, nodes[i]);
                    // If this assertion fails, that means the nodes are not totally ordered
                    AssertFatal(order != NodeOrdering::Undefined, "nodes are not totally ordered");
                    if(order != NodeOrdering::LeftFirst
                       and order != NodeOrdering::LeftInBodyOfRight)
                    {
                        firstNode = nodes[i];
                    }

                    order = graph.control.compareNodes(rocRoller::IgnoreCache, lastNode, nodes[i]);
                    // If this assertion fails, that means the nodes are not totally ordered
                    AssertFatal(order != NodeOrdering::Undefined, "nodes are not totally ordered");
                    if(order == NodeOrdering::LeftFirst or order == NodeOrdering::LeftInBodyOfRight)
                    {
                        lastNode = nodes[i];
                    }
                }

                return {firstNode, lastNode};
            }

            /**
             * @brief Order a new group (nodes) with existing groups. Each group consists of
             *        a pair of nodes which are the first and last nodes of memory nodes in a
             *        for-loop.
             *
             */
            static void orderGroups(rocRoller::KernelGraph::KernelGraph& graph,
                                    std::set<std::pair<int, int>>&       groups,
                                    std::vector<int>&                    nodes)

            {
                if(nodes.empty())
                    return;

                //
                // Create a new group using the first and last node of `nodes`.
                // An assumption here is `nodes` should be totally ordered.
                //
                auto [firstNode, lastNode] = getFirstAndLastNodes(graph, nodes);
                if(groups.empty())
                {
                    groups.emplace(firstNode, lastNode);
                    return;
                }

                auto [new_group, inserted] = groups.emplace(firstNode, lastNode); // Add new group
                AssertFatal(inserted); // Should not have identical group

                //
                // Order (insert sequence edges) the new group with existing groups.
                // An important assumption here is groups should not overlap.
                //
                if(new_group != groups.begin())
                {
                    auto prev_group = std::prev(new_group);
                    AssertFatal(prev_group->second < firstNode);
                    auto sc1 = getTopSetCoordinate(graph, prev_group->second);
                    auto sc2 = getTopSetCoordinate(graph, firstNode);
                    graph.control.addElement(Sequence(), {sc1}, {sc2});
                }

                if(*new_group != *groups.rbegin())
                {
                    auto next_group = std::next(new_group);
                    AssertFatal(next_group->first > lastNode);
                    auto sc1 = getTopSetCoordinate(graph, lastNode);
                    auto sc2 = getTopSetCoordinate(graph, next_group->first);
                    graph.control.addElement(Sequence(), {sc1}, {sc2});
                }
            }

            /**
             * @brief General routine to fuse one node into another
             *
             * @param graph
             * @param fusedNodeTag
             * @param nodeTag
             */
            void fuseNode(KernelGraph& graph, int fusedNodeTag, int nodeTag)
            {
                // Move operations that come after `nodeTag` to follow `fusedNodeTag`.
                for(auto const& child :
                    graph.control.getOutputNodeIndices<Sequence>(nodeTag).to<std::vector>())
                {
                    if(fusedNodeTag != child)
                    {
                        graph.control.addElement(Sequence(), {fusedNodeTag}, {child});
                    }
                    graph.control.deleteElement<Sequence>(std::vector<int>{nodeTag},
                                                          std::vector<int>{child});
                    // Make sure we don't introduce a cycle
                    std::unordered_set<int> toDelete;
                    for(auto descSeqOfChild :
                        filter(graph.control.isElemType<Sequence>(),
                               graph.control.depthFirstVisit(child, GD::Downstream)))
                    {
                        if(graph.control.getNeighbours<GD::Downstream>(descSeqOfChild)
                               .to<std::unordered_set>()
                               .contains(fusedNodeTag))
                        {
                            toDelete.insert(descSeqOfChild);
                        }
                    }
                    for(auto edge : toDelete)
                    {
                        graph.control.deleteElement(edge);
                    }
                }

                // Fuse the bodies
                for(auto const& child :
                    graph.control.getOutputNodeIndices<Body>(nodeTag).to<std::vector>())
                {
                    graph.control.addElement(Body(), {fusedNodeTag}, {child});
                    graph.control.deleteElement<Body>(std::vector<int>{nodeTag},
                                                      std::vector<int>{child});
                }

                // Make sure dependencies are satisfied
                for(auto const& parent :
                    graph.control.getInputNodeIndices<Sequence>(nodeTag).to<std::vector>())
                {
                    auto descOfFusedLoop
                        = graph.control
                              .depthFirstVisit(fusedNodeTag,
                                               graph.control.isElemType<Sequence>(),
                                               GD::Downstream)
                              .to<std::unordered_set>();

                    if(!descOfFusedLoop.contains(parent))
                    {
                        graph.control.addElement(Sequence(), {parent}, {fusedNodeTag});
                    }
                    graph.control.deleteElement<Sequence>(std::vector<int>{parent},
                                                          std::vector<int>{nodeTag});
                }

                for(auto const& parent :
                    graph.control.getInputNodeIndices<Body>(nodeTag).to<std::vector>())
                {
                    graph.control.addElement(Body(), {parent}, {fusedNodeTag});
                    graph.control.deleteElement<Body>(std::vector<int>{parent},
                                                      std::vector<int>{nodeTag});
                }
            }

            /**
             * @brief Visitor to determine if two nodes are the "same" operation of the purposes of fusion
             *
             */
            struct IsSameOperationVisitor
            {
                template <CConcreteOperation OpA, CConcreteOperation OpB>
                bool operator()(int, OpA const&, int, OpB const&)
                {
                    return false;
                }

                bool operator()(int tagA, SetCoordinate const& A, int tagB, SetCoordinate const& B)
                {
                    auto connA = graph.mapper.getConnections(tagA);
                    auto connB = graph.mapper.getConnections(tagB);

                    if(connA.size() != connB.size())
                    {
                        return false;
                    }
                    for(auto iterA = connA.begin(), iterB = connB.begin(); iterA != connA.end();
                        iterA++, iterB++)
                    {
                        if(iterA->coordinate != iterB->coordinate)
                            return false;
                        if(iterA->connection != iterB->connection)
                            return false;
                    }
                    return identical(A.value, B.value);
                }

                bool call(int tagA, int tagB)
                {
                    return std::visit(*this,
                                      std::variant<int>(tagA),
                                      graph.control.getNode(tagA),
                                      std::variant<int>(tagB),
                                      graph.control.getNode(tagB));
                }

                KernelGraph const& graph;
            };

            /**
             * @brief Walks up the tree, fusing nodes which contain the start in their body and are the same operation
             *
             * @param graph
             * @param tag
             */
            void fuseScopes(KernelGraph& graph, int tag)
            {
                auto parentsWithEdges
                    = graph.control.getInputNodeIndices<Body>(tag).template to<std::vector>();
                std::set<std::set<int>> nodeSetsToMerge;
                IsSameOperationVisitor  visitor{graph};

                for(auto const& A : parentsWithEdges)
                {
                    std::set<int> sameAsThis;
                    for(auto const& B : parentsWithEdges)
                    {
                        if(A == B)
                            continue;
                        if(visitor.call(A, B))
                        {
                            sameAsThis.insert(B);
                        }
                    }
                    if(!sameAsThis.empty())
                    {
                        sameAsThis.insert(A);
                        nodeSetsToMerge.insert(sameAsThis);
                    }
                }

                for(auto mergeSet : nodeSetsToMerge)
                {
                    auto mergedNodeTag = *mergeSet.begin();
                    for(auto const& nodeTag : mergeSet)
                    {
                        if(nodeTag == mergedNodeTag)
                            continue;
                        fuseNode(graph, mergedNodeTag, nodeTag);

                        graph.control.deleteElement(nodeTag);
                        graph.mapper.purge(nodeTag);
                    }
                    fuseScopes(graph, mergedNodeTag);
                }
            }

            void fuseLoops(KernelGraph& graph, int tag)
            {
                rocRoller::Log::getLogger()->debug("KernelGraph::fuseLoops({})", tag);

                auto loopGroupsToFuse = gatherForLoops(graph, tag);
                for(auto forLoopsToFuse : loopGroupsToFuse)
                {
                    if(forLoopsToFuse.size() <= 1)
                        return;

                    auto dontWalkPastForLoop = [&](int tag) -> bool {
                        for(auto neighbour : graph.control.getNeighbours(tag, GD::Downstream))
                        {
                            if(graph.control.get<ForLoopOp>(neighbour))
                            {
                                return false;
                            }
                        }
                        return true;
                    };

                    auto fusedLoopTag = *forLoopsToFuse.begin();

                    auto fusedLoopBodyChildren
                        = graph.control.getOutputNodeIndices<Body>(fusedLoopTag).to<std::vector>();

                    auto initializeGroups = [&]<typename T>() {
                        std::set<std::pair<int, int>> groups;
                        auto                          nodes
                            = filter(graph.control.isElemType<T>(),
                                     graph.control.depthFirstVisit(
                                         fusedLoopBodyChildren, dontWalkPastForLoop, GD::Downstream))
                                  .template to<std::vector>();
                        if(not nodes.empty())
                            groups.emplace(getFirstAndLastNodes(graph, nodes));
                        return groups;
                    };

                    auto groups_loads     = initializeGroups.template operator()<LoadTiled>();
                    auto groups_ldsLoads  = initializeGroups.template operator()<LoadLDSTile>();
                    auto groups_stores    = initializeGroups.template operator()<StoreTiled>();
                    auto groups_ldsStores = initializeGroups.template operator()<StoreLDSTile>();

                    for(auto const& forLoopTag : forLoopsToFuse)
                    {
                        if(forLoopTag == fusedLoopTag)
                            continue;

                        //
                        // Extract the memory nodes in forLoopTag, which will be used
                        // at the end to order with memory nodes in fusedLoopTag.
                        //
                        std::vector<int> loads;
                        std::vector<int> ldsLoads;
                        std::vector<int> stores;
                        std::vector<int> ldsStores;
                        {
                            auto children = graph.control.getOutputNodeIndices<Body>(forLoopTag)
                                                .to<std::vector>();

                            loads = filter(graph.control.isElemType<LoadTiled>(),
                                        graph.control.depthFirstVisit(
                                            children, dontWalkPastForLoop, GD::Downstream))
                                        .to<std::vector>();
                            ldsLoads = filter(graph.control.isElemType<LoadLDSTile>(),
                                            graph.control.depthFirstVisit(
                                                children, dontWalkPastForLoop, GD::Downstream))
                                        .to<std::vector>();
                            stores = filter(graph.control.isElemType<StoreTiled>(),
                                            graph.control.depthFirstVisit(
                                                children, dontWalkPastForLoop, GD::Downstream))
                                        .to<std::vector>();
                            ldsStores = filter(graph.control.isElemType<StoreLDSTile>(),
                                            graph.control.depthFirstVisit(
                                                children, dontWalkPastForLoop, GD::Downstream))
                                            .to<std::vector>();
                        }

                        fuseNode(graph, fusedLoopTag, forLoopTag);
                        purgeFor(graph, forLoopTag);

                        //
                        // Order the memory nodes in forLoopTag with memory
                        // nodes in fusedLoopTag.
                        //
                        // An important assumption here is the memory nodes of
                        // forLoopTag should be ordered totally already, and
                        // orderGroups leverages this fact to connect the first and
                        // last nodes with other groups to achieve total ordering.
                        //
                        orderGroups(graph, groups_loads, loads);
                        orderGroups(graph, groups_ldsLoads, ldsLoads);
                        orderGroups(graph, groups_stores, stores);
                        orderGroups(graph, groups_ldsStores, ldsStores);
                    }

                    fuseScopes(graph, fusedLoopTag);
                }
            }
        }

        KernelGraph FuseLoops::apply(KernelGraph const& k)
        {
            TIMER(t, "KernelGraph::fuseLoops");

            auto newGraph = k;

            for(const auto node :
                newGraph.control.depthFirstVisit(*newGraph.control.roots().begin()))
            {
                if(isOperation<ForLoopOp>(newGraph.control.getElement(node)))
                {
                    FuseLoopsNS::fuseLoops(newGraph, node);
                }
            }

            return newGraph;
        }

        /**
         * @brief Ensure no node is a child of two nodes which are not contained within each other
         *
         * @param graph
         * @return ConstraintStatus
         */
        ConstraintStatus BodyOfOnlyOneNode(const KernelGraph& graph)
        {
            ConstraintStatus retval;
            for(auto tag : graph.control.leaves().to<std::vector>())
            {
                auto containing = graph.control.nodesContaining(tag).to<std::unordered_set>();
                for(auto contA : containing)
                {
                    for(auto contB : containing)
                    {
                        if(contA == contB)
                        {
                            continue;
                        }
                        auto order = graph.control.compareNodes(UseCacheIfAvailable, contA, contB);
                        if(!(order == NodeOrdering::LeftInBodyOfRight
                             || order == NodeOrdering::RightInBodyOfLeft))
                        {
                            retval.combine(
                                false,
                                concatenate(
                                    "Nodes ", contA, " and ", contB, "have intersecting bodies."));
                        }
                    }
                }
            }
            return retval;
        }

        std::vector<GraphConstraint> FuseLoops::postConstraints() const
        {
            return {BodyOfOnlyOneNode};
        }
    }
}
