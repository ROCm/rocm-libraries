// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/KernelGraph/ControlGraph/LastRWTracer.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops.hpp>
#include <rocRoller/KernelGraph/Transforms/FuseLoops_detail.hpp>
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
        namespace FuseLoopsDetail
        {
            using GD = rocRoller::Graph::Direction;

            void OrderGroups(rocRoller::KernelGraph::KernelGraph& graph,
                             std::set<std::pair<int, int>>&       groups,
                             std::vector<int>&                    nodes)

            {
                if(nodes.empty())
                    return;

                // Create a new group using the first and last node of `nodes`.
                // An assumption here is `nodes` should be totally ordered.
                auto [firstNode, lastNode] = getFirstAndLastNodes(graph, nodes);
                if(groups.empty())
                {
                    groups.emplace(firstNode, lastNode);
                    return;
                }

                auto [new_group, inserted] = groups.emplace(firstNode, lastNode); // Add new group
                AssertFatal(inserted); // Should not have identical group

                // Order (insert sequence edges) the new group with existing groups.
                // An important assumption here is groups should not overlap.
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

            std::optional<std::unordered_set<int>> IdentifyFusibleLoops(
                KernelGraph& graph, int tag, std::unordered_map<int, LoopBodyInfo>& loopInfo)
            {
                std::unordered_set<int>   forLoopsToFuse;
                Expression::ExpressionPtr loopIncrement;
                Expression::ExpressionPtr loopLength;

                for(auto const forLoop : loopInfo.at(tag).childLoops)
                {
                    if(forLoopsToFuse.count(forLoop) != 0)
                        return std::nullopt;

                    // Check to see if loops are all the same length
                    auto forLoopDim = getSize(std::get<CT::Dimension>(graph.coordinates.getElement(
                        graph.mapper.get(forLoop, NaryArgument::DEST))));
                    if(loopLength)
                    {
                        if(!identical(forLoopDim, loopLength))
                            return std::nullopt;
                    }
                    else
                    {
                        loopLength = forLoopDim;
                    }

                    // Check to see if loops are incremented by the same value
                    auto [dataTag, increment] = getForLoopIncrement(graph, forLoop);
                    if(loopIncrement)
                    {
                        if(!identical(loopIncrement, increment))
                            return std::nullopt;
                    }
                    else
                    {
                        loopIncrement = increment;
                    }

                    forLoopsToFuse.insert(forLoop);
                }

                if(forLoopsToFuse.size() <= 1)
                    return std::nullopt;

                return forLoopsToFuse;
            }

            void FuseLoops(KernelGraph&                           graph,
                           int                                    tag,
                           std::unordered_map<int, LoopBodyInfo>& loopInfo,
                           std::unordered_set<int> const&         forLoopsToFuse)
            {
                Log::debug("KernelGraph::fuseLoops({})", tag);

                if(forLoopsToFuse.empty())
                    return;

                auto fusedLoopTag = *forLoopsToFuse.begin();
                Log::debug("KernelGraph::fuseLoops({}) fusing {} into {}",
                           tag,
                           forLoopsToFuse,
                           fusedLoopTag);

                auto dontWalkPastForLoop = [&](Graph::Direction direction) {
                    return [&, direction](int tag) -> bool {
                        for(auto neighbour : graph.control.getNeighbours(tag, direction))
                        {
                            if(graph.control.get<ForLoopOp>(neighbour))
                                return false;
                        }
                        return true;
                    };
                };

                auto fusedLoopBodyChildren
                    = graph.control.getOutputNodeIndices<Body>(fusedLoopTag).to<std::vector>();

                // cppcheck-suppress internalAstError
                auto initializeGroups = [&]<typename T>() {
                    std::set<std::pair<int, int>> groups;
                    auto                          nodes
                        = filter(graph.control.isElemType<T>(),
                                 graph.control.depthFirstVisit(fusedLoopBodyChildren,
                                                               dontWalkPastForLoop(GD::Downstream),
                                                               GD::Downstream))
                              .template to<std::vector>();
                    if(not nodes.empty())
                        groups.emplace(getFirstAndLastNodes(graph, nodes));
                    return groups;
                };

                auto groups_loads     = initializeGroups.template operator()<LoadTiled>();
                auto groups_ldsLoads  = initializeGroups.template operator()<LoadLDSTile>();
                auto groups_stores    = initializeGroups.template operator()<StoreTiled>();
                auto groups_ldsStores = initializeGroups.template operator()<StoreLDSTile>();

                for(auto const forLoopTag : forLoopsToFuse)
                {
                    if(forLoopTag == fusedLoopTag)
                        continue;

                    auto forLoop = graph.control.get<ForLoopOp>(forLoopTag);
                    if(forLoop->loopName == rocRoller::KLOOPTAIL)
                    {
                        Log::debug("KernelGraph::fuseLoops({}) removing redundant SetCoordinate "
                                   "chain for tail loop {}",
                                   tag,
                                   forLoopTag);
                        for(auto setCoordTag :
                            filter(graph.control.isElemType<
                                       rocRoller::KernelGraph::ControlGraph::SetCoordinate>(),
                                   graph.control.depthFirstVisit(
                                       forLoopTag, dontWalkPastForLoop(GD::Upstream), GD::Upstream))
                                .to<std::vector>())
                        {
                            deleteControlNode(graph, setCoordTag);
                        }
                    }

                    for(auto const child :
                        graph.control.getOutputNodeIndices<Sequence>(forLoopTag).to<std::vector>())
                    {
                        if(fusedLoopTag != child)
                        {
                            graph.control.addElement(Sequence(), {fusedLoopTag}, {child});
                        }
                        graph.control.deleteElement<Sequence>(std::vector<int>{forLoopTag},
                                                              std::vector<int>{child});
                        std::unordered_set<int> toDelete;
                        for(auto descSeqOfChild :
                            filter(graph.control.isElemType<Sequence>(),
                                   graph.control.depthFirstVisit(child, GD::Downstream)))
                        {
                            auto neighbours
                                = graph.control.getNeighbours<GD::Downstream>(descSeqOfChild);
                            if(std::find(neighbours.begin(), neighbours.end(), fusedLoopTag)
                               != neighbours.end())
                            {
                                toDelete.insert(descSeqOfChild);
                            }
                        }
                        for(auto edge : toDelete)
                        {
                            graph.control.deleteElement(edge);
                        }
                    }

                    // Extract the memory nodes in forLoopTag, which will be used
                    // at the end to order with memory nodes in fusedLoopTag.
                    std::vector<int> loads;
                    std::vector<int> ldsLoads;
                    std::vector<int> stores;
                    std::vector<int> ldsStores;
                    {
                        auto children = graph.control.getOutputNodeIndices<Body>(forLoopTag)
                                            .to<std::vector>();

                        loads = filter(graph.control.isElemType<LoadTiled>(),
                                       graph.control.depthFirstVisit(
                                           children,
                                           dontWalkPastForLoop(GD::Downstream),
                                           GD::Downstream))
                                    .to<std::vector>();
                        ldsLoads = filter(graph.control.isElemType<LoadLDSTile>(),
                                          graph.control.depthFirstVisit(
                                              children,
                                              dontWalkPastForLoop(GD::Downstream),
                                              GD::Downstream))
                                       .to<std::vector>();
                        stores = filter(graph.control.isElemType<StoreTiled>(),
                                        graph.control.depthFirstVisit(
                                            children,
                                            dontWalkPastForLoop(GD::Downstream),
                                            GD::Downstream))
                                     .to<std::vector>();
                        ldsStores = filter(graph.control.isElemType<StoreLDSTile>(),
                                           graph.control.depthFirstVisit(
                                               children,
                                               dontWalkPastForLoop(GD::Downstream),
                                               GD::Downstream))
                                        .to<std::vector>();
                    }

                    for(auto const child :
                        graph.control.getOutputNodeIndices<Body>(forLoopTag).to<std::vector>())
                    {
                        graph.control.addElement(Body(), {fusedLoopTag}, {child});
                        graph.control.deleteElement<Body>(std::vector<int>{forLoopTag},
                                                          std::vector<int>{child});
                    }

                    // Set the children loops of forLoopTag to be the children of fusedLoopTag
                    for(auto const child : loopInfo.at(forLoopTag).childLoops)
                    {
                        if(loopInfo.contains(child))
                        {
                            loopInfo.at(child).parentLoops.insert(fusedLoopTag);
                            loopInfo.at(fusedLoopTag).childLoops.insert(child);
                            loopInfo.at(child).parentLoops.erase(forLoopTag);
                        }
                    }

                    for(auto const parent :
                        graph.control.getInputNodeIndices<Sequence>(forLoopTag).to<std::vector>())
                    {
                        auto descOfFusedLoop
                            = graph.control
                                  .depthFirstVisit(fusedLoopTag,
                                                   graph.control.isElemType<Sequence>(),
                                                   GD::Downstream)
                                  .to<std::unordered_set>();
                        if(!descOfFusedLoop.contains(parent))
                        {
                            graph.control.addElement(Sequence(), {parent}, {fusedLoopTag});
                        }
                        graph.control.deleteElement<Sequence>(std::vector<int>{parent},
                                                              std::vector<int>{forLoopTag});
                    }

                    for(auto const parent :
                        graph.control.getInputNodeIndices<Body>(forLoopTag).to<std::vector>())
                    {
                        graph.control.addElement(Body(), {parent}, {fusedLoopTag});
                        graph.control.deleteElement<Body>(std::vector<int>{parent},
                                                          std::vector<int>{forLoopTag});
                    }

                    // Set the parent loops of forLoopTag to be the parent of fusedLoopTag
                    // and remove forLoopTag from loopInfo.
                    for(auto const parent : loopInfo.at(forLoopTag).parentLoops)
                    {
                        loopInfo.at(parent).childLoops.insert(fusedLoopTag);
                        loopInfo.at(parent).childLoops.erase(forLoopTag);
                        loopInfo.at(fusedLoopTag).parentLoops.insert(parent);
                    }
                    loopInfo.erase(forLoopTag);

                    purgeFor(graph, forLoopTag);

                    // Order the memory nodes in forLoopTag with memory
                    // nodes in fusedLoopTag.
                    //
                    // An important assumption here is the memory nodes of
                    // forLoopTag should be ordered totally already, and
                    // orderGroups leverages this fact to connect the first and
                    // last nodes with other groups to achieve total ordering.
                    OrderGroups(graph, groups_loads, loads);
                    OrderGroups(graph, groups_ldsLoads, ldsLoads);
                    OrderGroups(graph, groups_stores, stores);
                    OrderGroups(graph, groups_ldsStores, ldsStores);

                    Log::debug("KernelGraph::fuseLoops({}) done fusing {} into {}",
                               tag,
                               forLoopTag,
                               fusedLoopTag);
                }
            }

            std::optional<int>
                GetChildLoop(KernelGraph const& graph, int tag, std::unordered_set<int>& visited)
            {
                if(isOperation<ForLoopOp>(graph.control.getElement(tag)))
                    return tag;

                std::optional<int> ret;
                for(auto node :
                    graph.control.getOutputNodeIndices(tag, [](ControlEdge) { return true; }))
                {
                    if(visited.contains(node))
                        continue;

                    visited.insert(node);
                    auto childLoop = GetChildLoop(graph, node, visited);
                    if(childLoop.has_value())
                    {
                        AssertFatal(not ret.has_value(),
                                    "Each edge should contain at most only one loop");
                        ret = childLoop;
                    }
                }
                return ret;
            }

            void PopulateParentLoops(KernelGraph const&                     graph,
                                     std::unordered_map<int, LoopBodyInfo>& loopInfo)
            {
                for(auto& [loop, info] : loopInfo)
                {
                    for(auto& child : info.childLoops)
                        loopInfo.at(child).parentLoops.insert(loop);
                }
            }

            void IdentifyAndFuseLoops(KernelGraph&                           graph,
                                      std::unordered_map<int, LoopBodyInfo>& loopInfo)
            {
                bool changed = true;
                while(changed)
                {
                    changed = false;
                    std::unordered_set<int> loopTags;
                    for(auto const& [loopTag, _] : loopInfo)
                        loopTags.insert(loopTag);

                    for(auto const loopTag : loopTags)
                    {
                        if(loopInfo.find(loopTag) == loopInfo.end())
                            continue;

                        auto fusibleLoops = IdentifyFusibleLoops(graph, loopTag, loopInfo);
                        if(!fusibleLoops)
                            continue;

                        FuseLoops(graph, loopTag, loopInfo, *fusibleLoops);
                        changed = true;
                    }
                }
            }
        } // namespace FuseLoopsDetail

        KernelGraph FuseLoops::apply(KernelGraph const& k)
        {
            auto newGraph = k;

            std::unordered_map<int, FuseLoopsDetail::LoopBodyInfo> loopInfo;
            for(auto loopTag : newGraph.control.getNodes<ForLoopOp>().to<std::vector>())
                PopulateChildLoops<Body>(newGraph, loopTag, loopInfo);

            PopulateParentLoops(newGraph, loopInfo);
            IdentifyAndFuseLoops(newGraph, loopInfo);

            // Fuse Tail KLoops
            std::unordered_set<int> KLoopTags;
            for(auto const& [loopTag, _] : loopInfo)
            {
                auto forLoop = newGraph.control.get<ForLoopOp>(loopTag);
                if(forLoop->loopName.starts_with(rocRoller::KLOOP))
                    KLoopTags.insert(loopTag);
            }

            for(auto loopTag : KLoopTags)
                PopulateChildLoops<Sequence>(newGraph, loopTag, loopInfo);

            PopulateParentLoops(newGraph, loopInfo);
            IdentifyAndFuseLoops(newGraph, loopInfo);

            return newGraph;
        }
    }
}
