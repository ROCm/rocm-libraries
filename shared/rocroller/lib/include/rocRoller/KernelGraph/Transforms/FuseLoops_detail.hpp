// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Graph/Hypergraph.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace FuseLoopsDetail
        {
            /**
             * @brief A struct to record a loop's parent loops (the loops that
             *        contain this loop inside their Body) and child loops (the
             *        loops that are inside the Body of this loop)
             */
            struct LoopBodyInfo
            {
                std::unordered_set<int> parentLoops;
                std::set<int>           childLoops;
            };

            /**
             * @brief Order a new group (nodes) with existing groups. Each group consists of
             *        a pair of nodes which are the first and last nodes of memory nodes in a
             *        for-loop.
             */
            void OrderGroups(rocRoller::KernelGraph::KernelGraph& graph,
                             std::set<std::pair<int, int>>&       groups,
                             std::vector<int>&                    nodes);

            /**
             * @brief Identifies whether the set of child loops within a given loop can be fused.
             *
             * This function analyzes the child loops of the specified parent loop (tag),
             * checking whether they have identical loop lengths and increments. If all child
             * loops meet these criteria and are not already marked for fusion, their identifiers
             * are collected and returned. If any child loop fails the checks or if there is only
             * one child loop, fusion is deemed impossible and std::nullopt is returned.
             *
             * @param graph KernelGraph containing the loops.
             * @param tag Parent loop tag.
             * @param loopInfo A mapping from loop identifiers to their associated LoopBodyInfo,
             *                 including child loops.
             * @return std::optional<std::unordered_set<int>>
             *         - The set of child loop identifiers that can be fused, if fusion is possible.
             *         - std::nullopt if fusion is not possible due to mismatched lengths, increments,
             *           or insufficient loops.
             * TODO: Consider if this should find subsets of fusible loops instead of requiring
             *       all child loops be fusible
             */
            std::optional<std::unordered_set<int>> IdentifyFusibleLoops(
                KernelGraph& graph, int tag, std::unordered_map<int, LoopBodyInfo>& loopInfo);

            /**
             * @brief Fuses multiple ForLoop operations into a single loop node.
             *
             * Merges the bodies and control flow of the specified for-loops into one.
             * Ensures correct ordering of memory operations and removes redundant loop nodes.
             *
             * @param graph KernelGraph containing the loops.
             * @param tag Parent loop tag.
             * @param loopInfo Map of loop tags to their parent/child relationships.
             * @param forLoopsToFuse Set of loop tags to be fused; the first is retained as the fused loop.
             */
            void FuseLoops(KernelGraph&                           graph,
                           int                                    tag,
                           std::unordered_map<int, LoopBodyInfo>& loopInfo,
                           std::unordered_set<int> const&         forLoopsToFuse);

            /**
             * @brief Recursively searches the graph downstream of a given tag to find a child ForLoopOp.
             *
             * @param graph KernelGraph containing the loops.
             * @param tag Parent loop tag.
             * @param visited A set of tags that have already been visited during the search.
             * @return std::optional<int> The tag of a child ForLoopOp if found, else std::nullopt.
             */
            std::optional<int>
                GetChildLoop(KernelGraph const& graph, int tag, std::unordered_set<int>& visited);

            /**
             * @brief Populates child loops for the specified loop tag in the kernel graph.
             *
             * @tparam ControlEdgeType Type of control edge used for traversal.
             * @param graph KernelGraph containing the loops.
             * @param tag Loop identifier whose child loops are populated.
             * @param loopInfo Map from loop tags to their LoopBodyInfo, updated with child loops.
             */
            template <typename ControlEdgeType>
            void PopulateChildLoops(KernelGraph const&                     graph,
                                    int                                    tag,
                                    std::unordered_map<int, LoopBodyInfo>& loopInfo)
            {
                std::unordered_set<int> visited;
                loopInfo[tag];

                for(auto node : graph.control.getOutputNodeIndices<ControlEdgeType>(tag))
                {
                    visited.clear();
                    auto childLoop = GetChildLoop(graph, node, visited);
                    if(childLoop.has_value())
                    {
                        loopInfo.at(tag).childLoops.insert(childLoop.value());
                    }
                }
            }

            /**
             * @brief Populates the parentLoops set for each loop in the provided loopInfo map.
             *
             * Iterates through each loop and its associated LoopBodyInfo, then for each child loop,
             * inserts the current loop as a parent into the child loop's parentLoops set.
             *
             * @param graph KernelGraph containing the loops.
             * @param loopInfo   Map from loop identifiers to their corresponding LoopBodyInfo objects.
             */
            void PopulateParentLoops(KernelGraph const&                     graph,
                                     std::unordered_map<int, LoopBodyInfo>& loopInfo);

            /**
             * @brief Identifies and fuses compatible loops within the given kernel graph.
             *
             * Iterates over the provided loop information, detects sets of loops that can be fused,
             * and applies loop fusion transformations to the kernel graph.
             *
             * @param graph KernelGraph containing the loops.
             * @param loopInfo A map containing information about each loop body in the graph.
             */
            void IdentifyAndFuseLoops(KernelGraph&                           graph,
                                      std::unordered_map<int, LoopBodyInfo>& loopInfo);
        } // namespace FuseLoopsDetail
    } // namespace KernelGraph
} // namespace rocRoller
