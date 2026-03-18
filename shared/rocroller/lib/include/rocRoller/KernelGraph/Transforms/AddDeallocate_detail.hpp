// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <set>

#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Utils.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        namespace AddDeallocateDetail
        {
            /**
             * @brief Add the next downstream Barrier to the list of dependencies.
             *
             * @param dependencies The set of dependencies for the Deallocate operation.
             * @param coordinate The coordinate for which to add the Deallocate operation.
             * @param lastRWOps The last read/write operations for the coordinate.
             * @param original The original kernel graph.
             */
            void addDownstreamBarrierInLoop(std::set<int>&       dependencies,
                                            int                  coordinate,
                                            std::set<int> const& lastRWOps,
                                            KernelGraph const&   original);

            /**
             * @brief Simplify dependencies by removing unnecessary ones.
             *
             * For example, if x and y are in the set of dependencies,
             * and x is `NodeOrdering::LeftFirst` of y, then x can be
             * removed from the dependencies.
             *
             * The set of dependencies is modified in-place.
             *
             * All dependencies must have the same body-parent.
             *
             * @param graph The kernel graph.
             * @param deps The set of dependencies to simplify.
             */
            void simplifyDependencies(KernelGraph const& graph, std::set<int>& deps);

            /**
             * @brief Merge deallocate nodes into a single node.
             */
            template <CInputRangeOf<int> Range>
            void mergeDeallocateNodes(KernelGraph& graph, int dstIdx, Range& srcs)
            {
                using namespace rocRoller::KernelGraph::CoordinateGraph;
                using namespace rocRoller::KernelGraph::ControlGraph;

                auto dst = graph.control.getNode<Deallocate>(dstIdx);

                auto connectionIdx = graph.mapper.getConnections(dstIdx).size();

                for(int srcIdx : srcs)
                {
                    auto src = graph.control.getNode<Deallocate>(srcIdx);

                    dst.arguments.insert(
                        dst.arguments.end(), src.arguments.begin(), src.arguments.end());

                    for(auto const& c : graph.mapper.getConnections(srcIdx))
                    {
                        graph.mapper.connect<Dimension>(dstIdx, c.coordinate, connectionIdx);
                        connectionIdx++;
                    }

                    deleteControlNode(graph, srcIdx);
                }
                graph.control.setElement(dstIdx, std::move(dst));
            }

            /**
             *  @brief Delete kernel arguments that are never used in the kernel.
             */
            void deleteUnusedArguments(AssemblyKernelPtr                kernel,
                                       ControlFlowArgumentTracer const& argTracer);
        }
    }
}
