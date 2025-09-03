/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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
             * @param compare The topological comparison function (original).
             */
            void addDownstreamBarrierInLoop(std::set<int>&            dependencies,
                                            int                       coordinate,
                                            std::set<int> const&      lastRWOps,
                                            KernelGraph const&        original,
                                            TopologicalCompare const& compare);

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
