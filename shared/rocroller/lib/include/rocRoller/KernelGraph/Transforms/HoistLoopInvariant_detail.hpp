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

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/ControlGraph/ControlFlowRWTracer.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>
#include <set>

namespace rocRoller
{
    namespace KernelGraph
    {
        using LoopToControlNodes = std::unordered_map<int, std::set<int>>;
        using CoordinateToLoops  = std::unordered_map<int, LoopToControlNodes>;

        /**
         * @brief Extract all DataFlowTags referenced in an expression
         * @param expr The expression to extract tags from
         * @return Set of DataFlowTag IDs found in the expression
         */
        std::set<int> extractDataFlowTags(Expression::Expression const& expr);

        /**
         * @brief Build a mapping of coordinates to loop groups to control nodes
         * 
         * This helper function processes tracer records to create a hierarchical mapping:
         * - For each coordinate accessed in the graph
         * - Groups control nodes by the loop they belong to
         * - Special key -1 is used for control nodes not in any loop
         * 
         * @param graph The kernel graph being analyzed
         * @param tracer The control flow read/write tracer with access records
         * @return Mapping of coordinate -> loop -> [control nodes]
         */
        CoordinateToLoops buildCoordinateLoopMapping(KernelGraph const&         graph,
                                                     ControlFlowRWTracer const& tracer);

        /**
         * @brief Hoist a single node out of a loop and insert it before the loop (for testing)
         * 
         * This function disconnects the given node from the loop body and inserts it
         * into the control flow before the loop starts.
         * 
         * @param kgraph The kernel graph to modify
         * @param nodeToHoist The node to hoist out of the loop
         * @param loopNode The loop from which to hoist the node
         * @param predecessorNode The node that should precede the hoisted node
         * @param sequenceEdge The sequence edge leading into the loop (will be deleted)
         * @return The hoisted node, which becomes the new predecessor for subsequent operations
         */
        int hoistNodeBeforeLoop(KernelGraph& kgraph, int nodeToHoist, int loopNode);

        /**
         * @brief Find the enclosing ForLoopOp for a given control node
         * 
         * This function traverses up the control graph to find the first
         * ForLoopOp that contains the given node in its body.
         * 
         * @param kgraph The kernel graph to search
         * @param controlNode The control node to find the enclosing loop for
         * @return The node index of the enclosing ForLoopOp, or std::nullopt if none exists
         */
        std::optional<int> findEnclosingLoop(KernelGraph const& kgraph, int controlNode);

        /**
         * @brief Check if a coordinate is written to within a ForLoopOp
         * 
         * This function uses the ControlFlowRWTracer to check if the specified
         * coordinate is written to by any control node that is a descendant
         * of the given loop node.
         * 
         * @param kgraph The kernel graph to search
         * @param loopNode The ForLoopOp node to check within
         * @param coordinate The coordinate to check for writes
         * @param tracer The ControlFlowRWTracer with read/write information
         * @return True if the coordinate is written within the loop, false otherwise
         */
        bool isCoordinateWrittenInLoop(KernelGraph const&         kgraph,
                                       int                        loopNode,
                                       int                        coordinate,
                                       ControlFlowRWTracer const& tracer);
    }
}
