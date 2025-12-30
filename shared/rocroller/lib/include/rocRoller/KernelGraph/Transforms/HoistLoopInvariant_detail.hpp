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
        /**
         * @brief Extract all DataFlowTags referenced in an expression
         * @return Set of DataFlowTags found in the expression
         */
        std::set<int> extractDataFlowTags(Expression::Expression const& expr);

        using LoopToControlNodes = std::unordered_map<int, std::set<int>>;
        using CoordinateToLoops  = std::unordered_map<int, LoopToControlNodes>;
        /**
         * @brief Build a mapping of coordinates to loop groups to control nodes
         * 
         * Note the loop here is the immediate loop containing the control node,
         * does not include nested loops/scopes.
         * 
         * @return Mapping of coordinate -> loop -> [control nodes]
         */
        CoordinateToLoops buildCoordinateLoopMapping(KernelGraph const&         graph,
                                                     ControlFlowRWTracer const& tracer);

        /**
         * @brief Hoist a single node out of a loop and insert it before the loop
         */
        int hoistNodeBeforeLoop(KernelGraph& kgraph, int nodeToHoist, int loopNode);

        /**
         * @brief Count how many times a coordinate is written within a ForLoopOp, including nested loops/scopes
         * 
         * @return The number of times the coordinate is written within the loop
         */
        int countCoordinateWritesInLoop(KernelGraph const&         kgraph,
                                        int                        loopNode,
                                        int                        coordinate,
                                        ControlFlowRWTracer const& tracer);

        /**
         * @brief Check if there's a read of the coordinate before the write within the loop
         * 
         * This function detects read-before-write patterns that would prevent hoisting,
         * including cases where reads and writes are in different conditional branches.
         * 
         * @param graph The kernel graph
         * @param loopNode The loop node to check within
         * @param coordinate The coordinate being checked
         * @param writeControlNode The control node performing the write
         * @param tracer The control flow tracer with read/write information
         * @return true if there's a read before the write, false otherwise
         */
        bool hasReadBeforeWriteInLoop(KernelGraph const&         graph,
                                      int                        loopNode,
                                      int                        coordinate,
                                      int                        writeControlNode,
                                      ControlFlowRWTracer const& tracer);
    }
}
