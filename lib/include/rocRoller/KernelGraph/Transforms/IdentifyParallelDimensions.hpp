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

/*
 * Command to KernelGraph translator
 */
#pragma once

#include <variant>
#include <vector>

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph_fwd.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Returns sets of dimensions which have been identified as parallel to
         * each other.
         *
         * The values in each set will be node IDs from the coordinate graph.
         *
         * This means that the kernel will co-iterate them, and that they must
         * have the same size.
         *
         * The returned sets may have dimensions in common with each other.
         * This means that the output should be run through mergeSets() before
         * using.
         */
        std::vector<std::set<int>> identifyParallelDimensionSets(KernelGraph const& graph);

        /**
         * Returns the set of LoadTiled nodes reachable from `start`
         * - Via Sequence edges only
         * - Without crossing any control nodes that modify dimensionality.
         * This currently means any nodes other than TensorContraction nodes.
         *
         * @param start a control node ID. Typically this should be a StoreTiled node.
         */
        std::set<int> loadNodesReachableWithoutDimensionModifyingNodes(
            ControlGraph::ControlGraph const& graph, int start);

        /**
         * @brief Performs the IdentifyParallelDimensions graph transformation.
         *
         * This transformation identifies parallel dimensions
         *  - Connected to a MatrixMultiply control op
         *  - Connected to LoadTiled/StoreTiled nodes that don't modify the dimensions
         *
         * Currently, it just reassigns the `size` field of the `SubDimension`s
         * such that they will have the same kernel argument.  We may in the
         * future want to identify this in the graph so that other code can
         * take advantage of this redundancy.  We will also want to add
         * predicates to the CommandKernel so that we check that these sizes are
         * consistent.
         */
        class IdentifyParallelDimensions : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "IdentifyParallelDimensions";
            }
        };

    }
}
