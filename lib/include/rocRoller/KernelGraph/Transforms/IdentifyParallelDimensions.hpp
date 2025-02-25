
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
