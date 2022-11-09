#pragma once

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>

#include "ControlGraph/ControlGraph.hpp"
#include "CoordinateTransform/HyperGraph.hpp"

#include "KernelHypergraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /*
         * Kernel graph containers
         */

        class KernelGraph
        {
        public:
            ControlGraph::ControlGraph      control;
            CoordinateTransform::HyperGraph coordinates;

            std::string toDOT() const;
        };

        /**
         * Translate from Command to (initial) KernelGraph.
         *
         * Resulting KernelGraph matches the Command operations
         * closely.
         */
        KernelGraph translate(std::shared_ptr<Command>);

        // Delete above and rename this to 'translate' when graph rearch complete
        KernelHypergraph translate2(std::shared_ptr<Command>);

        /**
         * Rewrite KernelGraph to distribute linear packets onto GPU.
         *
         * Linear dimensions are packed/flattened, tiled onto
         * workgroups and wavefronts, and then operated on.
         */
        KernelGraph lowerLinear(KernelGraph, ContextPtr);

        // TODO Delete above when graph rearch complete
        KernelHypergraph lowerLinear(KernelHypergraph, ContextPtr);

        /**
         * Rewrite KernelGraph to additionally distribute linear dimensions onto a For loop.
         */
        KernelGraph lowerLinearLoop(KernelGraph, Expression::ExpressionPtr loopSize, ContextPtr);

        /**
         * Rewrite KernelGraph to additionally unroll linear dimensions.
         */
        KernelGraph
            lowerLinearUnroll(KernelGraph, Expression::ExpressionPtr unrollSize, ContextPtr);

        /**
         * Rewrite KernelGraph to distribute tiled packets onto GPU.
         *
         * When loading tiles, the tile size, storage location (eg,
         * VGPRs vs LDS), and affinity (eg, owned by a thread vs
         * workgroup) of each tile is specified by the destination
         * tile.  These attributes do not need to be known at
         * translation time.  To specify these attributes, call
         * `setDimension`.
         */
        KernelGraph lowerTile(KernelGraph, std::shared_ptr<CommandParameters>, ContextPtr);

        // TODO Delete above when graph rearch complete
        KernelHypergraph
            lowerTile(KernelHypergraph, std::shared_ptr<CommandParameters>, ContextPtr);

        /**
         * Rewrite KernelGraphs to make sure no more CommandArgument
         * values are present within the graph.
         */
        KernelGraph cleanArguments(KernelGraph, std::shared_ptr<AssemblyKernel>);

        // replace above when new graph arch complete
        KernelHypergraph cleanArguments(KernelHypergraph, std::shared_ptr<AssemblyKernel>);

        /**
         * Removes all CommandArgruments found within an expression
         * with the appropriate AssemblyKernel Argument.
         */
        Expression::ExpressionPtr cleanArguments(Expression::ExpressionPtr,
                                                 std::shared_ptr<AssemblyKernel>);

        /**
         * Rewrite KernelGraphs to set dimension/operation perameters.
         */
        KernelGraph updateParameters(KernelGraph, std::shared_ptr<CommandParameters>);

        // TODO Delete above when graph rearch complete
        KernelHypergraph updateParameters(KernelHypergraph, std::shared_ptr<CommandParameters>);

        /*
         * Code generation
         */
        Generator<Instruction>
            generate(KernelGraph, std::shared_ptr<Command>, std::shared_ptr<AssemblyKernel>);

        Generator<Instruction>
            generate(KernelHypergraph, std::shared_ptr<Command>, std::shared_ptr<AssemblyKernel>);

    }
}
