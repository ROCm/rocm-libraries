#pragma once

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>

#include "KernelHypergraph.hpp"

namespace rocRoller
{
    namespace KernelGraph
    {
        /*
         * Kernel graph containers
         */

        /**
         * Translate from Command to (initial) KernelGraph.
         *
         * Resulting KernelGraph matches the Command operations
         * closely.
         */
        KernelHypergraph translate(std::shared_ptr<Command>);

        /**
         * Rewrite KernelGraph to distribute linear packets onto GPU.
         *
         * Linear dimensions are packed/flattened, tiled onto
         * workgroups and wavefronts, and then operated on.
         */
        KernelHypergraph lowerLinear(KernelHypergraph, ContextPtr);

        /**
         * Rewrite KernelGraph to additionally distribute linear dimensions onto a For loop.
         */
        KernelHypergraph lowerLinearLoop(KernelHypergraph const&,
                                         Expression::ExpressionPtr loopSize,
                                         ContextPtr);

        /**
         * Rewrite KernelGraph to additionally unroll linear dimensions.
         */

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
        KernelHypergraph
            lowerTile(KernelHypergraph, std::shared_ptr<CommandParameters>, ContextPtr);

        KernelHypergraph addComputeIndexAB(KernelHypergraph const&, int, int, int);
        KernelHypergraph addComputeIndexC(KernelHypergraph const&, int, int, bool);
        KernelHypergraph addComputeIndexVGPR(KernelHypergraph const&, int, int, bool);

        /**
         * @brief Rewrite KernelGraph to add Deallocate operations.
         *
         * The control graph is analysed to determine register
         * lifetimes.  Deallocate operations are added when registers
         * are no longer needed.
         */
        KernelHypergraph addDeallocate(KernelHypergraph const&);

        /**
         * Rewrite KernelGraphs to make sure no more CommandArgument
         * values are present within the graph.
         */
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
        KernelHypergraph updateParameters(KernelHypergraph, std::shared_ptr<CommandParameters>);

        /*
         * Code generation
         */
        Generator<Instruction> generate(KernelHypergraph, std::shared_ptr<AssemblyKernel>);

    }
}
