#pragma once

#include <rocRoller/AssemblyKernel_fwd.hpp>
#include <rocRoller/CommandSolution_fwd.hpp>
#include <rocRoller/Context.hpp>

#include <rocRoller/KernelGraph/ControlGraph/ControlGraph.hpp>
#include <rocRoller/KernelGraph/ControlToCoordinateMapper.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>

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
            ControlGraph::ControlGraph       control;
            CoordinateGraph::CoordinateGraph coordinates;
            ControlToCoordinateMapper        mapper;

            std::string toDOT(bool drawMappings = false, std::string title = "") const;

            template <typename T>
            std::pair<int, T> getDimension(int controlIndex, ConnectionSpec conn) const
            {
                int  tag     = mapper.get(controlIndex, conn);
                auto element = coordinates.getElement(tag);
                AssertFatal(std::holds_alternative<CoordinateGraph::Dimension>(element),
                            "Invalid connection: element isn't a Dimension.",
                            ShowValue(controlIndex));
                auto dim = std::get<CoordinateGraph::Dimension>(element);
                AssertFatal(std::holds_alternative<T>(dim),
                            "Invalid connection: Dimension type mismatch.",
                            ShowValue(controlIndex),
                            ShowValue(typeid(T).name()),
                            ShowValue(dim));
                return {tag, std::get<T>(dim)};
            }

            template <typename T>
            std::pair<int, T> getDimension(int controlIndex, int subDimension = 0) const
            {
                return getDimension<T>(controlIndex,
                                       Connections::TypeAndSubDimension{typeid(T), subDimension});
            }
        };

        /**
         * Translate from Command to (initial) KernelGraph.
         *
         * Resulting KernelGraph matches the Command operations
         * closely.
         */
        KernelGraph translate(std::shared_ptr<Command>);

        /**
         * Rewrite KernelGraph to distribute linear packets onto GPU.
         *
         * Linear dimensions are packed/flattened, tiled onto
         * workgroups and wavefronts, and then operated on.
         */
        KernelGraph lowerLinear(KernelGraph, ContextPtr);

        /**
         * Rewrite KernelGraph to additionally distribute linear dimensions onto a For loop.
         */
        KernelGraph
            lowerLinearLoop(KernelGraph const&, Expression::ExpressionPtr loopSize, ContextPtr);

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
        KernelGraph lowerTile(KernelGraph, std::shared_ptr<CommandParameters>, ContextPtr);

        /**
         * @brief Rewrite KernelGraph to add ComputeIndex operations.
         *
         * The ComputeIndex operations are used so that indices do not need to be
         * completely recalculated everytime when iterating through a tile of data.
         *
         * ComputeIndex operations are added before For Loops and calculate the
         * first index in the loop.
         *
         * A new ForLoopIncrement is added to the loop as well to increment the index
         * by the stride amount.
         *
         * Offset, Stride and Buffer edges are added to the DataFlow portion of the
         * Coordinate graph to keep track of the data needed to perform the operations.
         *
         * @param original
         * @return KernelGraph
         */
        KernelGraph addComputeIndexOperations(KernelGraph const& original);

        /**
         * @brief Rewrite KernelGraph to add Deallocate operations.
         *
         * The control graph is analysed to determine register
         * lifetimes.  Deallocate operations are added when registers
         * are no longer needed.
         */
        KernelGraph addDeallocate(KernelGraph const&);

        /**
         * @brief Rewrite KernelGraph to add LDS operations for loading data
         *
         * Modies the coordinate and control graphs to add LDS information.
         * This only adds nodes for loading data from global memory through LDS.
         * Nodes for storing data to global memory using LDS are added
         * in the lowerTile pass.
         *
         * @return KernelGraph
         */
        KernelGraph addLDS(KernelGraph const&, ContextPtr);

        /**
         * Rewrite KernelGraphs to make sure no more CommandArgument
         * values are present within the graph.
         */
        KernelGraph cleanArguments(KernelGraph, std::shared_ptr<AssemblyKernel>);

        /**
         * Removes all CommandArgruments found within an expression
         * with the appropriate AssemblyKernel Argument.
         */
        Expression::ExpressionPtr cleanArguments(Expression::ExpressionPtr,
                                                 std::shared_ptr<AssemblyKernel>);

        /**
         * @brief Performs the Loop Unrolling transformation.
         *
         * Unrolls every loop that does not have a previous iteration dependency by a value of 2.
         *
         * @return KernelGraph
         */
        KernelGraph unrollLoops(KernelGraph const&, ContextPtr);

        /**
         * @brief Performs the Loop Fusion transformation.
         *
         * Fuses multiple loops together if they iterate over the same length.
         *
         * @return KernelGraph
         */
        KernelGraph fuseLoops(KernelGraph const&, ContextPtr);

        /**
         * @brief Performs the loop cleaning transformation
         *
         * Removes forloops that only contain a single iterations
         *
         * @return KernelGraph
         */
        KernelGraph cleanLoops(KernelGraph const&);
        /**
         * @brief Performs the expression fusion transformation.
         *
         * Fuses neighbouring expressions where possible.
         *
         * @return KernelGraph
         */
        KernelGraph fuseExpressions(KernelGraph const&);

        /**
         * Rewrite KernelGraphs to set dimension/operation perameters.
         */
        KernelGraph updateParameters(KernelGraph, std::shared_ptr<CommandParameters>);

        /*
         * Code generation
         */
        Generator<Instruction> generate(KernelGraph, std::shared_ptr<AssemblyKernel>);

    }
}
