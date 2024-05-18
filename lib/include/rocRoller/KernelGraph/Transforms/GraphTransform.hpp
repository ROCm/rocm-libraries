/**
 * @ingroup KernelGraph
 * @defgroup Transformations Transformations
 * @brief Graph transformations (lowering passes).
 *
 * A graph transformation...
 *
 * Lowering passes include:
 * - AddComputeIndex
 * - AddConvert
 * - AddDeallocate
 * - AddLDS
 * - AddStreamK
 * - CleanArguments
 * - CleanLoops
 * - ConnectWorkgroups
 * - ConstantPropagation
 * - FuseExpressions
 * - FuseLoops
 * - InlineIncrements
 * - LoopOverTileNumbers
 * - LowerLinear
 * - LowerTensorContraction
 * - LowerTile
 * - OrderEpilogueBlocks
 * - OrderMemory
 * - Simplify
 * - UnrollLoops
 * - UpdateParameters
 */

#pragma once

#include <vector>

#include <rocRoller/KernelGraph/KernelGraph.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * @brief Base class for graph transformations.
         *
         * Contains an apply function, that takes in a KernelGraph and
         * returns a transformed kernel graph based on the
         * transformation.
         */
        class GraphTransform
        {
        public:
            GraphTransform()                                       = default;
            ~GraphTransform()                                      = default;
            virtual KernelGraph apply(KernelGraph const& original) = 0;
            virtual std::string name() const                       = 0;

            /**
             * @brief List of assumptions that must hold before
             * applying this transformation.
             */
            virtual std::vector<GraphConstraint> preConstraints() const
            {
                return {};
            }

            /**
             * @brief List of ongoing assumptions that can be made
             * after applying this transformation.
             */
            virtual std::vector<GraphConstraint> postConstraints() const
            {
                return {};
            }
        };
    }
}
