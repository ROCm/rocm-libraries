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
