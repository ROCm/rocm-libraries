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

#include "LoadPacked.hpp"
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>

namespace rocRoller::KernelGraph
{
    namespace LoadPackedDetail
    {

        /**
         * Returns the set of coordinates that will be statically set when
         * generating code at `controlNode`.
         *
         * Essentially this is the set of `SetCoordinate` nodes that contain
         * `controlNode`.  If multiple `SetCoordinate` nodes which contain
         * `controlNode` set the same coordinate, the closer value is correct.
         */
        std::map<int, Expression::ExpressionPtr> getStaticCoords(int                controlNode,
                                                                 KernelGraph const& graph);

        /**
         * Returns `Register::Value` expressions for each of the
         * `requiredCoords` that is of a dimension that is typically represented
         * by a register (Wavefront, Workitem, Workgroup, ForLoop).
         */
        std::map<int, Expression::ExpressionPtr>
            fillRegisterCoords(std::unordered_set<int> const& requiredCoords,
                               KernelGraph const&             graph,
                               ContextPtr                     context);

        /**
         * Creates a Transformer that is as close as possible to the
         * Transformer we will have when generating code for controlNode.
         *
         * \retval {Transformer, set<int>}
         *
         * - Statically set dimensions will be set to those expressions.
         * - Register dimensions will be set to unallocated registers.
         * - Returns other dimensions that cannot be determined in the set.
         */
        std::tuple<CoordinateGraph::Transformer, std::set<int>> getFakeTransformerForControlNode(
            int controlNode, KernelGraph const& graph, ContextPtr context);

        /**
         * Returns the fastest moving coordinate out of `coords`, which:
         * - must be non-empty
         * - If it has more than one element, they must all be ElementNumber nodes.
         */
        int getFastestMovingCoord(std::set<int> const& coords, KernelGraph const& graph);

    }

}
