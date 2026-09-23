// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/KernelGraph/Transforms/LoadPacked.hpp>

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
