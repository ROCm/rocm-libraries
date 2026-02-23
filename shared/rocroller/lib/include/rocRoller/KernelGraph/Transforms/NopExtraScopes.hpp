// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Changes Scope nodes (x) into a Nop node if it is possible.
         *
         * Criteria:
         * - x has no outgoing Sequence edges
         * - x is inside another Scope node (y)
         * - There are no nodes after x inside y
         *
         * The outgoing Body edges from the Scope will be replaced with Sequence edges.
         */
        class NopExtraScopes : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "NopExtraScopes";
            }
        };
    }
}
