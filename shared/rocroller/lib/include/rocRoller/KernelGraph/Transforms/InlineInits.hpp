// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * - Replaces Initialize edges from anything other than ForLoopOp and
         *   ConditionalOp with Body edges from that same node.
         * - Adds Sequence edges in order to preserve the relative order of
         *   the nodes (i.e. places those nodes at the beginning of the Body).
         */
        class InlineInits : public GraphTransform
        {
        public:
            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "InlineInits";
            }
        };
    }
}
