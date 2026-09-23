// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        /**
         * Removes sequence edges that aren't strictly needed for correctness while preserving
         * or re-adding edges that are required.
         *
         * Currently just does this for Multiply nodes.  It finds groups of Multiply nodes that
         * have direct connections and deletes any Sequence edges that aren't required to
         * preserve the accumulation in the K order.
         */
        class RemoveImplicitScheduling : public GraphTransform
        {
        public:
            RemoveImplicitScheduling() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "RemoveImplicitScheduling";
            }
        };
    }
}
