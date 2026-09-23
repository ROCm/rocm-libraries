// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        ConstraintStatus NoUnorderedMultiplyNodes(const KernelGraph& k);

        class OrderMultiplyNodes : public GraphTransform
        {
        public:
            OrderMultiplyNodes() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "OrderMultiplyNodes";
            }

            inline std::vector<GraphConstraint> preConstraints() const override
            {
                return {};
            }

            inline std::vector<GraphConstraint> postConstraints() const override
            {
                return {&NoUnorderedMultiplyNodes};
            }
        };
    }
}
