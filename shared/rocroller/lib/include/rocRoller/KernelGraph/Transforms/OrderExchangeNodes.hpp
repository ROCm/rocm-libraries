// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/Transforms/GraphTransform.hpp>

namespace rocRoller
{
    namespace KernelGraph
    {
        class OrderExchangeNodes : public GraphTransform
        {
        public:
            OrderExchangeNodes() = default;

            KernelGraph apply(KernelGraph const& original) override;
            std::string name() const override
            {
                return "OrderExchangeNodes";
            }
        };
    }
}
