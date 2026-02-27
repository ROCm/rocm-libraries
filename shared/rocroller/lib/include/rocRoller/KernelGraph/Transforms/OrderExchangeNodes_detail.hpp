// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/KernelGraph/Transforms/OrderExchangeNodes.hpp>

namespace rocRoller::KernelGraph::OrderMultiplyNodesDetail
{
    struct ExchangeOrder
    {
        int getMultiply(int exchange) const;

        bool operator()(int a, int b) const;

        KernelGraph const& graph;
    };
}
