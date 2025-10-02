// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_map>

namespace hipdnn_sdk::test_utilities
{

class IGraphNodePlanExecutor
{
public:
    virtual ~IGraphNodePlanExecutor() = default;

    virtual void execute(const std::unordered_map<int64_t, void*>& variantPack) = 0;
};

}
