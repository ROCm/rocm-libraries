// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace hipdnn_test_sdk::detail
{

class IGraphNodePlanExecutor
{
public:
    virtual ~IGraphNodePlanExecutor() = default;

    virtual void execute(const std::unordered_map<int64_t, void*>& variantPack) = 0;
    virtual std::vector<int64_t> getOutputTensorIds() const = 0;
};

} // namespace hipdnn_test_sdk::detail
