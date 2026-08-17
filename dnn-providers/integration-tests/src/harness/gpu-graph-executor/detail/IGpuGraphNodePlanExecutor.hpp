// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <unordered_map>

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

class IGpuGraphNodePlanExecutor
{
public:
    virtual ~IGpuGraphNodePlanExecutor() = default;

    virtual void execute(const std::unordered_map<int64_t, void*>& variantPack) = 0;
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
