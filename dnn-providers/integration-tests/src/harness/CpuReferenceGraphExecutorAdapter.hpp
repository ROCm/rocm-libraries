// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "IReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests
{

class CpuReferenceGraphExecutorAdapter : public IReferenceGraphExecutor
{
public:
    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override
    {
        _executor.execute(graphBuffer, size, variantPack);
    }

    bool requiresDeviceMemory() const override
    {
        return false;
    }

private:
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor _executor;
};

} // namespace hipdnn_integration_tests
