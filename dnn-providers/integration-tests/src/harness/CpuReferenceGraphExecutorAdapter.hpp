// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdexcept>
#include <string>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceNotApplicableError.hpp>

#include "IReferenceGraphExecutor.hpp"
#include "ReferenceCapabilityError.hpp"

namespace hipdnn_integration_tests
{

class CpuReferenceGraphExecutorAdapter : public IReferenceGraphExecutor
{
public:
    bool isApplicable(void* graphBuffer, size_t size) override
    {
        return _executor.isApplicable(graphBuffer, size);
    }

    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override
    {
        // The test_sdk CPU executor throws CpuReferenceNotApplicableError for
        // capability misses (no plan for this node type, or plan builder says
        // not applicable). We translate that into a ReferenceCapabilityError
        // (case A) so the harness can skip. Any other exception is a genuine
        // runtime failure (case C) and propagates unchanged — the harness
        // reports it as RUNTIME_ERROR → FAIL().
        try
        {
            _executor.execute(graphBuffer, size, variantPack);
        }
        catch(const hipdnn_test_sdk::utilities::CpuReferenceNotApplicableError& e)
        {
            throw ReferenceCapabilityError(std::string("CPU reference executor could not run "
                                                       "this graph: ")
                                           + e.what());
        }
    }

    bool requiresDeviceMemory() const override
    {
        return false;
    }

private:
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor _executor;
};

} // namespace hipdnn_integration_tests
