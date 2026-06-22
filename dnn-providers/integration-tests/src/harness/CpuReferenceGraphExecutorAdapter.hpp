// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdexcept>
#include <string>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "IReferenceGraphExecutor.hpp"
#include "ReferenceCapabilityError.hpp"

namespace hipdnn_integration_tests
{

class CpuReferenceGraphExecutorAdapter : public IReferenceGraphExecutor
{
public:
    void execute(void* graphBuffer,
                 size_t size,
                 const std::unordered_map<int64_t, void*>& variantPack) override
    {
        // The shared test_sdk CPU executor throws a plain std::runtime_error for
        // BOTH "no plan for this op" (capability miss, case A) and a genuine
        // runtime failure (case C) — it does not distinguish them by type. We
        // cannot tell them apart here, so we conservatively translate every throw
        // into a ReferenceCapabilityError (case A), carrying the original message
        // so a real failure still surfaces in the unverifiable report. Net effect:
        // a CPU-ref crash routes as "couldn't run" rather than a hard FAIL. The
        // GPU executor (our code) keeps full A-vs-C fidelity by throwing the right
        // type at the source.
        try
        {
            _executor.execute(graphBuffer, size, variantPack);
        }
        catch(const std::exception& e)
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
