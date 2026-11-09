// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"

#include <string>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceNotApplicableError.hpp>

namespace hipdnn_integration_tests
{

struct CpuReferenceGraphExecutorAdapter::Impl
{
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor executor;
};

CpuReferenceGraphExecutorAdapter::CpuReferenceGraphExecutorAdapter()
    : _impl(std::make_unique<Impl>())
{
}

CpuReferenceGraphExecutorAdapter::~CpuReferenceGraphExecutorAdapter() = default;

bool CpuReferenceGraphExecutorAdapter::isApplicable(void* graphBuffer, size_t size)
{
    return _impl->executor.isApplicable(graphBuffer, size);
}

void CpuReferenceGraphExecutorAdapter::execute(
    void* graphBuffer, size_t size, const std::unordered_map<int64_t, void*>& variantPack)
{
    // The test_sdk CPU executor throws CpuReferenceNotApplicableError for
    // capability misses (no plan for this node type, or plan builder says
    // not applicable). We translate that into a ReferenceCapabilityError
    // (case A) so the harness can skip. Any other exception is a genuine
    // runtime failure (case C) and propagates unchanged — the harness
    // reports it as RUNTIME_ERROR → FAIL().
    try
    {
        _impl->executor.execute(graphBuffer, size, variantPack);
    }
    catch(const hipdnn_test_sdk::utilities::CpuReferenceNotApplicableError& e)
    {
        throw ReferenceCapabilityError(std::string("CPU reference executor could not run "
                                                   "this graph: ")
                                       + e.what());
    }
}

} // namespace hipdnn_integration_tests
