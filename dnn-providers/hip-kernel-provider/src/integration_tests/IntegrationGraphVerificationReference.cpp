// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "IntegrationGraphVerificationHarness.hpp"

namespace hip_kernel_provider::test_utilities
{

void executeCpuReferenceGraph(void* graphBuffer,
                              size_t size,
                              const std::unordered_map<int64_t, void*>& variantPack)
{
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(graphBuffer, size, variantPack);
}

} // namespace hip_kernel_provider::test_utilities
