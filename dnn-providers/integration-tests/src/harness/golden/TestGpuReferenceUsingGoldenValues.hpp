// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests::golden
{

// Validates the GPU reference executor (the ALMIOPEN-1944 port) against golden
// bundle data. The base class owns load/compare and, via runReferenceExecutor,
// the device variant-pack handling; this subclass only selects the executor.
class TestGpuReferenceUsingGoldenValues : public IntegrationGraphGoldenReferenceVerificationHarness
{
protected:
    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        IntegrationGraphGoldenReferenceVerificationHarness::SetUp();
    }

    void executeUnderTest(hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors) override
    {
        gpu_graph_executor::GpuReferenceGraphExecutor executor;
        runReferenceExecutor(executor, graphAndTensors);
    }
};

} // namespace hipdnn_integration_tests::golden
