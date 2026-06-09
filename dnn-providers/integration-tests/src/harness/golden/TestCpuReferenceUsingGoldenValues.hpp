// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"

namespace hipdnn_integration_tests::golden
{

// Validates the CPU reference executor (the ALMIOPEN-1944 port) against golden
// bundle data. The base class owns load/compare; this subclass only selects
// which IReferenceGraphExecutor runs.
class TestCpuReferenceUsingGoldenValues
    : public IntegrationGraphGoldenReferenceVerificationHarness
{
protected:
    void executeUnderTest(
        hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors) override
    {
        CpuReferenceGraphExecutorAdapter executor;
        runReferenceExecutor(executor, graphAndTensors);
    }
};

} // namespace hipdnn_integration_tests::golden
