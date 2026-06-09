// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"

namespace hipdnn_integration_tests::golden
{

class TestCpuReferenceUsingGoldenValues
    : public IntegrationGraphGoldenReferenceVerificationHarness
{
protected:
    void executeUnderTest(
        hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors) override
    {
        auto hostBuffers = graphAndTensors.hostBufferMap();
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
            graphAndTensors.graphBuffer.data(),
            graphAndTensors.graphBuffer.size(),
            hostBuffers);
    }
};

} // namespace hipdnn_integration_tests::golden
