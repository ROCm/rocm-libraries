// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests::golden
{

class TestGpuReferenceUsingGoldenValues
    : public IntegrationGraphGoldenReferenceVerificationHarness
{
protected:
    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        IntegrationGraphGoldenReferenceVerificationHarness::SetUp();
    }

    void executeUnderTest(
        hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors) override
    {
        auto deviceBufferMap = toDeviceVariantPack(graphAndTensors);
        gpu_graph_executor::GpuReferenceGraphExecutor executor;
        executor.execute(
            graphAndTensors.graphBuffer.data(),
            graphAndTensors.graphBuffer.size(),
            deviceBufferMap);

        for(auto uid : graphAndTensors.outputTensorUids)
        {
            graphAndTensors.tensorMap.at(uid)->markDeviceModified();
        }
    }

private:
    static std::unordered_map<int64_t, void*> toDeviceVariantPack(
        hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors)
    {
        std::unordered_map<int64_t, void*> pack;
        for(auto& [uid, tensor] : graphAndTensors.tensorMap)
        {
            pack[uid] = tensor->rawDeviceData();
        }
        return pack;
    }
};

} // namespace hipdnn_integration_tests::golden
