// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/GraphTensorBundle.hpp>

#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"

namespace hipdnn_integration_tests::golden
{

class IntegrationGpuGoldenReferenceEngineValidation
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
        auto handle = getSharedHandle();

        const std::vector<uint8_t> graphBytes(
            graphAndTensors.graphBuffer.data(),
            graphAndTensors.graphBuffer.data() + graphAndTensors.graphBuffer.size());

        hipdnn_frontend::graph::Graph graph;
        auto err = graph.from_binary(handle, graphBytes);
        ASSERT_TRUE(err.is_good()) << "from_binary failed: " << err.get_message();

        std::vector<int64_t> engineIds;
        auto status = graph.get_ranked_engine_ids(engineIds);

        if(TestConfig::get().hasEngineName())
        {
            int64_t targetEngineId = TestConfig::get().getEngineId();
            if(status.is_bad()
               || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
            {
                GTEST_SKIP() << "Engine " << TestConfig::get().getEngineName()
                             << " does not support this graph";
            }
            graph.set_preferred_engine_id_ext(targetEngineId);
        }
        else
        {
            if(status.is_bad() || engineIds.empty())
            {
                GTEST_SKIP() << "No engine supports this graph";
            }
        }

        auto result = graph.create_execution_plans();
        ASSERT_TRUE(result.is_good()) << result.get_message();
        result = graph.check_support();
        ASSERT_TRUE(result.is_good()) << result.get_message();
        result = graph.build_plans();
        ASSERT_TRUE(result.is_good()) << result.get_message();

        int64_t workspaceSize = 0;
        result = graph.get_workspace_size(workspaceSize);
        ASSERT_TRUE(result.is_good()) << result.get_message();
        ASSERT_GE(workspaceSize, 0);
        const hipdnn_data_sdk::utilities::Workspace workspace(
            static_cast<size_t>(workspaceSize));

        std::unordered_map<int64_t, void*> variantPack;
        for(auto& [uid, tensor] : graphAndTensors.tensorMap)
        {
            variantPack[uid] = tensor->rawDeviceData();
        }

        result = graph.execute(handle, variantPack, workspace.get());
        ASSERT_TRUE(result.is_good()) << result.get_message();

        for(auto uid : graphAndTensors.outputTensorUids)
        {
            graphAndTensors.tensorMap.at(uid)->markDeviceModified();
        }
    }
};

} // namespace hipdnn_integration_tests::golden
