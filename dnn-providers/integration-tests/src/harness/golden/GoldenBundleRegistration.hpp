// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/GoldenBundleDiscovery.hpp"
#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests::golden
{

namespace detail
{

inline void registerBundlesForMode(
    const std::vector<DiscoveredBundle>& bundles,
    const std::string& runnerSuffix,
    const IntegrationGraphGoldenReferenceVerificationHarness::ExecuteFunc& executor,
    bool requiresDevice)
{
    for(const auto& bundle : bundles)
    {
        auto suiteName = bundle.suiteName + "_" + runnerSuffix;

        ::testing::RegisterTest(
            suiteName.c_str(),
            bundle.testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [path = bundle.jsonPath, executor, requiresDevice]() -> ::testing::Test* {
                auto* test = new IntegrationGraphGoldenReferenceVerificationHarness(executor,
                                                                                    requiresDevice);
                test->setBundlePath(path);
                return test;
            });
    }
}

} // namespace detail

inline std::filesystem::path resolveDataDir()
{
    auto& config = TestConfig::get();
    if(config.hasGoldenDataDir())
    {
        return config.getGoldenDataDir();
    }
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
           / "../lib/golden_reference_data";
}

inline void registerBundleTests()
{
    if(!TestConfig::get().allowBundles())
    {
        return;
    }

    auto goldenDataDir = resolveDataDir();
    if(!std::filesystem::exists(goldenDataDir))
    {
        std::cerr << "Warning: --allow-bundles enabled but golden data directory "
                     "does not exist: "
                  << goldenDataDir << '\n';
        return;
    }

    std::vector<DiscoveredBundle> bundles;
    try
    {
        bundles = discoverGoldenBundles(goldenDataDir);
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error during golden bundle discovery: " << e.what() << '\n';
        throw;
    }

    if(bundles.empty())
    {
        std::cerr << "Warning: --allow-bundles enabled but no golden bundles found in "
                  << goldenDataDir << '\n';
        return;
    }

    using Harness = IntegrationGraphGoldenReferenceVerificationHarness;

    auto cpuExecutor = [](hipdnn_test_sdk::utilities::GraphAndTensorMap& gat) {
        CpuReferenceGraphExecutorAdapter executor;
        Harness::runReferenceExecutor(executor, gat);
    };

    auto gpuExecutor = [](hipdnn_test_sdk::utilities::GraphAndTensorMap& gat) {
        gpu_graph_executor::GpuReferenceGraphExecutor executor;
        Harness::runReferenceExecutor(executor, gat);
    };

    auto engineExecutor = [](hipdnn_test_sdk::utilities::GraphAndTensorMap& gat) {
        auto handle = getSharedHandle();

        const std::vector<uint8_t> graphBytes(gat.graphBuffer.data(),
                                              gat.graphBuffer.data() + gat.graphBuffer.size());

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
        const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        std::unordered_map<int64_t, void*> variantPack;
        for(auto& [uid, tensor] : gat.tensorMap)
        {
            variantPack[uid] = tensor->rawDeviceData();
        }

        result = graph.execute(handle, variantPack, workspace.get());
        ASSERT_TRUE(result.is_good()) << result.get_message();

        for(auto uid : gat.outputTensorUids)
        {
            gat.tensorMap.at(uid)->markDeviceModified();
        }
    };

    detail::registerBundlesForMode(bundles, "CpuRef", cpuExecutor, false);
    detail::registerBundlesForMode(bundles, "GpuRef", gpuExecutor, true);
    detail::registerBundlesForMode(bundles, "Engine", engineExecutor, true);

    std::cout << "Registered " << bundles.size()
              << " golden bundle(s) across CpuRef, GpuRef, and Engine runners\n";
}

} // namespace hipdnn_integration_tests::golden
