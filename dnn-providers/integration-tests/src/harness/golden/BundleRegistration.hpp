// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/BundleDiscovery.hpp"
#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

namespace hipdnn_integration_tests::golden
{

namespace detail
{

// Registers one GTest test per discovered bundle for a single runner mode
// (CpuRef / GpuRef / Engine). This is the runtime, macro-free equivalent of
// TEST_F + INSTANTIATE_TEST_SUITE_P: the suite/test names and the "parameter"
// (each bundle's .json path) come from the filesystem scan, so they cannot be
// baked in at compile time the way the macros require.
inline void registerBundlesForMode(
    const std::vector<DiscoveredBundle>& bundles,
    const std::string& runnerSuffix,
    const IntegrationGraphGoldenReferenceVerificationHarness::ExecuteFunc& executor,
    bool requiresDevice)
{
    for(const auto& bundle : bundles)
    {
        // Suffix the suite so the same bundle registers as a distinct test under
        // each mode, e.g. "ConvFwd_nchw_fp16" -> "ConvFwd_nchw_fp16_CpuRef".
        auto suiteName = bundle.suiteName + "_" + runnerSuffix;

        ::testing::RegisterTest(
            suiteName.c_str(), // GTest suite name (before the '.')
            bundle.testName.c_str(), // GTest test name (after the '.')
            nullptr, // type_param: unused (not a TYPED_TEST)
            nullptr, // value_param: unused (no GetParam() string)
            __FILE__,
            __LINE__, // all bundles report this site; the harness prints the
            // actual bundle path on failure (see buildFailureReport)
            // Factory: GTest calls this to construct the test when it runs. Each
            // bundle's factory captures its own path/executor, so test N loads
            // bundle N. The fixture is the single shared harness, parameterized
            // by the injected executor + requiresDevice rather than subclassed.
            [path = bundle.jsonPath, executor, requiresDevice]() -> ::testing::Test* {
                auto* test = new IntegrationGraphGoldenReferenceVerificationHarness(executor,
                                                                                    requiresDevice);
                test->setBundlePath(path);
                return test;
            });
    }
}

// In short: turns the list of discovered bundles into live GTest cases for one
// runner mode at startup — each bundle becomes "{suite}_{mode}.{test}", built to
// load and verify its own bundle when run.

} // namespace detail

// Resolves the bundle data root: an explicit CLI/env override from the shared
// TestConfig singleton if one was provided, otherwise the conventional install
// location next to the test binary (../lib/golden_reference_data).
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

    auto dataDir = resolveDataDir();
    if(!std::filesystem::exists(dataDir))
    {
        HIPDNN_PLUGIN_LOG_WARN(
            "--allow-bundles enabled but data directory does not exist: " << dataDir);
        return;
    }

    std::vector<DiscoveredBundle> bundles;
    try
    {
        bundles = discoverBundles(dataDir);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Error during bundle discovery: " << e.what());
        throw;
    }

    if(bundles.empty())
    {
        HIPDNN_PLUGIN_LOG_WARN("--allow-bundles enabled but no bundles found in " << dataDir);
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

        // "Unsupported graph" is signalled by throwing, not GTEST_SKIP(): this
        // lambda runs inside TestBody (via the harness' _executeFunc), where a
        // GTEST_SKIP() would only return from the lambda and let the comparison
        // proceed against zeroed outputs. The harness catches this and issues the
        // real skip. We include the engine context the executor knows; the
        // harness adds the bundle path on its side.
        const auto graphSummary = [&] {
            return std::to_string(gat.outputTensorUids.size()) + " output tensor(s), "
                   + std::to_string(engineIds.size()) + " ranked engine(s)";
        };

        if(TestConfig::get().hasEngineName())
        {
            int64_t targetEngineId = TestConfig::get().getEngineId();
            if(status.is_bad()
               || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
            {
                throw std::runtime_error("Engine " + TestConfig::get().getEngineName()
                                         + " does not support this graph (" + graphSummary() + ")");
            }
            graph.set_preferred_engine_id_ext(targetEngineId);
        }
        else
        {
            if(status.is_bad() || engineIds.empty())
            {
                throw std::runtime_error("No engine supports this graph (" + graphSummary() + ")");
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

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles.size()
                                         << " bundle(s) across CpuRef, GpuRef, and Engine runners");
}

} // namespace hipdnn_integration_tests::golden
