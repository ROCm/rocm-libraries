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

#include "harness/SharedHandle.hpp"
#include "harness/TestConfig.hpp"
#include "harness/golden/BundleDiscovery.hpp"
#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"

namespace hipdnn_integration_tests::golden
{

namespace detail
{

// A discovered bundle paired with its eagerly-loaded contents. The bundle is
// loaded once at registration time (not per test run) and shared into the test
// factory via shared_ptr so the factory lambda stays copyable.
struct LoadedBundle
{
    std::filesystem::path jsonPath;
    std::string suiteName;
    std::string testName;
    std::shared_ptr<IntegrationTestBundle> bundle;
};

// The Engine executor: builds the graph from its serialized bytes, selects an
// engine (honouring an explicit --engine if given), builds plans, and executes
// into the variant pack. "Unsupported graph" is signalled by throwing (the
// harness translates that into a SKIP — a GTEST_SKIP() here would only return
// from the lambda, not TestBody). Genuine build/execute errors use ASSERT_* so
// they FAIL the test.
inline IntegrationGraphGoldenReferenceVerificationHarness::ExecuteFunc makeEngineExecutor()
{
    return [](IntegrationTestBundle& bundle) {
        auto handle = getSharedHandle();

        // SetUp guarantees tensors is present before the executor runs.
        auto& tensorMap = *bundle.tensors;

        const std::vector<uint8_t> graphBytes(
            bundle.graphBuffer.data(), bundle.graphBuffer.data() + bundle.graphBuffer.size());

        hipdnn_frontend::graph::Graph graph;
        auto err = graph.from_binary(handle, graphBytes);
        ASSERT_TRUE(err.is_good()) << "from_binary failed: " << err.get_message();

        std::vector<int64_t> engineIds;
        auto status = graph.get_ranked_engine_ids(engineIds);

        const auto graphSummary = [&] {
            return std::to_string(bundle.outputTensorUids.size()) + " output tensor(s), "
                   + std::to_string(engineIds.size()) + " ranked engine(s)";
        };

        if(TestConfig::get().hasEngineName())
        {
            int64_t targetEngineId = TestConfig::get().getEngineId();
            if(status.is_bad()
               || std::find(engineIds.begin(), engineIds.end(), targetEngineId) == engineIds.end())
            {
                throw std::runtime_error("Engine " + std::string(TestConfig::get().getEngineName())
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
        for(auto& [uid, tensor] : tensorMap)
        {
            variantPack[uid] = tensor->rawDeviceData();
        }

        result = graph.execute(handle, variantPack, workspace.get());
        ASSERT_TRUE(result.is_good()) << result.get_message();

        for(auto uid : bundle.outputTensorUids)
        {
            tensorMap.at(uid)->markDeviceModified();
        }
    };
}

// Registers one GTest test per preloaded bundle, run by the Engine executor.
// This is the runtime, macro-free equivalent of TEST_F + INSTANTIATE_TEST_SUITE_P:
// the suite/test names come from the filesystem scan, so they cannot be baked in
// at compile time the way the macros require. The bundle data is already loaded;
// each test's factory just hands its shared bundle to the harness.
//
// Engine is the only runner (CpuRef / GpuRef were removed — those executors are
// covered by the standalone pipeline tests), so the executor, the "_Engine"
// suite suffix, and the requires-device flag are fixed here rather than passed in.
inline void registerBundles(const std::vector<LoadedBundle>& bundles)
{
    const auto executor = makeEngineExecutor();

    for(const auto& bundle : bundles)
    {
        ::testing::RegisterTest(
            bundle.suiteName.c_str(), // GTest suite name (before the '.')
            bundle.testName.c_str(), // GTest test name (after the '.')
            nullptr, // type_param: unused (not a TYPED_TEST)
            nullptr, // value_param: unused (no GetParam() string)
            __FILE__,
            __LINE__, // all bundles report this site; the harness prints the
            // actual bundle path on failure (see buildFailureReport)
            // Factory: GTest calls this to construct the test when it runs. Each
            // bundle's factory captures its own preloaded bundle. The fixture is
            // the single shared harness, parameterized by the injected executor.
            [loaded = bundle.bundle, path = bundle.jsonPath, executor]() -> ::testing::Test* {
                auto* test = new IntegrationGraphGoldenReferenceVerificationHarness(
                    executor, /*requiresDevice=*/true);
                test->setBundle(loaded, path);
                return test;
            });
    }
}

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

    std::vector<DiscoveredBundle> discovered;
    try
    {
        discovered = discoverBundles(dataDir);
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Error during bundle discovery: " << e.what());
        throw;
    }

    if(discovered.empty())
    {
        HIPDNN_PLUGIN_LOG_WARN("--allow-bundles enabled but no bundles found in " << dataDir);
        return;
    }

    // Load all bundles eagerly, once, at registration time. A bundle that cannot
    // be loaded (malformed JSON, invalid graph, or missing/invalid metadata) is
    // logged and skipped — no test is registered for it. A bundle whose .bin
    // blobs are absent loads with tensors == nullopt; its test registers and the
    // harness SKIPs it at run time. A wrong-size blob throws here and is treated
    // the same as any other load failure (logged and skipped).
    std::vector<detail::LoadedBundle> bundles;
    bundles.reserve(discovered.size());
    for(const auto& disc : discovered)
    {
        LoadResult loadResult;
        try
        {
            loadResult = loadIntegrationTestBundle(disc.jsonPath);
        }
        catch(const std::exception& e)
        {
            HIPDNN_PLUGIN_LOG_ERROR("Skipping bundle " << disc.jsonPath << ": " << e.what());
            continue;
        }

        if(const auto* error = std::get_if<LoadError>(&loadResult))
        {
            HIPDNN_PLUGIN_LOG_ERROR("Skipping bundle " << disc.jsonPath << ": "
                                                       << toString(*error));
            continue;
        }

        bundles.push_back({disc.jsonPath,
                           disc.suiteName,
                           disc.testName,
                           std::make_shared<IntegrationTestBundle>(
                               std::move(std::get<IntegrationTestBundle>(loadResult)))});
    }

    if(bundles.empty())
    {
        HIPDNN_PLUGIN_LOG_WARN("No bundles could be loaded from " << dataDir);
        return;
    }

    detail::registerBundles(bundles);

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles.size() << " golden bundle test(s)");
}

} // namespace hipdnn_integration_tests::golden
