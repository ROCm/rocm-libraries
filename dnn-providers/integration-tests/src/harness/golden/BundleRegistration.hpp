// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

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

// Registers one GTest test per preloaded bundle, run by the Engine executor.
// This is the runtime, macro-free equivalent of TEST_F + INSTANTIATE_TEST_SUITE_P:
// the suite/test names come from the filesystem scan, so they cannot be baked in
// at compile time the way the macros require. The bundle data is already loaded;
// each test's factory just hands its shared bundle to the harness.
//
// Engine is the only runner (CpuRef / GpuRef were removed — those executors are
// covered by the standalone pipeline tests), so the executor and the
// requires-device flag are fixed here rather than passed in. The suite name is
// the discovered name as-is: with a single runner there is no second runner to
// disambiguate against, so no runner suffix is appended.
inline void registerBundles(const std::vector<LoadedBundle>& bundles)
{
    for(const auto& bundle : bundles)
    {
        ::testing::RegisterTest(
            bundle.suiteName.c_str(),
            bundle.testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [loaded = bundle.bundle, path = bundle.jsonPath]() -> ::testing::Test* {
                auto* test = new IntegrationGraphGoldenReferenceVerificationHarness(
                    /*requiresDevice=*/true);
                test->setBundle(loaded, path);
                return test;
            });
    }
}

} // namespace detail

// Resolves the bundle data root: an explicit CLI/env override from the shared
// TestConfig singleton if one was provided, otherwise the conventional install
// location next to the test binary (../lib/integration_test_bundles). This must
// match where the top-level integration-tests/CMakeLists.txt copies and installs
// the bundles (lib/integration_test_bundles).
inline std::filesystem::path resolveDataDir()
{
    auto& config = TestConfig::get();
    if(config.hasGoldenDataDir())
    {
        return config.getGoldenDataDir();
    }
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
           / "../lib/integration_test_bundles";
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
