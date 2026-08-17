// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"

namespace hipdnn_integration_tests::bundle
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

// A GTest test body that immediately fails with a stored diagnostic message.
// Registered in place of a bundle that failed to load, so the failure surfaces
// as a red test result — attributed to that bundle's suite/test name — instead
// of only an ERROR log line that nothing in CI asserts on.
//
// TestBody() is public (rather than the usual protected/private override) so
// tests can invoke it directly to verify the failure message it records;
// GTest's own dispatch through Test::Run() works the same regardless of
// access, since that call happens from within the base class.
class FailedBundleLoadTest : public ::testing::Test
{
public:
    explicit FailedBundleLoadTest(std::string message)
        : _message(std::move(message))
    {
    }

    void TestBody() override
    {
        ADD_FAILURE() << _message;
    }

private:
    std::string _message;
};

// Registers a synthetic failing test for a bundle that failed to load. Keeps
// registration of the other, unrelated bundles unaffected: this only replaces
// what would otherwise be a silently-dropped test with a failing one under the
// same suite/test name.
inline void registerFailedBundleLoad(const std::string& suiteName,
                                     const std::string& testName,
                                     const std::string& message)
{
    ::testing::RegisterTest(
        suiteName.c_str(),
        testName.c_str(),
        nullptr,
        nullptr,
        __FILE__,
        __LINE__,
        [message]() -> ::testing::Test* { return new FailedBundleLoadTest(message); });
}

// A bundle that failed to load, carrying enough information to register a
// FailedBundleLoadTest in its place: the suite/test name it would have used
// had it loaded, plus a diagnostic message describing why it didn't.
struct FailedLoad
{
    std::string suiteName;
    std::string testName;
    std::string message;
};

// A bundle that failed to load for an ordinary reason (malformed JSON, missing
// metadata, a bad sweep case, ...). No test is registered for it — only the
// diagnostic message to log. Kept distinct from FailedLoad so only the
// RuntimePassByValueInvariantError contradiction (see IntegrationTestBundle.hpp)
// turns the suite red; every other load failure keeps the original
// log-and-skip behavior.
struct SkippedLoad
{
    std::string message;
};

// The result of attempting to load one discovered bundle: it loaded
// successfully, it hit the PBV invariant and should hard-fail, or it failed to
// load for some other reason and should be skipped quietly.
using LoadOutcome = std::variant<LoadedBundle, FailedLoad, SkippedLoad>;

// Attempts to load one discovered bundle and classifies the outcome. Split out
// from registerBundleTests() so the decision (did this bundle load, and if
// not, why) is a pure function that can be unit tested without touching
// ::testing::RegisterTest, which is only valid to call before RUN_ALL_TESTS()
// runs and so can't be exercised from within a running test body.
inline LoadOutcome classifyBundle(const DiscoveredBundle& disc)
{
    const auto diagnosticPath = disc.diagnosticPath();
    LoadResult loadResult;
    try
    {
        loadResult = loadIntegrationTestBundle(disc);
    }
    catch(const RuntimePassByValueInvariantError& e)
    {
        return FailedLoad{disc.suiteName,
                          disc.testName,
                          "Failed to load bundle " + diagnosticPath.string() + ": " + e.what()};
    }
    catch(const std::exception& e)
    {
        return SkippedLoad{"Skipping bundle " + diagnosticPath.string() + ": " + e.what()};
    }

    if(const auto* error = std::get_if<LoadError>(&loadResult))
    {
        return SkippedLoad{"Skipping bundle " + diagnosticPath.string() + ": " + toString(*error)};
    }

    return LoadedBundle{diagnosticPath,
                        disc.suiteName,
                        disc.testName,
                        std::make_shared<IntegrationTestBundle>(
                            std::move(std::get<IntegrationTestBundle>(loadResult)))};
}

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
                auto* test = new IntegrationBundleVerificationHarness(
                    /*requiresDevice=*/true);
                test->setBundle(loaded, path);
                return test;
            });
    }
}

} // namespace detail

// Resolves the bundle data root: an explicit CLI/env override from the shared
// TestConfig singleton if one was provided, otherwise the conventional install
// location next to the test binary (../lib/integration-test-bundles). This must
// match where the top-level integration-tests/CMakeLists.txt copies and installs
// the bundles (lib/integration-test-bundles).
inline std::filesystem::path resolveDataDir()
{
    auto& config = TestConfig::get();
    if(config.hasGoldenDataDir())
    {
        return config.getGoldenDataDir();
    }
    return hipdnn_data_sdk::utilities::getCurrentExecutableDirectory()
           / "../lib/integration-test-bundles";
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

    // Load all bundles eagerly, once, at registration time. A bundle that
    // fails to load because of the runtime-pass-by-value invariant (see
    // RuntimePassByValueInvariantError in IntegrationTestBundle.hpp) gets a
    // synthetic failing test registered in its place — see
    // detail::registerFailedBundleLoad() — instead of just an ERROR log, so
    // that specific contradiction turns the suite red rather than quietly
    // shrinking it. Every other load failure (malformed JSON, invalid graph,
    // missing/invalid metadata, a bad sweep case, a wrong-size blob) keeps the
    // original behavior: logged and skipped, no test registered. A bundle
    // whose .bin blobs are absent loads with tensors == nullopt; its test
    // registers normally and the harness SKIPs it at run time.
    std::vector<detail::LoadedBundle> bundles;
    bundles.reserve(discovered.size());
    for(const auto& disc : discovered)
    {
        auto outcome = detail::classifyBundle(disc);
        if(auto* failed = std::get_if<detail::FailedLoad>(&outcome))
        {
            HIPDNN_PLUGIN_LOG_ERROR(failed->message);
            detail::registerFailedBundleLoad(failed->suiteName, failed->testName, failed->message);
            continue;
        }
        if(auto* skipped = std::get_if<detail::SkippedLoad>(&outcome))
        {
            HIPDNN_PLUGIN_LOG_ERROR(skipped->message);
            continue;
        }

        bundles.push_back(std::move(std::get<detail::LoadedBundle>(outcome)));
    }

    if(bundles.empty())
    {
        HIPDNN_PLUGIN_LOG_WARN("No bundles could be loaded from " << dataDir);
        return;
    }

    detail::registerBundles(bundles);

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles.size() << " bundle test(s)");
}

} // namespace hipdnn_integration_tests::bundle
