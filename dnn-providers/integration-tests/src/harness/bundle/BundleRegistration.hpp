// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/ReferenceExecutorPool.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/BundleReferenceValidationHarness.hpp"
#include "harness/bundle/HarnessDependencies.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/ReferenceOpCoverage.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace detail
{

inline std::filesystem::path sidecarPathFor(const DiscoveredBundle& disc)
{
    if(disc.isTemplateSweepCase())
    {
        return disc.jsonPath.parent_path() / "support.json";
    }
    return supportJsonPath(disc.diagnosticPath());
}

// A discovered bundle paired with its eagerly-loaded contents. The bundle is
// loaded once at registration time (not per test run) and shared into the test
// factory via shared_ptr so the factory lambda stays copyable.
struct LoadedBundle
{
    std::filesystem::path jsonPath;
    std::string suiteName;
    std::string testName;
    std::shared_ptr<IntegrationTestBundle> bundle;
    SupportClaimLocator claimLocator;
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

    SupportClaimLocator locator;
    locator.sidecarPath = sidecarPathFor(disc);
    locator.diagnosticPath = diagnosticPath.string();
    if(disc.isTemplateSweepCase())
    {
        locator.caseId = disc.sweep->caseId;
    }

    return LoadedBundle{diagnosticPath,
                        disc.suiteName,
                        disc.testName,
                        std::make_shared<IntegrationTestBundle>(
                            std::move(std::get<IntegrationTestBundle>(loadResult))),
                        locator};
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
inline void registerBundles(const std::vector<LoadedBundle>& bundles,
                            const std::optional<LoadedEngine>& engineUnderTest)
{
    for(const auto& bundle : bundles)
    {
        ::testing::RegisterTest(bundle.suiteName.c_str(),
                                bundle.testName.c_str(),
                                nullptr,
                                nullptr,
                                __FILE__,
                                __LINE__,
                                [loaded = bundle.bundle,
                                 path = bundle.jsonPath,
                                 locator = bundle.claimLocator,
                                 engineUnderTest]() -> ::testing::Test* {
                                    auto* test = new IntegrationBundleVerificationHarness(
                                        productionDependencies(TensorPlacement::DEVICE),
                                        engineUnderTest);
                                    test->setBundle(loaded, path, locator);
                                    return test;
                                });
    }
}

// Registers one validation test per bundle this reference is *required* to handle
// and for which golden data exists. Both conditions are checked here rather than in
// the body precisely so the harness has no skip path: if a test exists, it must run
// and pass.
//
// Bundles that fall outside the reference's supported-op set are simply absent from
// the suite, and the count is logged so the gap is visible rather than silent.
inline void registerReferenceValidationTests(const std::vector<LoadedBundle>& bundles,
                                             ReferenceExecutorType referenceType)
{
    const char* label = BundleReferenceValidationHarness::referenceLabel(referenceType);

    size_t registered = 0;
    size_t noGolden = 0;
    size_t uncovered = 0;
    std::set<std::string> uncoveredOps;

    for(const auto& bundle : bundles)
    {
        if(!bundle.bundle->hasGoldenOutputs)
        {
            noGolden++;
            continue;
        }
        if(!referenceCoversGraph(
               referenceType, bundle.bundle->graphBuffer.data(), bundle.bundle->graphBuffer.size()))
        {
            uncovered++;
            // Name the ops responsible, not just the tally. The op set is a
            // commitment (see ReferenceOpCoverage.hpp): "7 bundles excluded" says a
            // gap exists, "7 excluded: ConvolutionBwdData, Reduction" says which one
            // to close.
            for(auto& nodeType : uncoveredNodeTypes(referenceType,
                                                    bundle.bundle->graphBuffer.data(),
                                                    bundle.bundle->graphBuffer.size()))
            {
                uncoveredOps.insert(std::move(nodeType));
            }
            continue;
        }

        ::testing::RegisterTest(
            (bundle.suiteName + "_" + label).c_str(),
            bundle.testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [loaded = bundle.bundle, path = bundle.jsonPath, referenceType]() -> ::testing::Test* {
                // Only the GPU reference touches a device. Passing true for the CPU
                // lane made SetUp() run SKIP_IF_NO_DEVICES() on work that reads and
                // writes host memory, so CPU golden-data validation silently skipped
                // on any runner without a GPU.
                const bool requiresDevice = referenceType == ReferenceExecutorType::GPU;
                auto* test = new BundleReferenceValidationHarness(
                    referenceType, requiresDevice, sharedReferenceExecutors());
                test->setBundle(loaded, path);
                return test;
            });
        registered++;
    }

    std::cerr << "Golden-data validation (" << label << "): " << registered
              << " bundle(s) registered, " << noGolden << " without golden data, " << uncovered
              << " outside this reference's supported-op set" << formatUncoveredOps(uncoveredOps)
              << "\n";
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

// The engine this run tests, or nothing when --test-engine was not given. main()
// has already exited non-zero if it named an engine that is not loaded, so a
// non-empty result is always a loaded engine.
inline std::optional<LoadedEngine> resolveEngineUnderTest()
{
    if(!LoadedEngineTable::get().isBuilt() || !TestConfig::get().hasEngineName())
    {
        return std::nullopt;
    }

    if(const auto* engine = LoadedEngineTable::get().find(TestConfig::get().getEngineName()))
    {
        return *engine;
    }
    return std::nullopt;
}

namespace detail
{

// Discovery plus the eager load, shared by both entry points below. Returns
// nullopt when there is nothing to register; the reason is already on stderr.
//
// `countClaimCoverage` seeds the support-claim counters as bundles load. Only the
// engine binary enforces claims, so the golden-data binary passes false rather
// than seeding counters no one will ever satisfy.
inline std::optional<std::vector<LoadedBundle>> discoverAndLoadBundles(bool countClaimCoverage)
{
    if(!TestConfig::get().allowBundles())
    {
        return std::nullopt;
    }

    auto dataDir = resolveDataDir();
    if(!std::filesystem::exists(dataDir))
    {
        std::cerr << "WARNING: Bundle tests are enabled but the data directory does not exist: "
                  << dataDir << "\n";
        return std::nullopt;
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
        std::cerr << "WARNING: Bundle tests are enabled but no bundles were found in " << dataDir
                  << "\n";
        return std::nullopt;
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
    std::vector<LoadedBundle> bundles;
    bundles.reserve(discovered.size());

    for(const auto& disc : discovered)
    {
        auto outcome = classifyBundle(disc);

        if(auto* failed = std::get_if<FailedLoad>(&outcome))
        {
            HIPDNN_PLUGIN_LOG_ERROR(failed->message);
            registerFailedBundleLoad(failed->suiteName, failed->testName, failed->message);
            continue;
        }
        if(auto* skipped = std::get_if<SkippedLoad>(&outcome))
        {
            HIPDNN_PLUGIN_LOG_ERROR(skipped->message);
            continue;
        }

        // Counted only for bundles that actually register a test. A bundle that
        // failed to load can never be queried, so counting its sidecar would make
        // the coverage guard fire on a gap it cannot close.
        if(countClaimCoverage)
        {
            supportClaimCoverage().graphsFound++;
            if(std::filesystem::exists(sidecarPathFor(disc)))
            {
                supportClaimCoverage().graphsWithClaims++;
            }
        }

        bundles.push_back(std::move(std::get<LoadedBundle>(outcome)));
    }

    if(bundles.empty())
    {
        std::cerr << "WARNING: No bundles could be loaded from " << dataDir << "\n";
        return std::nullopt;
    }

    return bundles;
}

} // namespace detail

/// Registers the engine-verification suite: one test per bundle, driven against
/// the engine named by --test-engine.
inline void registerBundleTests()
{
    // Enforcement needs a named engine to check against, so a run without
    // --test-engine has nothing to count; seeding the coverage counters anyway
    // would trip verifiedNothing() on a run that never intended to enforce.
    const std::optional<LoadedEngine> engineUnderTest = resolveEngineUnderTest();
    const bool enforcing = TestConfig::get().enforceSupportClaims() && engineUnderTest.has_value();

    auto bundles = detail::discoverAndLoadBundles(enforcing);
    if(!bundles.has_value())
    {
        return;
    }

    detail::registerBundles(*bundles, engineUnderTest);

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles->size() << " bundle test(s)");
}

/// Registers the golden-data validation suite for one reference executor.
///
/// Two harnesses, never both: verifying an engine and validating our own golden
/// data are different jobs, so they live in different binaries. This one is the
/// entry point for hipdnn_golden_data_tests, which loads no plugin and creates no
/// handle -- see the binary's main() for why that separation is structural rather
/// than a flag.
inline void registerGoldenDataValidationTests(ReferenceExecutorType referenceType)
{
    auto bundles = detail::discoverAndLoadBundles(/*countClaimCoverage=*/false);
    if(!bundles.has_value())
    {
        return;
    }

    detail::registerReferenceValidationTests(*bundles, referenceType);
}

} // namespace hipdnn_integration_tests::bundle
