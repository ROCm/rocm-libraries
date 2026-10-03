// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/BundleMetadata.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/HarnessDependencies.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportClaims.hpp"

namespace hipdnn_integration_tests::bundle
{

namespace detail
{

// Where this bundle's support claim lives. Delegates to the SupportClaims
// factories rather than re-deriving the paths, so the writer, the enforcer and
// the round-trip tests all name a bundle's sidecar by one rule -- a second
// derivation here could drift from that rule and have the writer overwrite a
// file the enforcer never reads.
inline SupportClaimLocator claimLocatorFor(const DiscoveredBundle& disc)
{
    if(disc.isTemplateSweepCase())
    {
        return sweepCaseClaimLocator(disc.jsonPath, disc.sweep->caseId);
    }
    return singleGraphClaimLocator(disc.jsonPath);
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

// How a synthetic test ends. FAIL stands in for a bundle that failed to load, or
// for golden data nothing validated; SKIP for a declared coverage gap.
enum class SyntheticOutcome
{
    FAIL,
    SKIP,
};

// A GTest test body that immediately fails or skips with a stored diagnostic
// message. Registered in place of a real test, so the outcome surfaces as a test
// result attributed to that bundle's suite/test name instead of only a log line
// that nothing in CI asserts on.
//
// TestBody() is public (rather than the usual protected/private override) so
// tests can invoke it directly to verify what it records; GTest's own dispatch
// through Test::Run() works the same regardless of access, since that call
// happens from within the base class.
class SyntheticBundleTest : public ::testing::Test
{
public:
    SyntheticBundleTest(SyntheticOutcome outcome, std::string message)
        : _outcome(outcome)
        , _message(std::move(message))
    {
    }

    void TestBody() override
    {
        if(_outcome == SyntheticOutcome::SKIP)
        {
            GTEST_SKIP() << _message;
        }
        ADD_FAILURE() << _message;
    }

private:
    SyntheticOutcome _outcome;
    std::string _message;
};

// Registers a synthetic test under a bundle's suite/test name. Registration of
// other, unrelated bundles is unaffected: this only replaces what would otherwise
// be a silently-dropped test.
inline void registerSyntheticBundleTest(const std::string& suiteName,
                                        const std::string& testName,
                                        SyntheticOutcome outcome,
                                        const std::string& message)
{
    ::testing::RegisterTest(suiteName.c_str(),
                            testName.c_str(),
                            nullptr,
                            nullptr,
                            __FILE__,
                            __LINE__,
                            [outcome, message]() -> ::testing::Test* {
                                return new SyntheticBundleTest(outcome, message);
                            });
}

// A bundle that failed to load, carrying enough information to register a
// failing SyntheticBundleTest in its place: the suite/test name it would have used
// had it loaded, plus a diagnostic message describing why it didn't.
struct FailedLoad
{
    std::string suiteName;
    std::string testName;
    std::string message;
};

// A bundle that failed to load for an ordinary reason (malformed graph JSON, an
// absent sweep metadata block with no golden data to validate, a bad sweep
// case, ...). No test is registered for it — only the diagnostic message to
// log. Kept distinct from FailedLoad so only the failures that would otherwise
// shrink the suite behind our backs turn it red; every other load failure keeps
// the original log-and-skip behavior. Malformed metadata is never a SkippedLoad:
// it throws BundleMetadataError and becomes a FailedLoad.
struct SkippedLoad
{
    std::string message;
};

// The result of attempting to load one discovered bundle: it loaded
// successfully, it hit a contradiction that must hard-fail, or it failed to
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
    catch(const hipdnn_integration_tests::BundleMetadataError& e)
    {
        // Malformed metadata is an authoring error, not "metadata not recorded".
        // Skipping it would let a typo silently delete the test; failing it with
        // the parser's detail tells the author exactly what to fix.
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
        // Golden blobs on disk with no metadata is the one LoadError that must not
        // be a skip. Skipping it means pulling the DVC data *removes* a test
        // and the run still passes — a more complete checkout verifying strictly less.
        // Every other error describes a bundle that was already unusable.
        if(*error == LoadError::UNVALIDATABLE_GOLDEN_DATA)
        {
            return FailedLoad{disc.suiteName,
                              disc.testName,
                              "Failed to load bundle " + diagnosticPath.string() + ": "
                                  + toString(*error)};
        }
        return SkippedLoad{"Skipping bundle " + diagnosticPath.string() + ": " + toString(*error)};
    }

    return LoadedBundle{diagnosticPath,
                        disc.suiteName,
                        disc.testName,
                        std::make_shared<IntegrationTestBundle>(
                            std::move(std::get<IntegrationTestBundle>(loadResult))),
                        claimLocatorFor(disc)};
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

// The bundle root and every bundle discovered under it.
struct DiscoveredBundleSet
{
    std::filesystem::path dataDir;
    std::vector<DiscoveredBundle> bundles;
};

// Discovery, shared by both binaries. Returns nullopt when bundles are switched off
// or there is nothing to discover; the reason is already on stderr.
inline std::optional<DiscoveredBundleSet> discoverDataDirBundles()
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

    return DiscoveredBundleSet{std::move(dataDir), std::move(discovered)};
}

// The eager load, shared by both binaries. Returns nullopt when nothing loaded; the
// reason is already on stderr.
//
// `countFound` seeds graphsFound and `countClaims` seeds graphsWithClaims as bundles
// load. Only the engine binary enforces or authors claims, so the golden-data binary
// passes false for both rather than seeding counters no one will ever satisfy.
inline std::optional<std::vector<LoadedBundle>>
    loadDiscoveredBundles(const DiscoveredBundleSet& discovered, bool countFound, bool countClaims)
{
    // Load all bundles eagerly, once, at registration time; classifyBundle()
    // decides each outcome. Three failures get a synthetic failing test
    // registered in place of the bundle (see detail::registerSyntheticBundleTest())
    // instead of just an ERROR log, so they turn the suite red rather than
    // quietly shrinking it:
    //   - the runtime-pass-by-value invariant (RuntimePassByValueInvariantError
    //     in IntegrationTestBundle.hpp);
    //   - malformed metadata (BundleMetadataError: not an object, a bad
    //     format_version or enforcement_level, a non-numeric inputs key, or a
    //     .meta.json that is unreadable or not valid JSON). Red even for
    //     graph-only bundles: a metadata typo must never delete a test;
    //   - golden blobs with no metadata at all
    //     (LoadError::UNVALIDATABLE_GOLDEN_DATA): pulling the data must never
    //     delete a test.
    // Every other load failure (malformed graph JSON, invalid graph, a bad sweep
    // case, a wrong-size blob) keeps the original behavior: logged and skipped,
    // no test registered. A bundle whose .bin blobs are absent loads with
    // tensors == nullopt; its test registers normally and the harness SKIPs it
    // at run time.
    std::vector<LoadedBundle> bundles;
    bundles.reserve(discovered.bundles.size());

    for(const auto& disc : discovered.bundles)
    {
        auto outcome = classifyBundle(disc);

        if(auto* failed = std::get_if<FailedLoad>(&outcome))
        {
            HIPDNN_PLUGIN_LOG_ERROR(failed->message);
            registerSyntheticBundleTest(
                failed->suiteName, failed->testName, SyntheticOutcome::FAIL, failed->message);
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
        if(countFound)
        {
            supportClaimCoverage().graphsFound++;
        }
        // The locator the registered test will carry, not a second derivation of it --
        // the coverage number has to count the file the run reads.
        if(countClaims
           && std::filesystem::exists(std::get<LoadedBundle>(outcome).claimLocator.sidecarPath))
        {
            supportClaimCoverage().graphsWithClaims++;
        }

        bundles.push_back(std::move(std::get<LoadedBundle>(outcome)));
    }

    if(bundles.empty())
    {
        std::cerr << "WARNING: No bundles could be loaded from " << discovered.dataDir << "\n";
        return std::nullopt;
    }

    return bundles;
}

} // namespace detail

/// Registers the engine-verification suite: one test per bundle, driven against
/// the engine named by --test-engine.
inline void registerBundleTests()
{
    // A named engine is what makes a claim checkable, so a run without --test-engine
    // has nothing to count; seeding the coverage counters anyway would print a summary
    // reporting every claim on disk as unchecked by a run that was never going to check
    // one. Not keyed on the claim mode: a warn-only run reads the same sidecars and
    // needs the same denominators, and only the cost of a broken claim differs.
    //
    // `graphsWithClaims` uses the same predicate as the harness's shouldObserveClaims(),
    // with the per-graph sidecar check done as each bundle loads. Registration seeds
    // the denominators the summary divides by, so a mismatch here does not merely
    // miscount -- it reattributes every gap line to the wrong cause.
    const std::optional<LoadedEngine> engineUnderTest = resolveEngineUnderTest();
    const bool writing = TestConfig::get().writeSupportClaims();
    const bool observing = engineUnderTest.has_value() && !writing;

    const auto discovered = detail::discoverDataDirBundles();
    if(!discovered.has_value())
    {
        return;
    }

    // Write mode needs `graphsFound` as the denominator for what the observer
    // saw: SetUp() can skip a bundle before the observer runs, and such a graph
    // is invisible to the observation log.
    auto bundles = detail::loadDiscoveredBundles(*discovered,
                                                 /*countFound=*/observing || writing,
                                                 /*countClaims=*/observing);
    if(!bundles.has_value())
    {
        return;
    }

    detail::registerBundles(*bundles, engineUnderTest);

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles->size() << " bundle test(s)");
}

} // namespace hipdnn_integration_tests::bundle
