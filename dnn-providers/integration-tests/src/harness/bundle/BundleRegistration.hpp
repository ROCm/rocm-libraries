// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/TestConfig.hpp"
#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/GTestFilter.hpp"
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

    /// "Suite.Test": the name GTest filters on.
    std::string fullName() const
    {
        return suiteName + "." + testName;
    }
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

// A bundle that failed to load for an ordinary reason (malformed JSON, an
// absent sweep metadata block with no golden data to validate, a bad sweep
// case, ...). No test is registered for it — only the diagnostic message to
// log. Kept distinct from FailedLoad so only the failures that would otherwise
// shrink the suite behind our backs turn it red; every other load failure keeps
// the original log-and-skip behavior.
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
// runs and so can't be exercised from within a running test body. `sweeps` is
// shared across one load pass so each sweep manifest is parsed once.
inline LoadOutcome classifyBundle(const DiscoveredBundle& disc, SweepManifestCache& sweeps)
{
    const auto diagnosticPath = disc.diagnosticPath();
    LoadResult loadResult;
    try
    {
        loadResult = loadIntegrationTestBundle(disc, sweeps);
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
        // Golden blobs on disk with no usable metadata is the one load failure that
        // must not be a skip. Skipping it means pulling the DVC data *removes* a test
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
// at compile time the way the macros require. Only the graph is loaded already; each
// test's factory hands its shared bundle to the harness, which reads the tensors.
//
// Engine is the only runner (CpuRef / GpuRef were removed — those executors are
// covered by the standalone pipeline tests), so the executor and the
// requires-device flag are fixed here rather than passed in. The suite name is
// the discovered name as-is: with a single runner there is no second runner to
// disambiguate against, so no runner suffix is appended.
inline void registerBundles(const std::vector<LoadedBundle>& bundles,
                            const std::optional<LoadedEngine>& engineUnderTest,
                            const std::shared_ptr<DeviceInputFiller>& deviceFiller)
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
            [loaded = bundle.bundle,
             path = bundle.jsonPath,
             locator = bundle.claimLocator,
             engineUnderTest,
             filler = std::weak_ptr<DeviceInputFiller>(deviceFiller)]() -> ::testing::Test* {
                auto* test = new IntegrationBundleVerificationHarness(
                    productionDependencies(TensorPlacement::DEVICE, filler.lock()),
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

/// What registration did with the bundles it discovered, for the zero-tests diagnostic
/// in main(): "nothing discovered" and "everything filtered out before loading" are
/// different faults with different fixes, and the registered-test count alone cannot
/// tell them apart.
struct BundleRegistrationStats
{
    size_t discovered = 0;
    size_t excludedByFilter = 0;
};

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
    // Load all bundles eagerly, once, at registration time. A bundle that
    // fails to load because of the runtime-pass-by-value invariant (see
    // RuntimePassByValueInvariantError in IntegrationTestBundle.hpp) gets a
    // synthetic failing test registered in its place — see
    // detail::registerSyntheticBundleTest() — instead of just an ERROR log, so
    // that specific contradiction turns the suite red rather than quietly
    // shrinking it. The same applies to golden blobs whose metadata is missing or
    // unparseable (LoadError::UNVALIDATABLE_GOLDEN_DATA): pulling the data must never
    // delete a test. Every other load failure (malformed JSON, invalid graph, a bad
    // sweep case) keeps the original behavior: logged and skipped, no test registered.
    // A bundle's tensor blobs are not read here: it records where they are, and its
    // test reads them when it runs, so an absent, unreadable or wrong-size blob is that
    // test's own result.
    std::vector<LoadedBundle> bundles;
    bundles.reserve(discovered.bundles.size());

    // Loads the bundles it is handed: for the engine binary only the ones --gtest_filter
    // selects (see selectBundlesToLoad()), for the golden-data binary and
    // --write-support-claims all of them. Say how many: a large bundle root otherwise
    // looks like a hang.
    std::cerr << "Loading " << discovered.bundles.size() << " bundle test(s) from "
              << discovered.dataDir << "\n";

    SweepManifestCache sweeps;
    for(const auto& disc : discovered.bundles)
    {
        auto outcome = classifyBundle(disc, sweeps);

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

/// The discovered bundles a --gtest_filter would run, and the ones it would drop. Both
/// keep discovery order, so a sweep's cases stay adjacent for SweepManifestCache.
struct FilterSplit
{
    std::vector<DiscoveredBundle> selected;
    std::vector<DiscoveredBundle> excluded;
};

inline FilterSplit splitByGTestFilter(std::vector<DiscoveredBundle> discovered,
                                      std::string_view filter)
{
    FilterSplit split;
    for(auto& bundle : discovered)
    {
        auto& into
            = gtestFilterSelects(filter, bundle.fullName()) ? split.selected : split.excluded;
        into.push_back(std::move(bundle));
    }
    return split;
}

// Narrows the discovered bundles to the ones `filter` selects, and accounts for the rest,
// before any of them is loaded.
//
// GTest applies --gtest_filter only inside RUN_ALL_TESTS(), after every bundle would
// have been parsed, expanded and had its tensors read. Dropping what the filter is
// about to drop first makes a run's cost follow what it selects instead of the size of
// the bundle tree.
//
// `writing` (--write-support-claims) keeps every bundle: authoring needs every graph
// loaded, because `graphsFound` is the denominator for the graphs it did not observe.
//
// `observing` (an engine is named and claims are not being written) also counts the
// excluded bundles into `coverage`. The coverage ladder counts every claim-bearing
// graph on disk and attributes the gap to the filter (`not_selected`), so an excluded
// bundle is counted by sidecar presence only, which is what the load would have counted
// too. A bundle that would have failed to load is counted here and was not before; such
// a bundle is already a red test whenever a run does select it.
//
// These counters are the denominators the summary divides by, so a miscount here does
// not merely misreport -- it reattributes every gap line to the wrong cause. The
// counters are parameters so a test can check them.
inline std::vector<DiscoveredBundle> selectBundlesToLoad(std::vector<DiscoveredBundle> discovered,
                                                         std::string_view filter,
                                                         bool writing,
                                                         bool observing,
                                                         BundleRegistrationStats& stats,
                                                         SupportClaimCoverage& coverage)
{
    stats.discovered = discovered.size();
    stats.excludedByFilter = 0;

    if(writing)
    {
        return discovered;
    }

    auto split = splitByGTestFilter(std::move(discovered), filter);
    stats.excludedByFilter = split.excluded.size();

    if(!split.excluded.empty())
    {
        std::cerr << "--gtest_filter excluded " << split.excluded.size() << " of "
                  << stats.discovered << " discovered bundle test(s) before loading\n";
    }

    if(observing)
    {
        for(const auto& bundle : split.excluded)
        {
            coverage.graphsFound++;
            if(std::filesystem::exists(claimLocatorFor(bundle).sidecarPath))
            {
                coverage.graphsWithClaims++;
            }
        }
    }

    return std::move(split.selected);
}

} // namespace detail

/// Registers the engine-verification suite: one test per bundle, driven against
/// the engine named by --test-engine. `deviceFiller` generates large inputs on the
/// device; the registered tests hold it weakly, so its owner decides when it dies, and a
/// test that outlives it fills on the host. Returns what registration did with the
/// bundles it discovered.
inline BundleRegistrationStats
    registerBundleTests(const std::shared_ptr<DeviceInputFiller>& deviceFiller)
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

    BundleRegistrationStats stats;

    auto discovered = detail::discoverDataDirBundles();
    if(!discovered.has_value())
    {
        return stats;
    }

    discovered->bundles = detail::selectBundlesToLoad(std::move(discovered->bundles),
                                                      GTEST_FLAG_GET(filter),
                                                      writing,
                                                      observing,
                                                      stats,
                                                      supportClaimCoverage());

    if(discovered->bundles.empty())
    {
        return stats;
    }

    // Write mode needs `graphsFound` as the denominator for what the observer
    // saw: SetUp() can skip a bundle before the observer runs, and such a graph
    // is invisible to the observation log.
    auto bundles = detail::loadDiscoveredBundles(*discovered,
                                                 /*countFound=*/observing || writing,
                                                 /*countClaims=*/observing);
    if(!bundles.has_value())
    {
        return stats;
    }

    detail::registerBundles(*bundles, engineUnderTest, deviceFiller);

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles->size() << " bundle test(s)");
    return stats;
}

} // namespace hipdnn_integration_tests::bundle
