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

// A GTest test body that immediately skips with a stored diagnostic message.
//
// Used for a bundle the cost gate excluded on a run where no other reference
// lane will cover it. Running it is not an option -- these are the shapes the
// scalar CPU reference needs tens of minutes for, which is what the gate exists
// to avoid -- but dropping it silently is what this whole binary is about not
// doing. A skip carrying the reason lands in the GTest and ctest reports under
// the bundle's own name, so "nothing validated this today" is a line someone can
// read, at no runtime cost.
class UncoveredBundleTest : public ::testing::Test
{
public:
    explicit UncoveredBundleTest(std::string message)
        : _message(std::move(message))
    {
    }

    void TestBody() override
    {
        GTEST_SKIP() << _message;
    }

private:
    std::string _message;
};

// Registers a synthetic skipping test for a bundle no reference validated this
// run. Same shape as registerFailedBundleLoad(), different verdict: this is a
// declared coverage gap, not a broken bundle.
inline void registerUncoveredBundle(const std::string& suiteName,
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
        [message]() -> ::testing::Test* { return new UncoveredBundleTest(message); });
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
//
// Returns the number of bundles registered so the caller can tell "this reference
// verified nothing" apart from "this whole run had nothing to verify".
//
// `gpuLaneWillRun` says whether a GPU reference lane will actually execute in this
// process -- selected by --reference *and* backed by a device. It does not change
// *whether* the cost gate excludes a shape, only what the exclusion means: with
// that lane running the bundle is still validated and the exclusion is a pure cost
// trade; without it nothing validates the bundle, so a skipping test is registered
// in its place to say so.
inline size_t registerReferenceValidationTests(const std::vector<LoadedBundle>& bundles,
                                               ReferenceExecutorType referenceType,
                                               bool gpuLaneWillRun)
{
    const char* label = BundleReferenceValidationHarness::referenceLabel(referenceType);

    size_t registered = 0;
    size_t noGolden = 0;
    size_t uncovered = 0;
    size_t knownGaps = 0;
    size_t tooCostly = 0;
    size_t uncoveredOnCost = 0;
    std::set<std::string> uncoveredOps;

    for(const auto& bundle : bundles)
    {
        // Belt and braces: loadGoldenDataBundles() already filtered on this, so a
        // bundle without golden outputs reaching here is a filter bug, not data.
        // Counted rather than assumed away so it surfaces instead of skewing the
        // registered-of-total line below.
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

        const std::string bundleId = bundle.suiteName + "." + bundle.testName;

        // Deliberate cost exclusion, counted and printed rather than silent. It
        // applies unconditionally: these are the shapes the scalar CPU reference
        // needs tens of minutes for -- a 4096-token GQA bundle measured 19.6 min on
        // a CI runner and blew the ctest timeout -- so running them is never the
        // right answer, whatever else is or isn't covering them.
        //
        // What the GPU lane changes is the verdict, not the exclusion. With that
        // lane running the bundle IS validated, just not here, and the tally below
        // is the whole story. Without it -- `--reference cpu`, or a device-less
        // runner where the GPU harness SKIP_IF_NO_DEVICES()s in SetUp() -- nothing
        // validates this bundle, so it gets a skipping test under its own name
        // rather than vanishing into a counter.
        if(!referenceShapeIsAffordable(referenceType,
                                       bundleId,
                                       bundle.bundle->graphBuffer.data(),
                                       bundle.bundle->graphBuffer.size()))
        {
            tooCostly++;
            if(!gpuLaneWillRun)
            {
                uncoveredOnCost++;
                registerUncoveredBundle(
                    bundle.suiteName + "_" + label,
                    bundle.testName,
                    std::string("Excluded from the ") + label
                        + " lane on cost (see referenceShapeIsAffordable), and no GpuRef lane "
                          "runs this session to cover it -- so nothing validated this bundle "
                          "against its golden data.\n  bundle: "
                        + bundle.jsonPath.string());
            }
            continue;
        }

        if(findKnownReferenceGap(referenceType, bundleId) != nullptr)
        {
            knownGaps++;
        }

        ::testing::RegisterTest(
            (bundle.suiteName + "_" + label).c_str(),
            bundle.testName.c_str(),
            nullptr,
            nullptr,
            __FILE__,
            __LINE__,
            [loaded = bundle.bundle, path = bundle.jsonPath, bundleId, referenceType]()
                -> ::testing::Test* {
                // Only the GPU reference touches a device. Passing true for the CPU
                // lane made SetUp() run SKIP_IF_NO_DEVICES() on work that reads and
                // writes host memory, so CPU golden-data validation silently skipped
                // on any runner without a GPU.
                const bool requiresDevice = referenceType == ReferenceExecutorType::GPU;
                auto* test = new BundleReferenceValidationHarness(
                    referenceType, requiresDevice, sharedReferenceExecutors());
                test->setBundle(loaded, path, bundleId);
                return test;
            });
        registered++;
    }

    // knownGaps is a subset of registered, not a third bucket: those tests still run,
    // they just assert the reference declines. Printing it separately keeps "38
    // registered" from reading as "38 validated against golden data".
    std::cerr << "Golden-data validation (" << label << "): " << registered << " of "
              << bundles.size() << " golden-bearing bundle(s) registered, " << uncovered
              << " outside this reference's supported-op set" << formatUncoveredOps(uncoveredOps);
    if(tooCostly > 0)
    {
        std::cerr << "\n       " << tooCostly
                  << " excluded as too costly for this reference (see "
                     "referenceShapeIsAffordable); ";
        if(uncoveredOnCost > 0)
        {
            std::cerr << uncoveredOnCost
                      << " of those are covered by NO reference this session (no GpuRef lane) "
                         "and are registered as skipping tests naming the gap";
        }
        else
        {
            std::cerr << "the GpuRef lane runs this session and covers them";
        }
    }
    if(knownGaps > 0)
    {
        std::cerr << "\n       " << knownGaps << " of the " << registered
                  << " registered are known reference gaps (see knownReferenceGaps()): they assert "
                     "the reference declines the graph, and are NOT validated against golden data";
    }
    if(noGolden > 0)
    {
        std::cerr << "\n       WARNING: " << noGolden
                  << " loaded bundle(s) had no golden outputs; loadGoldenDataBundles() should "
                     "already have excluded those";
    }
    std::cerr << "\n";

    // A reference lane that registered nothing while golden data was sitting right
    // there verified nothing, and the whole-binary guard in main() cannot see it: a
    // sibling reference that registered some bundles keeps the total non-zero. Every
    // bundle here carries golden data by construction -- the load filters on it --
    // so the only legitimate ways to register zero are the ones already printed
    // above: every bundle fell outside this reference's op set, or every one was
    // excluded on cost. Anything else is a bundle that should have been here and
    // isn't.
    if(registered == 0 && uncovered == 0 && tooCostly == 0)
    {
        registerFailedBundleLoad(std::string("GoldenDataValidation_") + label,
                                 "RegisteredAtLeastOneBundle",
                                 std::string("The ") + label
                                     + " reference registered no golden-data validation tests, but "
                                       "bundles carrying golden data were loaded. This lane "
                                       "verified nothing.");
    }

    return registered;
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

// Answers "could this bundle possibly carry golden outputs?" without parsing its
// graph, expanding its sweep template, or building a flatbuffer.
//
// The golden-data binary validates only bundles that have golden data, and in the
// checked-in tree that is 50 of 5711. Deciding it at registration time means
// paying the full load for the other 5661 first, which is the bulk of that
// binary's startup.
//
// The answers here are exact, not heuristic, because both sides of the real test
// bottom out in the same facts:
//
//   * a sweep case with no `golden` key resolves no golden directory
//     (resolveSweepGoldenDirectory), so goldenOutputsPresent is false;
//   * a direct bundle with no `<stem>.tensor*.bin` sibling has no output blob to
//     find, so blobsPresentFor() over its output uids is false.
//
// Both therefore end with hasGoldenOutputs == false, which is exactly what the
// registration-time filter drops.
//
// It errs toward loading whenever it cannot tell -- an unparseable or absent sweep
// manifest, a case id that is not there -- so a bundle that would report a load
// error still reaches classifyBundle() and still reports it. In particular
// UNVALIDATABLE_GOLDEN_DATA is unreachable from anything this skips: that error
// requires golden blobs to be present, and presence is what the probe tests.
class GoldenOutputProbe
{
public:
    bool mayCarryGoldenOutputs(const DiscoveredBundle& disc)
    {
        return disc.isTemplateSweepCase() ? sweepCaseDeclaresGolden(disc)
                                          : hasOutputBlobSibling(disc.jsonPath);
    }

private:
    // Parsed once per sweep.json rather than once per case: a single manifest can
    // carry thousands of cases, and re-parsing it for each is the cost this probe
    // exists to avoid.
    const nlohmann::json* sweepManifest(const std::filesystem::path& path)
    {
        const auto it = _manifests.find(path);
        if(it != _manifests.end())
        {
            return it->second.has_value() ? &*it->second : nullptr;
        }

        auto parsed = detail::parseJsonFile(path);
        const auto inserted = _manifests.emplace(path, std::move(parsed)).first;
        return inserted->second.has_value() ? &*inserted->second : nullptr;
    }

    bool sweepCaseDeclaresGolden(const DiscoveredBundle& disc)
    {
        const auto* manifest = sweepManifest(disc.jsonPath);
        if(manifest == nullptr)
        {
            return true;
        }
        const auto* caseJson = detail::findSweepCase(*manifest, disc.sweep->caseId);
        if(caseJson == nullptr)
        {
            return true;
        }
        return caseJson->contains("golden") && !caseJson->at("golden").is_null();
    }

    static bool hasOutputBlobSibling(const std::filesystem::path& jsonPath)
    {
        const auto prefix = jsonPath.stem().string() + ".tensor";
        std::error_code error;
        for(const auto& entry : std::filesystem::directory_iterator(jsonPath.parent_path(), error))
        {
            if(entry.path().extension() != ".bin")
            {
                continue;
            }
            const auto name = entry.path().filename().string();
            if(name.rfind(prefix, 0) == 0)
            {
                return true;
            }
        }
        // An unreadable directory is not evidence of absence.
        return static_cast<bool>(error);
    }

    std::map<std::filesystem::path, std::optional<nlohmann::json>> _manifests;
};

// Discovery plus the eager load, shared by both entry points below. Returns
// nullopt when there is nothing to register; the reason is already on stderr.
//
// `countClaimCoverage` seeds the support-claim counters as bundles load. Only the
// engine binary enforces claims, so the golden-data binary passes false rather
// than seeding counters no one will ever satisfy.
//
// `requireGoldenOutputs` drops, before loading them, the bundles that cannot carry
// golden data (see GoldenOutputProbe). The golden-data binary passes true: those
// bundles are ones it would load in full and then discard, and they outnumber the
// ones it validates by roughly a hundred to one. The engine binary passes false --
// it tests every bundle, golden data or not.
//
// Note this also stops the golden-data binary reporting load failures for bundles
// it has no business validating; those keep surfacing in the engine binary, which
// still loads everything. UNVALIDATABLE_GOLDEN_DATA is unaffected, since a bundle
// carrying golden blobs is never dropped by the probe.
inline std::optional<std::vector<LoadedBundle>> discoverAndLoadBundles(bool countClaimCoverage,
                                                                       bool requireGoldenOutputs)
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
    // shrinking it. The same applies to golden blobs whose metadata is missing or
    // unparseable (LoadError::UNVALIDATABLE_GOLDEN_DATA): pulling the data must never
    // delete a test. Every other load failure (malformed JSON, invalid graph, a bad
    // sweep case, a wrong-size blob) keeps the original behavior: logged and
    // skipped, no test registered. A bundle
    // whose .bin blobs are absent loads with tensors == nullopt; its test
    // registers normally and the harness SKIPs it at run time.
    std::vector<LoadedBundle> bundles;
    bundles.reserve(discovered.size());

    GoldenOutputProbe goldenProbe;
    size_t skippedWithoutGolden = 0;

    for(const auto& disc : discovered)
    {
        if(requireGoldenOutputs && !goldenProbe.mayCarryGoldenOutputs(disc))
        {
            skippedWithoutGolden++;
            continue;
        }

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

    if(skippedWithoutGolden > 0)
    {
        std::cerr << "Bundle discovery: " << bundles.size() << " bundle(s) with golden data, "
                  << skippedWithoutGolden << " without skipped before load\n";
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

    auto bundles = detail::discoverAndLoadBundles(enforcing, /*requireGoldenOutputs=*/false);
    if(!bundles.has_value())
    {
        return;
    }

    detail::registerBundles(*bundles, engineUnderTest);

    HIPDNN_PLUGIN_LOG_INFO("Registered " << bundles->size() << " bundle test(s)");
}

/// The bundles hipdnn_golden_data_tests validates: those carrying golden data.
///
/// Loaded once and handed to each reference lane, rather than rediscovered per
/// lane. Discovery plus load dominates this binary's startup, and doing it twice
/// also registered every failing-load test twice under the same name.
///
/// Returns nullopt when there is nothing to validate; the reason is already on
/// stderr.
inline std::optional<std::vector<detail::LoadedBundle>> loadGoldenDataBundles()
{
    return detail::discoverAndLoadBundles(/*countClaimCoverage=*/false,
                                          /*requireGoldenOutputs=*/true);
}

/// Registers the golden-data validation suite for one reference executor, and
/// returns how many bundles it registered.
///
/// Two harnesses, never both: verifying an engine and validating our own golden
/// data are different jobs, so they live in different binaries. This one is the
/// entry point for hipdnn_golden_data_tests, which loads no plugin and creates no
/// handle -- see the binary's main() for why that separation is structural rather
/// than a flag.
///
/// `gpuLaneWillRun` is the caller's answer to "will a GPU reference lane actually
/// execute this session" -- selected by --reference and backed by a device. It
/// decides what the CPU lane's cost exclusion means, not whether it applies.
inline size_t registerGoldenDataValidationTests(const std::vector<detail::LoadedBundle>& bundles,
                                                ReferenceExecutorType referenceType,
                                                bool gpuLaneWillRun)
{
    return detail::registerReferenceValidationTests(bundles, referenceType, gpuLaneWillRun);
}

} // namespace hipdnn_integration_tests::bundle
