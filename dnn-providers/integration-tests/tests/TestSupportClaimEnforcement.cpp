// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Enforcement lifecycle inside TestBody(): the claim query runs before anything can
// short-circuit, and an accepted claim is only published as confirmed support once
// the engine has actually run the graph green.
//
// These drive the real SetUp()/TestBody() against a real harness wired to mocked
// collaborators: the engine runner stands in for openGraph()/execute(), and the
// claim observer stands in for the sidecar query. Both seams exist precisely
// because the real ones need a handle, so the assertions below cover the routing
// and the two-phase commit, not the backend.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/bundle/IGraphEngineRunner.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/SupportClaimReport.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t ENGINE_ID = 11;
const std::string ENGINE_NAME = "ENGINE_UNDER_TEST";

LoadedEngine makeEngineUnderTest()
{
    return LoadedEngine{ENGINE_ID, ENGINE_NAME};
}

SupportResult makeVerdict(SupportVerdict verdict, std::string engineName = ENGINE_NAME)
{
    SupportResult result;
    result.verdict = verdict;
    result.bundlePath = "test/bundle";
    result.engineName = std::move(engineName);
    result.arch = "gfx942";
    result.platform = "linux";
    result.detail = "stub";
    return result;
}

// What the stubbed reference executor does when the fallback chain reaches it.
enum class RefBehavior
{
    MATCHES, ///< writes what the engine wrote, so the comparison passes
    CAPABILITY_MISS, ///< no oracle can run this op — the chain ends in a skip
    ERRORS, ///< the reference itself is broken, which is not the engine's fault
};

class TestSupportClaimEnforcement : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    testing_support::HarnessMocks _mocks;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(scratch::makeDir("claim_enforcement_"));
        _tempDir = _scopedDir->path();
    }

    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      bool includeGoldenOutput) const
    {
        return fixtures::loadBundle(_tempDir, name, includeGoldenOutput);
    }

    // A locator whose sidecar really exists, so shouldEnforceClaims() is satisfied.
    SupportClaimLocator makeLocator() const
    {
        const auto path = _tempDir / "Bundle.support.json";
        std::ofstream(path) << R"({"version": 1, "claims": {}})";

        SupportClaimLocator locator;
        locator.sidecarPath = path;
        locator.diagnosticPath = "test/bundle";
        return locator;
    }

    // Runs a fully configured harness against a bundle, with the reporter in place.
    void drive(IntegrationBundleVerificationHarness& harness,
               const std::shared_ptr<IntegrationTestBundle>& bundle,
               ::testing::TestPartResultArray* results)
    {
        harness.setBundle(bundle, "test/bundle", makeLocator());
        testing_support::driveHarness(harness, results);
    }

    // Applies `behavior` to both reference executors uniformly: nothing in this
    // file ever cares which lane (GPU/CPU) the fallback chain actually reaches.
    void setRefBehavior(RefBehavior behavior)
    {
        using ::testing::_;
        using ::testing::Throw;

        for(auto* executor : {&_mocks.cpuReference, &_mocks.gpuReference})
        {
            switch(behavior)
            {
            case RefBehavior::CAPABILITY_MISS:
                ON_CALL(*executor, execute(_, _, _))
                    .WillByDefault(Throw(ReferenceCapabilityError("stub: unsupported op")));
                break;
            case RefBehavior::ERRORS:
                ON_CALL(*executor, execute(_, _, _))
                    .WillByDefault(Throw(std::runtime_error("stub: reference exploded")));
                break;
            case RefBehavior::MATCHES:
            default:
                ON_CALL(*executor, execute(_, _, _))
                    .WillByDefault([](void*, size_t, const VariantPack& variantPack) {
                        auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
                        std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
                    });
                break;
            }
        }
    }

    // Runs the real TestBody() with a canned claim observation and a stubbed
    // engine — the common shape most cases below need.
    void run(VerificationMode mode,
             SupportObservation observation,
             bool includeGoldenOutput,
             bool engineSucceeds,
             ::testing::TestPartResultArray* results)
    {
        using ::testing::_;
        using ::testing::Return;

        ON_CALL(_mocks.claimObserver, observe(_, _, _, _, _))
            .WillByDefault(Return(std::move(observation)));
        testing_support::engineWrites(_mocks.engineRunner,
                                      &fixtures::writeOutput,
                                      engineSucceeds ? fixtures::K_OUTPUT_VALUE
                                                     : fixtures::K_OUTPUT_VALUE + 100.0f);

        IntegrationBundleVerificationHarness harness(
            _mocks.dependencies(testing_support::hostPolicy(mode, /*enforceSupportClaims=*/true)),
            makeEngineUnderTest());
        drive(harness, loadBundle("Bundle", includeGoldenOutput), results);
    }

    // The single verdict recorded for this run, for tests that care which one it is.
    static SupportVerdict onlyVerdict(const std::vector<SupportResult>& verdicts)
    {
        EXPECT_EQ(verdicts.size(), 1u);
        return verdicts.empty() ? SupportVerdict::CLAIM_BROKEN : verdicts.front().verdict;
    }

    static SupportObservation observed(std::vector<SupportResult> results)
    {
        return SupportObservation{SidecarState::CHECKED, std::move(results)};
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Coverage: the query happens above every short-circuit in runComparison().
//
// This is the regression for the original defect. Before the hoist, a FULL bundle
// with no golden blob under --verification-mode=golden returned at the mode
// dispatch, never queried its claims, contributed nothing to the summary, and the
// run still exited 0 as long as one other graph had been queried.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimEnforcement, GoldenModeWithoutDataStillQueriesClaims)
{
    using ::testing::_;

    ::testing::TestPartResultArray results;
    std::vector<CoverageUpdate> coverage;
    std::vector<SupportResult> verdicts;
    testing_support::captureCoverage(_mocks.reporter, coverage);
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    // GOLDEN mode with no golden data returns before ever touching the engine.
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(0);

    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
        /*includeGoldenOutput=*/false,
        /*engineSucceeds=*/true,
        &results);

    ASSERT_EQ(coverage.size(), 1u);
    EXPECT_TRUE(coverage.front().queried);
    EXPECT_EQ(verdicts.size(), 1u);
}

TEST_F(TestSupportClaimEnforcement, NonFullBundleStillQueriesClaims)
{
    using ::testing::_;
    using ::testing::Return;

    ::testing::TestPartResultArray results;
    std::vector<CoverageUpdate> coverage;
    std::vector<SupportResult> verdicts;
    testing_support::captureCoverage(_mocks.reporter, coverage);
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    ON_CALL(_mocks.claimObserver, observe(_, _, _, _, _))
        .WillByDefault(Return(observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)})));

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(
            testing_support::hostPolicy(VerificationMode::AUTO, /*enforceSupportClaims=*/true)),
        makeEngineUnderTest());
    auto bundle = loadBundle("Bundle", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::APPLICABILITY;
    drive(harness, bundle, &results);

    ASSERT_EQ(coverage.size(), 1u);
    EXPECT_TRUE(coverage.front().queried);
    EXPECT_EQ(verdicts.size(), 1u);
}

// A sidecar that was read but produced no enforceable verdict still counts as
// coverage; deriving the counter from the verdict vector would report a gap the run
// cannot close.
TEST_F(TestSupportClaimEnforcement, EvaluatedSidecarWithNoVerdictsStillCountsAsQueried)
{
    ::testing::TestPartResultArray results;
    std::vector<CoverageUpdate> coverage;
    std::vector<SupportResult> verdicts;
    testing_support::captureCoverage(_mocks.reporter, coverage);
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    run(VerificationMode::AUTO, observed({}), true, true, &results);

    ASSERT_EQ(coverage.size(), 1u);
    EXPECT_TRUE(coverage.front().queried);
    EXPECT_EQ(verdicts.size(), 0u);
}

// The per-graph check. The run-level guard only fires when nothing anywhere was
// queried, so a partial gap needs its own signal.
TEST_F(TestSupportClaimEnforcement, UnqueriedSidecarFailsTheRun)
{
    ::testing::TestPartResultArray results;
    std::vector<CoverageUpdate> coverage;
    testing_support::captureCoverage(_mocks.reporter, coverage);

    run(VerificationMode::AUTO, SupportObservation{SidecarState::NONE, {}}, true, true, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    ASSERT_EQ(coverage.size(), 1u);
    EXPECT_FALSE(coverage.front().queried);
}

// ---------------------------------------------------------------------------
// A broken claim is terminal before the comparison runs.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimEnforcement, BrokenClaimFailsBeforeReachingTheEngine)
{
    using ::testing::_;

    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    // Running the comparison after a decline only stacks a sentinel-buffer diff on
    // top of the real diagnostic.
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(0);

    run(VerificationMode::AUTO,
        observed({makeVerdict(SupportVerdict::CLAIM_BROKEN)}),
        true,
        true,
        &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    ASSERT_EQ(verdicts.size(), 1u);
    EXPECT_EQ(verdicts.front().verdict, SupportVerdict::CLAIM_BROKEN);
}

TEST_F(TestSupportClaimEnforcement, ErroredQueryFailsBeforeReachingTheEngine)
{
    using ::testing::_;

    ::testing::TestPartResultArray results;
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(0);

    run(VerificationMode::AUTO,
        observed({makeVerdict(SupportVerdict::QUERY_ERRORED)}),
        true,
        true,
        &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
}

// ---------------------------------------------------------------------------
// Two-phase commit: an accepted claim is an observation about the ranked list, not
// a verification. Only reaching the depth the bundle declares can promote it.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimEnforcement, AcceptedBecomesConfirmedWhenTheRunPasses)
{
    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/true,
        &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_EQ(onlyVerdict(verdicts), SupportVerdict::CLAIM_CONFIRMED);
}

// The engine ran the graph green and the comparison caught a mismatch. The claim
// held — the engine did accept the graph — but the cell must not be published as
// working support.
TEST_F(TestSupportClaimEnforcement, MismatchDemotesTheClaimToFailedInUse)
{
    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/false,
        &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_EQ(onlyVerdict(verdicts), SupportVerdict::CLAIM_FAILED_IN_USE);
}

// The regression this rework exists for. The engine executed the graph, then the
// fallback chain ran out of oracles and the test skipped. Nothing verified the
// outputs, so confirming the cell would publish support on the strength of a run
// that compared nothing.
TEST_F(TestSupportClaimEnforcement, ExecutedWithoutAnOracleStaysAccepted)
{
    using ::testing::_;
    using ::testing::Return;

    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    ON_CALL(_mocks.claimObserver, observe(_, _, _, _, _))
        .WillByDefault(Return(observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)})));
    testing_support::engineWrites(
        _mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    setRefBehavior(RefBehavior::CAPABILITY_MISS);
    EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(1);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(
            testing_support::hostPolicy(VerificationMode::AUTO, /*enforceSupportClaims=*/true)),
        makeEngineUnderTest());
    drive(harness, loadBundle("Bundle", /*includeGoldenOutput=*/false), &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_EQ(onlyVerdict(verdicts), SupportVerdict::CLAIM_ACCEPTED);
}

// A reference executor that blew up makes the run red without saying anything about
// the engine. Demoting the claim there would print "do not publish this cell" over
// somebody else's defect.
TEST_F(TestSupportClaimEnforcement, ReferenceErrorDoesNotDemoteTheClaim)
{
    using ::testing::_;
    using ::testing::Return;

    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    ON_CALL(_mocks.claimObserver, observe(_, _, _, _, _))
        .WillByDefault(Return(observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)})));
    testing_support::engineWrites(
        _mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    setRefBehavior(RefBehavior::ERRORS);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(
            testing_support::hostPolicy(VerificationMode::CPU, /*enforceSupportClaims=*/true)),
        makeEngineUnderTest());
    drive(harness, loadBundle("Bundle", /*includeGoldenOutput=*/false), &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "a broken reference must still fail the test";
    EXPECT_EQ(onlyVerdict(verdicts), SupportVerdict::CLAIM_ACCEPTED);
}

// Confirmation is measured against the depth the bundle declares, not against a
// fixed "outputs were compared". A buildable bundle whose plans compiled has done
// everything it promises, so its claim is confirmed rather than stuck at accepted.
TEST_F(TestSupportClaimEnforcement, BuildableBundleConfirmsAtItsOwnDepth)
{
    using ::testing::_;
    using ::testing::Return;

    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    ON_CALL(_mocks.claimObserver, observe(_, _, _, _, _))
        .WillByDefault(Return(observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)})));
    // The default ranked-list acceptance plus a compiling plan is everything the
    // BUILDABLE rung asks for.
    ON_CALL(_mocks.engineRunner, buildPlans(_, _))
        .WillByDefault(Return(EngineOpResult::succeeded()));

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(
            testing_support::hostPolicy(VerificationMode::AUTO, /*enforceSupportClaims=*/true)),
        makeEngineUnderTest());
    auto bundle = loadBundle("Bundle", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::BUILDABLE;
    drive(harness, bundle, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_EQ(onlyVerdict(verdicts), SupportVerdict::CLAIM_CONFIRMED);
}

// An enforcement rung that could not even establish applicability leaves the claim
// exactly where the ranked-list query put it.
TEST_F(TestSupportClaimEnforcement, UnreachedEnforcementRungStaysAccepted)
{
    using ::testing::_;
    using ::testing::Return;

    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    ON_CALL(_mocks.claimObserver, observe(_, _, _, _, _))
        .WillByDefault(Return(observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)})));
    // The rung cannot even establish applicability when the engine never took the
    // graph, which is what leaves this rung unreached.
    ON_CALL(_mocks.engineRunner, openGraph(_, _))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return testing_support::declinedSession();
        });

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(
            testing_support::hostPolicy(VerificationMode::AUTO, /*enforceSupportClaims=*/true)),
        makeEngineUnderTest());
    auto bundle = loadBundle("Bundle", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::BUILDABLE;
    drive(harness, bundle, &results);

    EXPECT_EQ(onlyVerdict(verdicts), SupportVerdict::CLAIM_ACCEPTED);
}

// The promotion policy itself is pinned in TestSupportVerdict.cpp, where it needs no
// fixture; these cases cover the combinations the deviceless harness can actually
// drive end to end.

// Only the engine this test drove can be promoted. Another engine's claim was
// decided from the same ranked list but never executed, so the run has no
// evidence either way about it.
TEST_F(TestSupportClaimEnforcement, OnlyTheDrivenEngineIsPromoted)
{
    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED),
                  makeVerdict(SupportVerdict::CLAIM_ACCEPTED, "OTHER_ENGINE")}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/true,
        &results);

    ASSERT_EQ(verdicts.size(), 2u);
    EXPECT_EQ(std::count_if(verdicts.begin(),
                            verdicts.end(),
                            [](const SupportResult& r) {
                                return r.verdict == SupportVerdict::CLAIM_CONFIRMED;
                            }),
              1);
    EXPECT_EQ(std::count_if(verdicts.begin(),
                            verdicts.end(),
                            [](const SupportResult& r) {
                                return r.verdict == SupportVerdict::CLAIM_ACCEPTED;
                            }),
              1);
}

// Positive drift keeps its verdict — there is no claim to promote — but picks up how
// far the run got, which is what separates "this cell works, write it down" from
// "the ranked list said so and nothing tried it".
TEST_F(TestSupportClaimEnforcement, UnclaimedSupportKeepsItsVerdictAndGainsTheDepth)
{
    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    run(VerificationMode::GOLDEN,
        observed({makeVerdict(SupportVerdict::UNCLAIMED_SUPPORT)}),
        /*includeGoldenOutput=*/true,
        /*engineSucceeds=*/true,
        &results);

    ASSERT_EQ(verdicts.size(), 1u);
    EXPECT_EQ(verdicts.front().verdict, SupportVerdict::UNCLAIMED_SUPPORT);
    EXPECT_NE(verdicts.front().detail.find("verified"), std::string::npos);
}

// Every observed verdict reaches the report exactly once, including on the terminal
// failure path — the report is the run's dashboard and must not lose rows.
TEST_F(TestSupportClaimEnforcement, EveryVerdictIsRecordedExactlyOnce)
{
    ::testing::TestPartResultArray results;
    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);

    run(VerificationMode::AUTO,
        observed({makeVerdict(SupportVerdict::CLAIM_BROKEN),
                  makeVerdict(SupportVerdict::UNCLAIMED_SUPPORT, "OTHER_ENGINE")}),
        true,
        true,
        &results);

    EXPECT_EQ(verdicts.size(), 2u);
}

// NOLINTEND(readability-identifier-naming)
