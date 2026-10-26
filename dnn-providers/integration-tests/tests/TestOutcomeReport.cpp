// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// RFC 0023 §9: what the real harness publishes to the outcome report, one row per test.
//
// These drive the real SetUp()/TestBody() against mocked collaborators (the same wiring
// as the enforcement and rung suites) and read what reached the reporter seam. They pin
// the promise the driver relies on: one `started` and one `finished` record per test,
// whichever way it ends, with the outcome RFC 0023 §9.2 derives from what the harness
// decided.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/bundle/CaseOutcome.hpp"
#include "harness/bundle/IGraphEngineRunner.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "mocks/MockVerificationReporter.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using namespace hipdnn_integration_tests::bundle::testing_support;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
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

SupportResult makeVerdict(SupportVerdict verdict)
{
    SupportResult result;
    result.verdict = verdict;
    result.bundlePath = "test/bundle";
    result.engineName = ENGINE_NAME;
    result.arch = "gfx942";
    result.platform = "linux";
    result.detail = "stub";
    return result;
}

SupportObservation observed(std::vector<SupportResult> results)
{
    return SupportObservation{SidecarState::CHECKED, std::move(results)};
}

class TestOutcomeReport : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    HarnessMocks _mocks;

    // Every record the harness published, in publication order. `_events` interleaves
    // both kinds so the started-before-finished promise can be asserted directly.
    std::vector<CaseIdentity> _started;
    std::vector<CaseOutcomeRecord> _finished;
    std::vector<std::string> _events;

    void SetUp() override
    {
        ensureTestConfigInitialized();
        _scopedDir.emplace(claimScratchDirectory("outcome_report"));
        _tempDir = _scopedDir->path();

        using ::testing::_;
        ON_CALL(_mocks.reporter, recordCaseStarted(_))
            .WillByDefault([this](const CaseIdentity& id) {
                _started.push_back(id);
                _events.emplace_back("started");
            });
        ON_CALL(_mocks.reporter, recordCaseOutcome(_))
            .WillByDefault([this](const CaseOutcomeRecord& record) {
                _finished.push_back(record);
                _events.emplace_back("finished");
            });
    }

    std::shared_ptr<IntegrationTestBundle> bundle(bool includeGoldenOutput = true,
                                                  EnforcementLevel level = EnforcementLevel::FULL)
    {
        auto loaded = fixtures::loadBundle(_tempDir, "Bundle", includeGoldenOutput);
        loaded->metadata.enforcementLevel = level;
        return loaded;
    }

    // A sidecar that really exists, so the claim phase runs.
    SupportClaimLocator locator() const
    {
        const auto path = _tempDir / "Bundle.support.json";
        std::ofstream(path) << R"({"version": 1, "claims": {}})";

        SupportClaimLocator result;
        result.sidecarPath = path;
        result.diagnosticPath = "test/bundle";
        return result;
    }

    // Drives SetUp() and, unless SetUp() skipped, TestBody() once.
    void drive(IntegrationBundleVerificationHarness& harness,
               std::shared_ptr<IntegrationTestBundle> loaded,
               ::testing::TestPartResultArray* results,
               SupportClaimLocator claimLocator = {})
    {
        harness.setBundle(std::move(loaded), "test/bundle", std::move(claimLocator));
        driveHarness(harness, results);
    }

    // The one record every row below reads.
    const CaseOutcomeRecord& onlyFinished() const
    {
        EXPECT_EQ(_finished.size(), 1u);
        static const CaseOutcomeRecord EMPTY{};
        return _finished.empty() ? EMPTY : _finished.front();
    }

    // Runs a pinned harness over the default bundle with `configure` applied to the mocks
    // first, and returns nothing: the answer is in _finished.
    template <typename Configure>
    void runPinned(Configure&& configure,
                   EnforcementLevel level = EnforcementLevel::FULL,
                   HarnessPolicy policy = hostPolicy())
    {
        configure(_mocks);
        IntegrationBundleVerificationHarness harness(_mocks.dependencies(std::move(policy)),
                                                     makeEngineUnderTest());
        ::testing::TestPartResultArray results;
        drive(harness, bundle(true, level), &results);
    }
};

} // namespace

// ---------------------------------------------------------------------------
// The happy path, and the promise every row relies on
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, MatchingOutputIsPassWithItsStartedRecordFirst)
{
    runPinned([](HarnessMocks& mocks) {
        engineWrites(mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    });

    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished"}));
    ASSERT_EQ(_started.size(), 1u);
    EXPECT_NE(_started.front().test.find("MatchingOutputIsPassWithItsStartedRecordFirst"),
              std::string::npos);

    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::PASS);
    EXPECT_EQ(record.stage, OutcomeStage::NONE);
    EXPECT_EQ(record.depth, VerificationDepth::VERIFIED);
    EXPECT_EQ(record.verifier, Verifier::GOLDEN);
    EXPECT_TRUE(record.reason.empty());
    EXPECT_EQ(record.identity.test, _started.front().test);
    EXPECT_EQ(record.identity.bundle, "test/bundle");
    EXPECT_EQ(record.identity.engine, ENGINE_NAME);
    EXPECT_EQ(record.identity.arch, "gfx942");
    EXPECT_EQ(record.identity.platform, "linux");
}

TEST_F(TestOutcomeReport, PurposeComesFromTheCaseMetadata)
{
    auto tagged = bundle();
    tagged->metadata.purpose = CasePurpose::PROBE;
    engineWrites(_mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, tagged, &results);

    ASSERT_EQ(onlyFinished().identity.purpose.has_value(), true);
    EXPECT_EQ(*onlyFinished().identity.purpose, CasePurpose::PROBE);
    ASSERT_EQ(_started.size(), 1u);
    EXPECT_EQ(_started.front().purpose, CasePurpose::PROBE);
}

TEST_F(TestOutcomeReport, ARunTwiceReportsTwice)
{
    engineWrites(_mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results);
    ::testing::TestPartResultArray again;
    driveHarness(harness, &again);

    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished", "started", "finished"}));
}

// ---------------------------------------------------------------------------
// DECLINED, and the rule that unknown is not declined
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, ResolvedQueryWithTheEngineAbsentIsDeclined)
{
    runPinned([](HarnessMocks& mocks) {
        ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
            .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                return declinedSession();
            });
    });

    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::DECLINED);
    EXPECT_EQ(record.stage, OutcomeStage::QUERY);
    EXPECT_EQ(record.depth, VerificationDepth::NOT_REACHED);
    EXPECT_NE(record.reason.find("does not support this graph"), std::string::npos)
        << record.reason;
}

TEST_F(TestOutcomeReport, UnresolvedQueryIsUnexercisedNotDeclined)
{
    runPinned([](HarnessMocks& mocks) {
        ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
            .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                GraphSession session;
                session.engines.status = hipdnn_frontend::Error(
                    hipdnn_frontend::ErrorCode::HEURISTIC_QUERY_FAILED, "plugin crashed");
                session.engines.accepted = false;
                return session;
            });
    });

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::QUERY);
}

TEST_F(TestOutcomeReport, UnresolvedQueryAtAnEnforcementRungIsUnexercisedNotDeclined)
{
    runPinned(
        [](HarnessMocks& mocks) {
            ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
                .WillByDefault(
                    [](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                        GraphSession session;
                        session.engines.status = hipdnn_frontend::Error(
                            hipdnn_frontend::ErrorCode::HEURISTIC_QUERY_FAILED, "plugin crashed");
                        session.engines.accepted = false;
                        return session;
                    });
        },
        EnforcementLevel::APPLICABILITY);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
}

TEST_F(TestOutcomeReport, ApplicabilityRungDeclineIsDeclined)
{
    runPinned(
        [](HarnessMocks& mocks) {
            ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
                .WillByDefault(
                    [](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                        return declinedSession();
                    });
        },
        EnforcementLevel::APPLICABILITY);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::DECLINED);
}

TEST_F(TestOutcomeReport, ARunWithNoEngineNeverReportsADecline)
{
    ON_CALL(_mocks.engineRunner, openGraph(::testing::_, ::testing::_))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return declinedSession();
        });
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()));
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
}

TEST_F(TestOutcomeReport, LateDeclineFromTheEngineIsDeclined)
{
    runPinned([](HarnessMocks& mocks) {
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(::testing::Return(EngineOpResult::declinedBy("late decline")));
    });

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::DECLINED);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::BUILD);
}

TEST_F(TestOutcomeReport, DeclineWhileCompilingAtTheBuildableRungIsDeclined)
{
    runPinned(
        [](HarnessMocks& mocks) {
            ON_CALL(mocks.engineRunner, buildPlans(::testing::_, ::testing::_))
                .WillByDefault(::testing::Return(EngineOpResult::declinedBy("late decline")));
        },
        EnforcementLevel::BUILDABLE);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::DECLINED);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::BUILD);
}

// ---------------------------------------------------------------------------
// CRASH: accepted, then failed. A late NOT_SUPPORTED is one.
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, BuildErrorAfterAcceptanceIsCrash)
{
    runPinned([](HarnessMocks& mocks) {
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(::testing::Return(EngineOpResult::failed("kernel failed to compile")));
    });

    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::CRASH);
    EXPECT_EQ(record.stage, OutcomeStage::BUILD);
    EXPECT_EQ(record.origin, FailureOrigin::ENGINE);
    EXPECT_NE(record.reason.find("kernel failed to compile"), std::string::npos) << record.reason;
}

TEST_F(TestOutcomeReport, LateNotSupportedAfterAcceptanceIsCrash)
{
    runPinned([](HarnessMocks& mocks) {
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(
                ::testing::Return(EngineOpResult::failed("NOT_SUPPORTED: no plan for this graph")));
    });

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::CRASH);
}

TEST_F(TestOutcomeReport, ExecuteErrorAfterPlansBuiltIsCrashAtTheExecuteStage)
{
    runPinned([](HarnessMocks& mocks) {
        EngineOpResult failedAfterBuild = EngineOpResult::failed("launch failed");
        failedAfterBuild.plansBuilt = true;
        ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(::testing::Return(failedAfterBuild));
    });

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::CRASH);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::EXECUTE);
}

TEST_F(TestOutcomeReport, BuildErrorAtTheBuildableRungIsCrash)
{
    runPinned(
        [](HarnessMocks& mocks) {
            ON_CALL(mocks.engineRunner, buildPlans(::testing::_, ::testing::_))
                .WillByDefault(::testing::Return(EngineOpResult::failed("kaboom")));
        },
        EnforcementLevel::BUILDABLE);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::CRASH);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::BUILD);
}

// ---------------------------------------------------------------------------
// WRONG
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, ComparisonMismatchIsWrong)
{
    runPinned([](HarnessMocks& mocks) {
        engineWrites(mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE + 100.0f);
    });

    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::WRONG);
    EXPECT_EQ(record.stage, OutcomeStage::COMPARE);
    EXPECT_EQ(record.origin, FailureOrigin::COMPARISON);
    EXPECT_EQ(record.verifier, Verifier::GOLDEN);
}

// ---------------------------------------------------------------------------
// UNEXERCISED
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, GraphBuildFailureIsUnexercised)
{
    runPinned([](HarnessMocks& mocks) {
        ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
            .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                return buildErrorSession("bad flatbuffer");
            });
    });

    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(record.stage, OutcomeStage::GRAPH);
    EXPECT_NE(record.reason.find("bad flatbuffer"), std::string::npos) << record.reason;
}

TEST_F(TestOutcomeReport, AnUnverifiableSkipIsUnexercised)
{
    // An enforcement rung with no engine to check against is unverifiable.
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()));
    ::testing::TestPartResultArray results;
    drive(harness, bundle(true, EnforcementLevel::BUILDABLE), &results);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
    EXPECT_NE(onlyFinished().reason.find("--test-engine"), std::string::npos);
}

TEST_F(TestOutcomeReport, NoOracleForAnExecutedGraphIsUnexercised)
{
    // No golden data, and both references decline: the engine ran and nothing checked it.
    ON_CALL(_mocks.cpuReference, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(false));
    ON_CALL(_mocks.gpuReference, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(false));

    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(/*includeGoldenOutput=*/false), &results);

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::ORACLE);
}

TEST_F(TestOutcomeReport, PassWithAHarnessComplaintIsUnexercised)
{
    // A sidecar that exists but was never queried is a harness complaint: a failure that
    // sits beside an outcome that is otherwise a pass.
    ON_CALL(_mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(SupportObservation{SidecarState::NONE, {}}));
    engineWrites(_mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(hostPolicy(VerificationMode::AUTO, ClaimMode::ENFORCE)),
        makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results, locator());

    EXPECT_TRUE(anyFailed(results));
    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::UNEXERCISED);
    EXPECT_NE(record.reason.find("another failure"), std::string::npos) << record.reason;
}

TEST_F(TestOutcomeReport, MetadataGuardSkipInSetUpStillReports)
{
    auto locked = bundle();
    locked->metadata.gpuArchitecture = "gfx900";
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, locked, &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished"}));
    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(record.stage, OutcomeStage::SETUP);
    EXPECT_NE(record.reason.find("gfx900"), std::string::npos) << record.reason;
}

TEST_F(TestOutcomeReport, MissingBundleSkipInSetUpStillReports)
{
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished"}));
    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(onlyFinished().reason, "No bundle set");
}

TEST_F(TestOutcomeReport, ThrowWhileOpeningTheGraphIsRecordedOnceAndRethrown)
{
    ON_CALL(_mocks.engineRunner, openGraph(::testing::_, ::testing::_))
        .WillByDefault(::testing::Invoke(
            [](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) -> GraphSession {
                throw std::runtime_error("plugin exploded");
            }));

    IntegrationBundleVerificationHarness harness(_mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundle(), "test/bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        EXPECT_THROW(harness.TestBody(), std::runtime_error);
    }

    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished"}));
    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(record.stage, OutcomeStage::GRAPH);
    EXPECT_NE(record.reason.find("plugin exploded"), std::string::npos) << record.reason;
}

// ---------------------------------------------------------------------------
// Support claims
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, BrokenClaimIsDeclined)
{
    ON_CALL(_mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(observed({makeVerdict(SupportVerdict::CLAIM_BROKEN)})));

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(hostPolicy(VerificationMode::AUTO, ClaimMode::ENFORCE)),
        makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results, locator());

    // The run still fails it: the claim is the gate, the outcome only says what the
    // engine did.
    EXPECT_TRUE(anyFailed(results));
    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::DECLINED);
    EXPECT_EQ(record.stage, OutcomeStage::CLAIM);
}

TEST_F(TestOutcomeReport, ClaimBlockedByAnUnresolvedQueryIsUnexercised)
{
    ON_CALL(_mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(observed({makeVerdict(SupportVerdict::QUERY_ERRORED)})));

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(hostPolicy(VerificationMode::AUTO, ClaimMode::ENFORCE)),
        makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results, locator());

    EXPECT_TRUE(anyFailed(results));
    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(onlyFinished().stage, OutcomeStage::CLAIM);
}

TEST_F(TestOutcomeReport, SupportWithoutAClaimThatPassesIsUnclaimedPass)
{
    ON_CALL(_mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            ::testing::Return(observed({makeVerdict(SupportVerdict::UNCLAIMED_SUPPORT)})));
    engineWrites(_mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(hostPolicy(VerificationMode::AUTO, ClaimMode::ENFORCE)),
        makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results, locator());

    EXPECT_FALSE(anyFailed(results));
    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNCLAIMED_PASS);
}

TEST_F(TestOutcomeReport, ConfirmedClaimThatPassesIsPass)
{
    ON_CALL(_mocks.claimObserver,
            observe(::testing::_, ::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(observed({makeVerdict(SupportVerdict::CLAIM_ACCEPTED)})));
    engineWrites(_mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(hostPolicy(VerificationMode::AUTO, ClaimMode::ENFORCE)),
        makeEngineUnderTest());
    ::testing::TestPartResultArray results;
    drive(harness, bundle(), &results, locator());

    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::PASS);
}

// ---------------------------------------------------------------------------
// Synthetic tests: a bundle that failed to load still owes the report its two records
// ---------------------------------------------------------------------------

TEST_F(TestOutcomeReport, SyntheticLoadFailureReportsStartedAndFinished)
{
    CaseIdentity identity;
    identity.test = "suite.bad_case";
    identity.bundle = "dir/sweep.json#bad_case";

    detail::SyntheticBundleTest synthetic(detail::SyntheticOutcome::FAIL,
                                          "Failed to load bundle: bad purpose",
                                          identity,
                                          nonOwning<IVerificationReporter>(_mocks.reporter));
    EXPECT_NONFATAL_FAILURE(synthetic.TestBody(), "bad purpose");

    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished"}));
    EXPECT_EQ(_started.front().test, "suite.bad_case");
    const auto& record = onlyFinished();
    EXPECT_EQ(record.outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(record.identity.bundle, "dir/sweep.json#bad_case");
    EXPECT_NE(record.reason.find("bad purpose"), std::string::npos);
}

TEST_F(TestOutcomeReport, SyntheticSkipReportsStartedAndFinished)
{
    CaseIdentity identity;
    identity.test = "suite.skipped_case";

    detail::SyntheticBundleTest synthetic(detail::SyntheticOutcome::SKIP,
                                          "declared coverage gap",
                                          identity,
                                          nonOwning<IVerificationReporter>(_mocks.reporter));
    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        synthetic.TestBody();
    }

    EXPECT_TRUE(anySkipped(results));
    EXPECT_EQ(_events, (std::vector<std::string>{"started", "finished"}));
    EXPECT_EQ(onlyFinished().outcome, CaseOutcome::UNEXERCISED);
}

TEST_F(TestOutcomeReport, FailedLoadNamesTheBundleForTheReport)
{
    // The synthetic test's `bundle` identity comes from FailedLoad::bundle; pin that the
    // identity helper carries it and the suite/test names through unchanged.
    const auto identity = detail::syntheticIdentity("suite", "case", "dir/sweep.json#case");
    EXPECT_EQ(identity.test, "suite.case");
    EXPECT_EQ(identity.bundle, "dir/sweep.json#case");
}

// NOLINTEND(readability-identifier-naming)
