// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Expected-rejection bundles (metadata `expected_rejection`): the engine under test
// must refuse the graph, and nothing is ever filled, executed or compared. These drive
// the real SetUp()/TestBody() against a real harness wired to mocked collaborators, the
// same way TestSupportClaimEnforcement does; the engine runner mock stands in for the
// ranked query and plan build, which is all this path touches.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/SupportObservationLog.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t ENGINE_ID = 11;
const std::string ENGINE_NAME = "ENGINE_UNDER_TEST";

// What the ranked query said: `resolved` is OK / GRAPH_NOT_SUPPORTED, and `ranked` says
// whether the engine under test was in the list. A pass needs the first, and the
// second only decides where the engine refused.
GraphSession rankedQuery(bool resolved, bool ranked)
{
    GraphSession session;
    session.engines.status
        = resolved ? Error() : Error(ErrorCode::INVALID_VALUE, "stub: ranked query exploded");
    session.engines.accepted = resolved && ranked;
    return session;
}

class TestExpectedRejection : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    testing_support::HarnessMocks _mocks;
    std::vector<Verifier> _verifiers;
    std::vector<std::string> _verifierDetails;
    std::vector<std::string> _unverifiable;
    HarnessPolicy _policy = testing_support::hostPolicy(VerificationMode::AUTO, ClaimMode::ENFORCE);

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(claimScratchDirectory("expected_rejection"));
        _tempDir = _scopedDir->path();
        testing_support::captureVerifiers(_mocks.reporter, _verifiers, &_verifierDetails);
        testing_support::captureUnverifiable(_mocks.reporter, _unverifiable);
        SupportObservationLog::get().reset();
    }

    void TearDown() override
    {
        SupportObservationLog::get().reset();
    }

    std::shared_ptr<IntegrationTestBundle> rejectedBundle() const
    {
        auto bundle = fixtures::loadBundle(_tempDir, "Rejected", /*includeGoldenOutput=*/false);
        bundle->metadata.expectedRejection = true;
        return bundle;
    }

    void answerRankedQueryWith(bool resolved, bool ranked)
    {
        using ::testing::_;
        ON_CALL(_mocks.engineRunner, openGraph(_, _))
            .WillByDefault([resolved, ranked](const IntegrationTestBundle&,
                                              const std::optional<LoadedEngine>&) {
                return rankedQuery(resolved, ranked);
            });
    }

    // Nothing but the ranked query and the plan build may be touched: an expected
    // rejection is never run, and its claims are nobody's business.
    void expectNeverExecutedOrObserved()
    {
        using ::testing::_;
        EXPECT_CALL(_mocks.engineRunner, execute(_, _, _)).Times(0);
        EXPECT_CALL(_mocks.claimObserver, observe(_, _, _, _, _)).Times(0);
        EXPECT_CALL(_mocks.referenceExecutors, get(_)).Times(0);
        EXPECT_CALL(_mocks.reporter, recordSelectedWithClaims()).Times(0);
        EXPECT_CALL(_mocks.reporter, recordReachedBody()).Times(0);
        EXPECT_CALL(_mocks.reporter, recordCoverage(_)).Times(0);
    }

    // A locator whose sidecar really exists, so the claim counters would count a bundle
    // that was not excluded.
    SupportClaimLocator writeSidecar(const std::string& content, const std::string& caseId = {})
    {
        const auto path = _tempDir / (caseId.empty() ? "Rejected.support.json" : "support.json");
        std::ofstream(path) << content;

        SupportClaimLocator locator;
        locator.sidecarPath = path;
        locator.caseId = caseId;
        locator.diagnosticPath = "test/bundle";
        return locator;
    }

    void run(std::shared_ptr<IntegrationTestBundle> bundle,
             std::optional<LoadedEngine> engine,
             ::testing::TestPartResultArray* results,
             SupportClaimLocator locator = {})
    {
        IntegrationBundleVerificationHarness harness(_mocks.dependencies(_policy),
                                                     std::move(engine));
        harness.setBundle(std::move(bundle), "test/rejected", std::move(locator));
        testing_support::driveHarness(harness, results);
    }

    void runPinned(::testing::TestPartResultArray* results, SupportClaimLocator locator = {})
    {
        run(rejectedBundle(), LoadedEngine{ENGINE_ID, ENGINE_NAME}, results, std::move(locator));
    }
};

const char* const SINGLE_GRAPH_CLAIM
    = R"({"version": 1, "claims": {"ENGINE_UNDER_TEST": {"gfx942": ["linux"]}}})";

} // namespace

// An engine that is not in a resolved ranked list has refused the graph. Both codes
// that resolve the list count, and the pass is not a shallow one and not unverifiable.
TEST_F(TestExpectedRejection, EngineAbsentFromTheRankedListPassesAtThatStage)
{
    using ::testing::_;
    for(const auto code : {ErrorCode::OK, ErrorCode::GRAPH_NOT_SUPPORTED})
    {
        SCOPED_TRACE(static_cast<int>(code));
        _verifiers.clear();
        _verifierDetails.clear();
        ON_CALL(_mocks.engineRunner, openGraph(_, _))
            .WillByDefault(
                [code](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
                    GraphSession session;
                    session.engines.status = Error(code, "stub");
                    return session;
                });
        EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(0);
        expectNeverExecutedOrObserved();

        ::testing::TestPartResultArray results;
        runPinned(&results, writeSidecar(R"({"version": 1, "claims": {}})"));

        EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
        EXPECT_FALSE(testing_support::anySkipped(results));
        EXPECT_THAT(_verifiers, ::testing::ElementsAre(Verifier::EXPECTED_REJECTION));
        EXPECT_THAT(_verifierDetails, ::testing::ElementsAre("ranked_list"));
        EXPECT_TRUE(_unverifiable.empty());
    }
}

// Ranked is not accepted: the engine may still refuse to build the graph, and that is
// a rejection too, whether it declines or breaks.
TEST_F(TestExpectedRejection, RankedEngineThatDeclinesOrFailsToBuildPlansPasses)
{
    using ::testing::_;
    using ::testing::Return;
    const std::vector<std::pair<EngineOpResult, const char*>> answers{
        {EngineOpResult::declinedBy("stub: not mine"), "plan_build_declined"},
        {EngineOpResult::failed("stub: cannot build"), "plan_build_failed"}};
    for(const auto& [answer, stage] : answers)
    {
        SCOPED_TRACE(stage);
        _verifiers.clear();
        _verifierDetails.clear();
        answerRankedQueryWith(/*resolved=*/true, /*ranked=*/true);
        EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).WillOnce(Return(answer));
        expectNeverExecutedOrObserved();

        ::testing::TestPartResultArray results;
        runPinned(&results);

        EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
        EXPECT_FALSE(testing_support::anySkipped(results));
        EXPECT_THAT(_verifiers, ::testing::ElementsAre(Verifier::EXPECTED_REJECTION));
        EXPECT_THAT(_verifierDetails, ::testing::ElementsAre(stage));
    }
}

// The one way to fail: the engine took a graph it must refuse.
TEST_F(TestExpectedRejection, EngineThatBuildsPlansFailsNamingTheEngine)
{
    using ::testing::_;
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/true);
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(1);
    expectNeverExecutedOrObserved();

    ::testing::TestPartResultArray results;
    runPinned(&results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    const auto messages = testing_support::allMessages(results);
    EXPECT_THAT(messages, ::testing::HasSubstr("Engine " + ENGINE_NAME));
    EXPECT_THAT(messages, ::testing::HasSubstr("accepted expected-rejection bundle"));
    EXPECT_THAT(_verifiers, ::testing::ElementsAre(Verifier::NONE));
}

// A ranked query that did not resolve leaves "does this engine take it" unknown, so
// it cannot be read as the engine refusing it.
TEST_F(TestExpectedRejection, UnresolvedRankedQueryFailsInsteadOfPassing)
{
    using ::testing::_;
    answerRankedQueryWith(/*resolved=*/false, /*ranked=*/false);
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(0);

    ::testing::TestPartResultArray results;
    runPinned(&results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_THAT(testing_support::allMessages(results),
                ::testing::HasSubstr("stub: ranked query exploded"));
    EXPECT_THAT(_verifiers, ::testing::ElementsAre(Verifier::NONE));
}

// A graph that never loaded into the frontend never reached an engine, so no engine
// refused it.
TEST_F(TestExpectedRejection, GraphThatFailsFromBinaryFailsInsteadOfPassing)
{
    using ::testing::_;
    ON_CALL(_mocks.engineRunner, openGraph(_, _))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return testing_support::buildErrorSession("stub: bad blob");
        });
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(0);

    ::testing::TestPartResultArray results;
    runPinned(&results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    const auto messages = testing_support::allMessages(results);
    EXPECT_THAT(messages, ::testing::HasSubstr("never reached an engine"));
    EXPECT_THAT(messages, ::testing::HasSubstr("stub: bad blob"));
}

// The branch runs inside TestBody's try, so a throw is one FAILED outcome with its one
// VERIFIER line, not an exception that leaves the summary a row short.
TEST_F(TestExpectedRejection, PlanBuildThrowIsOneFailureWithOneVerifierLine)
{
    using ::testing::_;
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/true);
    ON_CALL(_mocks.engineRunner, buildPlans(_, _))
        .WillByDefault(::testing::Throw(std::runtime_error("stub: build threw")));

    ::testing::TestPartResultArray results;
    runPinned(&results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_THAT(testing_support::allMessages(results), ::testing::HasSubstr("stub: build threw"));
    EXPECT_THAT(_verifiers, ::testing::ElementsAre(Verifier::NONE));
}

// "Every engine rejects it" is checked one pinned lane at a time. Without a pinned
// engine there is nothing to judge, and the run says why instead of passing.
TEST_F(TestExpectedRejection, UnpinnedRunSkipsAsUnverifiable)
{
    using ::testing::_;
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(0);

    ::testing::TestPartResultArray results;
    run(rejectedBundle(), std::nullopt, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    ASSERT_TRUE(testing_support::anySkipped(results));
    EXPECT_THAT(testing_support::allMessages(results),
                ::testing::HasSubstr("expected_rejection requires --test-engine"));
    EXPECT_EQ(_unverifiable.size(), 1u);
}

// What the bundle's other metadata asks for is ignored: --skip-graph-validation caps
// every other bundle at applicability, but an expected rejection compiles plans and
// never executes, so it keeps judging the plan build.
TEST_F(TestExpectedRejection, SkipGraphValidationDoesNotCapTheVerdict)
{
    using ::testing::_;
    _policy.skipGraphValidation = true;
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/true);
    EXPECT_CALL(_mocks.engineRunner, buildPlans(_, _)).Times(1);

    ::testing::TestPartResultArray results;
    auto bundle = rejectedBundle();
    bundle->metadata.enforcementLevel = EnforcementLevel::APPLICABILITY;
    run(std::move(bundle), LoadedEngine{ENGINE_ID, ENGINE_NAME}, &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "an engine that builds the plans accepted the bundle, whatever enforcement_level says";
}

// A claim that names the bundle contradicts it: no engine may claim a graph every
// engine must reject.
TEST_F(TestExpectedRejection, ClaimNamingTheBundleFails)
{
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/false);

    ::testing::TestPartResultArray results;
    runPinned(&results, writeSidecar(SINGLE_GRAPH_CLAIM));

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_THAT(testing_support::allMessages(results), ::testing::HasSubstr("support claim"));
}

TEST_F(TestExpectedRejection, SweepClaimGroupListingTheCaseFails)
{
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/false);

    ::testing::TestPartResultArray results;
    runPinned(&results,
              writeSidecar(R"({"version": 1, "claims": {"ENGINE_UNDER_TEST": [
                  {"cases": ["sibling", "rejected_case"], "support": {"gfx942": ["linux"]}}]}})",
                           "rejected_case"));

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_THAT(testing_support::allMessages(results), ::testing::HasSubstr("support claim"));
}

// A sweep's cases share one support.json, so its presence says nothing about this
// case: claimed siblings leave an unclaimed expected-rejection case alone, and the case
// is still kept out of the claim counters the sidecar would otherwise put it in.
TEST_F(TestExpectedRejection, ClaimedSweepSiblingDoesNotFailAnUnclaimedCase)
{
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/false);
    expectNeverExecutedOrObserved();

    ::testing::TestPartResultArray results;
    runPinned(&results,
              writeSidecar(R"({"version": 1, "claims": {"ENGINE_UNDER_TEST": [
                  {"cases": ["sibling"], "support": {"gfx942": ["linux"]}}]}})",
                           "rejected_case"));

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_THAT(_verifiers, ::testing::ElementsAre(Verifier::EXPECTED_REJECTION));
}

// The control for the counters above: the same bundle and sidecar without the flag IS
// counted, so the exclusion is the flag's doing and not an artifact of the setup.
TEST_F(TestExpectedRejection, UnflaggedBundleWithASidecarIsStillCounted)
{
    using ::testing::_;
    EXPECT_CALL(_mocks.reporter, recordSelectedWithClaims()).Times(1);
    EXPECT_CALL(_mocks.reporter, recordReachedBody()).Times(1);

    ::testing::TestPartResultArray results;
    run(fixtures::loadBundle(_tempDir, "Unflagged", /*includeGoldenOutput=*/true),
        LoadedEngine{ENGINE_ID, ENGINE_NAME},
        &results,
        writeSidecar(R"({"version": 1, "claims": {}})"));
}

// An authoring run (--write-support-claims) refreshes claims, and an expected rejection
// has none to refresh: it is skipped, counted as skipped before observation so the
// run's totals still reconcile, and never observed.
TEST_F(TestExpectedRejection, AuthoringRunSkipsWithoutObservingAndCountsTheSkip)
{
    _policy.writeSupportClaims = true;
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/true);

    ::testing::TestPartResultArray results;
    runPinned(&results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_EQ(SupportObservationLog::get().graphsObserved(), 0u);
    EXPECT_EQ(SupportObservationLog::get().graphsUnobserved(), 0u);
    EXPECT_EQ(SupportObservationLog::get().graphsSkippedBeforeObservation(), 1u);
}

// The control: the same authoring run on an ordinary bundle observes it, so the skip
// above is the flag's doing.
TEST_F(TestExpectedRejection, AuthoringRunObservesAnOrdinaryBundle)
{
    _policy.writeSupportClaims = true;
    answerRankedQueryWith(/*resolved=*/true, /*ranked=*/true);

    ::testing::TestPartResultArray results;
    run(fixtures::loadBundle(_tempDir, "Ordinary", /*includeGoldenOutput=*/true),
        LoadedEngine{ENGINE_ID, ENGINE_NAME},
        &results);

    EXPECT_EQ(SupportObservationLog::get().graphsObserved(), 1u);
    EXPECT_EQ(SupportObservationLog::get().graphsSkippedBeforeObservation(), 0u);
}

// NOLINTEND(readability-identifier-naming)
