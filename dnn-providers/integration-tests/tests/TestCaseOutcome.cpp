// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// RFC 0023 §9.2, one row per test: the pure classification of what a test body decided
// into one of six outcomes, the report records built from it, and the JSON Lines sink
// that carries them. Nothing here touches a harness, a device or GTest's result state.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

#include "harness/bundle/CaseOutcome.hpp"
#include "harness/bundle/OutcomeReportLog.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_test_sdk::utilities::claimScratchDirectory;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

const std::string ENGINE_NAME = "ENGINE_UNDER_TEST";
constexpr VerificationDepth REQUIRED = VerificationDepth::VERIFIED;

SupportResult verdictFor(SupportVerdict verdict, const std::string& engine = ENGINE_NAME)
{
    SupportResult result;
    result.verdict = verdict;
    result.engineName = engine;
    return result;
}

CaseOutcome classify(const VerificationOutcome& outcome,
                     const std::vector<SupportResult>& claims = {},
                     VerificationDepth required = REQUIRED,
                     bool hadOtherFailure = false)
{
    return classifyCaseOutcome(outcome, claims, ENGINE_NAME, required, hadOtherFailure);
}

VerificationOutcome passedAt(VerificationDepth depth)
{
    return VerificationOutcome::passed(depth);
}

CaseIdentity makeIdentity()
{
    CaseIdentity identity;
    identity.test = "suite.case";
    identity.bundle = "dir/sweep.json#case";
    identity.engine = ENGINE_NAME;
    identity.arch = "gfx950";
    identity.platform = "linux";
    return identity;
}

} // namespace

// ---------------------------------------------------------------------------
// RFC 0023 §9.2: PASS and UNCLAIMED_PASS
// ---------------------------------------------------------------------------

TEST(TestCaseOutcome, PassedAtTheRequiredDepthIsPass)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::VERIFIED)), CaseOutcome::PASS);
}

TEST(TestCaseOutcome, PassedAtTheDepthABuildableBundleRequiresIsPass)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::BUILDABLE), {}, VerificationDepth::BUILDABLE),
              CaseOutcome::PASS);
}

TEST(TestCaseOutcome, PassedWithAnUnclaimedVerdictIsUnclaimedPass)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::VERIFIED),
                       {verdictFor(SupportVerdict::UNCLAIMED_SUPPORT)}),
              CaseOutcome::UNCLAIMED_PASS);
}

TEST(TestCaseOutcome, AnotherEnginesUnclaimedVerdictDoesNotMakeAnUnclaimedPass)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::VERIFIED),
                       {verdictFor(SupportVerdict::UNCLAIMED_SUPPORT, "OTHER_ENGINE")}),
              CaseOutcome::PASS);
}

TEST(TestCaseOutcome, PassedWithAConfirmedClaimIsPass)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::VERIFIED),
                       {verdictFor(SupportVerdict::CLAIM_CONFIRMED)}),
              CaseOutcome::PASS);
}

TEST(TestCaseOutcome, PassedWithAnotherGtestFailureIsUnexercised)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::VERIFIED), {}, REQUIRED, /*other=*/true),
              CaseOutcome::UNEXERCISED);
}

TEST(TestCaseOutcome, PassedBelowTheRequiredDepthIsUnexercised)
{
    EXPECT_EQ(classify(passedAt(VerificationDepth::EXECUTED)), CaseOutcome::UNEXERCISED);
}

// ---------------------------------------------------------------------------
// DECLINED
// ---------------------------------------------------------------------------

TEST(TestCaseOutcome, SkipAfterAResolvedQueryWithTheEngineAbsentIsDeclined)
{
    const auto outcome = VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, "no")
                             .atStage(OutcomeStage::QUERY)
                             .asDecline();
    EXPECT_EQ(classify(outcome), CaseOutcome::DECLINED);
}

TEST(TestCaseOutcome, SkipWithoutAResolvedQueryIsUnexercised)
{
    const auto outcome = VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, "unknown")
                             .atStage(OutcomeStage::QUERY);
    EXPECT_EQ(classify(outcome), CaseOutcome::UNEXERCISED);
}

TEST(TestCaseOutcome, ABrokenClaimIsDeclined)
{
    const auto outcome = VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                                     FailureOrigin::ENGINE,
                                                     "claimed but absent")
                             .atStage(OutcomeStage::CLAIM)
                             .asDecline();
    EXPECT_EQ(classify(outcome), CaseOutcome::DECLINED);
}

TEST(TestCaseOutcome, AClaimBlockedByAnUnresolvedQueryIsUnexercised)
{
    const auto outcome = VerificationOutcome::failed(
                             VerificationDepth::NOT_REACHED, FailureOrigin::ENGINE, "query errored")
                             .atStage(OutcomeStage::CLAIM);
    EXPECT_EQ(classify(outcome), CaseOutcome::UNEXERCISED);
}

TEST(TestCaseOutcome, ALateDeclineWhileBuildingIsDeclined)
{
    const auto outcome = VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, "late")
                             .atStage(OutcomeStage::BUILD)
                             .asDecline();
    EXPECT_EQ(classify(outcome), CaseOutcome::DECLINED);
}

// ---------------------------------------------------------------------------
// WRONG and CRASH
// ---------------------------------------------------------------------------

TEST(TestCaseOutcome, ComparisonFailureIsWrong)
{
    const auto outcome = VerificationOutcome::alreadyReportedFailure(VerificationDepth::VERIFIED,
                                                                     FailureOrigin::COMPARISON)
                             .atStage(OutcomeStage::COMPARE);
    EXPECT_EQ(classify(outcome), CaseOutcome::WRONG);
}

TEST(TestCaseOutcome, EngineFailureWhileBuildingIsCrash)
{
    const auto outcome = VerificationOutcome::failed(
                             VerificationDepth::NOT_REACHED, FailureOrigin::ENGINE, "build failed")
                             .atStage(OutcomeStage::BUILD);
    EXPECT_EQ(classify(outcome), CaseOutcome::CRASH);
}

TEST(TestCaseOutcome, EngineFailureWhileExecutingIsCrash)
{
    const auto outcome = VerificationOutcome::failed(
                             VerificationDepth::BUILDABLE, FailureOrigin::ENGINE, "execute failed")
                             .atStage(OutcomeStage::EXECUTE);
    EXPECT_EQ(classify(outcome), CaseOutcome::CRASH);
}

// ---------------------------------------------------------------------------
// UNEXERCISED
// ---------------------------------------------------------------------------

TEST(TestCaseOutcome, GraphRejectedByTheFrontendIsUnexercised)
{
    const auto outcome = VerificationOutcome::failed(VerificationDepth::NOT_REACHED,
                                                     FailureOrigin::ENGINE,
                                                     "from_binary failed")
                             .atStage(OutcomeStage::GRAPH);
    EXPECT_EQ(classify(outcome), CaseOutcome::UNEXERCISED);
}

TEST(TestCaseOutcome, OracleAndHarnessFailuresAreUnexercised)
{
    for(const auto origin : {FailureOrigin::ORACLE, FailureOrigin::HARNESS})
    {
        for(const auto stage : {OutcomeStage::ORACLE, OutcomeStage::SETUP})
        {
            const auto outcome
                = VerificationOutcome::failed(VerificationDepth::EXECUTED, origin, "broken")
                      .atStage(stage);
            EXPECT_EQ(classify(outcome), CaseOutcome::UNEXERCISED)
                << toString(origin) << "/" << toString(stage);
        }
    }
}

TEST(TestCaseOutcome, SkipRuleOrUnverifiableSkipIsUnexercised)
{
    const auto outcome = VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, "toml skip")
                             .atStage(OutcomeStage::SETUP);
    EXPECT_EQ(classify(outcome), CaseOutcome::UNEXERCISED);
}

// ---------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------

TEST(TestCaseOutcome, RecordCarriesTheOutcomeFacts)
{
    auto outcome = VerificationOutcome::alreadyReportedFailure(VerificationDepth::VERIFIED,
                                                               FailureOrigin::COMPARISON)
                       .atStage(OutcomeStage::COMPARE);
    outcome.verifier = Verifier::GPU_REFERENCE;

    const auto record = makeCaseOutcomeRecord(makeIdentity(), outcome, CaseOutcome::WRONG, false);

    EXPECT_EQ(record.outcome, CaseOutcome::WRONG);
    EXPECT_EQ(record.stage, OutcomeStage::COMPARE);
    EXPECT_EQ(record.depth, VerificationDepth::VERIFIED);
    EXPECT_EQ(record.origin, FailureOrigin::COMPARISON);
    EXPECT_EQ(record.verifier, Verifier::GPU_REFERENCE);
    EXPECT_FALSE(record.reason.empty());
}

TEST(TestCaseOutcome, PassRecordHasNoReason)
{
    const auto record = makeCaseOutcomeRecord(
        makeIdentity(), passedAt(VerificationDepth::VERIFIED), CaseOutcome::PASS, false);
    EXPECT_TRUE(record.reason.empty());
}

TEST(TestCaseOutcome, PassWithAnotherFailureSaysSo)
{
    const auto record = makeCaseOutcomeRecord(makeIdentity(),
                                              passedAt(VerificationDepth::VERIFIED),
                                              CaseOutcome::UNEXERCISED,
                                              /*hadOtherFailure=*/true);
    EXPECT_NE(record.reason.find("another failure"), std::string::npos) << record.reason;
}

TEST(TestCaseOutcome, UnexercisedRecordNamesItsStageAndReason)
{
    const auto record = makeUnexercisedRecord(makeIdentity(), OutcomeStage::SETUP, "no devices");
    EXPECT_EQ(record.outcome, CaseOutcome::UNEXERCISED);
    EXPECT_EQ(record.stage, OutcomeStage::SETUP);
    EXPECT_EQ(record.reason, "no devices");
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------

TEST(TestCaseOutcome, StartedRecordNamesTheTestAndBundleOnly)
{
    const auto line = startedRecordJson(makeIdentity());
    EXPECT_EQ(line.find('\n'), std::string::npos);

    const auto json = nlohmann::json::parse(line);
    EXPECT_EQ(json.at("event"), "started");
    EXPECT_EQ(json.at("test"), "suite.case");
    EXPECT_EQ(json.at("bundle"), "dir/sweep.json#case");
    EXPECT_FALSE(json.contains("outcome"));
}

TEST(TestCaseOutcome, FinishedRecordCarriesEveryField)
{
    auto identity = makeIdentity();
    identity.purpose = CasePurpose::PROBE;

    const auto outcome
        = VerificationOutcome::skipped(VerificationDepth::NOT_REACHED, "does not take it")
              .atStage(OutcomeStage::QUERY)
              .asDecline();
    const auto record
        = makeCaseOutcomeRecord(identity, outcome, CaseOutcome::DECLINED, /*other=*/false);

    const auto json = nlohmann::json::parse(finishedRecordJson(record));
    EXPECT_EQ(json.at("event"), "finished");
    EXPECT_EQ(json.at("test"), "suite.case");
    EXPECT_EQ(json.at("bundle"), "dir/sweep.json#case");
    EXPECT_EQ(json.at("engine"), ENGINE_NAME);
    EXPECT_EQ(json.at("arch"), "gfx950");
    EXPECT_EQ(json.at("platform"), "linux");
    EXPECT_EQ(json.at("purpose"), "probe");
    EXPECT_EQ(json.at("outcome"), "DECLINED");
    EXPECT_EQ(json.at("stage"), "query");
    EXPECT_EQ(json.at("depth"), "not-reached");
    EXPECT_EQ(json.at("origin"), "none");
    EXPECT_EQ(json.at("verifier"), "none");
    EXPECT_EQ(json.at("reason"), "does not take it");
}

TEST(TestCaseOutcome, FinishedRecordOmitsPurposeWhenUnset)
{
    const auto record = makeUnexercisedRecord(makeIdentity(), OutcomeStage::SETUP, "no bundle set");
    const auto json = nlohmann::json::parse(finishedRecordJson(record));
    EXPECT_FALSE(json.contains("purpose"));
}

TEST(TestCaseOutcome, EachOutcomeHasItsRfcSpelling)
{
    EXPECT_STREQ(toString(CaseOutcome::PASS), "PASS");
    EXPECT_STREQ(toString(CaseOutcome::DECLINED), "DECLINED");
    EXPECT_STREQ(toString(CaseOutcome::WRONG), "WRONG");
    EXPECT_STREQ(toString(CaseOutcome::CRASH), "CRASH");
    EXPECT_STREQ(toString(CaseOutcome::UNCLAIMED_PASS), "UNCLAIMED_PASS");
    EXPECT_STREQ(toString(CaseOutcome::UNEXERCISED), "UNEXERCISED");
}

TEST(TestCaseOutcome, ReasonWithNewlinesAndBadUtf8StaysOneParseableLine)
{
    const std::string reason = std::string("first\nsecond \xC3\x28 third");
    const auto record = makeUnexercisedRecord(makeIdentity(), OutcomeStage::SETUP, reason);

    const auto line = finishedRecordJson(record);
    EXPECT_EQ(line.find('\n'), std::string::npos);

    const auto json = nlohmann::json::parse(line);
    EXPECT_NE(json.at("reason").get<std::string>().find("first\nsecond"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The JSON Lines sink
// ---------------------------------------------------------------------------

namespace
{

class TestOutcomeReportLog : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _path;

    void SetUp() override
    {
        _scopedDir.emplace(claimScratchDirectory("outcome_report_log"));
        _path = _scopedDir->path() / "report.jsonl";
        OutcomeReportLog::get().close();
    }

    void TearDown() override
    {
        OutcomeReportLog::get().close();
    }

    std::vector<std::string> lines() const
    {
        std::vector<std::string> result;
        std::ifstream file(_path);
        for(std::string line; std::getline(file, line);)
        {
            result.push_back(line);
        }
        return result;
    }
};

} // namespace

TEST_F(TestOutcomeReportLog, IsANoOpUntilOpened)
{
    OutcomeReportLog::get().append("ignored");
    EXPECT_FALSE(OutcomeReportLog::get().isOpen());
    EXPECT_FALSE(std::filesystem::exists(_path));
}

TEST_F(TestOutcomeReportLog, EveryRecordIsOnDiskBeforeAppendReturns)
{
    ASSERT_TRUE(OutcomeReportLog::get().open(_path));

    OutcomeReportLog::get().append(startedRecordJson(makeIdentity()));
    // Read with the writer still open and unflushed-by-close: a crash at this point
    // must have kept the line.
    ASSERT_EQ(lines().size(), 1u);
    EXPECT_EQ(nlohmann::json::parse(lines()[0]).at("event"), "started");

    OutcomeReportLog::get().append(
        finishedRecordJson(makeUnexercisedRecord(makeIdentity(), OutcomeStage::SETUP, "r")));
    ASSERT_EQ(lines().size(), 2u);
    EXPECT_EQ(nlohmann::json::parse(lines()[1]).at("event"), "finished");
}

TEST_F(TestOutcomeReportLog, RecordsKeepTheirOrder)
{
    ASSERT_TRUE(OutcomeReportLog::get().open(_path));
    for(int i = 0; i < 5; ++i)
    {
        OutcomeReportLog::get().append("line " + std::to_string(i));
    }

    const auto written = lines();
    ASSERT_EQ(written.size(), 5u);
    for(int i = 0; i < 5; ++i)
    {
        EXPECT_EQ(written[static_cast<size_t>(i)], "line " + std::to_string(i));
    }
}

TEST_F(TestOutcomeReportLog, ReopeningAppendsRatherThanTruncating)
{
    ASSERT_TRUE(OutcomeReportLog::get().open(_path));
    OutcomeReportLog::get().append("first run");
    OutcomeReportLog::get().close();

    ASSERT_TRUE(OutcomeReportLog::get().open(_path));
    OutcomeReportLog::get().append("second run");

    EXPECT_EQ(lines(), (std::vector<std::string>{"first run", "second run"}));
}

TEST_F(TestOutcomeReportLog, OpenFailsForAnUnwritablePath)
{
    EXPECT_FALSE(OutcomeReportLog::get().open(_scopedDir->path() / "no-such-dir" / "report.jsonl"));
    EXPECT_FALSE(OutcomeReportLog::get().isOpen());
    // Never throws, even closed.
    OutcomeReportLog::get().append("dropped");
}

// NOLINTEND(readability-identifier-naming)
