// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"

using hipdnn_integration_tests::bundle::buildSupportClaimSummary;
using hipdnn_integration_tests::bundle::ClaimMode;
using hipdnn_integration_tests::bundle::countersAreConsistent;
using hipdnn_integration_tests::bundle::coverageFor;
using hipdnn_integration_tests::bundle::CoverageUpdate;
using hipdnn_integration_tests::bundle::missedQueryComplaint;
using hipdnn_integration_tests::bundle::printSupportClaimSummary;
using hipdnn_integration_tests::bundle::reportBundlePath;
using hipdnn_integration_tests::bundle::SidecarState;
using hipdnn_integration_tests::bundle::SupportClaimCoverage;
using hipdnn_integration_tests::bundle::supportClaimCoverage;
using hipdnn_integration_tests::bundle::SupportClaimRunContext;
using hipdnn_integration_tests::bundle::SupportClaimVerdicts;
using hipdnn_integration_tests::bundle::SupportObservation;
using hipdnn_integration_tests::bundle::SupportResult;
using hipdnn_integration_tests::bundle::SupportVerdict;
using hipdnn_integration_tests::bundle::VerificationDepth;
using hipdnn_integration_tests::bundle::verifiedNothing;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

SupportResult makeResult(SupportVerdict v)
{
    SupportResult result;
    result.verdict = v;
    result.bundlePath = "test/bundle";
    result.engineName = "ENGINE_A";
    result.arch = "gfx942";
    result.platform = "linux";
    result.detail = "detail";
    return result;
}

// The same engine, arch and platform makeResult() records, so an entry only carries
// them when a test changes one on purpose.
SupportClaimRunContext testRun()
{
    SupportClaimRunContext run;
    run.engine = "ENGINE_A";
    run.arch = "gfx942";
    run.platform = "linux";
    return run;
}

std::string summary(ClaimMode claims = ClaimMode::ENFORCE)
{
    std::ostringstream oss;
    printSupportClaimSummary(
        supportClaimCoverage(), SupportClaimVerdicts::get(), claims, testRun(), oss);
    return oss.str();
}

// The document itself, for the tests that care what it says rather than how it
// prints. Fails the calling test if the run would have printed nothing.
nlohmann::json summaryJson(ClaimMode claims = ClaimMode::ENFORCE)
{
    const auto document = buildSupportClaimSummary(
        supportClaimCoverage(), SupportClaimVerdicts::get(), claims, testRun());
    EXPECT_TRUE(document.has_value()) << "expected a summary, got none";
    return document.value_or(nlohmann::json::object());
}

class TestSupportClaimReport : public ::testing::Test
{
protected:
    void SetUp() override
    {
        clearAll();
    }
    void TearDown() override
    {
        clearAll();
    }

private:
    static void clearAll()
    {
        supportClaimCoverage() = {};
        SupportClaimVerdicts::get().clear();
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Zero records → no output
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintIsNoOpWhenEmpty)
{
    EXPECT_TRUE(summary().empty());
}

// ---------------------------------------------------------------------------
// Single-verdict recording
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, RecordsAccepted)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_ACCEPTED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_ACCEPTED), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsConfirmed)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_CONFIRMED), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsClaimBroken)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 1u);
    EXPECT_TRUE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsQueryErrored)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::QUERY_ERRORED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::QUERY_ERRORED), 1u);
    EXPECT_TRUE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsUnclaimedSupport)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::UNCLAIMED_SUPPORT), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

// The claim held; the run is already red from whatever actually broke. Failing it
// again here would double-report one defect.
TEST_F(TestSupportClaimReport, FailedInUseIsNotAClaimFailure)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_FAILED_IN_USE));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_FAILED_IN_USE), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

// A verdict the log has never seen counts zero rather than misreporting.
TEST_F(TestSupportClaimReport, CountIsZeroForUnseenVerdict)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_ACCEPTED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 0u);
}

// ---------------------------------------------------------------------------
// Multiple records aggregate correctly
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, MultipleRecordsAccumulate)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));

    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_CONFIRMED), 2u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::UNCLAIMED_SUPPORT), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 4u);
}

// ---------------------------------------------------------------------------
// Clearing each accumulator
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, ClearEmptiesTheVerdictLog)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 2u);

    SupportClaimVerdicts::get().clear();

    EXPECT_EQ(SupportClaimVerdicts::get().total(), 0u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, CoverageResetsToZero)
{
    supportClaimCoverage().graphsFound = 3;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsQueried = 1;

    supportClaimCoverage() = {};

    EXPECT_EQ(supportClaimCoverage().graphsFound, 0u);
    EXPECT_EQ(supportClaimCoverage().graphsWithClaims, 0u);
    EXPECT_EQ(supportClaimCoverage().graphsQueried, 0u);
}

// ---------------------------------------------------------------------------
// The nesting invariant: queried ⊆ withClaims ⊆ found. The queried count is its
// own counter, driven by SupportObservation::sidecar, because one graph can produce
// several verdicts (one per engine that had something to say).
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, QueriedCountIsIndependentOfVerdictCount)
{
    supportClaimCoverage().graphsFound = 5;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsQueried = 1;

    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));

    EXPECT_EQ(supportClaimCoverage().graphsQueried, 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 2u);
}

TEST_F(TestSupportClaimReport, MultiEngineQueriedCountIsPerGraph)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;

    SupportResult r1 = makeResult(SupportVerdict::CLAIM_CONFIRMED);
    r1.engineName = "ENGINE_A";
    SupportResult r2 = makeResult(SupportVerdict::UNCLAIMED_SUPPORT);
    r2.engineName = "ENGINE_B";

    SupportClaimVerdicts::get().record(r1);
    SupportClaimVerdicts::get().record(r2);

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("graphs").at("queried"), 1);
    EXPECT_EQ(doc.at("verdicts").at("confirmed"), 1);
    EXPECT_EQ(doc.at("verdicts").at("unclaimed"), 1);
}

// ---------------------------------------------------------------------------
// The summary document
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, SummaryShowsCountersAndVerdictTallies)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    const nlohmann::json expectedGraphs
        = {{"found", 2}, {"with_claims", 1}, {"selected", 1}, {"ran", 1}, {"queried", 1}};
    EXPECT_EQ(doc.at("graphs"), expectedGraphs) << doc.dump(2);

    const nlohmann::json expectedVerdicts = {{"confirmed", 1},
                                             {"accepted", 0},
                                             {"failed_in_use", 0},
                                             {"broken", 0},
                                             {"errored", 0},
                                             {"unclaimed", 0}};
    EXPECT_EQ(doc.at("verdicts"), expectedVerdicts) << doc.dump(2);
}

// A machine reading this should not have to guess whether a missing key means zero,
// so every key and every list is there even when it is empty.
TEST_F(TestSupportClaimReport, SummaryKeepsItsShapeWhenThereIsNothingToList)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("schema_version"), 1);
    EXPECT_EQ(doc.at("mode"), "enforcing");
    EXPECT_EQ(doc.at("run"),
              (nlohmann::json{{"engine", "ENGINE_A"}, {"arch", "gfx942"}, {"platform", "linux"}}));
    EXPECT_TRUE(doc.at("claim_failures").is_array());
    EXPECT_TRUE(doc.at("claim_failures").empty());
    EXPECT_TRUE(doc.at("failed_in_use").is_array());
    EXPECT_TRUE(doc.at("failed_in_use").empty());
    EXPECT_TRUE(doc.at("unclaimed_support").is_array());
    EXPECT_TRUE(doc.at("unclaimed_support").empty());
}

// A summary scraped out of a CI log has to say on its own face whether the failures
// under it were fatal, because a warn-only lane prints the same shape and the same
// failure list. Asserted on a run that has failures, since that is the case where the
// header is load-bearing rather than decorative.
TEST_F(TestSupportClaimReport, PrintHeaderNamesEnforcement)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto output = summary(ClaimMode::ENFORCE);

    EXPECT_NE(output.find("==== SUPPORT CLAIM SUMMARY (ENFORCING) ===="), std::string::npos);
    // The failures are listed in full under that header. Withholding them would make
    // the mode a coverage difference rather than an exit-code one.
    EXPECT_EQ(summaryJson(ClaimMode::ENFORCE).at("claim_failures").size(), 1u);
}

TEST_F(TestSupportClaimReport, PrintHeaderNamesWarnOnly)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto output = summary(ClaimMode::WARN);

    EXPECT_NE(output.find("==== SUPPORT CLAIM SUMMARY (WARNING ONLY -- NOT ENFORCED) ===="),
              std::string::npos)
        << output;
    EXPECT_EQ(summaryJson(ClaimMode::WARN).at("mode"), "warning_only");
    EXPECT_EQ(summaryJson(ClaimMode::WARN).at("claim_failures").size(), 1u);
}

// The printed block is the document under one named key, so it still says what it
// is once someone has cut it out of a log, and it parses back to what was built.
TEST_F(TestSupportClaimReport, PrintWrapsTheDocumentInANamedKey)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto output = summary();
    const auto body = output.find('{');
    ASSERT_NE(body, std::string::npos) << output;

    const auto printed = nlohmann::json::parse(output.substr(body));

    EXPECT_EQ(printed.at("support_claim_summary"), summaryJson()) << output;
}

// "accepted" and "confirmed" are different facts and the tallies have to say so,
// because only one of them reached the depth its bundle declares.
TEST_F(TestSupportClaimReport, SummaryDistinguishesAcceptedFromConfirmed)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_ACCEPTED));

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("verdicts").at("accepted"), 1);
    EXPECT_EQ(doc.at("verdicts").at("confirmed"), 0);
}

TEST_F(TestSupportClaimReport, SummaryShowsFailureDetail)
{
    auto broken = makeResult(SupportVerdict::CLAIM_BROKEN);
    broken.queryStatus = hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED;
    SupportClaimVerdicts::get().record(broken);

    const auto failures = summaryJson().at("claim_failures");

    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures[0].at("bundle"), "test/bundle");
    EXPECT_EQ(failures[0].at("verdict"), "CLAIM_BROKEN");
    EXPECT_EQ(failures[0].at("reason"), "detail");
    EXPECT_EQ(failures[0].at("status"),
              hipdnn_frontend::to_string(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED));
}

// An unresolved query has a code but no reason of its own, so the backend's words are
// the only explanation there is. A resolved one has neither field.
TEST_F(TestSupportClaimReport, SummaryCarriesTheQueryMessageOnlyWhenThereIsOne)
{
    auto errored = makeResult(SupportVerdict::QUERY_ERRORED);
    errored.bundlePath = "a/errored";
    errored.queryStatus = hipdnn_frontend::ErrorCode::HEURISTIC_QUERY_FAILED;
    errored.queryMessage = "backend said no";
    SupportClaimVerdicts::get().record(errored);
    auto broken = makeResult(SupportVerdict::CLAIM_BROKEN);
    broken.bundlePath = "b/broken";
    SupportClaimVerdicts::get().record(broken);

    const auto failures = summaryJson().at("claim_failures");

    ASSERT_EQ(failures.size(), 2u);
    EXPECT_EQ(failures[0].at("query_message"), "backend said no");
    EXPECT_TRUE(failures[0].contains("status"));
    EXPECT_FALSE(failures[1].contains("query_message"));
    EXPECT_FALSE(failures[1].contains("status"));
}

TEST_F(TestSupportClaimReport, SummaryListsUnclaimedBundles)
{
    auto unclaimed = makeResult(SupportVerdict::UNCLAIMED_SUPPORT);
    unclaimed.reachedDepth = VerificationDepth::VERIFIED;
    unclaimed.requiredDepth = VerificationDepth::VERIFIED;
    SupportClaimVerdicts::get().record(unclaimed);

    const auto list = summaryJson().at("unclaimed_support");

    // A bare count is not actionable — the bundle has to be named, and how far the
    // run got is what says whether the claim is ready to be written.
    const nlohmann::json expected
        = {{{"bundle", "test/bundle"}, {"reached", "verified"}, {"required", "verified"}}};
    EXPECT_EQ(list, expected) << list.dump(2);
}

// A sweep the engine takes whole would otherwise repeat one bundle once per case.
// Cases at different depths stay apart, since they are not ready for the same edit.
TEST_F(TestSupportClaimReport, SummaryGroupsUnclaimedSweepCasesByBundleAndDepth)
{
    const auto sweepCase = [](const std::string& caseId, VerificationDepth reached) {
        auto r = makeResult(SupportVerdict::UNCLAIMED_SUPPORT);
        r.bundlePath = "sweep.json#" + caseId;
        r.caseId = caseId;
        r.reachedDepth = reached;
        r.requiredDepth = VerificationDepth::VERIFIED;
        return r;
    };
    SupportClaimVerdicts::get().record(sweepCase("case_b", VerificationDepth::VERIFIED));
    SupportClaimVerdicts::get().record(sweepCase("case_a", VerificationDepth::VERIFIED));
    SupportClaimVerdicts::get().record(sweepCase("case_c", VerificationDepth::EXECUTED));

    const auto list = summaryJson().at("unclaimed_support");

    const nlohmann::json expected = {{{"bundle", "sweep.json"},
                                      {"reached", "verified"},
                                      {"required", "verified"},
                                      {"cases", nlohmann::json::array({"case_a", "case_b"})}},
                                     {{"bundle", "sweep.json"},
                                      {"reached", "executed"},
                                      {"required", "verified"},
                                      {"cases", nlohmann::json::array({"case_c"})}}};
    EXPECT_EQ(list, expected) << list.dump(2);
}

// Two runs over the same tree must print the same document, or a diff between them
// is noise. Record order is test order, which --gtest_shuffle changes.
TEST_F(TestSupportClaimReport, SummaryListsAreSortedRegardlessOfRecordOrder)
{
    for(const char* path : {"z/bundle", "a/bundle", "m/bundle"})
    {
        auto broken = makeResult(SupportVerdict::CLAIM_BROKEN);
        broken.bundlePath = path;
        SupportClaimVerdicts::get().record(broken);
        auto unclaimed = makeResult(SupportVerdict::UNCLAIMED_SUPPORT);
        unclaimed.bundlePath = path;
        SupportClaimVerdicts::get().record(unclaimed);
    }

    const auto doc = summaryJson();

    for(const char* list : {"claim_failures", "unclaimed_support"})
    {
        ASSERT_EQ(doc.at(list).size(), 3u) << list;
        EXPECT_EQ(doc.at(list)[0].at("bundle"), "a/bundle") << list;
        EXPECT_EQ(doc.at(list)[1].at("bundle"), "m/bundle") << list;
        EXPECT_EQ(doc.at(list)[2].at("bundle"), "z/bundle") << list;
    }
}

// One lane tests one engine on one machine, so the run block says it once. An entry
// repeats a field only when it differs, which is when it is worth noticing.
TEST_F(TestSupportClaimReport, SummaryEntriesNameOnlyWhatDiffersFromTheRun)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    auto other = makeResult(SupportVerdict::QUERY_ERRORED);
    other.bundlePath = "z/bundle";
    other.engineName = "ENGINE_B";
    other.arch = "gfx90a";
    other.platform = "windows";
    SupportClaimVerdicts::get().record(other);

    const auto failures = summaryJson().at("claim_failures");

    // Still in bundle order: the extra "arch" key must not pull the second entry ahead.
    ASSERT_EQ(failures.size(), 2u);
    EXPECT_EQ(failures[0].at("bundle"), "test/bundle");
    EXPECT_FALSE(failures[0].contains("engine"));
    EXPECT_FALSE(failures[0].contains("arch"));
    EXPECT_FALSE(failures[0].contains("platform"));
    EXPECT_EQ(failures[1].at("engine"), "ENGINE_B");
    EXPECT_EQ(failures[1].at("arch"), "gfx90a");
    EXPECT_EQ(failures[1].at("platform"), "windows");
}

// A count with no bundle names is not actionable, and this is the list that tells an
// operator which cells must not be published as working support.
TEST_F(TestSupportClaimReport, SummaryNamesBundlesThatFailedInUse)
{
    auto failed = makeResult(SupportVerdict::CLAIM_FAILED_IN_USE);
    failed.reachedDepth = VerificationDepth::EXECUTED;
    failed.requiredDepth = VerificationDepth::VERIFIED;
    SupportClaimVerdicts::get().record(failed);

    const auto doc = summaryJson();

    const nlohmann::json expected = {{{"bundle", "test/bundle"},
                                      {"reached", "executed"},
                                      {"required", "verified"},
                                      {"reason", "detail"}}};
    EXPECT_EQ(doc.at("failed_in_use"), expected) << doc.dump(2);
    // Not a claim failure, so it must not appear in the failure list.
    EXPECT_TRUE(doc.at("claim_failures").empty());
}

// A filtered run discovers more claim-bearing graphs than it selects. That gap --
// discovered minus selected -- is the filter's doing and nothing else's, so the
// summary names it instead of leaving a mismatch to be misread as a harness gap.
TEST_F(TestSupportClaimReport, SummaryAttributesUnselectedGraphsToTheFilter)
{
    supportClaimCoverage().graphsFound = 3;
    supportClaimCoverage().graphsWithClaims = 3;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto unenforced = summaryJson().at("unenforced");

    EXPECT_EQ(unenforced.at("not_selected"), 2);
    // Everything selected ran, so none of it is the skip-list's doing.
    EXPECT_EQ(unenforced.at("skipped_before_run"), 0);
}

// The other half of the split. These graphs *were* selected -- the filter let them
// through -- and then SetUp() skipped them before running. Blaming --gtest_filter
// for them would send a reader to edit the one knob that is already correct.
TEST_F(TestSupportClaimReport, SummarySeparatesSelectedButSkippedFromTheFilterRemainder)
{
    supportClaimCoverage().graphsFound = 5;
    supportClaimCoverage().graphsWithClaims = 5;
    supportClaimCoverage().graphsSelectedWithClaims = 3;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto unenforced = summaryJson().at("unenforced");

    EXPECT_EQ(unenforced.at("skipped_before_run"), 2);
    EXPECT_EQ(unenforced.at("not_selected"), 2);
}

// The arch-skipped lane, which is the common case this split exists for: the filter
// selected everything and SetUp() skipped all of it.
TEST_F(TestSupportClaimReport, SummaryBlamesTheSkipWhenTheFilterSelectedEverything)
{
    supportClaimCoverage().graphsFound = 4;
    supportClaimCoverage().graphsWithClaims = 4;
    supportClaimCoverage().graphsSelectedWithClaims = 4;
    supportClaimCoverage().graphsReachedBody = 0;

    const auto unenforced = summaryJson().at("unenforced");

    EXPECT_EQ(unenforced.at("skipped_before_run"), 4);
    EXPECT_EQ(unenforced.at("not_selected"), 0)
        << "the filter selected every claim-bearing graph; the skip is what stopped them";
}

// A body that ran, opened its graph and still never queried is the one shortfall no
// configuration can produce. It is a harness defect rather than an unenforced graph,
// because sending a reader to the skip-list for a harness bug costs them the afternoon.
TEST_F(TestSupportClaimReport, SummaryNamesAMissedQueryAsAHarnessDefect)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 2;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("harness_defects").at("missed_query"), 1);
    EXPECT_EQ(doc.at("unenforced").at("skipped_before_run"), 0);
}

// A graph that never opened ran and failed; it is already accounted for by its own
// counter, so it must not also be counted as a skip or as a missed query.
TEST_F(TestSupportClaimReport, SummaryDoesNotCountUnopenedGraphsAsSkipped)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 2;
    supportClaimCoverage().graphsQueried = 1;
    supportClaimCoverage().graphsNotOpened = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("unenforced").at("not_opened"), 1);
    EXPECT_EQ(doc.at("unenforced").at("skipped_before_run"), 0);
    EXPECT_EQ(doc.at("harness_defects").at("missed_query"), 0);
}

TEST_F(TestSupportClaimReport, SummaryShowsNoShortfallWhenEverythingRan)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    const nlohmann::json expectedUnenforced = {{"not_opened", 0},
                                               {"no_applicable_claim", 0},
                                               {"not_selected", 0},
                                               {"skipped_before_run", 0}};
    EXPECT_EQ(doc.at("unenforced"), expectedUnenforced) << doc.dump(2);
    EXPECT_EQ(doc.at("harness_defects"), (nlohmann::json{{"missed_query", 0}}));
}

// Otherwise invisible: a sidecar read in full that promised nothing for this cell
// leaves no verdict, so the tallies look identical to a graph nobody ever claimed.
TEST_F(TestSupportClaimReport, SummaryCountsGraphsWhoseSidecarClaimsNothingHere)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 2;
    supportClaimCoverage().graphsQueried = 2;
    supportClaimCoverage().graphsWithNoApplicableClaim = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    EXPECT_EQ(summaryJson().at("unenforced").at("no_applicable_claim"), 1);
}

// ---------------------------------------------------------------------------
// Default-off inertness: a run over a tree with no sidecars anywhere must stay
// completely silent.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintIsSilentWhenGraphsFoundButNoSidecars)
{
    supportClaimCoverage().graphsFound = 100;

    EXPECT_TRUE(summary().empty());
}

// The run that trips the guard must still print. Its summary is all zeros except
// the discovery, selection and body counts, and those counts are the only thing that
// distinguishes it from a run with nothing to enforce. Ran is 1 and queried is 0
// because that pair — reached them, asked nothing — is exactly what trips it.
TEST_F(TestSupportClaimReport, PrintShowsDiscoveryCountsWhenNothingWasQueried)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;

    EXPECT_NE(summary().find("SUPPORT CLAIM SUMMARY"), std::string::npos);

    const auto graphs = summaryJson().at("graphs");

    EXPECT_EQ(graphs.at("with_claims"), 1);
    EXPECT_EQ(graphs.at("selected"), 1);
    EXPECT_EQ(graphs.at("ran"), 1);
    EXPECT_EQ(graphs.at("queried"), 0);
}

// ---------------------------------------------------------------------------
// Empty-query guard (RFC 0015 §7.2)
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenNothingDiscovered)
{
    // (0, 0) → false: no graph carried a claim, so there was nothing to enforce.
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardTrippedWhenBodiesRanButNoQueries)
{
    // (N, 0) → true: claim-bearing graphs ran and not one was ever queried.
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    EXPECT_TRUE(verifiedNothing(supportClaimCoverage()));
}

// The shape every category suite has: a filter narrows the run onto bundles that
// carry no sidecar, so nothing was queried and nothing should have been. Against
// the registration-time denominator this is indistinguishable from the case
// above, which is why the guard counts what ran instead of what was found.
TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenFilterSelectedNoClaimedGraphs)
{
    supportClaimCoverage().graphsWithClaims = 18;
    supportClaimCoverage().graphsSelectedWithClaims = 0;
    supportClaimCoverage().graphsReachedBody = 0;
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

// The guard counts bodies, not selections, and this is the difference. An
// arch-guarded lane selects its claim-bearing bundles and then skips every one of
// them — doing exactly what it is configured to do. Keying the guard on selection
// would turn that lane from green to fatal, which is why it is keyed on what ran.
TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenEverySelectedGraphWasSkipped)
{
    supportClaimCoverage().graphsWithClaims = 18;
    supportClaimCoverage().graphsSelectedWithClaims = 4;
    supportClaimCoverage().graphsReachedBody = 0;
    supportClaimCoverage().graphsQueried = 0;
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

// The case the guard exists for, and the one that separates it from the test
// above: the sidecars reached a test body and sat there untouched because no
// engine was ever there to ask.
TEST_F(TestSupportClaimReport, EmptyQueryGuardTrippedWhenRunClaimsWentUnasked)
{
    supportClaimCoverage().graphsWithClaims = 18;
    supportClaimCoverage().graphsSelectedWithClaims = 4;
    supportClaimCoverage().graphsReachedBody = 4;
    supportClaimCoverage().graphsQueried = 0;
    EXPECT_TRUE(verifiedNothing(supportClaimCoverage()));
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenQueriesObserved)
{
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

// An errored query is still an observed query. Counting only the ones that
// resolved would make a total-backend-failure run look like a no-sidecar run and
// hand it a green exit code — the precise silence this guard exists to break.
TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenEveryQueryErrored)
{
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::QUERY_ERRORED));
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWithOnlyQueries)
{
    // (0, N) → false: queries ran but no graph carried a claim.
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

// ---------------------------------------------------------------------------
// The nesting invariant, checked rather than described. Each counter is a subset
// of the one before it, so every attribution line in the summary is a difference
// between two adjacent sets. Break the nesting and those differences stop
// describing any set of graphs at all — while still printing a confident sentence
// about what happened to them.
// ---------------------------------------------------------------------------

TEST(TestCountersAreConsistent, ANestedLadderIsConsistent)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 10;
    coverage.graphsWithClaims = 8;
    coverage.graphsSelectedWithClaims = 5;
    coverage.graphsReachedBody = 4;
    coverage.graphsQueried = 3;
    coverage.graphsNotOpened = 1;

    EXPECT_TRUE(countersAreConsistent(coverage));
}

// The all-zero run: no engine named, or a tree with no sidecars. Every relation holds
// on equality, so the check must not read "nothing happened" as a defect.
TEST(TestCountersAreConsistent, AllZeroIsConsistent)
{
    EXPECT_TRUE(countersAreConsistent(SupportClaimCoverage{}));
}

TEST(TestCountersAreConsistent, MoreClaimsThanGraphsIsInconsistent)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 1;
    coverage.graphsWithClaims = 2;

    EXPECT_FALSE(countersAreConsistent(coverage));
}

// The shape the counter keying is there to prevent. Registration seeds the two
// discovery counters only when an engine was named; key a later bump on the sidecar
// alone and it selects graphs that were never counted as discovered. The summary
// reads that as a harness defect, which sends a reader after the wrong bug -- so the
// invariant is asserted here rather than left to be noticed in a log.
TEST(TestCountersAreConsistent, SelectingMoreThanWasDiscoveredIsInconsistent)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 0;
    coverage.graphsWithClaims = 0;
    coverage.graphsSelectedWithClaims = 100;
    coverage.graphsReachedBody = 100;
    coverage.graphsQueried = 0;

    EXPECT_FALSE(countersAreConsistent(coverage));
}

TEST(TestCountersAreConsistent, RunningMoreThanWasSelectedIsInconsistent)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 4;
    coverage.graphsWithClaims = 4;
    coverage.graphsSelectedWithClaims = 1;
    coverage.graphsReachedBody = 2;

    EXPECT_FALSE(countersAreConsistent(coverage));
}

// queried and notOpened are disjoint halves of the bodies that ran, so their sum
// cannot exceed it. Double-counting one graph as both is the way this breaks.
TEST(TestCountersAreConsistent, QueriesPlusUnopenedExceedingBodiesIsInconsistent)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 2;
    coverage.graphsWithClaims = 2;
    coverage.graphsSelectedWithClaims = 2;
    coverage.graphsReachedBody = 2;
    coverage.graphsQueried = 2;
    coverage.graphsNotOpened = 1;

    EXPECT_FALSE(countersAreConsistent(coverage));
}

// The other direction of the same relation is *not* an inconsistency. A body that
// ran, opened its graph and never queried is a real harness defect -- and one the
// summary already names on its own line. Folding it in here would suppress that
// line at exactly the moment it is true.
TEST(TestCountersAreConsistent, AMissedQueryIsAShortfallAndNotAnInconsistency)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 2;
    coverage.graphsWithClaims = 2;
    coverage.graphsSelectedWithClaims = 2;
    coverage.graphsReachedBody = 2;
    coverage.graphsQueried = 1;

    EXPECT_TRUE(countersAreConsistent(coverage));
}

// graphsWithNoApplicableClaim is deliberately outside the ladder: it is a subset of
// queried rather than a rung, and today both it and queried derive from the same
// read flag, so a relation over it would assert on the shape of one `if`.

// The counters are chosen so that all three subtraction lines would fire: only the
// topmost relation is broken, and every rung below it still descends. That is the
// dangerous shape -- one impossible number upstream, and three downstream counts
// that each look locally reasonable.
TEST_F(TestSupportClaimReport, SummarySuppressesAttributionsWhenCountersDoNotNest)
{
    supportClaimCoverage().graphsFound = 0;
    supportClaimCoverage().graphsWithClaims = 5;
    supportClaimCoverage().graphsSelectedWithClaims = 4;
    supportClaimCoverage().graphsReachedBody = 3;
    supportClaimCoverage().graphsQueried = 0;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("counters_consistent"), false);
    // Each would otherwise be a count of graphs that corresponds to no set of graphs,
    // under a key naming a cause.
    EXPECT_FALSE(doc.at("unenforced").contains("not_selected")) << doc.dump(2);
    EXPECT_FALSE(doc.at("unenforced").contains("skipped_before_run")) << doc.dump(2);
    EXPECT_TRUE(doc.at("harness_defects").empty()) << doc.dump(2);
}

// The counters themselves and the verdicts are still there. They are the evidence: one
// says which number is impossible, the other comes from the claim records and never
// touched the ladder at all.
TEST_F(TestSupportClaimReport, SummaryKeepsCountersAndVerdictsWhenCountersDoNotNest)
{
    supportClaimCoverage().graphsFound = 0;
    supportClaimCoverage().graphsWithClaims = 0;
    supportClaimCoverage().graphsSelectedWithClaims = 3;
    supportClaimCoverage().graphsReachedBody = 3;
    supportClaimCoverage().graphsNotOpened = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto doc = summaryJson();

    const nlohmann::json expectedGraphs
        = {{"found", 0}, {"with_claims", 0}, {"selected", 3}, {"ran", 3}, {"queried", 0}};
    EXPECT_EQ(doc.at("graphs"), expectedGraphs) << doc.dump(2);
    EXPECT_EQ(doc.at("claim_failures").size(), 1u) << doc.dump(2);
    // A direct read of one counter, not a difference between two, so a miscount
    // elsewhere cannot turn it into a wrong claim about which graphs these were.
    EXPECT_EQ(doc.at("unenforced").at("not_opened"), 1) << doc.dump(2);
}

TEST_F(TestSupportClaimReport, SummaryKeepsAttributionsWhenCountersNest)
{
    supportClaimCoverage().graphsFound = 3;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto doc = summaryJson();

    EXPECT_EQ(doc.at("counters_consistent"), true);
    // And the attribution the sound ladder earns is still there.
    EXPECT_EQ(doc.at("unenforced").at("skipped_before_run"), 1) << doc.dump(2);
}

// ---------------------------------------------------------------------------
// coverageFor(): the rules behind the counters, without the process-wide singleton.
//
// The one that matters is that `queried` follows the sidecar state and never
// results.empty(): a sidecar read in full can legally leave no verdicts, and
// counting those as gaps fails healthy runs.
// ---------------------------------------------------------------------------

TEST(TestSupportClaimCoverageRules, NoSidecarCountsNothing)
{
    const auto update = coverageFor(SupportObservation{SidecarState::NONE, {}},
                                    /*observationExpected=*/false);

    EXPECT_FALSE(update.queried);
    EXPECT_FALSE(update.noApplicableClaim);
    EXPECT_FALSE(update.missedQuery);
}

TEST(TestSupportClaimCoverageRules, ReadSidecarWithAVerdictCountsAsQueried)
{
    const auto update = coverageFor(
        SupportObservation{SidecarState::CHECKED, {makeResult(SupportVerdict::CLAIM_ACCEPTED)}},
        /*observationExpected=*/true);

    EXPECT_TRUE(update.queried);
    EXPECT_FALSE(update.noApplicableClaim);
    EXPECT_FALSE(update.missedQuery);
}

// Read in full, but silent about this cell. Covered, and separately counted so it
// does not read as "claimed and holds".
TEST(TestSupportClaimCoverageRules, ReadSidecarWithNoVerdictsIsQueriedButUnclaimed)
{
    const auto update = coverageFor(SupportObservation{SidecarState::CHECKED, {}},
                                    /*observationExpected=*/true);

    EXPECT_TRUE(update.queried);
    EXPECT_TRUE(update.noApplicableClaim);
    EXPECT_FALSE(update.missedQuery);
}

// Drift is not a promise, so a sidecar that only produced UNCLAIMED_SUPPORT still
// promised nothing about this cell.
TEST(TestSupportClaimCoverageRules, DriftAloneStillCountsAsNothingPromised)
{
    const auto update = coverageFor(
        SupportObservation{SidecarState::CHECKED, {makeResult(SupportVerdict::UNCLAIMED_SUPPORT)}},
        /*observationExpected=*/true);

    EXPECT_TRUE(update.queried);
    EXPECT_TRUE(update.noApplicableClaim);
}

// The per-graph gap signal: enforcement was expected and the sidecar was never read.
TEST(TestSupportClaimCoverageRules, ExpectedButUnreadSidecarIsAHarnessBug)
{
    const auto update = coverageFor(SupportObservation{SidecarState::NONE, {}},
                                    /*observationExpected=*/true);

    EXPECT_FALSE(update.queried);
    EXPECT_TRUE(update.missedQuery);
}

// A graph that never opened is not an enforcement gap. The run is already failing
// on the graph, and "enforcement would have passed without checking" would be a
// false statement pointing at a bug that is not there.
TEST(TestSupportClaimCoverageRules, UnopenedGraphIsUncoveredButNotAHarnessBug)
{
    const auto update = coverageFor(SupportObservation{SidecarState::NOT_QUERIED, {}},
                                    /*observationExpected=*/true);

    EXPECT_FALSE(update.queried);
    EXPECT_FALSE(update.missedQuery);
    EXPECT_FALSE(update.noApplicableClaim);
    EXPECT_TRUE(update.notOpened) << "the shortfall must be attributable to the graph, "
                                     "not left for the summary to blame on --gtest_filter";
}

// graphsReachedBody is deliberately absent from CoverageUpdate: it is true before
// the observation exists, and deriving it here would lose it whenever the read
// throws. The harness publishes it directly, and TestSupportClaimEnforcement's
// ReachedBodyIsCountedEvenWhenTheClaimReadThrows pins that.

// Its own counter, so the summary can subtract it before attributing the rest of
// the shortfall to --gtest_filter. A graph that never opened did run.
TEST(TestSupportClaimSummary, UnopenedGraphsAreNotBlamedOnTheFilter)
{
    SupportClaimCoverage coverage;
    coverage.graphsFound = 4;
    coverage.graphsWithClaims = 4;
    coverage.graphsSelectedWithClaims = 4;
    coverage.graphsReachedBody = 4;
    coverage.graphsQueried = 3;
    coverage.graphsNotOpened = 1;

    const auto doc = buildSupportClaimSummary(
                         coverage, SupportClaimVerdicts::get(), ClaimMode::ENFORCE, testRun())
                         .value_or(nlohmann::json::object());

    ASSERT_TRUE(doc.contains("unenforced")) << doc.dump(2);
    const auto& unenforced = doc.at("unenforced");
    EXPECT_EQ(unenforced.at("not_opened"), 1) << doc.dump(2);
    EXPECT_EQ(unenforced.at("not_selected"), 0)
        << "every claim-bearing graph is accounted for, so nothing is the filter's doing";
    EXPECT_EQ(unenforced.at("skipped_before_run"), 0);
    EXPECT_EQ(doc.at("harness_defects").at("missed_query"), 0);
}

// ---------------------------------------------------------------------------
// missedQueryComplaint(): the per-graph gap.
//
// The run-level guard only fires when *no* graph anywhere was queried, so a gap on
// one graph out of many needs its own signal or it is silently absorbed.
// ---------------------------------------------------------------------------

TEST(TestMissedQueryComplaint, NoGapIsSilent)
{
    const CoverageUpdate update; // missedQuery defaults false

    EXPECT_FALSE(missedQueryComplaint(update, "test/bundle").has_value());
}

TEST(TestMissedQueryComplaint, AGapNamesTheBundleItIsAbout)
{
    CoverageUpdate update;
    update.missedQuery = true;

    const auto complaint = missedQueryComplaint(update, "test/bundle");

    ASSERT_TRUE(complaint.has_value());
    // The message is the whole payload -- a complaint carries no severity to inspect,
    // and one that cannot say which bundle it came from is unactionable in a CI log.
    EXPECT_NE(complaint->message.find("test/bundle"), std::string::npos) << complaint->message;
}

// ---------------------------------------------------------------------------
// reportBundlePath(): the path a summary entry names.
//
// Relative to the bundle root's parent, so the same bundle reads the same on every
// machine and a script can find it in the source tree.
// ---------------------------------------------------------------------------

TEST(TestReportBundlePath, APathUnderTheRootStartsAtTheRootFolder)
{
    EXPECT_EQ(reportBundlePath("/opt/rocm/lib/integration-test-bundles/quick/ConvFwd/a.json",
                               "",
                               "/opt/rocm/lib/integration-test-bundles"),
              "integration-test-bundles/quick/ConvFwd/a.json");
}

// The case has its own field, so the path names the file a reader opens.
TEST(TestReportBundlePath, TheCaseSuffixIsDropped)
{
    EXPECT_EQ(reportBundlePath("/data/integration-test-bundles/quick/Sdpa/sweep.json#case_a",
                               "case_a",
                               "/data/integration-test-bundles"),
              "integration-test-bundles/quick/Sdpa/sweep.json");
}

TEST(TestReportBundlePath, OnlyTheRecordsOwnCaseIsDropped)
{
    EXPECT_EQ(reportBundlePath("dir/a.json#case_a", "case_b", ""), "dir/a.json#case_a");
}

TEST(TestReportBundlePath, AnEmptyRootLeavesThePathAsRecorded)
{
    EXPECT_EQ(reportBundlePath("/data/integration-test-bundles/quick/a.json", "", ""),
              "/data/integration-test-bundles/quick/a.json");
}

// A "../" chain is harder to follow than the path it replaces.
TEST(TestReportBundlePath, APathOutsideTheRootIsLeftAlone)
{
    EXPECT_EQ(reportBundlePath("/elsewhere/a.json", "", "/data/integration-test-bundles"),
              "/elsewhere/a.json");
}

TEST(TestReportBundlePath, ATrailingSlashOnTheRootChangesNothing)
{
    EXPECT_EQ(reportBundlePath("/data/integration-test-bundles/quick/a.json",
                               "",
                               "/data/integration-test-bundles/"),
              "integration-test-bundles/quick/a.json");
}

// The installed root is found relative to the binary, so it arrives with "..".
TEST(TestReportBundlePath, TheRootIsNormalizedBeforeMatching)
{
    EXPECT_EQ(reportBundlePath("/opt/rocm/lib/integration-test-bundles/quick/a.json",
                               "",
                               "/opt/rocm/bin/../lib/integration-test-bundles"),
              "integration-test-bundles/quick/a.json");
}

// The summary is where the rewrite is actually applied; the tests above only pin the
// helper.
TEST_F(TestSupportClaimReport, SummaryReportsBundlesRelativeToTheRoot)
{
    auto broken = makeResult(SupportVerdict::CLAIM_BROKEN);
    broken.bundlePath = "/data/integration-test-bundles/quick/ConvFwd/a.json";
    SupportClaimVerdicts::get().record(broken);

    auto run = testRun();
    run.bundleRoot = "/data/integration-test-bundles";
    const auto doc = buildSupportClaimSummary(
        supportClaimCoverage(), SupportClaimVerdicts::get(), ClaimMode::ENFORCE, run);

    ASSERT_TRUE(doc.has_value());
    EXPECT_EQ(doc->at("claim_failures").at(0).at("bundle"),
              "integration-test-bundles/quick/ConvFwd/a.json");
}

// NOLINTEND(readability-identifier-naming)
