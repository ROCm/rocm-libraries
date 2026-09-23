// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"

using hipdnn_integration_tests::bundle::ClaimMode;
using hipdnn_integration_tests::bundle::countersAreConsistent;
using hipdnn_integration_tests::bundle::coverageFor;
using hipdnn_integration_tests::bundle::CoverageUpdate;
using hipdnn_integration_tests::bundle::missedQueryComplaint;
using hipdnn_integration_tests::bundle::printSupportClaimSummary;
using hipdnn_integration_tests::bundle::SidecarState;
using hipdnn_integration_tests::bundle::SupportClaimCoverage;
using hipdnn_integration_tests::bundle::supportClaimCoverage;
using hipdnn_integration_tests::bundle::SupportClaimVerdicts;
using hipdnn_integration_tests::bundle::SupportObservation;
using hipdnn_integration_tests::bundle::SupportResult;
using hipdnn_integration_tests::bundle::SupportVerdict;
using hipdnn_integration_tests::bundle::verifiedNothing;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

SupportResult makeResult(SupportVerdict v)
{
    return SupportResult{v,
                         "test/bundle",
                         "ENGINE_A",
                         "gfx942",
                         "linux",
                         "detail",
                         hipdnn_frontend::ErrorCode::OK,
                         {}};
}

std::string summary(ClaimMode claims = ClaimMode::ENFORCE)
{
    std::ostringstream oss;
    printSupportClaimSummary(supportClaimCoverage(), SupportClaimVerdicts::get(), claims, oss);
    return oss.str();
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

    EXPECT_NE(summary().find("1 queried (2 verdicts)"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Progressive print levels
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintLevel1ShowsCounters)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("2 found, 1 with claims, 1 selected, 1 ran, 1 queried"),
              std::string::npos);
    EXPECT_NE(output.find("confirmed: 1"), std::string::npos);
    EXPECT_NE(output.find("broken: 0"), std::string::npos);
}

// A summary scraped out of a CI log has to say on its own face whether the failures
// under it were fatal, because a report-only lane prints the same shape and the same
// failure list. Asserted on a run that has failures, since that is the case where the
// header is load-bearing rather than decorative.
TEST_F(TestSupportClaimReport, PrintHeaderNamesEnforcement)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto output = summary(ClaimMode::ENFORCE);

    EXPECT_NE(output.find("==== SUPPORT CLAIM SUMMARY (ENFORCING) ===="), std::string::npos);
    // The failures are listed in full under that header. Withholding them would make
    // the mode a coverage difference rather than an exit-code one.
    EXPECT_NE(output.find("CLAIM FAILURES (1)"), std::string::npos);
}

// "accepted" and "confirmed" are different facts and the header has to say so,
// because only one of them reached the depth its bundle declares.
TEST_F(TestSupportClaimReport, PrintDistinguishesAcceptedFromConfirmed)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_ACCEPTED));

    const auto output = summary();

    EXPECT_NE(output.find("accepted: 1"), std::string::npos);
    EXPECT_NE(output.find("confirmed: 0"), std::string::npos);
    EXPECT_NE(output.find("confirmed = the run reached the depth"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintLevel2ShowsFailureDetail)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto output = summary();

    EXPECT_NE(output.find("CLAIM FAILURES"), std::string::npos);
    EXPECT_NE(output.find("test/bundle"), std::string::npos);
    EXPECT_NE(output.find("ENGINE_A"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintLevel3ListsUnclaimedBundles)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));

    const auto output = summary();

    EXPECT_NE(output.find("UNCLAIMED SUPPORT"), std::string::npos);
    // A bare count is not actionable — the bundle has to be named.
    EXPECT_NE(output.find("test/bundle"), std::string::npos);
    EXPECT_NE(output.find("ENGINE_A"), std::string::npos);
}

// A count with no bundle names is not actionable, and this is the section that
// tells an operator which cells must not be published as working support.
TEST_F(TestSupportClaimReport, PrintNamesBundlesThatFailedInUse)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_FAILED_IN_USE));

    const auto output = summary();

    // The header names the same bucket as the counter line ("failed-in-use"), so a
    // reader never has to translate between two words for one verdict.
    EXPECT_NE(output.find("FAILED IN USE"), std::string::npos);
    EXPECT_NE(output.find("test/bundle"), std::string::npos);
    EXPECT_NE(output.find("ENGINE_A"), std::string::npos);
    // Not a claim failure, so it must not appear under the failure header.
    EXPECT_EQ(output.find("CLAIM FAILURES"), std::string::npos);
}

// A filtered run discovers more claim-bearing graphs than it selects. That gap --
// discovered minus selected -- is the filter's doing and nothing else's, so the
// summary names it instead of leaving a mismatch to be misread as a harness gap.
TEST_F(TestSupportClaimReport, PrintAttributesUnselectedGraphsToTheFilter)
{
    supportClaimCoverage().graphsFound = 3;
    supportClaimCoverage().graphsWithClaims = 3;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("2 claim-bearing graph(s) were discovered but not selected"),
              std::string::npos);
    // Everything selected ran, so none of it is the skip-list's doing.
    EXPECT_EQ(output.find("skipped before running"), std::string::npos) << output;
}

// The other half of the split. These graphs *were* selected -- the filter let them
// through -- and then SetUp() skipped them before running. Blaming --gtest_filter
// for them would send a reader to edit the one knob that is already correct.
TEST_F(TestSupportClaimReport, PrintSeparatesSelectedButSkippedFromTheFilterRemainder)
{
    supportClaimCoverage().graphsFound = 5;
    supportClaimCoverage().graphsWithClaims = 5;
    supportClaimCoverage().graphsSelectedWithClaims = 3;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("2 claim-bearing graph(s) were selected but skipped before running"),
              std::string::npos)
        << output;
    EXPECT_NE(output.find("2 claim-bearing graph(s) were discovered but not selected"),
              std::string::npos)
        << output;
}

// The arch-skipped lane, which is the common case this split exists for: the filter
// selected everything and SetUp() skipped all of it. One line, naming the skip, and
// no mention of a filter that did nothing wrong.
TEST_F(TestSupportClaimReport, PrintBlamesTheSkipWhenTheFilterSelectedEverything)
{
    supportClaimCoverage().graphsFound = 4;
    supportClaimCoverage().graphsWithClaims = 4;
    supportClaimCoverage().graphsSelectedWithClaims = 4;
    supportClaimCoverage().graphsReachedBody = 0;

    const auto output = summary();

    EXPECT_NE(output.find("4 claim-bearing graph(s) were selected but skipped before running"),
              std::string::npos)
        << output;
    EXPECT_EQ(output.find("--gtest_filter"), std::string::npos)
        << "the filter selected every claim-bearing graph; the skip is what stopped them\n"
        << output;
}

// A body that ran, opened its graph and still never queried is the one shortfall no
// configuration can produce. It gets its own line saying so, because sending a reader
// to the skip-list for a harness bug costs them the afternoon.
TEST_F(TestSupportClaimReport, PrintNamesAMissedQueryAsAHarnessDefect)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 2;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("1 claim-bearing graph(s) ran without ever being queried"),
              std::string::npos)
        << output;
    EXPECT_NE(output.find("harness defect"), std::string::npos) << output;
    EXPECT_EQ(output.find("skipped before running"), std::string::npos) << output;
}

// A graph that never opened ran and failed; it is already accounted for by its own
// line, so it must not also be counted as a skip or as a missed query.
TEST_F(TestSupportClaimReport, PrintDoesNotCountUnopenedGraphsAsSkipped)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 2;
    supportClaimCoverage().graphsQueried = 1;
    supportClaimCoverage().graphsNotOpened = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("1 claim-bearing graph(s) could not be opened"), std::string::npos);
    EXPECT_EQ(output.find("skipped before running"), std::string::npos) << output;
    EXPECT_EQ(output.find("harness defect"), std::string::npos) << output;
}

TEST_F(TestSupportClaimReport, PrintOmitsEveryShortfallNoteWhenEverythingRan)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_EQ(output.find("not selected"), std::string::npos);
    EXPECT_EQ(output.find("skipped before running"), std::string::npos);
    EXPECT_EQ(output.find("harness defect"), std::string::npos);
}

// Otherwise invisible: a sidecar read in full that promised nothing for this cell
// leaves no verdict, so the tallies look identical to a graph nobody ever claimed.
TEST_F(TestSupportClaimReport, PrintNamesGraphsWhoseSidecarClaimsNothingHere)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 2;
    supportClaimCoverage().graphsQueried = 2;
    supportClaimCoverage().graphsWithNoApplicableClaim = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("1 queried graph(s) carry a sidecar that claims nothing"),
              std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintOmitsTheNoteWhenEveryQueriedGraphWasClaimed)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsSelectedWithClaims = 1;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    EXPECT_EQ(summary().find("claims nothing"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintShowsNoFailureSectionWhenOnlyConfirmed)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    EXPECT_EQ(summary().find("CLAIM FAILURES"), std::string::npos);
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

    const auto output = summary();

    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("1 with claims, 1 selected, 1 ran, 0 queried"), std::string::npos)
        << output;
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
// dangerous shape -- one impossible number upstream, and three downstream sentences
// that each look locally reasonable.
TEST_F(TestSupportClaimReport, PrintSuppressesAttributionsWhenCountersDoNotNest)
{
    supportClaimCoverage().graphsFound = 0;
    supportClaimCoverage().graphsWithClaims = 5;
    supportClaimCoverage().graphsSelectedWithClaims = 4;
    supportClaimCoverage().graphsReachedBody = 3;
    supportClaimCoverage().graphsQueried = 0;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_NE(output.find("do not nest"), std::string::npos) << output;
    // Each would otherwise print a count of graphs that corresponds to no set of
    // graphs, next to a sentence naming a cause.
    EXPECT_EQ(output.find("harness defect"), std::string::npos) << output;
    EXPECT_EQ(output.find("skipped before running"), std::string::npos) << output;
    EXPECT_EQ(output.find("not selected"), std::string::npos) << output;
}

// The counters themselves and the verdicts still print. They are the evidence: one
// says which number is impossible, the other comes from the claim records and never
// touched the ladder at all.
TEST_F(TestSupportClaimReport, PrintKeepsCountersAndVerdictsWhenCountersDoNotNest)
{
    supportClaimCoverage().graphsFound = 0;
    supportClaimCoverage().graphsWithClaims = 0;
    supportClaimCoverage().graphsSelectedWithClaims = 3;
    supportClaimCoverage().graphsReachedBody = 3;
    supportClaimCoverage().graphsNotOpened = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    const auto output = summary();

    EXPECT_NE(output.find("0 found, 0 with claims, 3 selected, 3 ran, 0 queried"),
              std::string::npos)
        << output;
    EXPECT_NE(output.find("CLAIM FAILURES"), std::string::npos) << output;
    // A direct read of one counter, not a difference between two, so a miscount
    // elsewhere cannot turn it into a wrong claim about which graphs these were.
    EXPECT_NE(output.find("1 claim-bearing graph(s) could not be opened"), std::string::npos)
        << output;
}

TEST_F(TestSupportClaimReport, PrintOmitsTheWarningWhenCountersNest)
{
    supportClaimCoverage().graphsFound = 3;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsSelectedWithClaims = 2;
    supportClaimCoverage().graphsReachedBody = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_CONFIRMED));

    const auto output = summary();

    EXPECT_EQ(output.find("do not nest"), std::string::npos) << output;
    // And the attribution the sound ladder earns is still there.
    EXPECT_NE(output.find("1 claim-bearing graph(s) were selected but skipped"), std::string::npos)
        << output;
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

    std::ostringstream os;
    printSupportClaimSummary(coverage, SupportClaimVerdicts::get(), ClaimMode::ENFORCE, os);
    const std::string out = os.str();

    EXPECT_NE(out.find("could not be opened"), std::string::npos) << out;
    EXPECT_EQ(out.find("--gtest_filter"), std::string::npos)
        << "every claim-bearing graph is accounted for, so nothing is the filter's doing\n"
        << out;
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

// NOLINTEND(readability-identifier-naming)
