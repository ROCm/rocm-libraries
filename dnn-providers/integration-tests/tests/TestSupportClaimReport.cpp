// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"

using hipdnn_integration_tests::bundle::printSupportClaimSummary;
using hipdnn_integration_tests::bundle::supportClaimCoverage;
using hipdnn_integration_tests::bundle::SupportClaimVerdicts;
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

std::string summary()
{
    std::ostringstream oss;
    printSupportClaimSummary(supportClaimCoverage(), SupportClaimVerdicts::get(), oss);
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

TEST_F(TestSupportClaimReport, RecordsSatisfied)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::SATISFIED), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 1u);
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

TEST_F(TestSupportClaimReport, RecordsEngineNotLoaded)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::ENGINE_NOT_LOADED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::ENGINE_NOT_LOADED), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsUnclaimedSupport)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::UNCLAIMED_SUPPORT), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsNotEnforced)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::NOT_ENFORCED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::NOT_ENFORCED), 1u);
    EXPECT_FALSE(SupportClaimVerdicts::get().hasFailures());
}

// A verdict the log has never seen counts zero rather than misreporting.
TEST_F(TestSupportClaimReport, CountIsZeroForUnseenVerdict)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 0u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::NOT_ENFORCED), 0u);
}

// ---------------------------------------------------------------------------
// Multiple records aggregate correctly
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, MultipleRecordsAccumulate)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::NOT_ENFORCED));

    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::SATISFIED), 2u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::NOT_ENFORCED), 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 4u);
}

// ---------------------------------------------------------------------------
// Clearing each accumulator
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, ClearEmptiesTheVerdictLog)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    SupportClaimVerdicts::get().clear();

    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::SATISFIED), 0u);
    EXPECT_EQ(SupportClaimVerdicts::get().count(SupportVerdict::CLAIM_BROKEN), 0u);
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
// own counter, not derived from the verdict log, because multi-engine
// enforcement produces N verdicts per graph.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, QueriedCountIsIndependentOfVerdictCount)
{
    supportClaimCoverage().graphsFound = 5;
    supportClaimCoverage().graphsWithClaims = 2;
    supportClaimCoverage().graphsQueried = 1;

    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::NOT_ENFORCED));

    EXPECT_EQ(supportClaimCoverage().graphsFound, 5u);
    EXPECT_EQ(supportClaimCoverage().graphsWithClaims, 2u);
    EXPECT_EQ(supportClaimCoverage().graphsQueried, 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 2u);
}

// ---------------------------------------------------------------------------
// Progressive print levels
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintLevel1ShowsCounters)
{
    supportClaimCoverage().graphsFound = 2;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));

    const auto output = summary();

    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("2 found, 1 with claims, 1 queried"), std::string::npos);
    EXPECT_NE(output.find("satisfied: 1"), std::string::npos);
    EXPECT_NE(output.find("not-enforced: 0"), std::string::npos);
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

TEST_F(TestSupportClaimReport, PrintShowsNoFailureSectionForEngineNotLoaded)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::ENGINE_NOT_LOADED));

    EXPECT_EQ(summary().find("CLAIM FAILURES"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintShowsNoFailureSectionWhenOnlySatisfied)
{
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));

    EXPECT_EQ(summary().find("CLAIM FAILURES"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Default-off inertness: a run over a tree with no sidecars anywhere must stay
// completely silent. That is every run in the tree today, and the report has to
// add nothing to their output.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintIsSilentWhenGraphsFoundButNoSidecars)
{
    supportClaimCoverage().graphsFound = 100;

    // No sidecars, so no records: every one of those graphs evaluated to
    // NO_SIDECAR, and NO_SIDECAR is never recorded.
    EXPECT_TRUE(summary().empty());
}

TEST_F(TestSupportClaimReport, PrintShowsNotEnforcedOnceASidecarExists)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::NOT_ENFORCED));

    const auto output = summary();

    // Enforcing nothing is a result, not silence that reads as success.
    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("not-enforced: 1"), std::string::npos);
}

// The run that trips the guard must still print. Its summary is all zeros except
// the discovery counts, and those counts are the only thing that distinguishes it
// from a run with nothing to enforce.
TEST_F(TestSupportClaimReport, PrintShowsDiscoveryCountsWhenNothingWasQueried)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;

    const auto output = summary();

    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("1 with claims, 0 queried"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Empty-query guard (RFC 0015 §7.2)
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenNothingDiscovered)
{
    // (0, 0) → false: no graph carried a claim, so there was nothing to enforce.
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardTrippedWhenDiscoveredButNoQueries)
{
    // (N, 0) → true: claim-bearing graphs exist but not one was ever queried.
    supportClaimCoverage().graphsWithClaims = 1;
    EXPECT_TRUE(verifiedNothing(supportClaimCoverage()));
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenQueriesObserved)
{
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

// An errored query is still an observed query. Counting only the ones that
// resolved would make a total-backend-failure run look like a no-sidecar run and
// hand it a green exit code — the precise silence this guard exists to break.
TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenEveryQueryErrored)
{
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::QUERY_ERRORED));
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWithOnlyQueries)
{
    // (0, N) → false: queries ran but no graph carried a claim.
    supportClaimCoverage().graphsQueried = 1;
    SupportClaimVerdicts::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_FALSE(verifiedNothing(supportClaimCoverage()));
}

// ---------------------------------------------------------------------------
// Multi-engine: 1 graph queried produces N verdicts (one per engine).
// The queried count must be 1, not N.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, MultiEngineQueriedCountIsPerGraph)
{
    supportClaimCoverage().graphsFound = 1;
    supportClaimCoverage().graphsWithClaims = 1;
    supportClaimCoverage().graphsQueried = 1;

    SupportResult r1 = makeResult(SupportVerdict::SATISFIED);
    r1.engineName = "ENGINE_A";
    SupportResult r2 = makeResult(SupportVerdict::NOT_ENFORCED);
    r2.engineName = "ENGINE_B";

    SupportClaimVerdicts::get().record(r1);
    SupportClaimVerdicts::get().record(r2);

    EXPECT_EQ(supportClaimCoverage().graphsQueried, 1u);
    EXPECT_EQ(SupportClaimVerdicts::get().total(), 2u);

    EXPECT_NE(summary().find("1 queried (2 verdicts)"), std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
