// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <sstream>
#include <string>

#include "harness/bundle/SupportClaimReport.hpp"
#include "harness/bundle/SupportVerdict.hpp"

using hipdnn_integration_tests::bundle::SupportClaimReport;
using hipdnn_integration_tests::bundle::SupportResult;
using hipdnn_integration_tests::bundle::SupportVerdict;

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

class TestSupportClaimReport : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SupportClaimReport::get().reset();
    }
    void TearDown() override
    {
        SupportClaimReport::get().reset();
    }
};

} // namespace

// ---------------------------------------------------------------------------
// Zero records → no output
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintIsNoOpWhenEmpty)
{
    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    EXPECT_TRUE(oss.str().empty());
}

// ---------------------------------------------------------------------------
// Single-verdict recording
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, RecordsSatisfied)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::SATISFIED), 1u);
    EXPECT_EQ(SupportClaimReport::get().getTotalRecorded(), 1u);
    EXPECT_FALSE(SupportClaimReport::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsClaimBroken)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::CLAIM_BROKEN), 1u);
    EXPECT_TRUE(SupportClaimReport::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsQueryErrored)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::QUERY_ERRORED));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::QUERY_ERRORED), 1u);
    EXPECT_TRUE(SupportClaimReport::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsEngineNotLoaded)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::ENGINE_NOT_LOADED));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::ENGINE_NOT_LOADED), 1u);
    EXPECT_FALSE(SupportClaimReport::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsUnclaimedSupport)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::UNCLAIMED_SUPPORT), 1u);
    EXPECT_FALSE(SupportClaimReport::get().hasFailures());
}

TEST_F(TestSupportClaimReport, RecordsNotEnforced)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::NOT_ENFORCED));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::NOT_ENFORCED), 1u);
    EXPECT_FALSE(SupportClaimReport::get().hasFailures());
}

// A verdict the report has never seen counts zero rather than misreporting.
TEST_F(TestSupportClaimReport, CountIsZeroForUnseenVerdict)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::CLAIM_BROKEN), 0u);
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::NOT_ENFORCED), 0u);
}

// ---------------------------------------------------------------------------
// Multiple records aggregate correctly
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, MultipleRecordsAccumulate)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimReport::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    SupportClaimReport::get().record(makeResult(SupportVerdict::NOT_ENFORCED));

    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::SATISFIED), 2u);
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::CLAIM_BROKEN), 1u);
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::NOT_ENFORCED), 1u);
    EXPECT_EQ(SupportClaimReport::get().getTotalRecorded(), 4u);
}

// ---------------------------------------------------------------------------
// Reset clears everything
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, ResetClearsAll)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimReport::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));
    SupportClaimReport::get().recordGraphFound();
    SupportClaimReport::get().recordGraphWithClaims();

    SupportClaimReport::get().reset();

    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::SATISFIED), 0u);
    EXPECT_EQ(SupportClaimReport::get().count(SupportVerdict::CLAIM_BROKEN), 0u);
    EXPECT_EQ(SupportClaimReport::get().getTotalRecorded(), 0u);
    EXPECT_EQ(SupportClaimReport::get().getGraphsFound(), 0u);
    EXPECT_EQ(SupportClaimReport::get().getGraphsWithClaims(), 0u);
    EXPECT_EQ(SupportClaimReport::get().getGraphsWithClaimsVerified(), 0u);
    EXPECT_FALSE(SupportClaimReport::get().hasFailures());
}

// ---------------------------------------------------------------------------
// The nesting invariant: queried ⊆ withClaimsFound ⊆ found. The queried count
// is an explicit counter (recordGraphWithClaimsVerified), not derived from _records.size(),
// because multi-engine enforcement produces N records per graph.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, QueriedCountTracksExplicitCalls)
{
    for(int i = 0; i < 5; ++i)
    {
        SupportClaimReport::get().recordGraphFound();
    }
    SupportClaimReport::get().recordGraphWithClaims();
    SupportClaimReport::get().recordGraphWithClaims();

    SupportClaimReport::get().recordGraphWithClaimsVerified();
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    SupportClaimReport::get().record(makeResult(SupportVerdict::NOT_ENFORCED));

    EXPECT_EQ(SupportClaimReport::get().getGraphsFound(), 5u);
    EXPECT_EQ(SupportClaimReport::get().getGraphsWithClaims(), 2u);
    EXPECT_EQ(SupportClaimReport::get().getGraphsWithClaimsVerified(), 1u);
    EXPECT_EQ(SupportClaimReport::get().getTotalRecorded(), 2u);
}

// ---------------------------------------------------------------------------
// Progressive print levels
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintLevel1ShowsCounters)
{
    SupportClaimReport::get().recordGraphFound();
    SupportClaimReport::get().recordGraphFound();
    SupportClaimReport::get().recordGraphWithClaims();
    SupportClaimReport::get().recordGraphWithClaimsVerified();
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("2 found, 1 with claims, 1 queried"), std::string::npos);
    EXPECT_NE(output.find("satisfied: 1"), std::string::npos);
    EXPECT_NE(output.find("not-enforced: 0"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintLevel2ShowsFailureDetail)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::CLAIM_BROKEN));

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("CLAIM FAILURES"), std::string::npos);
    EXPECT_NE(output.find("test/bundle"), std::string::npos);
    EXPECT_NE(output.find("ENGINE_A"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintLevel3ListsUnclaimedBundles)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::UNCLAIMED_SUPPORT));

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("UNCLAIMED SUPPORT"), std::string::npos);
    // A bare count is not actionable — the bundle has to be named.
    EXPECT_NE(output.find("test/bundle"), std::string::npos);
    EXPECT_NE(output.find("ENGINE_A"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintShowsNoFailureSectionForEngineNotLoaded)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::ENGINE_NOT_LOADED));

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_EQ(output.find("CLAIM FAILURES"), std::string::npos);
}

TEST_F(TestSupportClaimReport, PrintShowsNoFailureSectionWhenOnlySatisfied)
{
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_EQ(output.find("CLAIM FAILURES"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Default-off inertness: a run over a tree with no sidecars anywhere must stay
// completely silent. That is every run in the tree today, and the report has to
// add nothing to their output.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, PrintIsSilentWhenGraphsFoundButNoSidecars)
{
    for(int i = 0; i < 100; ++i)
    {
        SupportClaimReport::get().recordGraphFound();
    }

    // No sidecars, so no records: every one of those graphs evaluated to
    // NO_SIDECAR, and NO_SIDECAR is never recorded.
    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    EXPECT_TRUE(oss.str().empty());
}

TEST_F(TestSupportClaimReport, PrintShowsNotEnforcedOnceASidecarExists)
{
    SupportClaimReport::get().recordGraphFound();
    SupportClaimReport::get().recordGraphWithClaims();
    SupportClaimReport::get().record(makeResult(SupportVerdict::NOT_ENFORCED));

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    // Enforcing nothing is a result, not silence that reads as success.
    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("not-enforced: 1"), std::string::npos);
}

// The run that trips the guard must still print. Its summary is all zeros except
// the discovery counts, and those counts are the only thing that distinguishes it
// from a run with nothing to enforce.
TEST_F(TestSupportClaimReport, PrintShowsDiscoveryCountsWhenNothingWasQueried)
{
    SupportClaimReport::get().recordGraphFound();
    SupportClaimReport::get().recordGraphWithClaims();

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("SUPPORT CLAIM SUMMARY"), std::string::npos);
    EXPECT_NE(output.find("1 with claims, 0 queried"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Empty-query guard (RFC 0015 §7.2)
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenNothingDiscovered)
{
    // (0, 0) → false: no graph carried a claim, so there was nothing to enforce.
    EXPECT_FALSE(SupportClaimReport::get().claimsFoundButNoneVerified());
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardTrippedWhenDiscoveredButNoQueries)
{
    // (N, 0) → true: claim-bearing graphs exist but not one was ever queried.
    SupportClaimReport::get().recordGraphWithClaims();
    EXPECT_TRUE(SupportClaimReport::get().claimsFoundButNoneVerified());
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenQueriesObserved)
{
    SupportClaimReport::get().recordGraphWithClaims();
    SupportClaimReport::get().recordGraphWithClaimsVerified();
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_FALSE(SupportClaimReport::get().claimsFoundButNoneVerified());
}

// An errored query is still an observed query. Counting only the ones that
// resolved would make a total-backend-failure run look like a no-sidecar run and
// hand it a green exit code — the precise silence this guard exists to break.
TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWhenEveryQueryErrored)
{
    SupportClaimReport::get().recordGraphWithClaims();
    SupportClaimReport::get().recordGraphWithClaimsVerified();
    SupportClaimReport::get().record(makeResult(SupportVerdict::QUERY_ERRORED));
    EXPECT_FALSE(SupportClaimReport::get().claimsFoundButNoneVerified());
}

TEST_F(TestSupportClaimReport, EmptyQueryGuardNotTrippedWithOnlyQueries)
{
    // (0, N) → false: queries ran but no graph carried a claim.
    SupportClaimReport::get().recordGraphWithClaimsVerified();
    SupportClaimReport::get().record(makeResult(SupportVerdict::SATISFIED));
    EXPECT_FALSE(SupportClaimReport::get().claimsFoundButNoneVerified());
}

// ---------------------------------------------------------------------------
// Multi-engine: 1 graph queried produces N verdicts (one per engine).
// The queried count must be 1, not N.
// ---------------------------------------------------------------------------

TEST_F(TestSupportClaimReport, MultiEngineQueriedCountIsPerGraph)
{
    SupportClaimReport::get().recordGraphFound();
    SupportClaimReport::get().recordGraphWithClaims();
    SupportClaimReport::get().recordGraphWithClaimsVerified();

    SupportResult r1 = makeResult(SupportVerdict::SATISFIED);
    r1.engineName = "ENGINE_A";
    SupportResult r2 = makeResult(SupportVerdict::NOT_ENFORCED);
    r2.engineName = "ENGINE_B";

    SupportClaimReport::get().record(r1);
    SupportClaimReport::get().record(r2);

    EXPECT_EQ(SupportClaimReport::get().getGraphsWithClaimsVerified(), 1u);
    EXPECT_EQ(SupportClaimReport::get().getTotalRecorded(), 2u);

    std::ostringstream oss;
    SupportClaimReport::get().print(oss);
    const auto output = oss.str();

    EXPECT_NE(output.find("1 queried (2 verdicts)"), std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
