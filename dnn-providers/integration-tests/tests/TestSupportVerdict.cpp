// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <vector>

#include "harness/bundle/LoadedEngineTable.hpp"
#include "harness/bundle/SupportVerdict.hpp"

using hipdnn_frontend::ErrorCode;
using hipdnn_integration_tests::bundle::baseArchToken;
using hipdnn_integration_tests::bundle::evaluateSupport;
using hipdnn_integration_tests::bundle::isFailure;
using hipdnn_integration_tests::bundle::LoadedEngine;
using hipdnn_integration_tests::bundle::observeAllSupport;
using hipdnn_integration_tests::bundle::SupportResult;
using hipdnn_integration_tests::bundle::SupportVerdict;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t ENGINE_A = 42;
constexpr int64_t ENGINE_B = 7;

const std::string BUNDLE = "bundles/ConvFprop_fp16";
const std::string ENGINE = "MIOPEN_ENGINE";
const std::string ARCH = "gfx942";
const std::string PLAT = "linux";

SupportResult eval(ErrorCode code,
                   const std::vector<int64_t>& ids,
                   int64_t engineId,
                   bool claimed,
                   bool hasSidecar,
                   std::string_view queryMessage = {})
{
    return evaluateSupport(
        code, ids, engineId, claimed, hasSidecar, BUNDLE, ENGINE, ARCH, PLAT, queryMessage);
}

// A plausible stand-in for what a broken backend actually puts in err_msg.
const std::string QUERY_MSG = "hipdnnBackendFinalize failed: HIPDNN_STATUS_INTERNAL_ERROR";

} // namespace

// ---------------------------------------------------------------------------
// claimed + resolved + present → SATISFIED
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimedAndSupportedIsSatisfied)
{
    auto r = eval(ErrorCode::OK, {ENGINE_A, ENGINE_B}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::SATISFIED);
    EXPECT_FALSE(isFailure(r.verdict));
}

// ---------------------------------------------------------------------------
// claimed + resolved + absent → CLAIM_BROKEN
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimedButDeclinedIsClaimBroken)
{
    auto r = eval(ErrorCode::OK, {ENGINE_B}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::CLAIM_BROKEN);
    EXPECT_TRUE(isFailure(r.verdict));
}

TEST(TestSupportVerdict, ClaimedButGraphNotSupportedIsClaimBroken)
{
    auto r = eval(ErrorCode::GRAPH_NOT_SUPPORTED, {}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::CLAIM_BROKEN);
    EXPECT_TRUE(isFailure(r.verdict));
}

// ---------------------------------------------------------------------------
// claimed + unresolved → QUERY_ERRORED
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ClaimedButHeuristicQueryFailedIsErroredBeforeAssert)
{
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::QUERY_ERRORED);
    EXPECT_TRUE(isFailure(r.verdict));
}

TEST(TestSupportVerdict, ClaimedButBackendErrorIsErroredBeforeAssert)
{
    auto r = eval(ErrorCode::HIPDNN_BACKEND_ERROR, {}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::QUERY_ERRORED);
    EXPECT_TRUE(isFailure(r.verdict));
}

// Unresolved status dominates ranked-list membership: even if the id happens to
// be in the list, the query was not trustworthy.
TEST(TestSupportVerdict, UnresolvedWithIdPresentIsStillErroredBeforeAssert)
{
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {ENGINE_A}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::QUERY_ERRORED);
    EXPECT_TRUE(isFailure(r.verdict));
}

// ---------------------------------------------------------------------------
// unclaimed + hasSidecar + supported → UNCLAIMED_SUPPORT
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, UnclaimedButSupportedIsUnclaimedSupport)
{
    auto r = eval(ErrorCode::OK, {ENGINE_A}, ENGINE_A, false, true);
    EXPECT_EQ(r.verdict, SupportVerdict::UNCLAIMED_SUPPORT);
    EXPECT_FALSE(isFailure(r.verdict));
}

// ---------------------------------------------------------------------------
// unclaimed + hasSidecar + not supported → NOT_ENFORCED
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, UnclaimedAndDeclinedIsNotEnforced)
{
    auto r = eval(ErrorCode::OK, {ENGINE_B}, ENGINE_A, false, true);
    EXPECT_EQ(r.verdict, SupportVerdict::NOT_ENFORCED);
    EXPECT_FALSE(isFailure(r.verdict));
}

TEST(TestSupportVerdict, UnclaimedAndUnresolvedIsNotEnforced)
{
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {}, ENGINE_A, false, true);
    EXPECT_EQ(r.verdict, SupportVerdict::NOT_ENFORCED);
    EXPECT_FALSE(isFailure(r.verdict));

    // The two unclaimed-and-not-supported paths share a verdict but not a story:
    // one read the ranked list and did not find the engine, the other never got a
    // trustworthy list to read. The detail must not claim the former when it was
    // the latter — that would be an unverified fact in a diagnostic.
    EXPECT_EQ(r.detail.find("not in ranked list"), std::string::npos);
    EXPECT_NE(r.detail.find("did not resolve"), std::string::npos);
}

// ---------------------------------------------------------------------------
// queryStatus + queryMessage survive on the unclaimed-unresolved path
//
// This is the path that goes lossy today: the verdict is NOT_ENFORCED, shared
// with healthy "engine doesn't do this graph", and the error code survives only
// as prose in `detail`. On a bootstrap run every cell is unclaimed, so every
// error lands here. The fields let a consumer separate the two without parsing.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, UnclaimedUnresolvedCarriesQueryStatusAndMessage)
{
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {}, ENGINE_A, false, true, QUERY_MSG);
    EXPECT_EQ(r.verdict, SupportVerdict::NOT_ENFORCED);
    EXPECT_EQ(r.queryStatus, ErrorCode::HEURISTIC_QUERY_FAILED);
    EXPECT_EQ(r.queryMessage, QUERY_MSG);
}

TEST(TestSupportVerdict, QueryErroredCarriesQueryStatusAndMessage)
{
    auto r = eval(ErrorCode::HIPDNN_BACKEND_ERROR, {}, ENGINE_A, true, true, QUERY_MSG);
    EXPECT_EQ(r.verdict, SupportVerdict::QUERY_ERRORED);
    EXPECT_EQ(r.queryStatus, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(r.queryMessage, QUERY_MSG);
}

// Resolved queries have nothing to explain — the message is not stored.
TEST(TestSupportVerdict, ResolvedQueryDoesNotStoreMessage)
{
    auto r = eval(ErrorCode::OK, {ENGINE_A}, ENGINE_A, true, true, "should be dropped");
    EXPECT_EQ(r.verdict, SupportVerdict::SATISFIED);
    EXPECT_EQ(r.queryStatus, ErrorCode::OK);
    EXPECT_TRUE(r.queryMessage.empty());
}

// queryStatus is set even on the NO_SIDECAR short-circuit, because the caller
// did make the query and the observation is true regardless of what the sidecar
// says. The message is still conditional on unresolved.
TEST(TestSupportVerdict, NoSidecarStillCarriesQueryStatus)
{
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {}, ENGINE_A, false, false, QUERY_MSG);
    EXPECT_EQ(r.verdict, SupportVerdict::NO_SIDECAR);
    EXPECT_EQ(r.queryStatus, ErrorCode::HEURISTIC_QUERY_FAILED);
    EXPECT_EQ(r.queryMessage, QUERY_MSG);
}

// ---------------------------------------------------------------------------
// formatVerdictMessage includes queryMessage when present
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, FormatVerdictMessageIncludesQueryMessage)
{
    using hipdnn_integration_tests::bundle::formatVerdictMessage;
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {}, ENGINE_A, true, true, QUERY_MSG);
    const auto msg = formatVerdictMessage(r);
    EXPECT_NE(msg.find(QUERY_MSG), std::string::npos);
    EXPECT_NE(msg.find("query:"), std::string::npos);
}

TEST(TestSupportVerdict, FormatVerdictMessageOmitsQueryLineWhenEmpty)
{
    using hipdnn_integration_tests::bundle::formatVerdictMessage;
    auto r = eval(ErrorCode::OK, {ENGINE_A}, ENGINE_A, true, true);
    const auto msg = formatVerdictMessage(r);
    EXPECT_EQ(msg.find("query:"), std::string::npos);
}

// ---------------------------------------------------------------------------
// no sidecar at all → NO_SIDECAR regardless of query result
//
// Kept distinct from NOT_ENFORCED, which its caller also treats as "nothing was
// asserted": NOT_ENFORCED means a sidecar exists and is silent about this
// (engine, arch, platform), while NO_SIDECAR means there is no promise to check
// at all. Only the latter is dropped instead of recorded, so collapsing the two
// would flood the report with the graphs that carry no sidecar — nearly all of
// them — and break the report's derived queried-count.
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, NoSidecarSupportedIsNoSidecar)
{
    auto r = eval(ErrorCode::OK, {ENGINE_A}, ENGINE_A, false, false);
    EXPECT_EQ(r.verdict, SupportVerdict::NO_SIDECAR);
    EXPECT_FALSE(isFailure(r.verdict));
}

TEST(TestSupportVerdict, NoSidecarDeclinedIsNoSidecar)
{
    auto r = eval(ErrorCode::OK, {ENGINE_B}, ENGINE_A, false, false);
    EXPECT_EQ(r.verdict, SupportVerdict::NO_SIDECAR);
}

TEST(TestSupportVerdict, NoSidecarUnresolvedIsNoSidecar)
{
    auto r = eval(ErrorCode::HEURISTIC_QUERY_FAILED, {}, ENGINE_A, false, false);
    EXPECT_EQ(r.verdict, SupportVerdict::NO_SIDECAR);
}

// The absent sidecar short-circuits before the query is read at all. Pinned
// because it is what makes the verdict safe to compute on a run with no backend:
// an unresolved status cannot turn a sidecar-free graph into a failure.
TEST(TestSupportVerdict, NoSidecarShortCircuitsAheadOfEveryOtherSignal)
{
    auto r = eval(static_cast<ErrorCode>(9999), {ENGINE_A}, ENGINE_A, /*claimed=*/true, false);
    EXPECT_EQ(r.verdict, SupportVerdict::NO_SIDECAR);
    EXPECT_FALSE(isFailure(r.verdict));
}

// ---------------------------------------------------------------------------
// fail-closed: a future/unknown ErrorCode is unresolved
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, UnknownErrorCodeTreatedAsUnresolved)
{
    auto r = eval(static_cast<ErrorCode>(9999), {ENGINE_A}, ENGINE_A, true, true);
    EXPECT_EQ(r.verdict, SupportVerdict::QUERY_ERRORED);
}

// ---------------------------------------------------------------------------
// multi-engine: one query, two engines, independent verdicts
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, TwoEnginesSameQueryIndependentVerdicts)
{
    const std::vector<int64_t> rankedIds = {ENGINE_A}; // only A ranked

    auto rA = evaluateSupport(
        ErrorCode::OK, rankedIds, ENGINE_A, true, true, BUNDLE, "ENGINE_A", ARCH, PLAT);
    auto rB = evaluateSupport(
        ErrorCode::OK, rankedIds, ENGINE_B, true, true, BUNDLE, "ENGINE_B", ARCH, PLAT);

    EXPECT_EQ(rA.verdict, SupportVerdict::SATISFIED);
    EXPECT_EQ(rB.verdict, SupportVerdict::CLAIM_BROKEN);
}

// ---------------------------------------------------------------------------
// metadata flows through to the result struct
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ResultCarriesMetadata)
{
    auto r = evaluateSupport(ErrorCode::OK,
                             {ENGINE_A},
                             ENGINE_A,
                             true,
                             true,
                             "path/to/bundle",
                             "MY_ENGINE",
                             "gfx90a",
                             "windows");

    EXPECT_EQ(r.bundlePath, "path/to/bundle");
    EXPECT_EQ(r.engineName, "MY_ENGINE");
    EXPECT_EQ(r.arch, "gfx90a");
    EXPECT_EQ(r.platform, "windows");
    EXPECT_FALSE(r.detail.empty());
}

// ---------------------------------------------------------------------------
// toString covers every enum value
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, ToStringCoversAllValues)
{
    using hipdnn_integration_tests::bundle::toString;
    EXPECT_STREQ(toString(SupportVerdict::NO_SIDECAR), "NO_SIDECAR");
    EXPECT_STREQ(toString(SupportVerdict::SATISFIED), "SATISFIED");
    EXPECT_STREQ(toString(SupportVerdict::CLAIM_BROKEN), "CLAIM_BROKEN");
    EXPECT_STREQ(toString(SupportVerdict::QUERY_ERRORED), "QUERY_ERRORED");
    EXPECT_STREQ(toString(SupportVerdict::ENGINE_NOT_LOADED), "ENGINE_NOT_LOADED");
    EXPECT_STREQ(toString(SupportVerdict::NOT_ENFORCED), "NOT_ENFORCED");
    EXPECT_STREQ(toString(SupportVerdict::UNCLAIMED_SUPPORT), "UNCLAIMED_SUPPORT");
}

// ---------------------------------------------------------------------------
// baseArchToken strips target features, keeps base arch
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, BaseArchTokenStripsFeatures)
{
    EXPECT_EQ(baseArchToken("gfx942:sramecc+:xnack-"), "gfx942");
}

TEST(TestSupportVerdict, BaseArchTokenIdempotentOnBare)
{
    EXPECT_EQ(baseArchToken("gfx942"), "gfx942");
}

TEST(TestSupportVerdict, BaseArchTokenEmptyInput)
{
    EXPECT_EQ(baseArchToken(""), "");
}

TEST(TestSupportVerdict, BaseArchTokenColonOnly)
{
    EXPECT_EQ(baseArchToken(":"), "");
}

// ---------------------------------------------------------------------------
// formatVerdictMessage produces non-empty, structured output
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, FormatVerdictMessageContainsAllFields)
{
    using hipdnn_integration_tests::bundle::formatVerdictMessage;
    auto r = evaluateSupport(ErrorCode::OK,
                             {ENGINE_A},
                             ENGINE_A,
                             true,
                             true,
                             "path/to/bundle",
                             "MY_ENGINE",
                             "gfx90a",
                             "linux");

    const auto msg = formatVerdictMessage(r);
    EXPECT_NE(msg.find("SATISFIED"), std::string::npos);
    EXPECT_NE(msg.find("path/to/bundle"), std::string::npos);
    EXPECT_NE(msg.find("MY_ENGINE"), std::string::npos);
    EXPECT_NE(msg.find("gfx90a"), std::string::npos);
    EXPECT_NE(msg.find("linux"), std::string::npos);
}

TEST(TestSupportVerdict, FormatVerdictMessageShowsClaimBroken)
{
    using hipdnn_integration_tests::bundle::formatVerdictMessage;
    auto r = eval(ErrorCode::OK, {ENGINE_B}, ENGINE_A, true, true);

    const auto msg = formatVerdictMessage(r);
    EXPECT_NE(msg.find("CLAIM_BROKEN"), std::string::npos);
}

// ---------------------------------------------------------------------------
// observeAllSupport: empty bundle path → empty results (no sidecar to load)
// ---------------------------------------------------------------------------

TEST(TestSupportVerdict, CheckAllEmptyWhenNoBundlePath)
{
    const std::vector<LoadedEngine> engines = {{ENGINE_A, "ENGINE_A"}, {ENGINE_B, "ENGINE_B"}};
    const auto results = observeAllSupport(ErrorCode::OK, {ENGINE_A}, {}, engines);
    EXPECT_TRUE(results.empty());
}

TEST(TestSupportVerdict, EngineNotLoadedIsNotFailure)
{
    EXPECT_FALSE(isFailure(SupportVerdict::ENGINE_NOT_LOADED));
}

TEST(TestSupportVerdict, ToStringCoversEngineNotLoaded)
{
    using hipdnn_integration_tests::bundle::toString;
    EXPECT_STREQ(toString(SupportVerdict::ENGINE_NOT_LOADED), "ENGINE_NOT_LOADED");
}

TEST(TestSupportVerdict, CheckAllEmptyWhenSidecarDoesNotExist)
{
    using hipdnn_integration_tests::bundle::SupportClaimLocator;
    const SupportClaimLocator locator{"/no/such/file.support.json", {}, "Bundle.json"};
    const std::vector<LoadedEngine> engines = {{ENGINE_A, "ENGINE_A"}};
    const auto results = observeAllSupport(ErrorCode::OK, {ENGINE_A}, locator, engines);
    EXPECT_TRUE(results.empty());
}

// NOLINTEND(readability-identifier-naming)
