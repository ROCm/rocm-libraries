// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Error.hpp>

#include "harness/bundle/SupportClaimEnforcement.hpp"
#include "harness/bundle/SupportClaims.hpp"

using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_integration_tests::bundle::archToken;
using hipdnn_integration_tests::bundle::classifyEngineVerdict;
using hipdnn_integration_tests::bundle::EngineVerdict;
using hipdnn_integration_tests::bundle::evaluateClaimedEngines;
using hipdnn_integration_tests::bundle::findUnclaimedSupportedEngines;
using hipdnn_integration_tests::bundle::isResolvedSupportQuery;
using hipdnn_integration_tests::bundle::SupportClaims;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

SupportClaims makeClaims(const std::string& engine,
                         const std::string& arch,
                         const std::vector<std::string>& platforms)
{
    SupportClaims claims;
    claims.version = 1;
    for(const auto& platform : platforms)
    {
        claims.claims[engine][arch].insert(platform);
    }
    return claims;
}

} // namespace

// ---------------------------------------------------------------------------
// archToken
// ---------------------------------------------------------------------------

TEST(TestArchToken, StripsSuffixAfterFirstColon)
{
    EXPECT_EQ(archToken("gfx942:sramecc+:xnack-"), "gfx942");
}

TEST(TestArchToken, PassesThroughArchWithNoColon)
{
    EXPECT_EQ(archToken("gfx942"), "gfx942");
}

// ---------------------------------------------------------------------------
// isResolvedSupportQuery / classifyEngineVerdict
// ---------------------------------------------------------------------------

TEST(TestIsResolvedSupportQuery, OkAndGraphNotSupportedAreResolved)
{
    EXPECT_TRUE(isResolvedSupportQuery(Error(ErrorCode::OK, "")));
    EXPECT_TRUE(isResolvedSupportQuery(Error(ErrorCode::GRAPH_NOT_SUPPORTED, "")));
}

TEST(TestIsResolvedSupportQuery, OtherErrorCodesAreUnresolved)
{
    EXPECT_FALSE(isResolvedSupportQuery(Error(ErrorCode::HIPDNN_BACKEND_ERROR, "boom")));
    EXPECT_FALSE(isResolvedSupportQuery(Error(ErrorCode::HANDLE_ERROR, "boom")));
}

TEST(TestClassifyEngineVerdict, ResolvedAndPresentIsSupported)
{
    const Error status(ErrorCode::OK, "");
    EXPECT_EQ(classifyEngineVerdict(status, {1, 2, 3}, 2), EngineVerdict::Supported);
}

TEST(TestClassifyEngineVerdict, ResolvedButAbsentIsDeclined)
{
    const Error status(ErrorCode::OK, "");
    EXPECT_EQ(classifyEngineVerdict(status, {1, 3}, 2), EngineVerdict::Declined);
}

TEST(TestClassifyEngineVerdict, GraphNotSupportedWithEmptyListIsDeclined)
{
    const Error status(ErrorCode::GRAPH_NOT_SUPPORTED, "");
    EXPECT_EQ(classifyEngineVerdict(status, {}, 2), EngineVerdict::Declined);
}

TEST(TestClassifyEngineVerdict, UnresolvedQueryIsUnknownRegardlessOfIdList)
{
    const Error status(ErrorCode::HIPDNN_BACKEND_ERROR, "device lost");
    EXPECT_EQ(classifyEngineVerdict(status, {2}, 2), EngineVerdict::Unknown);
}

// ---------------------------------------------------------------------------
// evaluateClaimedEngines
// ---------------------------------------------------------------------------

TEST(TestEvaluateClaimedEngines, SupportedClaimYieldsNoRow)
{
    const auto claims = makeClaims("MIOPEN_ENGINE", "gfx942", {"linux"});
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    const Error status(ErrorCode::OK, "");

    const auto rows
        = evaluateClaimedEngines({"MIOPEN_ENGINE"}, "gfx942", "linux", claims, status, {engineId});
    EXPECT_TRUE(rows.empty());
}

TEST(TestEvaluateClaimedEngines, DeclinedClaimedEngineYieldsDeclinedRow)
{
    const auto claims = makeClaims("MIOPEN_ENGINE", "gfx942", {"linux"});
    const Error status(ErrorCode::GRAPH_NOT_SUPPORTED, "");

    const auto rows
        = evaluateClaimedEngines({"MIOPEN_ENGINE"}, "gfx942", "linux", claims, status, {});
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].engine, "MIOPEN_ENGINE");
    EXPECT_EQ(rows[0].verdict, EngineVerdict::Declined);
}

TEST(TestEvaluateClaimedEngines, UnresolvedQueryOnClaimedEngineYieldsUnknownRow)
{
    const auto claims = makeClaims("MIOPEN_ENGINE", "gfx942", {"linux"});
    const Error status(ErrorCode::HIPDNN_BACKEND_ERROR, "backend error");

    const auto rows
        = evaluateClaimedEngines({"MIOPEN_ENGINE"}, "gfx942", "linux", claims, status, {});
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].verdict, EngineVerdict::Unknown);
}

TEST(TestEvaluateClaimedEngines, UnclaimedEngineIsNeverEvaluated)
{
    // Engine is loaded and declined, but the bundle claims nothing for it:
    // absence of a claim means "not enforced", never a verdict.
    const SupportClaims noClaims;
    const Error status(ErrorCode::GRAPH_NOT_SUPPORTED, "");

    const auto rows
        = evaluateClaimedEngines({"MIOPEN_ENGINE"}, "gfx942", "linux", noClaims, status, {});
    EXPECT_TRUE(rows.empty());
}

TEST(TestEvaluateClaimedEngines, EngineNotLoadedIsNeverEvaluatedEvenIfClaimed)
{
    // A claim for an engine not loaded in the current run is not enforced
    // (RFC 0015 §7.3/§7.4) -- the engine simply never appears in
    // loadedEngineNames.
    const auto claims = makeClaims("MIOPEN_ENGINE", "gfx942", {"linux"});
    const Error status(ErrorCode::GRAPH_NOT_SUPPORTED, "");

    const auto rows
        = evaluateClaimedEngines({/* no engines loaded */}, "gfx942", "linux", claims, status, {});
    EXPECT_TRUE(rows.empty());
}

TEST(TestEvaluateClaimedEngines, NewArchWithNoClaimIsNotEnforced)
{
    const auto claims = makeClaims("MIOPEN_ENGINE", "gfx942", {"linux"});
    const Error status(ErrorCode::GRAPH_NOT_SUPPORTED, "");

    // Current arch (gfx1100) has no claim entry at all -- bring-up case, not
    // a refuse-to-run.
    const auto rows
        = evaluateClaimedEngines({"MIOPEN_ENGINE"}, "gfx1100", "linux", claims, status, {});
    EXPECT_TRUE(rows.empty());
}

TEST(TestEvaluateClaimedEngines, TwoLoadedEnginesOneDeclinedOneSupportedAreAttributedIndependently)
{
    // §7.4: one engine declined while another is supported yields a FAIL row
    // for the first and nothing for the second, from a single query.
    SupportClaims claims;
    claims.claims["MIOPEN_ENGINE"]["gfx942"].insert("linux");
    claims.claims["HIPBLASLT_ENGINE"]["gfx942"].insert("linux");

    const auto supportedId = hipdnn_data_sdk::utilities::engineNameToId("HIPBLASLT_ENGINE");
    const Error status(ErrorCode::OK, "");

    const auto rows = evaluateClaimedEngines(
        {"MIOPEN_ENGINE", "HIPBLASLT_ENGINE"}, "gfx942", "linux", claims, status, {supportedId});
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows[0].engine, "MIOPEN_ENGINE");
    EXPECT_EQ(rows[0].verdict, EngineVerdict::Declined);
}

// ---------------------------------------------------------------------------
// findUnclaimedSupportedEngines
// ---------------------------------------------------------------------------

TEST(TestFindUnclaimedSupportedEngines, SupportedButUnclaimedEngineIsReported)
{
    const SupportClaims noClaims;
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    const Error status(ErrorCode::OK, "");

    const auto unclaimed = findUnclaimedSupportedEngines(
        {"MIOPEN_ENGINE"}, "gfx942", "linux", noClaims, status, {engineId});
    ASSERT_EQ(unclaimed.size(), 1u);
    EXPECT_EQ(unclaimed[0], "MIOPEN_ENGINE");
}

TEST(TestFindUnclaimedSupportedEngines, ClaimedEngineIsNotReportedAsUnclaimed)
{
    const auto claims = makeClaims("MIOPEN_ENGINE", "gfx942", {"linux"});
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    const Error status(ErrorCode::OK, "");

    const auto unclaimed = findUnclaimedSupportedEngines(
        {"MIOPEN_ENGINE"}, "gfx942", "linux", claims, status, {engineId});
    EXPECT_TRUE(unclaimed.empty());
}

TEST(TestFindUnclaimedSupportedEngines, DeclinedEngineIsNotReportedAsUnclaimedSupport)
{
    const SupportClaims noClaims;
    const Error status(ErrorCode::GRAPH_NOT_SUPPORTED, "");

    const auto unclaimed
        = findUnclaimedSupportedEngines({"MIOPEN_ENGINE"}, "gfx942", "linux", noClaims, status, {});
    EXPECT_TRUE(unclaimed.empty());
}

TEST(TestFindUnclaimedSupportedEngines, UnresolvedQueryReportsNothing)
{
    // Support is unknown, not "supported" -- never surfaced as unclaimed
    // support either way.
    const SupportClaims noClaims;
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    const Error status(ErrorCode::HIPDNN_BACKEND_ERROR, "boom");

    const auto unclaimed = findUnclaimedSupportedEngines(
        {"MIOPEN_ENGINE"}, "gfx942", "linux", noClaims, status, {engineId});
    EXPECT_TRUE(unclaimed.empty());
}

// NOLINTEND(readability-identifier-naming)
