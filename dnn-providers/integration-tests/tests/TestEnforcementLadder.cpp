// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for RFC 0015's enforcement ladder as wired into
// IntegrationBundleVerificationHarness::TestBody(): which rung runs for a
// given `enforcement_level`, and how a support-claim verdict turns into a
// GTest outcome (FAIL vs pass) at each rung. Drives the harness's full
// SetUp()/TestBody() lifecycle under a ScopedFakeTestPartResultReporter, the
// same technique TestBundleVerificationHarness.cpp uses for the comparison
// pipeline, so these tests run without a real handle, plugin, or GPU.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>

#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/bundle/SupportClaims.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::bundle;
using hipdnn_integration_tests::BundleMetadata;
using hipdnn_integration_tests::EnforcementLevel;

namespace
{

class TestableEnforcementHarness : public IntegrationBundleVerificationHarness
{
public:
    TestableEnforcementHarness(SupportQueryResult queryResult,
                               std::vector<std::string> loadedEngines)
        : IntegrationBundleVerificationHarness(/*requiresDevice=*/false)
        , _queryResult(std::move(queryResult))
        , _loadedEngines(std::move(loadedEngines))
    {
    }

    void SetUp() override {}

    using IntegrationBundleVerificationHarness::TestBody;

    static SupportQueryResult makeQuery(hipdnn_frontend::Error status,
                                        std::vector<int64_t> rankedIds)
    {
        SupportQueryResult result;
        result.status = std::move(status);
        result.rankedEngineIds = std::move(rankedIds);
        return result;
    }

    bool planBuildCalled = false;
    bool executeCalled = false;
    bool planBuildShouldFail = false;

protected:
    SupportQueryResult queryGraphSupport() override
    {
        return _queryResult;
    }

    std::vector<std::string> listLoadedEngines() const override
    {
        return _loadedEngines;
    }

    std::string currentArchToken() const override
    {
        return "gfx942";
    }

    std::string currentPlatform() const override
    {
        return "linux";
    }

    void runPlanBuildOnly() override
    {
        planBuildCalled = true;
        if(planBuildShouldFail)
        {
            ADD_FAILURE() << "stub plan-build failure";
        }
    }

    void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& /*variantPack*/) override
    {
        executeCalled = true;
    }

    void applyMetadataGuards() const override {}

private:
    SupportQueryResult _queryResult;
    std::vector<std::string> _loadedEngines;
};

std::shared_ptr<IntegrationTestBundle> makeLadderBundle(EnforcementLevel level,
                                                        std::optional<SupportClaims> claims)
{
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->metadata.enforcementLevel = level;
    bundle->metadata.enforcementLevelExplicit = true;
    bundle->supportClaims = std::move(claims);
    return bundle;
}

SupportClaims
    claimEngine(const std::string& engine, const std::string& arch, const std::string& platform)
{
    SupportClaims claims;
    claims.version = 1;
    claims.claims[engine][arch].insert(platform);
    return claims;
}

// Drives SetUp()+TestBody() under a ScopedFakeTestPartResultReporter,
// capturing every TestPartResult instead of letting them affect *this* test.
void runCapturing(TestableEnforcementHarness& harness, ::testing::TestPartResultArray* results)
{
    const ::testing::ScopedFakeTestPartResultReporter reporter(
        ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
    harness.SetUp();
    harness.TestBody();
}

bool anyFailed(const ::testing::TestPartResultArray& results)
{
    for(int i = 0; i < results.size(); ++i)
    {
        if(results.GetTestPartResult(i).failed())
        {
            return true;
        }
    }
    return false;
}

std::string firstMessage(const ::testing::TestPartResultArray& results)
{
    return results.size() > 0 ? results.GetTestPartResult(0).message() : std::string{};
}

} // namespace

// ---------------------------------------------------------------------------
// applicability rung
// ---------------------------------------------------------------------------

TEST(TestEnforcementLadderApplicability, PassesWithNoClaims)
{
    auto bundle = makeLadderBundle(EnforcementLevel::Applicability, std::nullopt);
    auto query = TestableEnforcementHarness::makeQuery(hipdnn_frontend::Error(), {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(harness.planBuildCalled);
    EXPECT_FALSE(harness.executeCalled);
}

TEST(TestEnforcementLadderApplicability, PassesWhenClaimedEngineIsSupported)
{
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    auto bundle = makeLadderBundle(EnforcementLevel::Applicability,
                                   claimEngine("MIOPEN_ENGINE", "gfx942", "linux"));
    auto query = TestableEnforcementHarness::makeQuery(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::OK, ""), {engineId});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    EXPECT_FALSE(anyFailed(results));
}

TEST(TestEnforcementLadderApplicability, FailsWhenClaimedEngineIsDeclined)
{
    auto bundle = makeLadderBundle(EnforcementLevel::Applicability,
                                   claimEngine("MIOPEN_ENGINE", "gfx942", "linux"));
    auto query = TestableEnforcementHarness::makeQuery(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED, ""), {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    ASSERT_TRUE(anyFailed(results));
    EXPECT_NE(firstMessage(results).find("applicability"), std::string::npos);
    EXPECT_FALSE(harness.planBuildCalled);
    EXPECT_FALSE(harness.executeCalled);
}

TEST(TestEnforcementLadderApplicability, FailsAsErroredBeforeAssertWhenQueryUnresolved)
{
    auto bundle = makeLadderBundle(EnforcementLevel::Applicability,
                                   claimEngine("MIOPEN_ENGINE", "gfx942", "linux"));
    auto query = TestableEnforcementHarness::makeQuery(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::HIPDNN_BACKEND_ERROR, "device lost"),
        {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    ASSERT_TRUE(anyFailed(results));
    EXPECT_NE(firstMessage(results).find("errored"), std::string::npos);
}

TEST(TestEnforcementLadderApplicability, UnclaimedEngineDeclinedDoesNotFail)
{
    // No support.json at all -- absence of a claim means "not enforced".
    auto bundle = makeLadderBundle(EnforcementLevel::Applicability, std::nullopt);
    auto query = TestableEnforcementHarness::makeQuery(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED, ""), {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    EXPECT_FALSE(anyFailed(results));
}

// ---------------------------------------------------------------------------
// buildable rung
// ---------------------------------------------------------------------------

TEST(TestEnforcementLadderBuildable, BuildsPlansWhenClaimHolds)
{
    const auto engineId = hipdnn_data_sdk::utilities::engineNameToId("MIOPEN_ENGINE");
    auto bundle = makeLadderBundle(EnforcementLevel::Buildable,
                                   claimEngine("MIOPEN_ENGINE", "gfx942", "linux"));
    auto query = TestableEnforcementHarness::makeQuery(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::OK, ""), {engineId});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_TRUE(harness.planBuildCalled);
    EXPECT_FALSE(harness.executeCalled);
}

TEST(TestEnforcementLadderBuildable, ClaimBrokenStopsBeforePlanBuild)
{
    auto bundle = makeLadderBundle(EnforcementLevel::Buildable,
                                   claimEngine("MIOPEN_ENGINE", "gfx942", "linux"));
    auto query = TestableEnforcementHarness::makeQuery(
        hipdnn_frontend::Error(hipdnn_frontend::ErrorCode::GRAPH_NOT_SUPPORTED, ""), {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    ASSERT_TRUE(anyFailed(results));
    EXPECT_FALSE(harness.planBuildCalled);
    EXPECT_FALSE(harness.executeCalled);
}

TEST(TestEnforcementLadderBuildable, PlanBuildFailureFailsRegardlessOfClaims)
{
    // Buildable's plan-build step is an unconditional hard-fail, same as the
    // pre-existing full-rung behavior -- claimed or not.
    auto bundle = makeLadderBundle(EnforcementLevel::Buildable, std::nullopt);
    auto query = TestableEnforcementHarness::makeQuery(hipdnn_frontend::Error(), {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");
    harness.planBuildShouldFail = true;

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    ASSERT_TRUE(anyFailed(results));
    EXPECT_TRUE(harness.planBuildCalled);
    EXPECT_FALSE(harness.executeCalled);
}

// ---------------------------------------------------------------------------
// full rung (default) -- dispatch only; the pipeline itself is covered by
// TestBundleVerificationHarness.cpp.
// ---------------------------------------------------------------------------

TEST(TestEnforcementLadderFull, DefaultLevelDispatchesToComparisonNotLadder)
{
    // enforcement_level defaults to Full; TestBody() must route to the
    // pre-existing runComparison() pipeline, never runApplicabilityLevel()/
    // runBuildableLevel(). With no output tensors, runComparison() SKIPs
    // immediately via skipUnverifiable() -- a distinct outcome from any
    // ladder FAIL, proving the dispatch landed on the full-rung path.
    auto bundle = std::make_shared<IntegrationTestBundle>();
    ASSERT_EQ(bundle->metadata.enforcementLevel, EnforcementLevel::Full);

    auto query = TestableEnforcementHarness::makeQuery(hipdnn_frontend::Error(), {});
    TestableEnforcementHarness harness(std::move(query), {"MIOPEN_ENGINE"});
    harness.setBundle(bundle, "unit-test-bundle");

    ::testing::TestPartResultArray results;
    runCapturing(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    ASSERT_TRUE(results.size() > 0);
    EXPECT_TRUE(results.GetTestPartResult(0).skipped());
    EXPECT_FALSE(harness.planBuildCalled);
    EXPECT_FALSE(harness.executeCalled);
}

// NOLINTEND(readability-identifier-naming)
