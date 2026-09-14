// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Direct coverage for IntegrationBundleVerificationHarness::enforceAtLevel.
//
// Before the dependency-injection refactor this method had no seam of its own:
// BUILDABLE compiled plans against a real graph, so every suite that touched an
// enforcement rung stubbed enforceAtLevel() out wholesale rather than exercising it.
// Now IGraphEngineRunner is a mock collaborator, so these drive the real
// SetUp()/TestBody() with nothing above enforceAtLevel() stubbed, and additionally
// pin the FULL-mode engine-result routing (decline -> SKIP, failure -> FAILED with
// the frontend's own message) that replaced HasFatalFailure().

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/bundle/IGraphEngineRunner.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using namespace hipdnn_integration_tests::bundle::testing_support;
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

class TestEnforcementRungs : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        ensureTestConfigInitialized();
        _scopedDir.emplace(scratch::makeDir("enforcement_rungs_"));
        _tempDir = _scopedDir->path();
    }

    // One bundle on disk, tagged with the enforcement_level the test names. The
    // harness reads the level straight off metadata; nothing else about the bundle
    // matters to enforceAtLevel().
    std::shared_ptr<IntegrationTestBundle> bundleAt(EnforcementLevel level) const
    {
        auto bundle = fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/true);
        bundle->metadata.enforcementLevel = level;
        return bundle;
    }
};

} // namespace

// ---------------------------------------------------------------------------
// APPLICABILITY: accepted-into-the-ranked-list is the whole rung; no plan is ever
// compiled to answer it.
// ---------------------------------------------------------------------------

TEST_F(TestEnforcementRungs, ApplicabilityRungPassesWithoutCompilingPlans)
{
    using ::testing::_;

    HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, buildPlans(_, _)).Times(0);

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::APPLICABILITY), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestEnforcementRungs, ApplicabilityRungIsUnverifiableWhenTheEngineDeclines)
{
    HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, openGraph(::testing::_, ::testing::_))
        .WillByDefault([](const IntegrationTestBundle&, const std::optional<LoadedEngine>&) {
            return declinedSession();
        });

    std::vector<std::string> unverifiable;
    captureUnverifiable(mocks.reporter, unverifiable);

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::APPLICABILITY), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_TRUE(anySkipped(results));
    ASSERT_EQ(unverifiable.size(), 1u);
    EXPECT_NE(unverifiable.front().find("enforcement_level=applicability"), std::string::npos);
}

// ---------------------------------------------------------------------------
// BUILDABLE: additionally compiles a plan, exactly once, and never reaches execute.
// ---------------------------------------------------------------------------

TEST_F(TestEnforcementRungs, BuildableRungCompilesPlansExactlyOnce)
{
    using ::testing::_;

    HarnessMocks mocks;
    EXPECT_CALL(mocks.engineRunner, buildPlans(_, _)).Times(1);
    EXPECT_CALL(mocks.engineRunner, execute(_, _, _)).Times(0);

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::BUILDABLE), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestEnforcementRungs, BuildableRungFailureIsBlamedOnTheEngine)
{
    HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, buildPlans(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(EngineOpResult::failed("kaboom")));

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::BUILDABLE), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_TRUE(anyFailed(results));
    const std::string messages = allMessages(results);
    EXPECT_NE(messages.find("[rung=buildable]"), std::string::npos);
    EXPECT_NE(messages.find("kaboom"), std::string::npos);
}

// The case that previously escaped as an uncaught exception: a provider is allowed
// to decline later than the ranked list suggested, and that is the same answer as
// "not accepted", not a break the engine must answer for.
TEST_F(TestEnforcementRungs, BuildableRungDeclineIsUnverifiableNotAFailure)
{
    HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, buildPlans(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(EngineOpResult::declinedBy("late decline")));

    std::vector<std::string> unverifiable;
    captureUnverifiable(mocks.reporter, unverifiable);

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::BUILDABLE), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_TRUE(anySkipped(results));
    ASSERT_EQ(unverifiable.size(), 1u);
    EXPECT_NE(unverifiable.front().find("declined this graph while compiling plans"),
              std::string::npos);
}

// ---------------------------------------------------------------------------
// No named engine: the rung itself has nothing to check applicability against.
// ---------------------------------------------------------------------------

TEST_F(TestEnforcementRungs, EnforcementRungWithoutAnEngineIsUnverifiable)
{
    HarnessMocks mocks;

    std::vector<std::string> unverifiable;
    captureUnverifiable(mocks.reporter, unverifiable);

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()));
    harness.setBundle(bundleAt(EnforcementLevel::BUILDABLE), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_TRUE(anySkipped(results));
    ASSERT_EQ(unverifiable.size(), 1u);
    EXPECT_NE(unverifiable.front().find("--test-engine"), std::string::npos);
}

// ---------------------------------------------------------------------------
// FULL mode: engine-result routing during execute(), pinned directly. Before this
// refactor the harness read ::testing::Test::HasFatalFailure() after the fact to
// tell an engine break from a decline, which could not tell an engine assertion
// apart from any other fatal failure and produced an empty message either way.
// ---------------------------------------------------------------------------

TEST_F(TestEnforcementRungs, EngineDeclineDuringExecuteSkips)
{
    HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(EngineOpResult::declinedBy("nope")));

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::FULL), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_TRUE(anySkipped(results));
}

TEST_F(TestEnforcementRungs, EngineFailureDuringExecuteReportsTheEngineMessage)
{
    HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(EngineOpResult::failed("frontend said no")));

    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()),
                                                 makeEngineUnderTest());
    harness.setBundle(bundleAt(EnforcementLevel::FULL), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_TRUE(anyFailed(results));
    EXPECT_NE(allMessages(results).find("frontend said no"), std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
