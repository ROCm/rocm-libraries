// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "SupportClaimTestUtils.hpp"
#include "harness/bundle/GraphSession.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/LoadedEngine.hpp"
#include "harness/bundle/SupportClaims.hpp"
#include "harness/bundle/SupportObservationLog.hpp"

using hipdnn_frontend::ErrorCode;
using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using namespace hipdnn_integration_tests::bundle::testing_support;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

constexpr int64_t ENGINE_A_ID = 1;
constexpr int64_t ENGINE_B_ID = 2;

const LoadedEngine ENGINE_A{ENGINE_A_ID, "ENGINE_A"};
const LoadedEngine ENGINE_B{ENGINE_B_ID, "ENGINE_B"};

GraphSession resolvedSession(const std::vector<int64_t>& rankedIds)
{
    GraphSession session;
    session.engines.status = hipdnn_frontend::Error{ErrorCode::OK, ""};
    session.engines.rankedIds = rankedIds;
    return session;
}

GraphSession unresolvedSession()
{
    GraphSession session;
    session.engines.status
        = hipdnn_frontend::Error{ErrorCode::HIPDNN_BACKEND_ERROR, "backend timed out"};
    return session;
}

class TestObserveSupportOnly : public ::testing::Test
{
protected:
    testing_support::HarnessMocks _mocks;
};

TEST_F(TestObserveSupportOnly, BuildErrorReturnsEmpty)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = testing_support::buildErrorSession("from_binary failed");
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    EXPECT_TRUE(observations.empty());
}

TEST_F(TestObserveSupportOnly, UnresolvedQueryReturnsEmpty)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = unresolvedSession();
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    EXPECT_TRUE(observations.empty());
}

TEST_F(TestObserveSupportOnly, ResolvedQueryRecordsAllEngines)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = resolvedSession({ENGINE_A_ID});
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    ASSERT_EQ(observations.size(), 2u);
    EXPECT_EQ(observations[0].engineName, "ENGINE_A");
    EXPECT_TRUE(observations[0].engineIsSupported);
    EXPECT_EQ(observations[1].engineName, "ENGINE_B");
    EXPECT_FALSE(observations[1].engineIsSupported);
}

TEST_F(TestObserveSupportOnly, ModeCNarrowsToSelectedEngine)
{
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(testing_support::hostPolicy()),
                                                 ENGINE_A);
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = resolvedSession({ENGINE_A_ID});
    auto observations = harness.observeSupportOnly(session, {ENGINE_A, ENGINE_B});

    ASSERT_EQ(observations.size(), 1u);
    EXPECT_EQ(observations[0].engineName, "ENGINE_A");
    EXPECT_TRUE(observations[0].engineIsSupported);
}

TEST_F(TestObserveSupportOnly, CarriesArchAndPlatformFromPolicy)
{
    IntegrationBundleVerificationHarness harness(
        _mocks.dependencies(testing_support::hostPolicy()));
    harness.setBundle(nullptr, "fake/bundle.json", singleGraphClaimLocator("fake/bundle.json"));

    auto session = resolvedSession({});
    auto observations = harness.observeSupportOnly(session, {ENGINE_A});

    ASSERT_EQ(observations.size(), 1u);
    EXPECT_EQ(observations[0].arch, "gfx942");
    EXPECT_EQ(observations[0].platform, "linux");
    EXPECT_FALSE(observations[0].engineIsSupported);
}

// The log is a process-wide singleton, so each test starts from a known state
// rather than inheriting whatever the previous one filed.
class TestSupportObservationLogAccounting : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SupportObservationLog::get().reset();
    }
    void TearDown() override
    {
        SupportObservationLog::get().reset();
    }
};

// One graph, two engines, two cells: the reason the summary cannot use the
// observation count as its own denominator without overstating coverage.
TEST_F(TestSupportObservationLogAccounting, ObservedAndUnobservedGraphsAreCountedSeparately)
{
    auto& log = SupportObservationLog::get();
    log.recordGraph(
        {test_utils::singleGraphObservation("dir/good.json", "ENGINE_A", "gfx942", "linux", true),
         test_utils::singleGraphObservation(
             "dir/good.json", "ENGINE_B", "gfx942", "linux", false)});
    log.recordGraph({});
    log.recordGraph({});

    EXPECT_EQ(log.graphsObserved(), 1u);
    EXPECT_EQ(log.graphsUnobserved(), 2u);
    EXPECT_EQ(log.all().size(), 2u);
}

// A graph skipped in SetUp never reaches recordGraph(), so it must not land in
// either observation bucket -- authorSupportClaims() subtracts all three from the
// registered count and would double-count it.
TEST_F(TestSupportObservationLogAccounting, SkipsBeforeObservationAreCountedApartFromObservations)
{
    auto& log = SupportObservationLog::get();
    log.recordGraph(
        {test_utils::singleGraphObservation("dir/good.json", "ENGINE_A", "gfx942", "linux", true)});
    log.recordSkipBeforeObservation();
    log.recordSkipBeforeObservation();

    EXPECT_EQ(log.graphsSkippedBeforeObservation(), 2u);
    EXPECT_EQ(log.graphsObserved(), 1u);
    EXPECT_EQ(log.graphsUnobserved(), 0u);
    EXPECT_EQ(log.all().size(), 1u);
}

TEST_F(TestSupportObservationLogAccounting, ResetClearsTheSkipCount)
{
    auto& log = SupportObservationLog::get();
    log.recordSkipBeforeObservation();
    ASSERT_EQ(log.graphsSkippedBeforeObservation(), 1u);

    log.reset();

    EXPECT_EQ(log.graphsSkippedBeforeObservation(), 0u);
}

// Every SetUp() path that returns before observeSupportOnly() runs has to tell the
// log, or authorSupportClaims() sees the shortfall and calls a correct run a bug.
//
// Three gaps this fixture does not close, none of them fixable here:
//
//   - The [[test_skips]] path is untestable. checkTomlSkip() looks up the *unit
//     test's own* name in a TestConfig that ensureTestConfigInitialized() built from
//     defaults, and TestConfig::initialize() throws on a second call, so no test can
//     supply a skip list. The site is one line identical to the three covered below,
//     which is why all five route through the same helper.
//   - The no-device path is untestable. It only runs under a DEVICE policy, where the
//     outcome depends on whether the build machine has a GPU.
//   - Nothing here can key on ::testing::Test::IsSkipped(). driveHarness() installs a
//     ScopedFakeTestPartResultReporter(INTERCEPT_ALL_THREADS, ...), which diverts
//     dispositions away from the real TestResult, so IsSkipped() stays false. That is
//     why the counter is bumped by an explicit call at each site rather than by an
//     RAII object in SetUp() or a TearDown() hook -- either would be invisible here.
class TestSkipsBeforeObservation : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        ensureTestConfigInitialized();
        _scopedDir.emplace(
            hipdnn_test_sdk::utilities::claimScratchDirectory("skips_before_observation_"));
        _tempDir = _scopedDir->path();
        SupportObservationLog::get().reset();
    }

    void TearDown() override
    {
        SupportObservationLog::get().reset();
    }

    std::shared_ptr<IntegrationTestBundle> bundle() const
    {
        return fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/true);
    }
};

TEST_F(TestSkipsBeforeObservation, ArchGuardSkipIsCounted)
{
    HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()), ENGINE_A);

    auto guarded = bundle();
    guarded->metadata.gpuArchitecture = "gfx1100"; // hostPolicy() runs gfx942
    harness.setBundle(guarded, _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    ASSERT_TRUE(anySkipped(results));
    EXPECT_EQ(SupportObservationLog::get().graphsSkippedBeforeObservation(), 1u);
    EXPECT_EQ(SupportObservationLog::get().graphsObserved(), 0u);
    EXPECT_EQ(SupportObservationLog::get().graphsUnobserved(), 0u);
}

TEST_F(TestSkipsBeforeObservation, VramGuardSkipIsCounted)
{
    // checkVramRequirement() is inert while deviceVramMb is hostPolicy()'s 0 -- an
    // unknown device size cannot be too small -- so the guard has to be given a real
    // number before it will fire.
    auto policy = hostPolicy();
    policy.deviceVramMb = 1024;

    HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(policy), ENGINE_A);

    auto guarded = bundle();
    guarded->metadata.minimumVramMb = 16000;
    harness.setBundle(guarded, _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    ASSERT_TRUE(anySkipped(results));
    EXPECT_EQ(SupportObservationLog::get().graphsSkippedBeforeObservation(), 1u);
    EXPECT_EQ(SupportObservationLog::get().graphsObserved(), 0u);
    EXPECT_EQ(SupportObservationLog::get().graphsUnobserved(), 0u);
}

// Harness wiring, not policy: the counter records that a reason was filed, not that
// the reason was a good one. Left uncounted it would read as a vanished bundle.
TEST_F(TestSkipsBeforeObservation, NullBundleSkipIsCounted)
{
    HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()), ENGINE_A);
    harness.setBundle(nullptr, _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    ASSERT_TRUE(anySkipped(results));
    EXPECT_EQ(SupportObservationLog::get().graphsSkippedBeforeObservation(), 1u);
}

// The other direction: a SetUp() that runs to the end files nothing, so an ordinary
// run's residue is not quietly padded with skips that never happened.
TEST_F(TestSkipsBeforeObservation, SetUpThatCompletesCountsNothing)
{
    HarnessMocks mocks;
    IntegrationBundleVerificationHarness harness(mocks.dependencies(hostPolicy()), ENGINE_A);
    harness.setBundle(bundle(), _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveHarness(harness, &results);

    EXPECT_FALSE(anySkipped(results));
    EXPECT_EQ(SupportObservationLog::get().graphsSkippedBeforeObservation(), 0u);
}

} // namespace

// NOLINTEND(readability-identifier-naming)
