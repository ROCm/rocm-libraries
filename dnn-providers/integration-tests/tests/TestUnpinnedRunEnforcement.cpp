// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The other half of shouldEnforceClaims(): TestSupportClaimEnforcement.cpp pins the
// engine-present half of the guard, but nothing else in this binary states what
// happens when no engine was named at all. A run with no --test-engine still has to
// finish cleanly -- it just has to enforce nothing while doing it, even when a
// support sidecar exists and enforcement is on.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <optional>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "ScratchDirectory.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/SupportClaimReport.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestUnpinnedRunEnforcement : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    testing_support::HarnessMocks _mocks;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(scratch::makeDir("unpinned_run_"));
        _tempDir = _scopedDir->path();
    }

    // A locator whose sidecar really exists -- everything shouldEnforceClaims()
    // needs except an engine under test.
    SupportClaimLocator makeLocator() const
    {
        const auto path = _tempDir / "Bundle.support.json";
        std::ofstream(path) << R"({"version": 1, "claims": {}})";

        SupportClaimLocator locator;
        locator.sidecarPath = path;
        locator.diagnosticPath = "test/bundle";
        return locator;
    }
};

} // namespace

TEST_F(TestUnpinnedRunEnforcement, RunWithNoNamedEngineNeverQueriesClaimsOrRecordsAVerdict)
{
    using ::testing::_;

    // Never asked: shouldEnforceClaims() has to fail before the sidecar is even
    // opened, or this is only accidentally unenforced rather than structurally so.
    EXPECT_CALL(_mocks.claimObserver, observe(_, _, _, _, _)).Times(0);

    std::vector<SupportResult> verdicts;
    testing_support::captureVerdicts(_mocks.reporter, verdicts);
    testing_support::engineWrites(
        _mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);

    // GOLDEN mode with a matching golden blob and a green engine: nothing here
    // should fail on its own account, so any failure recorded below has to be a
    // symptom of the engine-less run trying to enforce something it should not.
    IntegrationBundleVerificationHarness harness(_mocks.dependencies(
        testing_support::hostPolicy(VerificationMode::GOLDEN, /*enforceSupportClaims=*/true)));
    harness.setBundle(fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/true),
                      "test/bundle",
                      makeLocator());

    ::testing::TestPartResultArray results;
    testing_support::driveHarness(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_TRUE(verdicts.empty());
}

// NOLINTEND(readability-identifier-naming)
