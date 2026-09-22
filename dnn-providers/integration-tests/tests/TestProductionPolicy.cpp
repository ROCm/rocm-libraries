// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// productionPolicy() is the single CLI/TestConfig -> HarnessPolicy translation in
// the program; every other unit test builds a HarnessPolicy by hand instead (see
// HarnessTestSupport.hpp's hostPolicy()), so nothing else in this binary would
// catch a transposed field here.
//
// TestConfig is a process-wide singleton initialized exactly once per binary, by
// whichever suite happens to run first (TestTestConfig.cpp's own comment covers
// why), so this file cannot pin specific values without racing that ordering.
// Instead it pins the wiring itself: every HarnessPolicy field must equal the
// TestConfig getter productionPolicy() is supposed to read, whatever that getter
// currently returns.

#include <gtest/gtest.h>

#include "HarnessTestSupport.hpp"
#include "common/PlatformUtils.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/ProductionPolicy.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestProductionPolicy : public ::testing::Test
{
protected:
    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
    }
};

} // namespace

TEST_F(TestProductionPolicy, EveryFieldMirrorsItsOwnConfigGetter)
{
    const HarnessPolicy policy = productionPolicy(TensorPlacement::DEVICE);

    EXPECT_EQ(policy.mode, TestConfig::get().getVerificationMode());
    // The one field fed by a helper rather than a getter. Wiring only -- the
    // precedence inside that helper is pinned by EnforcementSubsumesReporting below,
    // which this fixture cannot reach: the singleton is initialized with both flags
    // false, so claimMode() here can only ever answer OFF.
    EXPECT_EQ(policy.claims, claimMode());
    EXPECT_EQ(policy.arch, TestConfig::get().getCurrentArch());
    EXPECT_EQ(policy.platform, currentPlatform());
    EXPECT_EQ(policy.deviceVramMb, TestConfig::get().getCurrentDeviceVramMb());
}

// The flag-pair overload, which reads nothing, so every combination is reachable.
// main.cpp calls claimMode() for the summary header and the harness calls it through
// productionPolicy(); both are told not to re-derive this rule, which leaves it with
// exactly one definition and, without this test, no coverage at all.
TEST(TestClaimMode, EnforcementSubsumesReporting)
{
    EXPECT_EQ(claimMode(false, false), ClaimMode::OFF);
    EXPECT_EQ(claimMode(false, true), ClaimMode::REPORT);
    EXPECT_EQ(claimMode(true, true), ClaimMode::ENFORCE);

    // Unreachable from the CLI: TestConfig::reportSupportClaims() ORs the enforce
    // flag in, so enforcement always arrives with reporting already set. Pinned
    // anyway -- the rule is "enforce wins", not "enforce wins when report agrees".
    EXPECT_EQ(claimMode(true, false), ClaimMode::ENFORCE);
}

TEST_F(TestProductionPolicy, PlacementComesFromTheArgumentNotFromConfig)
{
    EXPECT_EQ(productionPolicy(TensorPlacement::HOST).placement, TensorPlacement::HOST);
    EXPECT_FALSE(productionPolicy(TensorPlacement::HOST).useDevice());

    EXPECT_EQ(productionPolicy(TensorPlacement::DEVICE).placement, TensorPlacement::DEVICE);
    EXPECT_TRUE(productionPolicy(TensorPlacement::DEVICE).useDevice());
}

// NOLINTEND(readability-identifier-naming)
