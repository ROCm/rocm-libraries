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
    EXPECT_EQ(policy.enforceSupportClaims, TestConfig::get().enforceSupportClaims());
    EXPECT_EQ(policy.arch, TestConfig::get().getCurrentArch());
    EXPECT_EQ(policy.platform, currentPlatform());
    EXPECT_EQ(policy.deviceVramMb, TestConfig::get().getCurrentDeviceVramMb());
}

TEST_F(TestProductionPolicy, PlacementComesFromTheArgumentNotFromConfig)
{
    EXPECT_EQ(productionPolicy(TensorPlacement::HOST).placement, TensorPlacement::HOST);
    EXPECT_FALSE(productionPolicy(TensorPlacement::HOST).useDevice());

    EXPECT_EQ(productionPolicy(TensorPlacement::DEVICE).placement, TensorPlacement::DEVICE);
    EXPECT_TRUE(productionPolicy(TensorPlacement::DEVICE).useDevice());
}

// NOLINTEND(readability-identifier-naming)
