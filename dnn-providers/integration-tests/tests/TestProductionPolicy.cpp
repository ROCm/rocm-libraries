// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// productionPolicy() is the single CLI/TestConfig -> HarnessPolicy translation in
// the program; every other unit test builds a HarnessPolicy by hand instead (see
// HarnessTestSupport.hpp's hostPolicy()), so nothing else in this binary would
// catch a transposed field here.
//
// TestConfig is a process-wide singleton initialized exactly once per binary, by
// whichever suite happens to run first (TestTestConfig.cpp's own comment covers
// why), so a test cannot pin specific values on it without racing that ordering.
// The first test below pins the wiring itself: every HarnessPolicy field must
// equal the TestConfig getter productionPolicy() is supposed to read, whatever
// that getter currently returns. The settings of an engine file are the one field
// the singleton cannot show here (this binary initializes it without a file), so
// two tests build a TestConfig of their own and pin the value it carries.

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <utility>

#include <hipdnn_test_sdk/utilities/ScratchDirectory.hpp>

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
    EXPECT_EQ(policy.validator, TestConfig::get().getValidatorDevice());
    // The one field fed by a helper rather than a getter, and the assertion is
    // wiring only: this fixture initializes the singleton with the flag false, so
    // both sides can only ever answer OFF. What it does catch is productionPolicy()
    // growing a second opinion about the mode instead of deferring to claimMode() --
    // the header main.cpp prints comes from that same call, and the two disagreeing
    // would label the run with a mode it did not use.
    EXPECT_EQ(policy.claims, claimMode());
    EXPECT_EQ(policy.arch, TestConfig::get().getCurrentArch());
    EXPECT_EQ(policy.platform, currentPlatform());
    EXPECT_EQ(policy.deviceVramMb, TestConfig::get().getCurrentDeviceVramMb());
    EXPECT_EQ(policy.testSettings, TestConfig::get().testSettings());
}

TEST_F(TestProductionPolicy, PlacementComesFromTheArgumentNotFromConfig)
{
    EXPECT_EQ(productionPolicy(TensorPlacement::HOST).placement, TensorPlacement::HOST);
    EXPECT_FALSE(productionPolicy(TensorPlacement::HOST).useDevice());

    EXPECT_EQ(productionPolicy(TensorPlacement::DEVICE).placement, TensorPlacement::DEVICE);
    EXPECT_TRUE(productionPolicy(TensorPlacement::DEVICE).useDevice());
}

// The settings file of an engine reaches the harness only through this one field: the coefficient
// of the float dynamic tolerance and every [[tolerance_overrides]] entry are read from it. A
// TestConfig of its own, built from a file, shows that production carries what the config
// loaded, which the process-wide one (initialized without a file in this binary) cannot.
TEST_F(TestProductionPolicy, SettingsLoadedFromAFileReachThePolicy)
{
    const auto dir = hipdnn_test_sdk::utilities::claimScratchDirectory("production_policy");
    const auto path = dir.path() / "engine.toml";
    std::ofstream(path) << R"([meta]
version = 1

[conv_fwd_tolerance.float]
summation_order = 7.5

[[tolerance_overrides]]
filters = ["*SomeCase*"]
atol = 0.5
rtol = 0.25
)";
    TestConfigOptions options;
    options.configPath = path;
    const TestConfig config(std::move(options));

    const HarnessPolicy policy = productionPolicy(TensorPlacement::HOST, config);

    ASSERT_NE(policy.testSettings, nullptr);
    EXPECT_EQ(policy.testSettings, config.testSettings());
    ASSERT_TRUE(policy.testSettings->convFwdSummationOrder("float").has_value());
    EXPECT_DOUBLE_EQ(*policy.testSettings->convFwdSummationOrder("float"), 7.5);
    const auto entry = policy.testSettings->findToleranceOverride("Suite.SomeCase");
    ASSERT_TRUE(entry.has_value());
    EXPECT_FLOAT_EQ(entry->atol, 0.5f);
    EXPECT_FLOAT_EQ(entry->rtol, 0.25f);
}

TEST_F(TestProductionPolicy, ConfigWithoutAFileGivesAPolicyWithoutSettings)
{
    const TestConfig config{TestConfigOptions{}};
    EXPECT_EQ(productionPolicy(TensorPlacement::HOST, config).testSettings, nullptr);
}

// NOLINTEND(readability-identifier-naming)
