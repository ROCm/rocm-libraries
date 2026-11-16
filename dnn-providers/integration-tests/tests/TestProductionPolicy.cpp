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
// currently returns. The one exception re-executes the binary in a death test to
// get a fresh singleton; see BenchmarkReferenceReachesConfigAndPolicy.

#include <cstdlib>

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
    EXPECT_EQ(policy.benchmarkReference, TestConfig::get().benchmarkReference());
}

TEST_F(TestProductionPolicy, PlacementComesFromTheArgumentNotFromConfig)
{
    EXPECT_EQ(productionPolicy(TensorPlacement::HOST).placement, TensorPlacement::HOST);
    EXPECT_FALSE(productionPolicy(TensorPlacement::HOST).useDevice());

    EXPECT_EQ(productionPolicy(TensorPlacement::DEVICE).placement, TensorPlacement::DEVICE);
    EXPECT_TRUE(productionPolicy(TensorPlacement::DEVICE).useDevice());
}

// The mirror check above can only see the flag off, so it cannot tell a wired
// --benchmark-reference from one that is dropped or inverted on the way. A
// threadsafe-style death test re-executes this binary for the statement alone, which
// gets a TestConfig nobody has initialized yet, so it can turn the flag on. Named
// "...Death" rather than "...DeathTest": the test-name validator reserves "Test" for
// the leading keyword.
namespace
{

[[noreturn]] void exitWithBenchmarkReferenceWiring()
{
    TestConfigOptions options;
    options.benchmarkReference = true;
    TestConfig::initialize(options);
    const bool wired = TestConfig::get().benchmarkReference()
                       && productionPolicy(TensorPlacement::HOST).benchmarkReference;
    std::exit(wired ? 0 : 1);
}

} // namespace

TEST(TestProductionPolicyDeath, BenchmarkReferenceReachesConfigAndPolicy)
{
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    // EXPECT_EXIT expands to a switch with no default label, attributed to this site.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-default"
    EXPECT_EXIT(exitWithBenchmarkReferenceWiring(), ::testing::ExitedWithCode(0), "");
#pragma clang diagnostic pop
}

// NOLINTEND(readability-identifier-naming)
