// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Covers IntegrationGraphVerificationHarness::reportFillFailure()'s FAIL-vs-SKIP
// split: an INVALID fill (the test's own recipe is malformed) fails the test; an
// UNSUPPORTED fill (the harness cannot fill this graph) skips it. Neither case
// reaches the real engine, so this never builds a graph.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include "HarnessTestSupport.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"
#include "harness/input-init/FillInputs.hpp"

using namespace hipdnn_integration_tests;

namespace
{

// Exists only to reach reportFillFailure() (protected in the base class) from a
// TEST_F body, which is itself a subclass of this fixture. runGraphTest() is the
// only pure-virtual member and is never invoked by these tests; it is overridden
// here only so the class is concrete. The constructor guarantees TestConfig is
// ready before gtest's own SetUp() call reaches checkTomlSkip(), regardless of
// what order other suites in this binary run in.
class TestGraphVerificationHarness : public IntegrationGraphVerificationHarness<float, int>
{
public:
    TestGraphVerificationHarness()
    {
        bundle::testing_support::ensureTestConfigInitialized();
    }

protected:
    void runGraphTest() override {}
};

} // namespace

TEST_F(TestGraphVerificationHarness, RejectsInvalidFill)
{
    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        reportFillFailure(FillResult::invalid("malformed fill recipe"));
    }

    EXPECT_TRUE(bundle::testing_support::anyFailed(results));
    EXPECT_FALSE(bundle::testing_support::anySkipped(results));
}

TEST_F(TestGraphVerificationHarness, SkipsUnsupportedFill)
{
    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        reportFillFailure(FillResult::unsupported("no input fill registered"));
    }

    EXPECT_FALSE(bundle::testing_support::anyFailed(results));
    EXPECT_TRUE(bundle::testing_support::anySkipped(results));
}
