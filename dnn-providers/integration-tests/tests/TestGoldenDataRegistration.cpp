// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The golden-data binary's cross-lane check (ReferenceLaneVerdict.hpp): a
// golden-bearing bundle must end the run with a test under its name in at least one
// reference lane, and one that does not becomes a failing <bundle>_Unvalidated
// test. Also the skipping test that stands in for a bundle no lane can afford.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "HarnessTestSupport.hpp"
#include "harness/bundle/BundleRegistration.hpp"
#include "harness/reference-validation/GoldenDataRegistration.hpp"
#include "harness/reference-validation/ReferenceLaneVerdict.hpp"

using namespace hipdnn_integration_tests::bundle;

// NOLINTBEGIN(readability-identifier-naming)

// The cross-lane check behind registerUnvalidatedGoldenDataFailures(): a golden-bearing
// bundle must end the run with a test under its name in at least one lane. Each
// row is one bundle's (CpuRef, GpuRef) verdict pair.
TEST(TestUnaccountedGoldenBundles, FlagsOnlyGoldenBundlesNoLaneRegistered)
{
    using Verdict = ReferenceLaneVerdict;
    const std::vector<Verdict> cpu{
        Verdict::REGISTERED, // 0: CPU-only op (e.g. BatchnormInference) -- covered
        Verdict::EXCLUDED_ON_COST, // 1: costly for CPU, GPU validates it -- covered
        Verdict::OUTSIDE_OP_SET, // 2: neither reference implements it -- dropped
        Verdict::EXCLUDED_ON_COST, // 3: left to a GPU lane that cannot run it -- dropped
        Verdict::NO_GOLDEN_OUTPUTS, // 4: nothing to validate
        Verdict::UNCOVERED_ON_COST, // 5: device-less run, skip test stands in -- covered
    };
    const std::vector<Verdict> gpu{
        Verdict::OUTSIDE_OP_SET,
        Verdict::REGISTERED,
        Verdict::OUTSIDE_OP_SET,
        Verdict::OUTSIDE_OP_SET,
        Verdict::NO_GOLDEN_OUTPUTS,
        Verdict::REGISTERED,
    };

    EXPECT_EQ(detail::bundlesNoLaneAccountsFor(cpu, gpu), (std::vector<size_t>{2, 3}));
}

// What the cross-lane check registers: a failing test (FailedLoad, not a skip) per
// unaccounted bundle, under the bundle's own tier-prefixed name, saying what each
// lane did with it.
TEST(TestUnaccountedGoldenBundles, UnaccountedBundleBecomesAFailureNamingBothLanes)
{
    using Verdict = ReferenceLaneVerdict;
    auto loaded = [](const std::string& suite, const std::string& test) {
        detail::LoadedBundle bundle;
        bundle.jsonPath = std::filesystem::path("bundles") / (suite + ".json");
        bundle.suiteName = suite;
        bundle.testName = test;
        return bundle;
    };
    const std::vector<detail::LoadedBundle> bundles{loaded("quick_SdpaFwd_a", "Small"),
                                                    loaded("quick_SdpaFwd_b", "Small")};
    const std::vector<Verdict> cpu{Verdict::REGISTERED, Verdict::EXCLUDED_ON_COST};
    const std::vector<Verdict> gpu{Verdict::OUTSIDE_OP_SET, Verdict::OUTSIDE_OP_SET};

    const auto failures = detail::unaccountedGoldenBundleFailures(bundles, cpu, gpu);

    ASSERT_EQ(failures.size(), 1u);
    EXPECT_EQ(failures[0].suiteName, "quick_SdpaFwd_b_Unvalidated");
    EXPECT_EQ(failures[0].testName, "Small");
    EXPECT_NE(failures[0].message.find(toString(Verdict::EXCLUDED_ON_COST)), std::string::npos)
        << failures[0].message;
    EXPECT_NE(failures[0].message.find(toString(Verdict::OUTSIDE_OP_SET)), std::string::npos)
        << failures[0].message;
    EXPECT_NE(failures[0].message.find(bundles[1].jsonPath.string()), std::string::npos)
        << failures[0].message;
}

// A deselected lane leaves its verdict vector empty; pairing it with a full one is
// a caller bug that must not be read as "no lane covered anything".
TEST(TestUnaccountedGoldenBundles, MismatchedVerdictSetsAreRejected)
{
    using Verdict = ReferenceLaneVerdict;
    const std::vector<detail::LoadedBundle> bundles(2);
    const std::vector<Verdict> both{Verdict::REGISTERED, Verdict::REGISTERED};
    const std::vector<Verdict> none;

    EXPECT_THROW(detail::bundlesNoLaneAccountsFor(both, none), std::invalid_argument);
    EXPECT_THROW(detail::unaccountedGoldenBundleFailures(bundles, none, none),
                 std::invalid_argument);
}

// A cost-excluded bundle that no lane will cover gets this test in its place. It
// must skip, naming the gap: a failure would make every device-less run red for a
// deliberate trade, and a pass would claim a validation that never happened.
TEST(TestUncoveredBundle, SkipsWithItsMessageRatherThanFailingOrPassing)
{
    detail::UncoveredBundleTest test("nothing validated this bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        test.TestBody();
    }

    EXPECT_TRUE(testing_support::anySkipped(results));
    EXPECT_FALSE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("nothing validated this bundle"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// NOLINTEND(readability-identifier-naming)
