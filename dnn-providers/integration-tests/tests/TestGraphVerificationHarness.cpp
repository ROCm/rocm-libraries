// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Covers IntegrationGraphVerificationHarness::reportFillFailure()'s FAIL-vs-SKIP
// split: an INVALID fill (the test's own recipe is malformed) fails the test; an
// UNSUPPORTED fill (the harness cannot fill this graph) skips it. Neither case
// reaches the real engine, so this never builds a graph. A host-only fixture also
// checks that a registered tolerance reaches the host validator with atol and rtol
// in their own places.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "HarnessTestSupport.hpp"
#include "harness/IntegrationGraphVerificationHarness.hpp"
#include "harness/ValidationSite.hpp"
#include "harness/input-init/FillInputs.hpp"
#include "harness/tolerance/ToleranceResolver.hpp"

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

// The base SetUp() needs a GPU. These tests make no HIP call, so they run on a host
// with no GPU.
class TestGraphVerificationHarnessHostOnly : public IntegrationGraphVerificationHarness<float, int>
{
public:
    TestGraphVerificationHarnessHostOnly()
    {
        bundle::testing_support::ensureTestConfigInitialized();
    }

protected:
    void SetUp() override {}
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

TEST_F(TestGraphVerificationHarnessHostOnly, Fp8ToleranceReachesValidatorWithAtolAndRtolInOrder)
{
    using hipdnn_data_sdk::types::fp8_e4m3;
    using hipdnn_data_sdk::utilities::Tensor;

    const auto fp8Step
        = tolerance::fp8OutputTolerance(hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3);
    ASSERT_TRUE(fp8Step.has_value());
    ASSERT_NE(fp8Step->atol, fp8Step->rtol);

    auto output = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    output->set_uid(7).set_name("y").set_data_type(hipdnn_frontend::DataType::FP8_E4M3);
    output->set_dim({1}).set_stride({1});

    registerValidator(output, *fp8Step);
    ASSERT_EQ(_deferredValidators.size(), 1u);
    _deferredValidators.front()(ValidationSite::HOST);

    const auto& entry = _tensorValidationMap.at(7);
    EXPECT_EQ(entry.tolerance.atol, fp8Step->atol);
    EXPECT_EQ(entry.tolerance.rtol, fp8Step->rtol);
    ASSERT_NE(entry.validator, nullptr) << entry.validatorError;

    // 9 is one E4M3 step above 8, and 10 is two steps above it. The one-step check
    // passes only when rtol scales the reference.
    const auto allCloseToEight = [&entry](float value) {
        Tensor<fp8_e4m3> reference(std::vector<int64_t>{1});
        Tensor<fp8_e4m3> actual(std::vector<int64_t>{1});
        reference.setHostValue(fp8_e4m3(8.0f), 0);
        actual.setHostValue(fp8_e4m3(value), 0);
        return entry.validator->allClose(reference, actual);
    };
    EXPECT_TRUE(allCloseToEight(9.0f));
    EXPECT_FALSE(allCloseToEight(10.0f));
}
