// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/exact_verification.hpp>

namespace {
using namespace roc::hostnumerics;

TEST(ExactVerification, ActivationUsesDivisionAndExplicitOutputRounding) {
    Tensor auxiliary(ScalarType::Float64, Layout(Shape{2, 2}, {1, 5}, 2));
    Tensor observed(ScalarType::Float64, Layout(Shape{2, 2}, {3, 1}, 1));
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) {
            auxiliary.storeFrom({i, j}, 5.0);
            observed.storeFrom({i, j}, 5.0 / 3.0);
        }
    auto result = verifyExactActivation(observed, auxiliary, Tensor(3.0), Tensor(1.0),
                                        IdentityActivation{}, 10);
    EXPECT_EQ(result.status, VerificationStatus::Passed) << result.reason;
    ASSERT_TRUE(result.maximumAbsolute);
    EXPECT_EQ(result.maximumAbsolute->item<double>(), 5.0 / 3.0);
    observed.storeFrom({1, 0}, 5.0 * (1.0 / 3.0));
    result = verifyExactActivation(observed, auxiliary, Tensor(3.0), Tensor(1.0),
                                   IdentityActivation{}, 10);
    EXPECT_EQ(result.status, VerificationStatus::Mismatch);
    ASSERT_EQ(result.comparison.reportedMismatches.size(), 1);
    EXPECT_EQ(result.comparison.reportedMismatches.front().coordinates,
              (std::vector<size_t>{1, 0}));
    // Halfway integer output is rounded to even; large results saturate.
    const auto five = Tensor(5.0).reshapeSharingStorage(Shape{1});
    auto two = Tensor(int8_t{2}).reshapeSharingStorage(Shape{1});
    EXPECT_EQ(
        verifyExactActivation(two, five, Tensor(2.0), Tensor(1.0), IdentityActivation{}, 10).status,
        VerificationStatus::Passed);
    two.storeFrom({0}, int8_t{127});
    EXPECT_EQ(verifyExactActivation(two, five, Tensor(1.0), Tensor(100.0), IdentityActivation{}, 10)
                  .status,
              VerificationStatus::Passed);
}

TEST(ExactVerification, ActivationRejectsUncertifiedAuxiliaryAndUnsupportedPolicies) {
    Tensor auxiliary(ScalarType::Float32, Shape{2}), observed(ScalarType::Float32, Shape{2});
    for (double bad : {0.5, 0x1p24, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        auxiliary.storeFrom({1}, bad);
        auto result = verifyExactActivation(observed, auxiliary, Tensor(1.0), Tensor(1.0),
                                            IdentityActivation{}, 10);
        EXPECT_EQ(result.status, VerificationStatus::Unsupported);
        EXPECT_EQ(result.uncertifiedElements, 1);
        EXPECT_FALSE(result.maximumAbsolute);
    }
    auxiliary.storeFrom({1}, 0);
    for (double scale : {0.0, std::numeric_limits<double>::infinity()})
        EXPECT_EQ(verifyExactActivation(observed, auxiliary, Tensor(scale), Tensor(1.0),
                                        IdentityActivation{}, 10)
                      .status,
                  VerificationStatus::Unsupported);
    EXPECT_EQ(
        verifyExactActivation(observed, auxiliary, Tensor(1.0), Tensor(1.0), GeluActivation{}, 10)
            .status,
        VerificationStatus::Unsupported);
    EXPECT_EQ(verifyExactActivation(observed, auxiliary, Tensor(1.0), Tensor(1.0),
                                    ClampActivation{2, -2}, 10)
                  .status,
              VerificationStatus::Unsupported);
    EXPECT_THROW(verifyExactActivation(Tensor(0.0), auxiliary, Tensor(1.0), Tensor(1.0),
                                       IdentityActivation{}, 10),
                 std::runtime_error);
    EXPECT_THROW(verifyExactActivation(observed, auxiliary, auxiliary, Tensor(1.0),
                                       IdentityActivation{}, 10),
                 std::runtime_error);
}

TEST(ExactVerification, ActivationHandlesNegativeScalesAndFractionalClamp) {
    Tensor auxiliary(ScalarType::Float32, Shape{1, 2});
    auxiliary.storeFrom({0, 0}, 6);
    auxiliary.storeFrom({0, 1}, -9);
    Tensor observed(ScalarType::Float32, Shape{1, 2});
    const double bound = double(0.1f);
    observed.storeFrom({0, 0}, 0);
    observed.storeFrom({0, 1}, -9 * bound);
    auto result = verifyExactActivation(observed, auxiliary, Tensor(-3.0), Tensor(-9.0),
                                        ClampActivation{0, bound}, 10);
    EXPECT_EQ(result.status, VerificationStatus::Passed);
    ASSERT_TRUE(result.maximumAbsolute);
    EXPECT_EQ(result.maximumAbsolute->item<double>(), bound);
    observed.storeFrom({0, 1}, -27);
    result = verifyExactActivation(observed, auxiliary, Tensor(-3.0), Tensor(-9.0),
                                   ReluActivation{}, 10);
    EXPECT_EQ(result.status, VerificationStatus::Passed);
    EXPECT_EQ(result.maximumAbsolute->item<double>(), 3);
}

TEST(ExactVerification, SumCertifiesAbsoluteBoundBeforeCancellingTerms) {
    Tensor input(ScalarType::Float32, Layout(Shape{2, 3}, {1, 4}, 2));
    for (size_t row = 0; row < 2; ++row) {
        input.storeFrom({row, 0}, 0x1p23);
        input.storeFrom({row, 1}, -0x1p23);
        input.storeFrom({row, 2}, 1);
    }
    Tensor output(ScalarType::Float32, Shape{2});
    output.storeFrom({0}, 1);
    output.storeFrom({1}, 1);
    EXPECT_EQ(verifyExactSum(output, input, {1}, ScalarType::Float32, 10).status,
              VerificationStatus::Unsupported);
    EXPECT_EQ(verifyExactSum(output, input, {1}, ScalarType::Float64, 10).status,
              VerificationStatus::Passed);
    input.storeFrom({0, 2}, 0.5);
    EXPECT_EQ(verifyExactSum(output, input, {1}, ScalarType::Float64, 10).status,
              VerificationStatus::Unsupported);
    input.storeFrom({0, 2}, std::numeric_limits<float>::quiet_NaN());
    EXPECT_EQ(verifyExactSum(output, input, {1}, ScalarType::Float64, 10).status,
              VerificationStatus::Unsupported);
}

TEST(ExactVerification, SumUsesLogicalAxesRoundingSaturationAndBoundedReports) {
    Tensor input(ScalarType::Float32, Shape{2, 2, 2});
    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j) {
            input.storeFrom({0, i, j}, 512);
            input.storeFrom({1, i, j}, -45);
        }
    input.storeFrom({0, 0, 0}, 513);
    Tensor half(ScalarType::Float16, Shape{2});
    half.storeFrom({0}, 2048);
    half.storeFrom({1}, -180);
    EXPECT_EQ(verifyExactSum(half, input, {1, 2}, ScalarType::Float32, 10).status,
              VerificationStatus::Passed);
    EXPECT_EQ(verifyExactSum(half, input, {1, 2}, ScalarType::Float16, 10).status,
              VerificationStatus::Unsupported);
    Tensor narrow(ScalarType::Int8, Shape{2});
    narrow.storeFrom({0}, 127);
    narrow.storeFrom({1}, -128);
    EXPECT_EQ(verifyExactSum(narrow, input, {2, 1}, ScalarType::Float32, 10).status,
              VerificationStatus::Passed);
    narrow.storeFrom({0}, 0);
    narrow.storeFrom({1}, 0);
    const auto result = verifyExactSum(narrow, input, {1, 2}, ScalarType::Float32, 1);
    EXPECT_EQ(result.status, VerificationStatus::Mismatch);
    EXPECT_EQ(result.comparison.mismatches, 2);
    EXPECT_EQ(result.comparison.reportedMismatches.size(), 1);
    EXPECT_THROW(verifyExactSum(narrow, input, {1, 1}, ScalarType::Float32, 10),
                 std::runtime_error);
    EXPECT_THROW(verifyExactSum(narrow, input, {3}, ScalarType::Float32, 10), std::runtime_error);
}

TEST(ExactVerification, MaximumRejectsNonfiniteFractionalAndSaturatedStorage) {
    Tensor output(ScalarType::Float32, Shape{2});
    output.storeFrom({0}, -6);
    output.storeFrom({1}, 4);
    auto result = recoverExactMaximumAbsolute(output, Tensor(-2.0));
    ASSERT_TRUE(std::holds_alternative<Tensor>(result));
    EXPECT_EQ(std::get<Tensor>(result).item<double>(), 3);
    for (double bad : {0.5, 0x1p24, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        output.storeFrom({1}, bad);
        EXPECT_TRUE(std::holds_alternative<VerificationFailure>(
            recoverExactMaximumAbsolute(output, Tensor(1.0))));
    }
    for (int8_t value : {int8_t{-128}, int8_t{127}})
        EXPECT_TRUE(std::holds_alternative<VerificationFailure>(
            recoverExactMaximumAbsolute(Tensor(value), Tensor(1.0))));
    EXPECT_TRUE(std::holds_alternative<VerificationFailure>(
        recoverExactMaximumAbsolute(Tensor(0.0), Tensor(0.0))));
    result = recoverExactMaximumAbsolute(Tensor(ScalarType::Float32, Shape{0, 3}), Tensor(1.0));
    ASSERT_TRUE(std::holds_alternative<Tensor>(result));
    EXPECT_EQ(std::get<Tensor>(result).item<double>(), 0);
}
}  // namespace
