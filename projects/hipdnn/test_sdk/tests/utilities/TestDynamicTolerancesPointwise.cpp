// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/PointwiseErrorClassification.hpp>
#include <vector>

using namespace hipdnn_data_sdk::types;

template <typename Out, typename In, typename Comp>
struct TypeTriple
{
    using OutputType = Out;
    using InputType = In;
    using ComputeType = Comp;
};

// =================================================================================================
// TestCalculatePointwiseTolerance
// =================================================================================================

using namespace hipdnn_test_sdk::utilities::pointwise;

struct PointwiseToleranceTestCase
{
    double scale;
    PointwiseErrorClass errorClass;
    double expectedTolerance;

    friend std::ostream& operator<<(std::ostream& os, const PointwiseToleranceTestCase& tc)
    {
        static constexpr std::array CLASS_NAMES = {"Bitwise",
                                                   "Linear",
                                                   "Rational",
                                                   "TranscendentalFwd",
                                                   "TranscendentalBwd",
                                                   "CompositeFwd",
                                                   "CompositeBwd"};
        os << "scale: " << tc.scale
           << ", errorClass: " << CLASS_NAMES[static_cast<uint8_t>(tc.errorClass)]
           << ", expectedTolerance: " << tc.expectedTolerance;
        return os;
    }
};

template <typename T>
std::vector<PointwiseToleranceTestCase> getPointwiseToleranceTestCases();

// Float / Float / Float — C_high: Bitwise=1, Linear=2, Rational=4,
//   TransFwd=8, TransBwd=12, CompFwd=16, CompBwd=24
// tolerance = C * 2^-23 * scale, no cast errors
template <>
std::vector<PointwiseToleranceTestCase>
    getPointwiseToleranceTestCases<TypeTriple<float, float, float>>()
{
    return {
        {1.0, PointwiseErrorClass::BITWISE, 1.0 * std::pow(2.0, -23)},
        {1.0, PointwiseErrorClass::LINEAR, 2.0 * std::pow(2.0, -23)},
        {1.0, PointwiseErrorClass::RATIONAL, 4.0 * std::pow(2.0, -23)},
        {1.0, PointwiseErrorClass::TRANSCENDENTAL_FWD, 8.0 * std::pow(2.0, -23)},
        {1.0, PointwiseErrorClass::TRANSCENDENTAL_BWD, 12.0 * std::pow(2.0, -23)},
        {1.0, PointwiseErrorClass::COMPOSITE_FWD, 16.0 * std::pow(2.0, -23)},
        {1.0, PointwiseErrorClass::COMPOSITE_BWD, 24.0 * std::pow(2.0, -23)},
        // CompositeBwd with large scale: C=24, scale=100. Tol = 2400 * 2^-23
        {100.0, PointwiseErrorClass::COMPOSITE_BWD, 2400.0 * std::pow(2.0, -23)},
    };
}

// Float / Double / Float — Input downcast (double -> float): +epsilon_compute * scale
// tolerance = (C + 1) * 2^-23 * scale
template <>
std::vector<PointwiseToleranceTestCase>
    getPointwiseToleranceTestCases<TypeTriple<float, double, float>>()
{
    return {
        // Bitwise: (1 + 1) * 2^-23 = 2 * 2^-23
        {1.0, PointwiseErrorClass::BITWISE, 2.0 * std::pow(2.0, -23)},
        // Linear: (2 + 1) * 2^-23 = 3 * 2^-23
        {1.0, PointwiseErrorClass::LINEAR, 3.0 * std::pow(2.0, -23)},
        // CompositeFwd: (16 + 1) * 2^-23 = 17 * 2^-23
        {1.0, PointwiseErrorClass::COMPOSITE_FWD, 17.0 * std::pow(2.0, -23)},
        // CompositeBwd: (24 + 1) * 2^-23 = 25 * 2^-23
        {1.0, PointwiseErrorClass::COMPOSITE_BWD, 25.0 * std::pow(2.0, -23)},
    };
}

// Bfloat16 / Float / Float — Output downcast (float -> bfloat16): +2^-7 * scale
// tolerance = C * 2^-23 * scale + 2^-7 * scale
template <>
std::vector<PointwiseToleranceTestCase>
    getPointwiseToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    return {
        // Bitwise: 1 * 2^-23 + 2^-7
        {1.0, PointwiseErrorClass::BITWISE, std::pow(2.0, -23) + std::pow(2.0, -7)},
        // TranscendentalFwd: 8 * 2^-23 + 2^-7
        {1.0,
         PointwiseErrorClass::TRANSCENDENTAL_FWD,
         8.0 * std::pow(2.0, -23) + std::pow(2.0, -7)},
        // TranscendentalBwd: 12 * 2^-23 + 2^-7
        {1.0,
         PointwiseErrorClass::TRANSCENDENTAL_BWD,
         12.0 * std::pow(2.0, -23) + std::pow(2.0, -7)},
        // CompositeBwd with large scale: (24 * 2^-23 + 2^-7) * 100
        {100.0,
         PointwiseErrorClass::COMPOSITE_BWD,
         100.0 * (24.0 * std::pow(2.0, -23) + std::pow(2.0, -7))},
    };
}

// Bfloat16 / Bfloat16 / Bfloat16 — C_low: Bitwise=2, Linear=4, Rational=8,
//   TransFwd=16, TransBwd=24, CompFwd=32, CompBwd=48
// tolerance = C_low * 2^-7 * scale, no cast errors
template <>
std::vector<PointwiseToleranceTestCase>
    getPointwiseToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    return {
        {1.0, PointwiseErrorClass::BITWISE, 2.0 * std::pow(2.0, -7)},
        {1.0, PointwiseErrorClass::LINEAR, 4.0 * std::pow(2.0, -7)},
        {1.0, PointwiseErrorClass::RATIONAL, 8.0 * std::pow(2.0, -7)},
        {1.0, PointwiseErrorClass::TRANSCENDENTAL_FWD, 16.0 * std::pow(2.0, -7)},
        {1.0, PointwiseErrorClass::TRANSCENDENTAL_BWD, 24.0 * std::pow(2.0, -7)},
        {1.0, PointwiseErrorClass::COMPOSITE_FWD, 32.0 * std::pow(2.0, -7)},
        {1.0, PointwiseErrorClass::COMPOSITE_BWD, 48.0 * std::pow(2.0, -7)},
    };
}

// Half / Float / Float — Output downcast (float -> half): +2^-10 * scale
// tolerance = C * 2^-23 * scale + 2^-10 * scale
template <>
std::vector<PointwiseToleranceTestCase>
    getPointwiseToleranceTestCases<TypeTriple<half, float, float>>()
{
    return {
        // Bitwise: 1 * 2^-23 + 2^-10
        {1.0, PointwiseErrorClass::BITWISE, std::pow(2.0, -23) + std::pow(2.0, -10)},
        // TranscendentalFwd: 8 * 2^-23 + 2^-10
        {1.0,
         PointwiseErrorClass::TRANSCENDENTAL_FWD,
         8.0 * std::pow(2.0, -23) + std::pow(2.0, -10)},
        // TranscendentalBwd: 12 * 2^-23 + 2^-10
        {1.0,
         PointwiseErrorClass::TRANSCENDENTAL_BWD,
         12.0 * std::pow(2.0, -23) + std::pow(2.0, -10)},
        // CompositeBwd with large scale: (24 * 2^-23 + 2^-10) * 100
        {100.0,
         PointwiseErrorClass::COMPOSITE_BWD,
         100.0 * (24.0 * std::pow(2.0, -23) + std::pow(2.0, -10))},
    };
}

// Half / Half / Half — C_low: Bitwise=2, Linear=4, Rational=8,
//   TransFwd=16, TransBwd=24, CompFwd=32, CompBwd=48
// tolerance = C_low * 2^-10 * scale, no cast errors
template <>
std::vector<PointwiseToleranceTestCase>
    getPointwiseToleranceTestCases<TypeTriple<half, half, half>>()
{
    return {
        {1.0, PointwiseErrorClass::BITWISE, 2.0 * std::pow(2.0, -10)},
        {1.0, PointwiseErrorClass::LINEAR, 4.0 * std::pow(2.0, -10)},
        {1.0, PointwiseErrorClass::RATIONAL, 8.0 * std::pow(2.0, -10)},
        {1.0, PointwiseErrorClass::TRANSCENDENTAL_FWD, 16.0 * std::pow(2.0, -10)},
        {1.0, PointwiseErrorClass::TRANSCENDENTAL_BWD, 24.0 * std::pow(2.0, -10)},
        {1.0, PointwiseErrorClass::COMPOSITE_FWD, 32.0 * std::pow(2.0, -10)},
        {1.0, PointwiseErrorClass::COMPOSITE_BWD, 48.0 * std::pow(2.0, -10)},
    };
}

template <typename Out, typename In, typename Comp>
class TestCalculatePointwiseTolerance : public ::testing::TestWithParam<PointwiseToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        auto tol = calculatePointwiseTolerance<Out, In, Comp>(params.scale, params.errorClass);

        auto expected = static_cast<float>(params.expectedTolerance);

        EXPECT_NEAR(tol, expected, 1e-10f);
    }
};

using TestCalculatePointwiseToleranceFp32 = TestCalculatePointwiseTolerance<float, float, float>;
TEST_P(TestCalculatePointwiseToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculatePointwiseToleranceFp32,
    ::testing::ValuesIn(getPointwiseToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculatePointwiseToleranceInputDouble
    = TestCalculatePointwiseTolerance<float, double, float>;
TEST_P(TestCalculatePointwiseToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculatePointwiseToleranceInputDouble,
    ::testing::ValuesIn(getPointwiseToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculatePointwiseToleranceComputeFloatBfp16
    = TestCalculatePointwiseTolerance<bfloat16, float, float>;
TEST_P(TestCalculatePointwiseToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculatePointwiseToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getPointwiseToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculatePointwiseToleranceBfp16
    = TestCalculatePointwiseTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculatePointwiseToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculatePointwiseToleranceBfp16,
    ::testing::ValuesIn(
        getPointwiseToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

using TestCalculatePointwiseToleranceComputeFloatFp16
    = TestCalculatePointwiseTolerance<half, float, float>;
TEST_P(TestCalculatePointwiseToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculatePointwiseToleranceComputeFloatFp16,
    ::testing::ValuesIn(getPointwiseToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculatePointwiseToleranceFp16 = TestCalculatePointwiseTolerance<half, half, half>;
TEST_P(TestCalculatePointwiseToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculatePointwiseToleranceFp16,
    ::testing::ValuesIn(getPointwiseToleranceTestCases<TypeTriple<half, half, half>>()));

// Bitwise must produce non-zero tolerance (epsilon floor prevents 0-tolerance fragility)
TEST(TestCalculatePointwiseTolerance, BitwiseHasNonZeroTolerance)
{
    // float/float/float: C=1 * 2^-23 * 1.0 > 0
    auto tol = calculatePointwiseTolerance<float, float, float>(1.0, PointwiseErrorClass::BITWISE);
    EXPECT_GT(tol, 0.0f);

    // half/half/half: C=1 * 2^-10 * 1.0 > 0
    auto tolHalf = calculatePointwiseTolerance<half, half, half>(1.0, PointwiseErrorClass::BITWISE);
    EXPECT_GT(tolHalf, 0.0f);

    // bfloat16/bfloat16/bfloat16: C=1 * 2^-7 * 1.0 > 0
    auto tolBf16 = calculatePointwiseTolerance<bfloat16, bfloat16, bfloat16>(
        1.0, PointwiseErrorClass::BITWISE);
    EXPECT_GT(tolBf16, 0.0f);
}

// Test that tolerance scales linearly with scale
TEST(TestCalculatePointwiseTolerance, ScalesWithScale)
{
    auto tol1 = calculatePointwiseTolerance<float, float, float>(1.0, PointwiseErrorClass::LINEAR);
    auto tol100
        = calculatePointwiseTolerance<float, float, float>(100.0, PointwiseErrorClass::LINEAR);

    EXPECT_NEAR(tol100 / tol1, 100.0f, 1e-3f);
}

// Test that C_low >= C_high for all error classes (invariant)
TEST(TestCalculatePointwiseTolerance, LowPrecisionMultiplierNotLessThanHigh)
{
    for(auto errorClass : {PointwiseErrorClass::BITWISE,
                           PointwiseErrorClass::LINEAR,
                           PointwiseErrorClass::RATIONAL,
                           PointwiseErrorClass::TRANSCENDENTAL_FWD,
                           PointwiseErrorClass::TRANSCENDENTAL_BWD,
                           PointwiseErrorClass::COMPOSITE_FWD,
                           PointwiseErrorClass::COMPOSITE_BWD})
    {
        // Compare tolerance for low-precision compute vs high-precision compute,
        // both with same type for input/output to isolate the C multiplier.
        auto tolHigh = calculatePointwiseTolerance<float, float, float>(1.0, errorClass);
        auto tolLowBf16
            = calculatePointwiseTolerance<bfloat16, bfloat16, bfloat16>(1.0, errorClass);
        auto tolLowHalf = calculatePointwiseTolerance<half, half, half>(1.0, errorClass);

        // Normalize by epsilon to extract C: tol = C * epsilon * inputScale
        // C_high = tolHigh / epsilon_float, C_low_bf16 = tolLowBf16 / epsilon_bf16
        auto epsFloat = static_cast<double>(std::numeric_limits<float>::epsilon());
        auto epsBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
        auto epsHalf = static_cast<double>(std::numeric_limits<half>::epsilon());

        const double cHigh = static_cast<double>(tolHigh) / epsFloat;
        const double cLowBf16 = static_cast<double>(tolLowBf16) / epsBf16;
        const double cLowHalf = static_cast<double>(tolLowHalf) / epsHalf;

        EXPECT_GE(cLowBf16, cHigh)
            << "C_low(bf16) < C_high for class " << static_cast<int>(errorClass);
        EXPECT_GE(cLowHalf, cHigh)
            << "C_low(half) < C_high for class " << static_cast<int>(errorClass);
    }
}

// Test that calculatePointwiseTolerance detects wrong outputs
TEST(TestCalculatePointwiseTolerance, DetectsFailure)
{
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};

    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

    // For float/float/float, TRANSCENDENTAL_FWD, scale=1.0:
    // tol = 8 * 2^-23 ≈ 9.54e-7 (no cast errors, clean C * epsilon * scale)
    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.0f + 1.0e-7f); // Error 1e-7 < 9.54e-7
    actualFailing->fillTensorWithValue(1.0f + 2.0e-6f); // Error 2e-6 > 9.54e-7

    auto tol = calculatePointwiseTolerance<float, float, float>(
        1.0, PointwiseErrorClass::TRANSCENDENTAL_FWD);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that backward ops always have higher tolerance than forward ops
TEST(TestCalculatePointwiseTolerance, BackwardToleranceExceedsForward)
{
    auto transFwd = calculatePointwiseTolerance<float, float, float>(
        1.0, PointwiseErrorClass::TRANSCENDENTAL_FWD);
    auto transBwd = calculatePointwiseTolerance<float, float, float>(
        1.0, PointwiseErrorClass::TRANSCENDENTAL_BWD);
    EXPECT_GT(transBwd, transFwd);

    auto compFwd
        = calculatePointwiseTolerance<float, float, float>(1.0, PointwiseErrorClass::COMPOSITE_FWD);
    auto compBwd
        = calculatePointwiseTolerance<float, float, float>(1.0, PointwiseErrorClass::COMPOSITE_BWD);
    EXPECT_GT(compBwd, compFwd);

    // Same invariant for low precision
    auto transFwdBf16 = calculatePointwiseTolerance<bfloat16, bfloat16, bfloat16>(
        1.0, PointwiseErrorClass::TRANSCENDENTAL_FWD);
    auto transBwdBf16 = calculatePointwiseTolerance<bfloat16, bfloat16, bfloat16>(
        1.0, PointwiseErrorClass::TRANSCENDENTAL_BWD);
    EXPECT_GT(transBwdBf16, transFwdBf16);
}

// =================================================================================================
// TestClassifyPointwiseOp — exercises every PointwiseMode branch in classifyPointwiseOp
// and isBoundedOutput for code coverage.
// =================================================================================================

using hipdnn_flatbuffers_sdk::data_objects::EnumNamesPointwiseMode;
using hipdnn_flatbuffers_sdk::data_objects::EnumValuesPointwiseMode;
using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

// Verify one representative per error class for correctness
TEST(TestClassifyPointwiseOp, OnePerClassCorrect)
{
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::ABS), PointwiseErrorClass::BITWISE);
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::ADD), PointwiseErrorClass::LINEAR);
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::DIV), PointwiseErrorClass::RATIONAL);
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::EXP), PointwiseErrorClass::TRANSCENDENTAL_FWD);
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::TANH_BWD),
              PointwiseErrorClass::TRANSCENDENTAL_BWD);
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::GELU_FWD), PointwiseErrorClass::COMPOSITE_FWD);
    EXPECT_EQ(classifyPointwiseOp(PointwiseMode::GELU_BWD), PointwiseErrorClass::COMPOSITE_BWD);
}

// Exercise every enum value for branch coverage; verify result is a valid class
TEST(TestClassifyPointwiseOp, AllModesReturnValidClass)
{
    for(auto mode : EnumValuesPointwiseMode())
    {
        if(mode == PointwiseMode::UNSET)
        {
            continue;
        }
        auto errorClass = classifyPointwiseOp(mode);
        EXPECT_LE(static_cast<uint8_t>(errorClass),
                  static_cast<uint8_t>(PointwiseErrorClass::COMPOSITE_BWD))
            << "invalid class for " << EnumNamesPointwiseMode()[static_cast<int>(mode)];
    }
}

// Only SIGMOID_FWD, TANH_FWD, ERF are bounded; all others are not
TEST(TestClassifyPointwiseOp, BoundedOutputCorrect)
{
    EXPECT_TRUE(isBoundedOutput(PointwiseMode::SIGMOID_FWD));
    EXPECT_TRUE(isBoundedOutput(PointwiseMode::TANH_FWD));
    EXPECT_TRUE(isBoundedOutput(PointwiseMode::ERF));

    for(auto mode : EnumValuesPointwiseMode())
    {
        if(mode == PointwiseMode::UNSET || mode == PointwiseMode::SIGMOID_FWD
           || mode == PointwiseMode::TANH_FWD || mode == PointwiseMode::ERF)
        {
            continue;
        }
        EXPECT_FALSE(isBoundedOutput(mode))
            << EnumNamesPointwiseMode()[static_cast<int>(mode)] << " should not be bounded";
    }
}
