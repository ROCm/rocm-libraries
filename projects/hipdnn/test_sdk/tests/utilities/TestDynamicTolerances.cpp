// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::conv;

// =================================================================================================
// TestCalculateConvWrwTolerance
// =================================================================================================

struct ConvWrwToleranceTestCase
{
    double inputMin;
    double inputMax;
    double dyMin;
    double dyMax;
    std::vector<int64_t> dyDims;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const ConvWrwToleranceTestCase& tc)
    {
        os << "inputMin: " << tc.inputMin << ", inputMax: " << tc.inputMax
           << ", dyMin: " << tc.dyMin << ", dyMax: " << tc.dyMax << ", dyDims: [";
        for(size_t i = 0; i < tc.dyDims.size(); ++i)
        {
            os << tc.dyDims[i] << (i < tc.dyDims.size() - 1 ? ", " : "");
        }
        os << "], expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename Out, typename In, typename Comp>
struct TypeTriple
{
    using OutputType = Out;
    using InputType = In;
    using ComputeType = Comp;
};

template <typename T>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases();

// Float / Float / Float (High Precision: Linear)
// Error = 2 * N^2 * u * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<float, float, float>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1. Tol = 2 * 1^2 * 2^-23 = 2 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            // N=2. Accum = 2. Tol = 2 * 2^2 * 2^-23 = 8 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 8.0 * std::pow(2.0, -23)},
            // N=10. Accum = 10. Tol = 2 * 10^2 * 2^-23 = 200 * 2^-23
            // Exact gamma: (20 * 2^-23) / (1 - 20 * 2^-23) * 10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {10, 1, 1, 1},
             (20.0 * std::pow(2.0, -23)) / (1.0 - 20.0 * std::pow(2.0, -23)) * 10.0},
            // Large values: range -1000, 1000. maxProduct = 10^6.
            // N=10. Accum = 10. Tol = gamma * 10^7
            {-1000.0,
             1000.0,
             -1000.0,
             1000.0,
             {10, 1, 1, 1},
             (20.0 * std::pow(2.0, -23)) / (1.0 - 20.0 * std::pow(2.0, -23)) * 1.0e7}};
}

// Float / Double / Float (Input casting error)
// Input is double, Compute is float. We lose precision.
// Accumulation Error = 2 * N^2 * u * maxProduct
// Casting Error = 2 * N * maxProduct * u
// Total = (2 * N^2 + 2 * N) * u * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<float, double, float>>()
{
    return {// N=1. Accum = 1. Tol = (2 + 2) * 2^-23 = 4 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 4.0 * std::pow(2.0, -23)},
            // N=10. Accum = 10. Tol = (200 + 20) * 2^-23 = 220 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 220.0 * std::pow(2.0, -23)}};
}

// HipBfloat16 / Float / Float (High Precision Compute: Linear)
// Accumulation Error = 2 * N^2 * u_float * maxProduct
// Output Cast Error = N * maxProduct * u_bfp16
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<hip_bfloat16, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // N=1. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23) + std::pow(2.0, -7)},
        // N=2. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 8.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -7)},
        // N=10. Accum = 10. Tol = 200 * 2^-23 + 10 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         200.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -7)}};
}

// HipBfloat16 / HipBfloat16 / HipBfloat16 (Lower Precision: Statistical)
// Error = K * sqrt(2N) * u * (N * maxProduct) = K * N * sqrt(2N) * u * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<hip_bfloat16, hip_bfloat16, hip_bfloat16>>()
{
    // Expected values are pre-rounded to Bfp16 to match implementation behavior
    // 2^-7 = 0.0078125
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-7 = 8.485 * 2^-7 = 0.06629...
            // Rounded to Bfp16: 0.06640625 (17/256)
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 0.06640625},
            // N=2. Accum = 2. Tol = 6 * 2 * sqrt(4) * 2^-7 = 24 * 2^-7 = 0.1875
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 0.1875},
            // N=10. Accum = 10. Tol = 6 * 10 * sqrt(20) * 2^-7 = 268.328 * 2^-7 = 2.0963...
            // Rounded to Bfp16: 2.09375 (134/64)
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 2.09375}};
}

// Half / Float / Float (High Precision Compute: Linear)
// Accumulation Error = 2 * N^2 * u_float * maxProduct
// Output Cast Error = N * maxProduct * u_half
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypeTriple<half, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // N=1. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23) + std::pow(2.0, -10)},
        // N=2. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 8.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -10)},
        // N=10. Accum = 10. Tol = 200 * 2^-23 + 10 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         200.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -10)}};
}

// Half / Half / Half (Lower Precision: Statistical)
// Error = K * N * sqrt(2N) * u * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypeTriple<half, half, half>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-10
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 6.0 * std::sqrt(2.0) * std::pow(2.0, -10)},
            // N=2. Accum = 2. Tol = 6 * 2 * sqrt(4) * 2^-10 = 24 * 2^-10
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 24.0 * std::pow(2.0, -10)},
            // N=10. Accum = 10. Tol = 6 * 10 * sqrt(20) * 2^-10
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 60.0 * std::sqrt(20.0) * std::pow(2.0, -10)}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateConvWrwTolerance : public ::testing::TestWithParam<ConvWrwToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateConvWrwTolerance<Out, In, Comp>(
                    params.inputMin, params.inputMax, params.dyMin, params.dyMax, params.dyDims)),
                std::invalid_argument)
                << "Failed to throw for dims size: " << params.dyDims.size();
        }
        else
        {
            auto tol = calculateConvWrwTolerance<Out, In, Comp>(
                params.inputMin, params.inputMax, params.dyMin, params.dyMax, params.dyDims);

            auto expected = hipdnn_data_sdk::utilities::staticCast<Out>(params.expectedTolerance);

            EXPECT_NEAR(tol, expected, 1e-5) << "Failed for dims size: " << params.dyDims.size();
        }
    }
};

using TestCalculateConvWrwToleranceFp32 = TestCalculateConvWrwTolerance<float, float, float>;
TEST_P(TestCalculateConvWrwToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceFp32,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateConvWrwToleranceInputDouble
    = TestCalculateConvWrwTolerance<float, double, float>;
TEST_P(TestCalculateConvWrwToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceInputDouble,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateConvWrwToleranceComputeFloatBfp16
    = TestCalculateConvWrwTolerance<hip_bfloat16, float, float>;
TEST_P(TestCalculateConvWrwToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<hip_bfloat16, float, float>>()));

using TestCalculateConvWrwToleranceBfp16
    = TestCalculateConvWrwTolerance<hip_bfloat16, hip_bfloat16, hip_bfloat16>;
TEST_P(TestCalculateConvWrwToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceBfp16,
    ::testing::ValuesIn(
        getConvWrwToleranceTestCases<TypeTriple<hip_bfloat16, hip_bfloat16, hip_bfloat16>>()));

using TestCalculateConvWrwToleranceComputeFloatFp16
    = TestCalculateConvWrwTolerance<half, float, float>;
TEST_P(TestCalculateConvWrwToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceComputeFloatFp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateConvWrwToleranceFp16 = TestCalculateConvWrwTolerance<half, half, half>;
TEST_P(TestCalculateConvWrwToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceFp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<half, half, half>>()));

// Test that calculateConvWrwTolerance catches simulated wrong outputs
TEST(TestCalculateConvWrwTolerance, DetectsFailure)
{
    // N=1, Spatial=10x10 => Accumulations = 100
    std::vector<int64_t> dims = {1, 1, 10, 10};
    std::vector<int64_t> strides = {100, 100, 10, 1};

    // Create tensors
    auto baseline = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);

    // Populate with values
    // Correct value: 1.0
    // Tolerance for N=100, Half/Float/Half is approx 0.1
    // Okay value: 1.05 (Error 0.05)
    // Wrong value: 1.2 (Error 0.2)

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.05f);
    actualFailing->fillTensorWithValue(1.2f);

    auto tol = calculateConvWrwTolerance<half, half, float>(-1.0, 1.0, -1.0, 1.0, dims);

    // tol approx 0.1
    EXPECT_LT(tol, 0.15_h);
    EXPECT_GT(tol, 0.09_h);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid) << "Validator should have passed";

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid) << "Validator should have failed";
}

// Test that calculateConvWrwTolerance throws when nU >= 1.0 (singularity)
TEST(TestCalculateConvWrwTolerance, ThrowsOnSingularity)
{
    // For float, epsilon is 2^-23 approx 1.19e-7.
    // nU = 2 * n * epsilon.
    // We need nU >= 1.0 => n >= 1 / (2 * epsilon) = 2^22 = 4,194,304.
    // Let's use 5,000,000.
    std::vector<int64_t> dims = {5000000, 1, 1, 1};

    EXPECT_THROW((calculateConvWrwTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, dims)),
                 std::overflow_error);
}

// Test that calculateConvWrwTolerance throws when tolerance exceeds OutputType max
TEST(TestCalculateConvWrwTolerance, ThrowsOnOutputOverflow)
{
    // OutputType = half (max approx 65504)
    // ComputeType = float
    // We need tolerance > 65504.
    // Let N=10.
    // maxProduct = 1e10 (input 1e5 * dy 1e5)
    // sumAbsProductBound = 1e11
    // epsilon (float) approx 1.19e-7
    // gamma approx 2 * 10 * 1.19e-7 approx 2.38e-6
    // accumulatedTolerance approx 2.38e-6 * 1e11 approx 2.38e5 = 238,000
    // 238,000 > 65,504 => Should throw.

    std::vector<int64_t> dims = {10, 1, 1, 1};
    double val = 1.0e5;

    EXPECT_THROW((calculateConvWrwTolerance<half, float, float>(-val, val, -val, val, dims)),
                 std::overflow_error);
}
