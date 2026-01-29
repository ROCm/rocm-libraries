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
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<float, float, float>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1.
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 1.0 * std::pow(2.0, -23)},
            // N=2. Accum = 2.
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            // N=10. Accum = 10.
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 10.0 * std::pow(2.0, -23)},
            // Large values: range -1000, 1000. maxProduct = 10^6.
            // N=10. Accum = 10.
            {-1000.0, 1000.0, -1000.0, 1000.0, {10, 1, 1, 1}, 10.0 * 1.0e6 * std::pow(2.0, -23)}};
}

// Float / Double / Float (Input casting error)
// Input is double, Compute is float. We lose precision.
// Error = (N * maxProduct * eps) + (2 * N * maxProduct * eps) = 3 * N * maxProduct * eps
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<float, double, float>>()
{
    return {// N=1. Accum = 1. Tol = 3 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 3.0 * std::pow(2.0, -23)},
            // N=10. Accum = 10. Tol = 3 * 10 * 2^-23 = 30 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 30.0 * std::pow(2.0, -23)}};
}

// HipBfloat16 / Float / Float (High Precision Compute: Linear)
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<hip_bfloat16, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // N=1. Accum = 1.
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -7)},
        // N=2. Accum = 2.
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 2.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -7)},
        // N=10. Accum = 10.
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         10.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -7)}};
}

// HipBfloat16 / HipBfloat16 / HipBfloat16 (Lower Precision: Statistical)
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<hip_bfloat16, hip_bfloat16, hip_bfloat16>>()
{
    // Expected values are pre-rounded to Bfp16 to match implementation behavior
    // 2^-7 = 0.0078125
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1. Tol = 6*1 * 2^-7 = 6 * 2^-7 = 0.046875
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 0.046875},
            // N=2. Accum = 2. Tol = 6*sqrt(2) * 2^-7 = 8.485 * 2^-7 = 0.06629...
            // Rounded to Bfp16: 0.06640625 (17/256)
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 0.06640625},
            // N=10. Accum = 10. Tol = 6*sqrt(10) * 2^-7 = 18.97 * 2^-7 = 0.14823...
            // Rounded to Bfp16: 0.1484375 (19/128)
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 0.1484375}};
}

// Half / Float / Float (High Precision Compute: Linear)
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypeTriple<half, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // N=1. Accum = 1.
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -10)},
        // N=2. Accum = 2.
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 2.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -10)},
        // N=10. Accum = 10.
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         10.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -10)}};
}

// Half / Half / Half (Lower Precision: Statistical)
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypeTriple<half, half, half>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1.
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 6.0 * std::pow(2.0, -10)},
            // N=2. Accum = 2.
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 6.0 * std::sqrt(2.0) * std::pow(2.0, -10)},
            // N=10. Accum = 10.
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 6.0 * std::sqrt(10.0) * std::pow(2.0, -10)}};
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
    // Setup
    std::vector<int64_t> dims = {10, 10, 10, 10};
    std::vector<int64_t> strides = {1000, 100, 10, 1};

    // Create tensors
    auto baseline = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);

    // Populate with values
    // Correct value: 1.0
    // Okay value: 1.5
    // Wrong value: 2.0

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.5f);
    actualFailing->fillTensorWithValue(2.0f);

    auto tol = calculateConvWrwTolerance<half, half, float>(-1.0, 1.0, -1.0, 1.0, dims);

    // tol approx .96~
    EXPECT_LT(tol, 1.0_h);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid) << "Validator should have passed";

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid) << "Validator should have failed";
}
