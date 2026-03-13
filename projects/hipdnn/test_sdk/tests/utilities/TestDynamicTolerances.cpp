// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::utilities::conv;
using namespace hipdnn_data_sdk::types;
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
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
            // N=2. Accum = 2. Tol = 2 * 2^2 * 2^-23 = 8 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 8.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
            // N=10. Accum = 10. Tol = 2 * 10^2 * 2^-23 = 200 * 2^-23
            // Exact gamma: (20 * 2^-23) / (1 - 20 * 2^-23) * 10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {10, 1, 1, 1},
             (20.0 * hipdnn_data_sdk::types::pow(2.0, -23))
                 / (1.0 - 20.0 * hipdnn_data_sdk::types::pow(2.0, -23)) * 10.0},
            // Large values: range -1000, 1000. maxProduct = 10^6.
            // N=10. Accum = 10. Tol = gamma * 10^7
            {-1000.0,
             1000.0,
             -1000.0,
             1000.0,
             {10, 1, 1, 1},
             (20.0 * hipdnn_data_sdk::types::pow(2.0, -23))
                 / (1.0 - 20.0 * hipdnn_data_sdk::types::pow(2.0, -23)) * 1.0e7}};
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
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 4.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
            // N=10. Accum = 10. Tol = (200 + 20) * 2^-23 = 220 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 220.0 * hipdnn_data_sdk::types::pow(2.0, -23)}};
}

// HipBfloat16 / Float / Float (High Precision Compute: Linear)
// Accumulation Error = 2 * N^2 * u_float * maxProduct
// Output Cast Error = N * maxProduct * u_bfp16
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // N=1. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1, 1},
         2.0 * hipdnn_data_sdk::types::pow(2.0, -23) + hipdnn_data_sdk::types::pow(2.0, -7)},
        // N=2. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {2, 1, 1, 1},
         8.0 * hipdnn_data_sdk::types::pow(2.0, -23) + 2.0 * hipdnn_data_sdk::types::pow(2.0, -7)},
        // N=10. Accum = 10. Tol = 200 * 2^-23 + 10 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         200.0 * hipdnn_data_sdk::types::pow(2.0, -23)
             + 10.0 * hipdnn_data_sdk::types::pow(2.0, -7)}};
}

// HipBfloat16 / HipBfloat16 / HipBfloat16 (Lower Precision: Statistical)
// Error = K * hipdnn_data_sdk::types::sqrt(2N) * u * (N * maxProduct) = K * N * hipdnn_data_sdk::types::sqrt(2N) * u * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    // 2^-7 = 0.0078125
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // N=1. Accum = 1. Tol = 6 * 1 * hipdnn_data_sdk::types::sqrt(2) * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1, 1},
         6.0 * hipdnn_data_sdk::types::sqrt(2.0) * hipdnn_data_sdk::types::pow(2.0, -7)},
        // N=2. Accum = 2. Tol = 6 * 2 * hipdnn_data_sdk::types::sqrt(4) * 2^-7 = 24 * 2^-7 = 0.1875
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 24.0 * hipdnn_data_sdk::types::pow(2.0, -7)},
        // N=10. Accum = 10. Tol = 6 * 10 * hipdnn_data_sdk::types::sqrt(20) * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         60.0 * hipdnn_data_sdk::types::sqrt(20.0) * hipdnn_data_sdk::types::pow(2.0, -7)}};
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
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1, 1},
         2.0 * hipdnn_data_sdk::types::pow(2.0, -23) + hipdnn_data_sdk::types::pow(2.0, -10)},
        // N=2. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {2, 1, 1, 1},
         8.0 * hipdnn_data_sdk::types::pow(2.0, -23) + 2.0 * hipdnn_data_sdk::types::pow(2.0, -10)},
        // N=10. Accum = 10. Tol = 200 * 2^-23 + 10 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         200.0 * hipdnn_data_sdk::types::pow(2.0, -23)
             + 10.0 * hipdnn_data_sdk::types::pow(2.0, -10)}};
}

// Half / Half / Half (Lower Precision: Statistical)
// Error = K * N * hipdnn_data_sdk::types::sqrt(2N) * u * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypeTriple<half, half, half>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1. Tol = 6 * 1 * hipdnn_data_sdk::types::sqrt(2) * 2^-10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 1, 1, 1},
             6.0 * hipdnn_data_sdk::types::sqrt(2.0) * hipdnn_data_sdk::types::pow(2.0, -10)},
            // N=2. Accum = 2. Tol = 6 * 2 * hipdnn_data_sdk::types::sqrt(4) * 2^-10 = 24 * 2^-10
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 24.0 * hipdnn_data_sdk::types::pow(2.0, -10)},
            // N=10. Accum = 10. Tol = 6 * 10 * hipdnn_data_sdk::types::sqrt(20) * 2^-10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {10, 1, 1, 1},
             60.0 * hipdnn_data_sdk::types::sqrt(20.0) * hipdnn_data_sdk::types::pow(2.0, -10)}};
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
                std::invalid_argument);
        }
        else
        {
            auto tol = calculateConvWrwTolerance<Out, In, Comp>(
                params.inputMin, params.inputMax, params.dyMin, params.dyMax, params.dyDims);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(tol, expected, 1e-5f);
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
    = TestCalculateConvWrwTolerance<bfloat16, float, float>;
TEST_P(TestCalculateConvWrwToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateConvWrwToleranceBfp16
    = TestCalculateConvWrwTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateConvWrwToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceBfp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

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
    const std::vector<int64_t> dims = {1, 1, 10, 10};
    const std::vector<int64_t> strides = {100, 100, 10, 1};

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
    EXPECT_LT(tol, 0.15f);
    EXPECT_GT(tol, 0.09f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateConvWrwTolerance throws when nU >= 1.0 (singularity)
TEST(TestCalculateConvWrwTolerance, ThrowsOnSingularity)
{
    // For float, epsilon is 2^-23 approx 1.19e-7.
    // nU = 2 * n * epsilon.
    // We need nU >= 1.0 => n >= 1 / (2 * epsilon) = 2^22 = 4,194,304.
    // Let's use 5,000,000.
    const std::vector<int64_t> dims = {5000000, 1, 1, 1};

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

    const std::vector<int64_t> dims = {10, 1, 1, 1};
    const double val = 1.0e5;

    EXPECT_THROW((calculateConvWrwTolerance<half, float, float>(-val, val, -val, val, dims)),
                 std::overflow_error);
}

// =================================================================================================
// TestCalculateConvDgradTolerance
// =================================================================================================

struct ConvDgradToleranceTestCase
{
    double dyMin;
    double dyMax;
    double wMin;
    double wMax;
    std::vector<int64_t> wDims;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const ConvDgradToleranceTestCase& tc)
    {
        os << "dyMin: " << tc.dyMin << ", dyMax: " << tc.dyMax << ", wMin: " << tc.wMin
           << ", wMax: " << tc.wMax << ", wDims: [";
        for(size_t i = 0; i < tc.wDims.size(); ++i)
        {
            os << tc.wDims[i] << (i < tc.wDims.size() - 1 ? ", " : "");
        }
        os << "], expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<ConvDgradToleranceTestCase> getConvDgradToleranceTestCases();

// Float / Float / Float (High Precision: Linear)
// For conv dgrad: dx[n, c, h, w] = sum_{k, r, s} dy[n, k, p, q] * w[k, c, r, s]
// Accumulations = K * R * S
// Error = 2 * Accum^2 * u * maxProduct
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<float, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, 0.0, true},
        // wDims=[K=1, C=1, R=1, S=1]. Accum = 1 * 1 * 1 = 1. Tol = 2 * 1^2 * 2^-23 = 2 * 2^-23
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
        // wDims=[K=2, C=1, R=1, S=1]. Accum = 2. Tol = 2 * 2^2 * 2^-23 = 8 * 2^-23
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 8.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
        // wDims=[K=1, C=1, R=3, S=3]. Accum = 1 * 3 * 3 = 9. Tol = 2 * 9^2 * 2^-23 = 162 * 2^-23
        // Exact gamma: (18 * 2^-23) / (1 - 18 * 2^-23) * 9
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 3, 3},
         (18.0 * hipdnn_data_sdk::types::pow(2.0, -23))
             / (1.0 - 18.0 * hipdnn_data_sdk::types::pow(2.0, -23)) * 9.0},
        // wDims=[K=16, C=16, R=3, S=3]. Accum = 16 * 3 * 3 = 144.
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {16, 16, 3, 3},
         (288.0 * hipdnn_data_sdk::types::pow(2.0, -23))
             / (1.0 - 288.0 * hipdnn_data_sdk::types::pow(2.0, -23)) * 144.0},
        // 3D Convolution: wDims=[K=8, C=8, D=3, R=3, S=3]. Accum = 8 * 3 * 3 * 3 = 216.
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {8, 8, 3, 3, 3},
         (432.0 * hipdnn_data_sdk::types::pow(2.0, -23))
             / (1.0 - 432.0 * hipdnn_data_sdk::types::pow(2.0, -23)) * 216.0},
        // Large values: range -1000, 1000. maxProduct = 10^6.
        // wDims=[K=16, C=16, R=3, S=3]. Accum = 144.
        {-1000.0,
         1000.0,
         -1000.0,
         1000.0,
         {16, 16, 3, 3},
         (288.0 * hipdnn_data_sdk::types::pow(2.0, -23))
             / (1.0 - 288.0 * hipdnn_data_sdk::types::pow(2.0, -23)) * 144.0 * 1.0e6}};
}

// Float / Double / Float (Input casting error)
// Input is double, Compute is float. We lose precision.
// Accumulation Error = 2 * Accum^2 * u * maxProduct
// Casting Error = 2 * Accum * maxProduct * u
// Total = (2 * Accum^2 + 2 * Accum) * u * maxProduct
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<float, double, float>>()
{
    return {// wDims=[1, 1, 1, 1]. Accum = 1. Tol = (2 + 2) * 2^-23 = 4 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 4.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
            // wDims=[1, 1, 3, 3]. Accum = 9. Tol = (162 + 18) * 2^-23 = 180 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 3, 3}, 180.0 * hipdnn_data_sdk::types::pow(2.0, -23)}};
}

// HipBfloat16 / Float / Float (High Precision Compute: Linear)
// Accumulation Error = 2 * Accum^2 * u_float * maxProduct
// Output Cast Error = Accum * maxProduct * u_bfp16
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, 0.0, true},
        // wDims=[1, 1, 1, 1]. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1, 1},
         2.0 * hipdnn_data_sdk::types::pow(2.0, -23) + hipdnn_data_sdk::types::pow(2.0, -7)},
        // wDims=[2, 1, 1, 1]. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {2, 1, 1, 1},
         8.0 * hipdnn_data_sdk::types::pow(2.0, -23) + 2.0 * hipdnn_data_sdk::types::pow(2.0, -7)},
        // wDims=[1, 1, 3, 3]. Accum = 9. Tol = 162 * 2^-23 + 9 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 3, 3},
         162.0 * hipdnn_data_sdk::types::pow(2.0, -23)
             + 9.0 * hipdnn_data_sdk::types::pow(2.0, -7)}};
}

// HipBfloat16 / HipBfloat16 / HipBfloat16 (Lower Precision: Statistical)
// Error = K * sqrt(2*Accum) * u * (Accum * maxProduct) = K * Accum * sqrt(2*Accum) * u * maxProduct
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, 0.0, true},
            // wDims=[1, 1, 1, 1]. Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-7
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 1, 1, 1},
             6.0 * hipdnn_data_sdk::types::sqrt(2.0) * hipdnn_data_sdk::types::pow(2.0, -7)},
            // wDims=[2, 1, 1, 1]. Accum = 2. Tol = 6 * 2 * sqrt(4) * 2^-7 = 24 * 2^-7
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 24.0 * hipdnn_data_sdk::types::pow(2.0, -7)},
            // wDims=[1, 1, 3, 3]. Accum = 9. Tol = 6 * 9 * sqrt(18) * 2^-7
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 1, 3, 3},
             54.0 * hipdnn_data_sdk::types::sqrt(18.0) * hipdnn_data_sdk::types::pow(2.0, -7)}};
}

// Half / Float / Float (High Precision Compute: Linear)
// Accumulation Error = 2 * Accum^2 * u_float * maxProduct
// Output Cast Error = Accum * maxProduct * u_half
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<half, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, 0.0, true},
        // wDims=[1, 1, 1, 1]. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1, 1},
         2.0 * hipdnn_data_sdk::types::pow(2.0, -23) + hipdnn_data_sdk::types::pow(2.0, -10)},
        // wDims=[2, 1, 1, 1]. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {2, 1, 1, 1},
         8.0 * hipdnn_data_sdk::types::pow(2.0, -23) + 2.0 * hipdnn_data_sdk::types::pow(2.0, -10)},
        // wDims=[1, 1, 3, 3]. Accum = 9. Tol = 162 * 2^-23 + 9 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 3, 3},
         162.0 * hipdnn_data_sdk::types::pow(2.0, -23)
             + 9.0 * hipdnn_data_sdk::types::pow(2.0, -10)}};
}

// Half / Half / Half (Lower Precision: Statistical)
// Error = K * Accum * sqrt(2*Accum) * u * maxProduct
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<half, half, half>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, 0.0, true},
            // wDims=[1, 1, 1, 1]. Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 1, 1, 1},
             6.0 * hipdnn_data_sdk::types::sqrt(2.0) * hipdnn_data_sdk::types::pow(2.0, -10)},
            // wDims=[2, 1, 1, 1]. Accum = 2. Tol = 6 * 2 * sqrt(4) * 2^-10 = 24 * 2^-10
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 24.0 * hipdnn_data_sdk::types::pow(2.0, -10)},
            // wDims=[1, 1, 3, 3]. Accum = 9. Tol = 6 * 9 * sqrt(18) * 2^-10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 1, 3, 3},
             54.0 * hipdnn_data_sdk::types::sqrt(18.0) * hipdnn_data_sdk::types::pow(2.0, -10)}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateConvDgradTolerance : public ::testing::TestWithParam<ConvDgradToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW((calculateConvDgradTolerance<Out, In, Comp>(
                             params.dyMin, params.dyMax, params.wMin, params.wMax, params.wDims)),
                         std::invalid_argument);
        }
        else
        {
            auto tol = calculateConvDgradTolerance<Out, In, Comp>(
                params.dyMin, params.dyMax, params.wMin, params.wMax, params.wDims);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(tol, expected, 1e-5f);
        }
    }
};

using TestCalculateConvDgradToleranceFp32 = TestCalculateConvDgradTolerance<float, float, float>;
TEST_P(TestCalculateConvDgradToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvDgradToleranceFp32,
    ::testing::ValuesIn(getConvDgradToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateConvDgradToleranceInputDouble
    = TestCalculateConvDgradTolerance<float, double, float>;
TEST_P(TestCalculateConvDgradToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvDgradToleranceInputDouble,
    ::testing::ValuesIn(getConvDgradToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateConvDgradToleranceComputeFloatBfp16
    = TestCalculateConvDgradTolerance<bfloat16, float, float>;
TEST_P(TestCalculateConvDgradToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvDgradToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getConvDgradToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateConvDgradToleranceBfp16
    = TestCalculateConvDgradTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateConvDgradToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvDgradToleranceBfp16,
    ::testing::ValuesIn(
        getConvDgradToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

using TestCalculateConvDgradToleranceComputeFloatFp16
    = TestCalculateConvDgradTolerance<half, float, float>;
TEST_P(TestCalculateConvDgradToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvDgradToleranceComputeFloatFp16,
    ::testing::ValuesIn(getConvDgradToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateConvDgradToleranceFp16 = TestCalculateConvDgradTolerance<half, half, half>;
TEST_P(TestCalculateConvDgradToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvDgradToleranceFp16,
    ::testing::ValuesIn(getConvDgradToleranceTestCases<TypeTriple<half, half, half>>()));

// Test that calculateConvDgradTolerance catches simulated wrong outputs
TEST(TestCalculateConvDgradTolerance, DetectsFailure)
{
    // wDims=[K=10, C=1, R=10, S=10] => Accumulations = 10 * 10 * 10 = 1000
    const std::vector<int64_t> dims = {1, 1, 10, 10};
    const std::vector<int64_t> strides = {100, 100, 10, 1};

    // Create tensors
    auto baseline = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);

    // Populate with values
    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.05f); // Small error
    actualFailing->fillTensorWithValue(2.5f); // Large error

    const std::vector<int64_t> wDims = {10, 1, 10, 10}; // Accum = 1000
    auto tol = calculateConvDgradTolerance<half, half, float>(-1.0, 1.0, -1.0, 1.0, wDims);

    // tol should be reasonable for 1000 accumulations with half/half/float
    // For half precision with statistical bound, can be > 1.0
    EXPECT_LT(tol, 2.0f);
    EXPECT_GT(tol, 0.1f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateConvDgradTolerance throws when nU >= 1.0 (singularity)
TEST(TestCalculateConvDgradTolerance, ThrowsOnSingularity)
{
    // For float, epsilon is 2^-23 approx 1.19e-7.
    // nU = 2 * n * epsilon.
    // We need nU >= 1.0 => n >= 1 / (2 * epsilon) = 2^22 = 4,194,304.
    // wDims=[K=2048, C=1, R=2048, S=1] => Accum = 2048 * 2048 = 4,194,304.
    const std::vector<int64_t> wDims = {2048, 1, 2048, 1};

    EXPECT_THROW((calculateConvDgradTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, wDims)),
                 std::overflow_error);
}

// Test that calculateConvDgradTolerance throws when tolerance exceeds OutputType max
TEST(TestCalculateConvDgradTolerance, ThrowsOnOutputOverflow)
{
    // OutputType = half (max approx 65504)
    // ComputeType = float
    // wDims=[K=10, C=1, R=1, S=1] => Accum = 10
    // maxProduct = 1e10 (dy 1e5 * w 1e5)
    // sumAbsProductBound = 1e11
    // epsilon (float) approx 1.19e-7
    // gamma approx 2 * 10 * 1.19e-7 approx 2.38e-6
    // accumulatedTolerance approx 2.38e-6 * 1e11 approx 2.38e5 = 238,000
    // 238,000 > 65,504 => Should throw.

    const std::vector<int64_t> wDims = {10, 1, 1, 1};
    const double val = 1.0e5;

    EXPECT_THROW((calculateConvDgradTolerance<half, float, float>(-val, val, -val, val, wDims)),
                 std::overflow_error);
}
// =================================================================================================
// TestCalculateConvFpropTolerance
// =================================================================================================

struct ConvFpropToleranceTestCase
{
    double inputMin;
    double inputMax;
    double wMin;
    double wMax;
    std::vector<int64_t> wDims;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const ConvFpropToleranceTestCase& tc)
    {
        os << "inputMin: " << tc.inputMin << ", inputMax: " << tc.inputMax << ", wMin: " << tc.wMin
           << ", wMax: " << tc.wMax << ", wDims: [";
        for(size_t i = 0; i < tc.wDims.size(); ++i)
        {
            os << tc.wDims[i] << (i < tc.wDims.size() - 1 ? ", " : "");
        }
        os << "], expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<ConvFpropToleranceTestCase> getConvFpropToleranceTestCases();

// Float / Float / Float (High Precision: Linear)
// Error = 2 * N^2 * u * maxProduct
// For fprop: N = C * R * S (input channels × filter spatial dims)
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<float, float, float>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // C=1, R=1, S=1. Accum = 1. Tol = 2 * 1^2 * 2^-23 = 2 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            // C=2, R=1, S=1. Accum = 2. Tol = 2 * 2^2 * 2^-23 = 8 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 2, 1, 1}, 8.0 * std::pow(2.0, -23)},
            // C=10, R=1, S=1. Accum = 10. Tol = (20 * 2^-23) / (1 - 20 * 2^-23) * 10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 10, 1, 1},
             (20.0 * std::pow(2.0, -23)) / (1.0 - 20.0 * std::pow(2.0, -23)) * 10.0},
            // C=3, R=3, S=3. Accum = 27. Tol = (54 * 2^-23) / (1 - 54 * 2^-23) * 27
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 3, 3, 3},
             (54.0 * std::pow(2.0, -23)) / (1.0 - 54.0 * std::pow(2.0, -23)) * 27.0},
            // Large values: range -1000, 1000. maxProduct = 10^6.
            // C=10, R=1, S=1. Accum = 10. Tol = gamma * 10^7
            {-1000.0,
             1000.0,
             -1000.0,
             1000.0,
             {1, 10, 1, 1},
             (20.0 * std::pow(2.0, -23)) / (1.0 - 20.0 * std::pow(2.0, -23)) * 1.0e7},
            // 3D Convolution (5D tensors): C=1, D=1, R=1, S=1. Accum = 1. Tol = 2 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            // 3D Convolution: C=2, D=2, R=2, S=2. Accum = 16. Tol = 2 * 16^2 * 2^-23 = 512 * 2^-23
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 2, 2, 2, 2},
             (32.0 * std::pow(2.0, -23)) / (1.0 - 32.0 * std::pow(2.0, -23)) * 16.0},
            // 3D Convolution: C=3, D=3, R=3, S=3. Accum = 81. Tol = gamma * 81
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 3, 3, 3, 3},
             (162.0 * std::pow(2.0, -23)) / (1.0 - 162.0 * std::pow(2.0, -23)) * 81.0}};
}

// Float / Double / Float (Input casting error)
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<float, double, float>>()
{
    return {// C=1, R=1, S=1. Accum = 1. Tol = (2 + 2) * 2^-23 = 4 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 4.0 * std::pow(2.0, -23)},
            // C=10, R=1, S=1. Accum = 10. Tol = (200 + 20) * 2^-23 = 220 * 2^-23
            {-1.0, 1.0, -1.0, 1.0, {1, 10, 1, 1}, 220.0 * std::pow(2.0, -23)}};
}

// HipBfloat16 / Float / Float (High Precision Compute: Linear)
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // C=1, R=1, S=1. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23) + std::pow(2.0, -7)},
        // C=2, R=1, S=1. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 2, 1, 1}, 8.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -7)},
        // C=10, R=1, S=1. Accum = 10. Tol = 200 * 2^-23 + 10 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 10, 1, 1},
         200.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -7)},
        // 3D Convolution: C=2, D=2, R=2, S=2. Accum = 16. Tol = 512 * 2^-23 + 16 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 2, 2, 2, 2},
         512.0 * std::pow(2.0, -23) + 16.0 * std::pow(2.0, -7)}};
}

// HipBfloat16 / HipBfloat16 / HipBfloat16 (Lower Precision: Statistical)
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // C=1, R=1, S=1. Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 6.0 * 1.0 * std::sqrt(2.0) * std::pow(2.0, -7)},
        // C=2, R=1, S=1. Accum = 2. Tol = 6 * 2 * sqrt(4) * 2^-7 = 24 * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 2, 1, 1}, 6.0 * 2.0 * std::sqrt(4.0) * std::pow(2.0, -7)},
        // C=10, R=1, S=1. Accum = 10. Tol = 6 * 10 * sqrt(20) * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 10, 1, 1}, 6.0 * 10.0 * std::sqrt(20.0) * std::pow(2.0, -7)},
        // 3D Convolution: C=2, D=2, R=2, S=2. Accum = 16. Tol = 6 * 16 * sqrt(32) * 2^-7
        {-1.0, 1.0, -1.0, 1.0, {1, 2, 2, 2, 2}, 6.0 * 16.0 * std::sqrt(32.0) * std::pow(2.0, -7)}};
}

// Half / Float / Float (High Precision Compute: Linear)
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<half, float, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // C=1, R=1, S=1. Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23) + std::pow(2.0, -10)},
        // C=2, R=1, S=1. Accum = 2. Tol = 8 * 2^-23 + 2 * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 2, 1, 1}, 8.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -10)},
        // C=10, R=1, S=1. Accum = 10. Tol = 200 * 2^-23 + 10 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 10, 1, 1},
         200.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -10)},
        // 3D Convolution: C=2, D=2, R=2, S=2. Accum = 16. Tol = 512 * 2^-23 + 16 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 2, 2, 2, 2},
         512.0 * std::pow(2.0, -23) + 16.0 * std::pow(2.0, -10)}};
}

// Half / Half / Half (Lower Precision: Statistical)
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<half, half, half>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        // C=1, R=1, S=1. Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 6.0 * std::sqrt(2.0) * std::pow(2.0, -10)},
        // C=2, R=1, S=1. Accum = 2. Tol = 6 * 2 * sqrt(4) * 2^-10 = 24 * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 2, 1, 1}, 24.0 * std::pow(2.0, -10)},
        // C=10, R=1, S=1. Accum = 10. Tol = 6 * 10 * sqrt(20) * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 10, 1, 1}, 60.0 * std::sqrt(20.0) * std::pow(2.0, -10)},
        // 3D Convolution: C=2, D=2, R=2, S=2. Accum = 16. Tol = 6 * 16 * sqrt(32) * 2^-10
        {-1.0, 1.0, -1.0, 1.0, {1, 2, 2, 2, 2}, 6.0 * 16.0 * std::sqrt(32.0) * std::pow(2.0, -10)}};
}

// Test fixture for ConvFprop tolerance
template <typename Out, typename In, typename Comp>
class TestCalculateConvFpropTolerance : public ::testing::TestWithParam<ConvFpropToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateConvFpropTolerance<Out, In, Comp>(
                    params.inputMin, params.inputMax, params.wMin, params.wMax, params.wDims)),
                std::invalid_argument)
                << "Failed to throw for dims size: " << params.wDims.size();
        }
        else
        {
            auto tol = calculateConvFpropTolerance<Out, In, Comp>(
                params.inputMin, params.inputMax, params.wMin, params.wMax, params.wDims);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(static_cast<float>(tol), expected, 1e-5f)
                << "Failed for dims size: " << params.wDims.size();
        }
    }
};

// Test cases for different type combinations
using TestCalculateConvFpropToleranceFp32 = TestCalculateConvFpropTolerance<float, float, float>;
TEST_P(TestCalculateConvFpropToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvFpropToleranceFp32,
    ::testing::ValuesIn(getConvFpropToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateConvFpropToleranceInputDouble
    = TestCalculateConvFpropTolerance<float, double, float>;
TEST_P(TestCalculateConvFpropToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvFpropToleranceInputDouble,
    ::testing::ValuesIn(getConvFpropToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateConvFpropToleranceComputeFloatBfp16
    = TestCalculateConvFpropTolerance<bfloat16, float, float>;
TEST_P(TestCalculateConvFpropToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvFpropToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getConvFpropToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateConvFpropToleranceBfp16
    = TestCalculateConvFpropTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateConvFpropToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvFpropToleranceBfp16,
    ::testing::ValuesIn(
        getConvFpropToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

using TestCalculateConvFpropToleranceComputeFloatFp16
    = TestCalculateConvFpropTolerance<half, float, float>;
TEST_P(TestCalculateConvFpropToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvFpropToleranceComputeFloatFp16,
    ::testing::ValuesIn(getConvFpropToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateConvFpropToleranceFp16 = TestCalculateConvFpropTolerance<half, half, half>;
TEST_P(TestCalculateConvFpropToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvFpropToleranceFp16,
    ::testing::ValuesIn(getConvFpropToleranceTestCases<TypeTriple<half, half, half>>()));

// Test that calculateConvFpropTolerance catches simulated wrong outputs
TEST(TestCalculateConvFpropTolerance, DetectsFailure)
{
    // C=10, R=10, S=1 => Accumulations = 100
    const std::vector<int64_t> dims = {1, 10, 10, 1};
    const std::vector<int64_t> strides = {100, 100, 10, 1};

    // Create tensors
    auto baseline = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, dims, strides);

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.05f);
    actualFailing->fillTensorWithValue(1.2f);

    auto tol = calculateConvFpropTolerance<half, half, float>(-1.0, 1.0, -1.0, 1.0, dims);

    // tol approx 0.1
    EXPECT_LT(tol, 0.15f);
    EXPECT_GT(tol, 0.09f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_data_sdk::data_objects::DataType::FLOAT, tol, 0);

    {
        SCOPED_TRACE("Validator should have passed");
        const bool valid = validator->allClose(*baseline, *actualPassing);
        EXPECT_TRUE(valid);
    }
    {
        SCOPED_TRACE("Validator should have failed");
        const bool valid = validator->allClose(*baseline, *actualFailing);
        EXPECT_FALSE(valid);
    }
}

// Test that calculateConvFpropTolerance throws when nU >= 1.0 (singularity)
TEST(TestCalculateConvFpropTolerance, ThrowsOnSingularity)
{
    // For float, epsilon is 2^-23 approx 1.19e-7.
    // nU = 2 * n * epsilon.
    // We need nU >= 1.0 => n >= 1 / (2 * epsilon) = 2^22 = 4,194,304.
    // Let's use C=5,000,000, R=1, S=1 => accumulations = 5,000,000.
    const std::vector<int64_t> dims = {1, 5000000, 1, 1};

    EXPECT_THROW((calculateConvFpropTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, dims)),
                 std::overflow_error);
}

// Test that calculateConvFpropTolerance throws when tolerance exceeds OutputType max
TEST(TestCalculateConvFpropTolerance, ThrowsOnOutputOverflow)
{
    // OutputType = half (max approx 65504)
    // ComputeType = float
    // We need tolerance > 65504.
    // Let C=10, R=1, S=1 => accumulations = 10.
    // maxProduct = 1e10 (input 1e5 * w 1e5)
    // sumAbsProductBound = 1e11
    // epsilon (float) approx 1.19e-7
    // gamma approx 2 * 10 * 1.19e-7 approx 2.38e-6
    // accumulatedTolerance approx 2.38e-6 * 1e11 approx 2.38e5 = 238,000
    // 238,000 > 65,504 => Should throw.

    const std::vector<int64_t> dims = {1, 10, 1, 1};
    const double val = 1.0e5;

    EXPECT_THROW((calculateConvFpropTolerance<half, float, float>(-val, val, -val, val, dims)),
                 std::overflow_error);
}

// =================================================================================================
// TestCalculateMatmulTolerance
// =================================================================================================

using namespace hipdnn_test_sdk::utilities::matmul;

struct MatmulToleranceTestCase
{
    std::vector<int64_t> aDims;
    std::vector<int64_t> bDims;
    double aValue;
    double bValue;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const MatmulToleranceTestCase& tc)
    {
        os << "aDims: [";
        for(size_t i = 0; i < tc.aDims.size(); ++i)
        {
            os << tc.aDims[i] << (i < tc.aDims.size() - 1 ? ", " : "");
        }
        os << "], bDims: [";
        for(size_t i = 0; i < tc.bDims.size(); ++i)
        {
            os << tc.bDims[i] << (i < tc.bDims.size() - 1 ? ", " : "");
        }
        os << "], aValue: " << tc.aValue << ", bValue: " << tc.bValue
           << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

// Helper to create a tensor filled with a constant value
template <typename T>
hipdnn_data_sdk::utilities::Tensor<T> createConstantTensor(const std::vector<int64_t>& dims,
                                                           double value)
{
    hipdnn_data_sdk::utilities::Tensor<T> tensor(dims);
    auto elementCount = tensor.elementCount();
    auto* data = static_cast<T*>(tensor.memory().hostData());

    for(size_t i = 0; i < elementCount; ++i)
    {
        data[i] = static_cast<T>(value);
    }

    return tensor;
}

using hipdnn_test_sdk::utilities::computeGamma;

template <typename T>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases();

// Float / Float / Float (High Precision: Linear)
// Tolerance = gamma(K, u_float) * ||A||_1 * ||B||_1
// All tensors filled with 1.0, so ||T||_1 = product(dims)
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = hipdnn_data_sdk::types::pow(2.0, -23);

    return {// K=1: A=2x1, B=1x2. ||A||=2, ||B||=2. Tol = gamma(1,u) * 4
            {{2, 1}, {1, 2}, 1.0, 1.0, computeGamma(1, u) * 2.0 * 2.0},
            // K=3: A=2x3, B=3x4. ||A||=6, ||B||=12. Tol = gamma(3,u) * 72
            {{2, 3}, {3, 4}, 1.0, 1.0, computeGamma(3, u) * 6.0 * 12.0},
            // K=10: A=2x10, B=10x2. ||A||=20, ||B||=20. Tol = gamma(10,u) * 400
            {{2, 10}, {10, 2}, 1.0, 1.0, computeGamma(10, u) * 20.0 * 20.0},
            // K=100: A=2x100, B=100x2. ||A||=200, ||B||=200. Tol = gamma(100,u) * 40000
            {{2, 100}, {100, 2}, 1.0, 1.0, computeGamma(100, u) * 200.0 * 200.0}};
}

// Float / Double / Float (Input casting error)
// Tolerance = gamma(K, u_float) * ||A|| * ||B|| + 2 * ||A|| * ||B|| * u_float
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = hipdnn_data_sdk::types::pow(2.0, -23);

    return {// K=1: ||A||=2, ||B||=2. Tol = gamma(1,u)*4 + 2*4*u
            {{2, 1}, {1, 2}, 1.0, 1.0, computeGamma(1, u) * 4.0 + 2.0 * 4.0 * u},
            // K=10: ||A||=20, ||B||=20. Tol = gamma(10,u)*400 + 2*400*u
            {{2, 10}, {10, 2}, 1.0, 1.0, computeGamma(10, u) * 400.0 + 2.0 * 400.0 * u}};
}

// Half / Float / Float (Output casting error)
// Tolerance = gamma(K, u_float) * ||A|| * ||B|| + ||A|| * ||B|| * u_half
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto uFloat = hipdnn_data_sdk::types::pow(2.0, -23);
    auto uHalf = hipdnn_data_sdk::types::pow(2.0, -10);

    return {// K=1: ||A||=2, ||B||=2. Tol = gamma(1,uFloat)*4 + 4*uHalf
            {{2, 1}, {1, 2}, 1.0, 1.0, computeGamma(1, uFloat) * 4.0 + 4.0 * uHalf},
            // K=10: ||A||=20, ||B||=20. Tol = gamma(10,uFloat)*400 + 400*uHalf
            {{2, 10}, {10, 2}, 1.0, 1.0, computeGamma(10, uFloat) * 400.0 + 400.0 * uHalf}};
}

// Half / Half / Half (Low Precision: Statistical)
// Tolerance = gamma(K, u_half) * ||A|| * ||B||
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = hipdnn_data_sdk::types::pow(2.0, -10);

    return {// K=1: ||A||=2, ||B||=2
            {{2, 1}, {1, 2}, 1.0, 1.0, computeGamma(1, u) * 4.0},
            // K=10: ||A||=20, ||B||=20
            {{2, 10}, {10, 2}, 1.0, 1.0, computeGamma(10, u) * 400.0},
            // K=100: ||A||=200, ||B||=200
            {{2, 100}, {100, 2}, 1.0, 1.0, computeGamma(100, u) * 40000.0}};
}

// Bfloat16 / Float / Float (Output casting error)
// Tolerance = gamma(K, u_float) * ||A|| * ||B|| + ||A|| * ||B|| * u_bf16
template <>
std::vector<MatmulToleranceTestCase>
    getMatmulToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto uFloat = hipdnn_data_sdk::types::pow(2.0, -23);
    auto uBf16 = hipdnn_data_sdk::types::pow(2.0, -7);

    return {// K=1: ||A||=2, ||B||=2. Tol = gamma(1,uFloat)*4 + 4*uBf16
            {{2, 1}, {1, 2}, 1.0, 1.0, computeGamma(1, uFloat) * 4.0 + 4.0 * uBf16},
            // K=10: ||A||=20, ||B||=20. Tol = gamma(10,uFloat)*400 + 400*uBf16
            {{2, 10}, {10, 2}, 1.0, 1.0, computeGamma(10, uFloat) * 400.0 + 400.0 * uBf16}};
}

// Bfloat16 / Bfloat16 / Bfloat16 (Low Precision: Statistical)
// Tolerance = gamma(K, u_bf16) * ||A|| * ||B||
template <>
std::vector<MatmulToleranceTestCase>
    getMatmulToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = hipdnn_data_sdk::types::pow(2.0, -7);

    return {// K=1: ||A||=2, ||B||=2
            {{2, 1}, {1, 2}, 1.0, 1.0, computeGamma(1, u) * 4.0},
            // K=10: ||A||=20, ||B||=20
            {{2, 10}, {10, 2}, 1.0, 1.0, computeGamma(10, u) * 400.0},
            // K=50: ||A||=100, ||B||=100 (K=100 exceeds gamma>=0.5 for bf16)
            {{2, 50}, {50, 2}, 1.0, 1.0, computeGamma(50, u) * 10000.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateMatmulTolerance : public ::testing::TestWithParam<MatmulToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& tc = GetParam();

        if(tc.expectThrow)
        {
            auto a = createConstantTensor<In>(tc.aDims, tc.aValue);
            auto b = createConstantTensor<In>(tc.bDims, tc.bValue);

            EXPECT_THROW((calculateMatmulTolerance<Out, In, Comp>(a, b)), std::exception) << tc;
        }
        else
        {
            auto a = createConstantTensor<In>(tc.aDims, tc.aValue);
            auto b = createConstantTensor<In>(tc.bDims, tc.bValue);

            auto tolerance = calculateMatmulTolerance<Out, In, Comp>(a, b);
            EXPECT_NEAR(tolerance, static_cast<float>(tc.expectedTolerance), 1e-10f) << tc;
        }
    }
};

using TestCalculateMatmulToleranceFp32 = TestCalculateMatmulTolerance<float, float, float>;
TEST_P(TestCalculateMatmulToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateMatmulToleranceFp32,
    ::testing::ValuesIn(getMatmulToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateMatmulToleranceInputDouble = TestCalculateMatmulTolerance<float, double, float>;
TEST_P(TestCalculateMatmulToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateMatmulToleranceInputDouble,
    ::testing::ValuesIn(getMatmulToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateMatmulToleranceComputeFloatFp16
    = TestCalculateMatmulTolerance<half, float, float>;
TEST_P(TestCalculateMatmulToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateMatmulToleranceComputeFloatFp16,
    ::testing::ValuesIn(getMatmulToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateMatmulToleranceFp16 = TestCalculateMatmulTolerance<half, half, half>;
TEST_P(TestCalculateMatmulToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateMatmulToleranceFp16,
    ::testing::ValuesIn(getMatmulToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalculateMatmulToleranceComputeFloatBfp16
    = TestCalculateMatmulTolerance<bfloat16, float, float>;
TEST_P(TestCalculateMatmulToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateMatmulToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getMatmulToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateMatmulToleranceBfp16
    = TestCalculateMatmulTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateMatmulToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateMatmulToleranceBfp16,
    ::testing::ValuesIn(getMatmulToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// Test dimension validation
TEST(TestCalculateMatmulTolerance, ThrowsOnInvalidDimensions)
{
    // Mismatched dimensions: A is 2x3, B is 4x5 (3 != 4)
    auto a = createConstantTensor<float>({2, 3}, 1.0);
    auto b = createConstantTensor<float>({4, 5}, 1.0);

    EXPECT_THROW((calculateMatmulTolerance<float, float, float>(a, b)), std::invalid_argument);
}

// Note: K=0 validation test removed. Tensor constructor's validateAllPositive() rejects
// dimension <= 0 before calculateMatmulTolerance is reached, so K=0 cannot be tested
// through the public API with real tensors.

// Test that large K with low-precision type causes gamma >= 0.5 overflow
TEST(TestCalculateMatmulTolerance, ThrowsOnSingularity)
{
    // For bfloat16, epsilon = 2^-7 ≈ 7.81e-3
    // K=100: nU = 2*100*7.81e-3 = 1.562 >= 0.01 → statistical bound
    // gamma = 6 * sqrt(200) * 7.81e-3 ≈ 0.663 >= 0.5 → overflow
    auto a = createConstantTensor<bfloat16>({2, 100}, 1.0);
    auto b = createConstantTensor<bfloat16>({100, 2}, 1.0);

    EXPECT_THROW((calculateMatmulTolerance<bfloat16, bfloat16, bfloat16>(a, b)),
                 std::overflow_error);
}

// Test that extreme values cause output overflow
TEST(TestCalculateMatmulTolerance, ThrowsOnOutputOverflow)
{
    // Create matrices with very large values that will cause tolerance > half::max
    // A = 2x10 filled with 1e5, B = 10x2 filled with 1e5
    // ||A||_1 = 20 * 1e5 = 2e6, ||B||_1 = 20 * 1e5 = 2e6
    // Product will exceed half max (65504)
    auto a = createConstantTensor<float>({2, 10}, 1.0e5);
    auto b = createConstantTensor<float>({10, 2}, 1.0e5);

    // OutputType = half, so max ≈ 65504
    EXPECT_THROW((calculateMatmulTolerance<half, float, float>(a, b)), std::overflow_error);
}
