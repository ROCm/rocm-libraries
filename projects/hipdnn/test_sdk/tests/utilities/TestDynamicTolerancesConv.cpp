// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <limits>
#include <random>
#include <stdexcept>
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

// Half / Half / Half
// Tolerance = computeGamma(k, u_half) * k * maxProduct
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());

    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // N=1. Accum = 1. sumAbsProductBound = 1
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, computeGamma(1, u) * 1.0},
            // N=2. Accum = 2. sumAbsProductBound = 2
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, computeGamma(2, u) * 2.0},
            // N=10. Accum = 10. sumAbsProductBound = 10
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, computeGamma(10, u) * 10.0}};
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
    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

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
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateConvWrwTolerance throws when gamma >= 0.5 (singularity)
TEST(TestCalculateConvWrwTolerance, ThrowsOnSingularity)
{
    // For bfloat16, epsilon = 2^-7 ≈ 7.81e-3
    // N=100: nU = 2*100*7.81e-3 = 1.562 >= 0.01 → statistical bound
    // gamma = 6 * sqrt(200) * 7.81e-3 ≈ 0.663 >= 0.5 → overflow
    const std::vector<int64_t> dims = {100, 1, 1, 1};

    EXPECT_THROW(
        (calculateConvWrwTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, dims)),
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
        // wDims=[K=1, C=1, R=1] (1D). Accum = K * R = 1. Tol = 2 * 1^2 * 2^-23 = 2 * 2^-23
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, 2.0 * hipdnn_data_sdk::types::pow(2.0, -23)},
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
        // wDims=[K=1, C=1, R=1] (1D). Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-7
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1},
         2.0 * hipdnn_data_sdk::types::pow(2.0, -23) + hipdnn_data_sdk::types::pow(2.0, -7)},
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
            // wDims=[K=1, C=1, R=1] (1D). Accum = 1. Tol = 6 * 1 * sqrt(2) * 2^-7
            {-1.0,
             1.0,
             -1.0,
             1.0,
             {1, 1, 1},
             6.0 * hipdnn_data_sdk::types::sqrt(2.0) * hipdnn_data_sdk::types::pow(2.0, -7)},
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
        // wDims=[K=1, C=1, R=1] (1D). Accum = 1. Tol = 2 * 2^-23 + 1 * 2^-10
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {1, 1, 1},
         2.0 * hipdnn_data_sdk::types::pow(2.0, -23) + hipdnn_data_sdk::types::pow(2.0, -10)},
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

// Half / Half / Half
// Tolerance = computeGamma(k, u_half) * k * maxProduct
template <>
std::vector<ConvDgradToleranceTestCase>
    getConvDgradToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());

    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1}, 0.0, true},
            // wDims=[K=1, C=1, R=1] (1D). Accum = 1. sumAbsProductBound = 1
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1}, computeGamma(1, u) * 1.0},
            // wDims=[1, 1, 1, 1]. Accum = 1. sumAbsProductBound = 1
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, computeGamma(1, u) * 1.0},
            // wDims=[2, 1, 1, 1]. Accum = 2. sumAbsProductBound = 2
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, computeGamma(2, u) * 2.0},
            // wDims=[1, 1, 3, 3]. Accum = 9. sumAbsProductBound = 9
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 3, 3}, computeGamma(9, u) * 9.0}};
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
    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

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
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateConvDgradTolerance throws when gamma >= 0.5 (singularity)
TEST(TestCalculateConvDgradTolerance, ThrowsOnSingularity)
{
    // For bfloat16, epsilon = 2^-7 ≈ 7.81e-3
    // wDims=[K=10, C=1, R=10, S=1] => Accum = 10 * 10 = 100
    // nU = 2*100*7.81e-3 = 1.562 >= 0.01 → statistical bound
    // gamma = 6 * sqrt(200) * 7.81e-3 ≈ 0.663 >= 0.5 → overflow
    const std::vector<int64_t> wDims = {10, 1, 10, 1};

    EXPECT_THROW(
        (calculateConvDgradTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, wDims)),
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

// Half / Half / Half
// Tolerance = computeGamma(k, u_half) * k * maxProduct
template <>
std::vector<ConvFpropToleranceTestCase>
    getConvFpropToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());

    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            // C=1, R=1, S=1. Accum = 1. sumAbsProductBound = 1
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, computeGamma(1, u) * 1.0},
            // C=2, R=1, S=1. Accum = 2. sumAbsProductBound = 2
            {-1.0, 1.0, -1.0, 1.0, {1, 2, 1, 1}, computeGamma(2, u) * 2.0},
            // C=10, R=1, S=1. Accum = 10. sumAbsProductBound = 10
            {-1.0, 1.0, -1.0, 1.0, {1, 10, 1, 1}, computeGamma(10, u) * 10.0},
            // 3D Convolution: C=2, D=2, R=2, S=2. Accum = 16. sumAbsProductBound = 16
            {-1.0, 1.0, -1.0, 1.0, {1, 2, 2, 2, 2}, computeGamma(16, u) * 16.0}};
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
    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.05f);
    actualFailing->fillTensorWithValue(1.2f);

    auto tol = calculateConvFpropTolerance<half, half, float>(-1.0, 1.0, -1.0, 1.0, dims);

    // tol approx 0.1
    EXPECT_LT(tol, 0.15f);
    EXPECT_GT(tol, 0.09f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

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

// Test that calculateConvFpropTolerance throws when gamma >= 0.5 (singularity)
TEST(TestCalculateConvFpropTolerance, ThrowsOnSingularity)
{
    // For bfloat16, epsilon = 2^-7 ≈ 7.81e-3
    // wDims=[K=1, C=100, R=1, S=1] => Accum = 100
    // nU = 2*100*7.81e-3 = 1.562 >= 0.01 → statistical bound
    // gamma = 6 * sqrt(200) * 7.81e-3 ≈ 0.663 >= 0.5 → overflow
    const std::vector<int64_t> dims = {1, 100, 1, 1};

    EXPECT_THROW(
        (calculateConvFpropTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, dims)),
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
// ConvFpropOutputToleranceFixture (calculateConvFpropOutputTolerance)
//
// One output element of a convolution is a dot product of K input-weight pairs. These tests
// build such dot products directly and grade two correct implementations, and a wrong one,
// against each other with the tolerance.
// =================================================================================================

namespace
{

struct DotProblem
{
    std::vector<float> x;
    std::vector<float> w;
};

// x and w uniform in [-1, 1], already representable in T, as the bundle fill produces them.
template <typename T>
DotProblem makeDotProblem(uint64_t k, unsigned seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    DotProblem p;
    p.x.resize(k);
    p.w.resize(k);
    for(uint64_t i = 0; i < k; ++i)
    {
        p.x[i] = static_cast<float>(static_cast<T>(dist(rng)));
        p.w[i] = static_cast<float>(static_cast<T>(dist(rng)));
    }
    return p;
}

// The first K/2 products followed by their exact negatives, so the exact sum is zero: an output
// near zero, which only the absolute term of the tolerance can cover. x and w are uniform in
// [-1, 1] and representable in T, as in makeDotProblem; the second half repeats x and negates w.
template <typename T>
DotProblem makeCancellingProblem(uint64_t k, unsigned seed)
{
    const auto firstHalf = makeDotProblem<T>(k / 2, seed);
    DotProblem p;
    p.x = firstHalf.x;
    p.w = firstHalf.w;
    for(uint64_t i = 0; i < k / 2; ++i)
    {
        p.x.push_back(firstHalf.x[i]);
        p.w.push_back(-firstHalf.w[i]);
    }
    return p;
}

// Float partial sums over `block` consecutive products, each rounded to half before they are
// summed in float: the half-precision intermediates some half kernels keep.
float blockedHalfDot(const DotProblem& p, uint64_t block)
{
    float total = 0.0f;
    for(uint64_t begin = 0; begin < p.x.size(); begin += block)
    {
        float partial = 0.0f;
        for(uint64_t i = begin; i < std::min<uint64_t>(begin + block, p.x.size()); ++i)
        {
            partial += p.x[i] * p.w[i];
        }
        total += static_cast<float>(static_cast<half>(partial));
    }
    return total;
}

// Float accumulation in index order, as the CPU reference does.
float sequentialDot(const DotProblem& p, uint64_t skip = UINT64_MAX)
{
    float acc = 0.0f;
    for(uint64_t i = 0; i < p.x.size(); ++i)
    {
        if(i != skip)
        {
            acc += p.x[i] * p.w[i];
        }
    }
    return acc;
}

// Float accumulation in a tree, as a GPU reduction does.
float pairwiseDot(const DotProblem& p, uint64_t begin, uint64_t end, uint64_t skip)
{
    if(end - begin <= 8)
    {
        float acc = 0.0f;
        for(uint64_t i = begin; i < end; ++i)
        {
            if(i != skip)
            {
                acc += p.x[i] * p.w[i];
            }
        }
        return acc;
    }
    const uint64_t mid = begin + (end - begin) / 2;
    return pairwiseDot(p, begin, mid, skip) + pairwiseDot(p, mid, end, skip);
}

double exactDot(const DotProblem& p)
{
    double acc = 0.0;
    for(uint64_t i = 0; i < p.x.size(); ++i)
    {
        acc += static_cast<double>(p.x[i]) * static_cast<double>(p.w[i]);
    }
    return acc;
}

ConvFpropErrorScale scaleOf(const DotProblem& p)
{
    double xx = 0.0;
    double ww = 0.0;
    for(uint64_t i = 0; i < p.x.size(); ++i)
    {
        xx += static_cast<double>(p.x[i]) * static_cast<double>(p.x[i]);
        ww += static_cast<double>(p.w[i]) * static_cast<double>(p.w[i]);
    }
    ConvFpropErrorScale s;
    s.accumulationLength = p.x.size();
    s.inputMeanSquare = xx / static_cast<double>(p.x.size());
    s.weightMeanSquare = ww / static_cast<double>(p.x.size());
    return s;
}

template <typename T>
bool withinTolerance(double impl, double ref, const AbsRelTolerance& tol)
{
    const auto implT
        = static_cast<double>(static_cast<float>(static_cast<T>(static_cast<float>(impl))));
    const auto refT
        = static_cast<double>(static_cast<float>(static_cast<T>(static_cast<float>(ref))));
    return std::abs(implT - refT) <= tol.atol + tol.rtol * std::abs(refT);
}

} // namespace

template <typename T>
class ConvFpropOutputToleranceFixture : public ::testing::Test
{
};

using ConvFpropOutputTypes = ::testing::Types<half, bfloat16>;
TYPED_TEST_SUITE(ConvFpropOutputToleranceFixture, ConvFpropOutputTypes, );

// A tree-ordered sum and the exact sum, each rounded once to the output type, stay within the
// tolerance of the reference's sequential sum at every reduction length. Both sides are rounded
// to the output type before they are compared, and with these draws the three float sums round to
// the same value in every seeded case, so this checks that the three summation orders agree after
// rounding; it does not constrain the width of the tolerance (it also passes at atol = rtol = 0).
// ConstantsAtAFixedScale and the near-zero cases below do that.
TYPED_TEST(ConvFpropOutputToleranceFixture, CorrectSumsInAnyOrderPass)
{
    using T = TypeParam;
    for(const uint64_t k : {uint64_t{16}, uint64_t{1152}, uint64_t{65536}})
    {
        for(unsigned seed = 0; seed < 32; ++seed)
        {
            const auto p = makeDotProblem<T>(k, seed);
            const auto tol = calculateConvFpropOutputTolerance<T>(scaleOf(p));
            const auto ref = static_cast<double>(sequentialDot(p));
            const auto tree = static_cast<double>(pairwiseDot(p, 0, k, UINT64_MAX));
            EXPECT_TRUE(withinTolerance<T>(tree, ref, tol)) << "K=" << k << " seed=" << seed;
            EXPECT_TRUE(withinTolerance<T>(exactDot(p), ref, tol)) << "K=" << k << " seed=" << seed;
        }
    }
}

// Leaving out the largest of 1152 products (an error of up to about 1: one whole filter tap is
// C = 128 such products, with a standard deviation of about 3.8) is detected. The old worst-case
// bound passes it always, and the fixed 0.1 passes it for about 44 % of draws.
TYPED_TEST(ConvFpropOutputToleranceFixture, MissingTermFails)
{
    using T = TypeParam;
    const uint64_t k = 1152;
    int detected = 0;
    for(unsigned seed = 0; seed < 32; ++seed)
    {
        const auto p = makeDotProblem<T>(k, seed);
        uint64_t largest = 0;
        for(uint64_t i = 0; i < k; ++i)
        {
            if(std::abs(p.x[i] * p.w[i]) > std::abs(p.x[largest] * p.w[largest]))
            {
                largest = i;
            }
        }
        const auto tol = calculateConvFpropOutputTolerance<T>(scaleOf(p));
        const auto withoutLargest = static_cast<double>(pairwiseDot(p, 0, k, largest));
        const auto ref = static_cast<double>(sequentialDot(p));
        detected += withinTolerance<T>(withoutLargest, ref, tol) ? 0 : 1;
    }
    EXPECT_EQ(detected, 32);
}

// The tolerance at a typical output, relative to that output, does not grow with the reduction
// length: the property the worst-case bound lacks. Only sqrt(K) * u32 and minNormal / sigma
// depend on K, and between K = 16 and K = 2^20 they move the width by under 0.5 %; 5 % leaves
// room for that but not for growth as sqrt(K), which would be 256-fold.
TYPED_TEST(ConvFpropOutputToleranceFixture, WidthRelativeToOutputIndependentOfK)
{
    using T = TypeParam;
    const auto widthAt = [](uint64_t k) {
        ConvFpropErrorScale s;
        s.accumulationLength = k;
        s.inputMeanSquare = 1.0 / 3.0;
        s.weightMeanSquare = 1.0 / 3.0;
        const double sigma = std::sqrt(static_cast<double>(k)) / 3.0;
        const auto tol = calculateConvFpropOutputTolerance<T>(s);
        return (tol.atol + tol.rtol * sigma) / sigma;
    };
    EXPECT_LT(widthAt(uint64_t{1} << 20), 1.05 * widthAt(16));
}

// With zero inputs only a difference below the smallest normal number passes: hardware that
// flushes denormals is not failed, any other difference is.
TYPED_TEST(ConvFpropOutputToleranceFixture, ZeroInputsGradeOnlyDenormals)
{
    using T = TypeParam;
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    const auto tol = calculateConvFpropOutputTolerance<T>(s);
    const auto minNormal = static_cast<double>(std::numeric_limits<T>::min());
    EXPECT_TRUE(withinTolerance<T>(0.5 * minNormal, 0.0, tol));
    EXPECT_FALSE(withinTolerance<T>(2.0 * minNormal, 0.0, tol));
}

// An output near zero has no relative term to lean on: two correct float sums of the same
// products in a different order (sequential against tree) differ by a random walk of roundings
// of about 0.4 * sqrt(K) * u32 * sigma, which only the summation-order term of atol covers. For
// bfloat16 that term is all of atol, so removing it fails this test.
TYPED_TEST(ConvFpropOutputToleranceFixture, NearZeroOutputsNeedTheSummationOrderTerm)
{
    using T = TypeParam;
    for(const uint64_t k : {uint64_t{1152}, uint64_t{65536}})
    {
        for(unsigned seed = 0; seed < 32; ++seed)
        {
            const auto p = makeCancellingProblem<T>(k, seed);
            const auto tol = calculateConvFpropOutputTolerance<T>(scaleOf(p));
            const auto ref = static_cast<double>(sequentialDot(p));
            const auto tree = static_cast<double>(pairwiseDot(p, 0, k, UINT64_MAX));
            EXPECT_TRUE(withinTolerance<T>(tree, ref, tol)) << "K=" << k << " seed=" << seed;
        }
    }
}

// Half kernels that keep partial sums in half differ from the float reference by about
// 0.3 * eps * sigma near zero. The lambda term allows that for half; without it (lambda 0, as
// bfloat16 has) the same implementation fails.
TEST(TestConvFpropOutputToleranceFp16, HalfPrecisionPartialSumsPass)
{
    // K/2 is not a multiple of the block, so the blocks of the cancelling half do not mirror
    // those of the first and their rounding errors do not cancel.
    const uint64_t k = 65528;
    for(unsigned seed = 0; seed < 8; ++seed)
    {
        const auto p = makeCancellingProblem<half>(k, seed);
        const auto tol = calculateConvFpropOutputTolerance<half>(scaleOf(p));
        const auto ref = static_cast<double>(sequentialDot(p));
        const auto blocked = static_cast<double>(blockedHalfDot(p, 8));
        EXPECT_TRUE(withinTolerance<half>(blocked, ref, tol)) << "seed=" << seed;

        // Check on the test itself: the half partial sums really are beyond what float
        // accumulation order explains, which is all that bfloat16's tolerance holds.
        const auto floatOnly = calculateConvFpropOutputTolerance<bfloat16>(scaleOf(p));
        EXPECT_GT(std::abs(blocked - ref), floatOnly.atol) << "seed=" << seed;
    }
}

// The calibrated constants at one scale: K = 144, mean squares 1/3, so sigma = 4. rtol is
// 2 eps; atol is (12 * u32 + lambda * eps) * 4 + minNormal with lambda 12 for half and 0 for
// bfloat16.
TYPED_TEST(ConvFpropOutputToleranceFixture, ConstantsAtAFixedScale)
{
    using T = TypeParam;
    constexpr bool IS_HALF = std::is_same_v<T, half>;
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto tol = calculateConvFpropOutputTolerance<T>(s);

    const double eps = std::ldexp(1.0, IS_HALF ? -10 : -7);
    const double minNormal = std::ldexp(1.0, IS_HALF ? -14 : -126);
    const double lambda = IS_HALF ? 12.0 : 0.0;
    const double expectedAtol = (12.0 * std::ldexp(1.0, -24) + lambda * eps) * 4.0 + minNormal;
    EXPECT_NEAR(tol.rtol, 2.0 * eps, 1e-12 * eps);
    EXPECT_NEAR(tol.atol, expectedAtol, 1e-9 * expectedAtol);
}

// Each rounding point adds slope * eps * magnitude to atol, summed over the points, and leaves
// rtol alone: the point of a value rounded to bfloat16 at 1.5 reaching the output doubled, and
// one rounded to half at 0.25 reaching it as it is.
TYPED_TEST(ConvFpropOutputToleranceFixture, RoundingPointsAddSlopeTimesEpsilonTimesMagnitude)
{
    using T = TypeParam;
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto without = calculateConvFpropOutputTolerance<T>(s);

    const double bfloat16Eps = std::ldexp(1.0, -7);
    const double halfEps = std::ldexp(1.0, -10);
    s.roundingPoints = {{bfloat16Eps, 1.5, 2.0}};
    const auto one = calculateConvFpropOutputTolerance<T>(s);
    EXPECT_NEAR(one.atol - without.atol, 2.0 * bfloat16Eps * 1.5, 1e-12);
    EXPECT_DOUBLE_EQ(one.rtol, without.rtol);

    s.roundingPoints.push_back({halfEps, 0.25, 1.0});
    const auto two = calculateConvFpropOutputTolerance<T>(s);
    EXPECT_NEAR(two.atol - without.atol, 2.0 * bfloat16Eps * 1.5 + halfEps * 0.25, 1e-12);
    EXPECT_DOUBLE_EQ(two.rtol, without.rtol);
}

// A value rounded to bfloat16 before an add that cancels it: the conv result 1.005 rounds to
// 1.0078125, and after adding -1 the output is 0.0078125 where the exact result is 0.005. No
// relative term of the output (0.005) or sum-order term covers the 0.0028 between them; the point
// of the rounding, at the size of the value rounded, does. A graph that has no such point keeps
// its band, and fails this. (Half has its lambda term, which is wider than this at this scale, so
// the test is for the outputs whose atol does not reach that far.)
TEST(TestConvFpropOutputToleranceRoundingPoints, BfloatRoundingBeforeACancellingAddIsCovered)
{
    using T = bfloat16;
    ConvFpropErrorScale s;
    s.accumulationLength = 16;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const double sigma = std::sqrt(16.0 / 9.0);

    const double convResult = 1.005;
    const double rounded = static_cast<double>(
        static_cast<float>(static_cast<bfloat16>(static_cast<float>(convResult))));
    ASSERT_EQ(rounded, 1.0078125);
    const double impl = rounded - 1.0;
    const double ref = convResult - 1.0;

    EXPECT_FALSE(withinTolerance<float>(impl, ref, calculateConvFpropOutputTolerance<T>(s)));

    s.roundingPoints = {{std::ldexp(1.0, -7), sigma, 1.0}};
    EXPECT_TRUE(withinTolerance<float>(impl, ref, calculateConvFpropOutputTolerance<T>(s)));
}

// A rounding point that is not a finite non-negative number is rejected, as the statistics are.
TYPED_TEST(ConvFpropOutputToleranceFixture, InvalidRoundingPointIsRejected)
{
    using T = TypeParam;
    ConvFpropErrorScale valid;
    valid.accumulationLength = 144;
    valid.inputMeanSquare = 1.0 / 3.0;
    valid.weightMeanSquare = 1.0 / 3.0;
    valid.roundingPoints = {{std::ldexp(1.0, -7), 1.0, 1.0}};
    ASSERT_NO_THROW(calculateConvFpropOutputTolerance<T>(valid));

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    for(const RoundingPoint& bad : {RoundingPoint{-1.0, 1.0, 1.0},
                                    RoundingPoint{nan, 1.0, 1.0},
                                    RoundingPoint{inf, 1.0, 1.0},
                                    RoundingPoint{0.01, -1.0, 1.0},
                                    RoundingPoint{0.01, nan, 1.0},
                                    RoundingPoint{0.01, inf, 1.0},
                                    RoundingPoint{0.01, 1.0, -1.0},
                                    RoundingPoint{0.01, 1.0, nan},
                                    RoundingPoint{0.01, 1.0, inf}})
    {
        auto scale = valid;
        scale.roundingPoints.push_back(bad);
        EXPECT_THROW(calculateConvFpropOutputTolerance<T>(scale), std::invalid_argument);
    }
}

// A fused multiply by a scale scales the typical output, and with it atol; rtol, a fraction of
// the reference itself, stays. Only the magnitude counts.
TYPED_TEST(ConvFpropOutputToleranceFixture, OutputScaleWidensAtolNotRtol)
{
    using T = TypeParam;
    const auto minNormal = static_cast<double>(std::numeric_limits<T>::min());
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto unit = calculateConvFpropOutputTolerance<T>(s);

    s.outputScale = 2.0;
    const auto doubled = calculateConvFpropOutputTolerance<T>(s);
    EXPECT_NEAR(doubled.atol - minNormal, 2.0 * (unit.atol - minNormal), 1e-9 * doubled.atol);
    EXPECT_DOUBLE_EQ(doubled.rtol, unit.rtol);

    s.outputScale = -2.0;
    const auto negative = calculateConvFpropOutputTolerance<T>(s);
    EXPECT_DOUBLE_EQ(negative.atol, doubled.atol);

    s.outputScale = 0.5;
    const auto halved = calculateConvFpropOutputTolerance<T>(s);
    EXPECT_NEAR(halved.atol - minNormal, 0.5 * (unit.atol - minNormal), 1e-9 * halved.atol);
}

// A scale that does not describe a convolution is rejected, not graded with a tolerance that is
// zero, not a number or infinite.
TYPED_TEST(ConvFpropOutputToleranceFixture, InvalidScaleIsRejected)
{
    using T = TypeParam;
    ConvFpropErrorScale valid;
    valid.accumulationLength = 144;
    valid.inputMeanSquare = 1.0 / 3.0;
    valid.weightMeanSquare = 1.0 / 3.0;
    ASSERT_NO_THROW(calculateConvFpropOutputTolerance<T>(valid));

    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(ConvFpropErrorScale{}),
                 std::invalid_argument);

    auto zeroLength = valid;
    zeroLength.accumulationLength = 0;
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(zeroLength), std::invalid_argument);

    auto negativeMoment = valid;
    negativeMoment.inputMeanSquare = -1.0;
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(negativeMoment), std::invalid_argument);

    auto nanMoment = valid;
    nanMoment.weightMeanSquare = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(nanMoment), std::invalid_argument);

    auto infiniteScale = valid;
    infiniteScale.outputScale = std::numeric_limits<double>::infinity();
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(infiniteScale), std::invalid_argument);
}

// An atol beyond the largest value of the output type means the output itself overflows.
TYPED_TEST(ConvFpropOutputToleranceFixture, ToleranceBeyondTheOutputRangeThrows)
{
    using T = TypeParam;
    ConvFpropErrorScale s;
    s.accumulationLength = uint64_t{1} << 40;
    s.inputMeanSquare = 1e36;
    s.weightMeanSquare = 1e36;
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(s), std::overflow_error);

    // The mean squares are finite, but their product overflows a double.
    s.inputMeanSquare = 1e300;
    s.weightMeanSquare = 1e300;
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(s), std::overflow_error);

    // Finite statistics whose product overflows against a zero factor: infinity times zero is
    // not a number, which no range check rejects.
    s.inputMeanSquare = 1e300;
    s.weightMeanSquare = 0.0;
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(s), std::overflow_error);

    s.inputMeanSquare = 1e300;
    s.weightMeanSquare = 1e300;
    s.outputScale = 0.0;
    EXPECT_THROW(calculateConvFpropOutputTolerance<T>(s), std::overflow_error);
}

// ── summation-order coefficient ─────────────────────────────────────────────

// An engine's coefficient multiplies the summation-order term of atol and leaves rtol alone for
// half and bfloat16, whose default is 1.
TYPED_TEST(ConvFpropOutputToleranceFixture, SummationOrderCoefficientScalesTheSumTermOfAtol)
{
    using T = TypeParam;
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto unset = calculateConvFpropOutputTolerance<T>(s);
    ConvFpropCoefficients one;
    one.summationOrder = 1.0;
    const auto explicitOne = calculateConvFpropOutputTolerance<T>(s, one);
    EXPECT_DOUBLE_EQ(explicitOne.atol, unset.atol);
    EXPECT_DOUBLE_EQ(explicitOne.rtol, unset.rtol);

    ConvFpropCoefficients five;
    five.summationOrder = 5.0;
    const auto wider = calculateConvFpropOutputTolerance<T>(s, five);
    // sqrt(144) * 2^-24 * sigma (4) = 48 * 2^-24 more per unit of coefficient
    EXPECT_NEAR(wider.atol - unset.atol, 4.0 * 12.0 * std::ldexp(1.0, -24) * 4.0, 1e-12);
    EXPECT_DOUBLE_EQ(wider.rtol, unset.rtol);
}

// A coefficient that is not finite and positive is rejected, for every output type.
TYPED_TEST(ConvFpropOutputToleranceFixture, InvalidSummationOrderCoefficientIsRejected)
{
    using T = TypeParam;
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    for(const double bad : {0.0,
                            -1.0,
                            std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity()})
    {
        ConvFpropCoefficients coefficients;
        coefficients.summationOrder = bad;
        EXPECT_THROW(calculateConvFpropOutputTolerance<T>(s, coefficients), std::invalid_argument)
            << bad;
    }
}

// ── float outputs ───────────────────────────────────────────────────────────
//
// For a float output the one rounding is of the float result itself: rtol is 2 eps of float plus
// the summation-order term, which atol carries on sigma: c * sqrt(K) * u32 on |ref| and on
// sigma, with c = DEFAULT_FLOAT_SUMMATION_ORDER (4) unless the engine sets it.

// K = 144, sigma = 4, c = 4: written out here, not taken from the library.
TEST(TestConvFpropOutputToleranceFloat, ConstantsAtAFixedScale)
{
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto tol = calculateConvFpropOutputTolerance<float>(s);
    const double sumTerm = 4.0 * 12.0 * std::ldexp(1.0, -24);
    EXPECT_NEAR(tol.rtol, std::ldexp(1.0, -22) + sumTerm, 1e-15);
    EXPECT_NEAR(tol.atol, sumTerm * 4.0 + std::ldexp(1.0, -126), 1e-15);
}

// With no coefficient set the default is used, and the default is 4.
TEST(TestConvFpropOutputToleranceFloat, BaseCoefficientIsUsedWhenTheEngineSetsNone)
{
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    ConvFpropCoefficients base;
    base.summationOrder = 4.0;
    const auto unset = calculateConvFpropOutputTolerance<float>(s);
    const auto four = calculateConvFpropOutputTolerance<float>(s, base);
    EXPECT_DOUBLE_EQ(unset.atol, four.atol);
    EXPECT_DOUBLE_EQ(unset.rtol, four.rtol);
    EXPECT_DOUBLE_EQ(DEFAULT_FLOAT_SUMMATION_ORDER, 4.0);
}

// An engine's coefficient replaces the default in atol and in rtol.
TEST(TestConvFpropOutputToleranceFloat, EngineCoefficientReplacesTheBaseOne)
{
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    ConvFpropCoefficients engine;
    engine.summationOrder = 9.0;
    const auto base = calculateConvFpropOutputTolerance<float>(s);
    const auto wider = calculateConvFpropOutputTolerance<float>(s, engine);
    const double perUnit = 12.0 * std::ldexp(1.0, -24);
    EXPECT_NEAR(wider.rtol - base.rtol, 5.0 * perUnit, 1e-15);
    EXPECT_NEAR(wider.atol - base.atol, 5.0 * perUnit * 4.0, 1e-15);
}

// The rounding points of a graph widen the atol of a float output as they do a half one's:
// slope * eps_p * magnitude each, added to the summation-order term. K = 144, c = 4 and the output
// scale 2 give sigma = 8; the points are written out here, not taken from the library.
TEST(TestConvFpropOutputToleranceFloat, RoundingPointsWidenTheAtolOnly)
{
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    s.outputScale = 2.0;
    const auto without = calculateConvFpropOutputTolerance<float>(s);
    s.roundingPoints = {{std::ldexp(1.0, -7), 8.0, 3.0}, {std::ldexp(1.0, -10), 5.0, 1.0}};
    const auto with = calculateConvFpropOutputTolerance<float>(s);

    const double sumTerm = 4.0 * 12.0 * std::ldexp(1.0, -24);
    EXPECT_NEAR(without.atol, sumTerm * 8.0 + std::ldexp(1.0, -126), 1e-15);
    EXPECT_NEAR(with.atol - without.atol,
                std::ldexp(1.0, -7) * 8.0 * 3.0 + std::ldexp(1.0, -10) * 5.0,
                1e-12);
    EXPECT_DOUBLE_EQ(with.rtol, without.rtol);
}

// The fused multiplies' scale multiplies sigma, and with it the atol of a float output (and not
// its rtol).
TEST(TestConvFpropOutputToleranceFloat, OutputScaleScalesTheAtolOnly)
{
    ConvFpropErrorScale s;
    s.accumulationLength = 144;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto plain = calculateConvFpropOutputTolerance<float>(s);
    s.outputScale = 0.5;
    const auto halved = calculateConvFpropOutputTolerance<float>(s);
    s.outputScale = -3.0;
    const auto negative = calculateConvFpropOutputTolerance<float>(s);

    const double minNormal = std::ldexp(1.0, -126);
    EXPECT_NEAR(halved.atol - minNormal, 0.5 * (plain.atol - minNormal), 1e-15);
    EXPECT_NEAR(negative.atol - minNormal, 3.0 * (plain.atol - minNormal), 1e-15);
    EXPECT_DOUBLE_EQ(halved.rtol, plain.rtol);
    EXPECT_DOUBLE_EQ(negative.rtol, plain.rtol);
}

// A float output has no lambda term: atol is the summation-order term and nothing else.
TEST(TestConvFpropOutputToleranceFloat, HasNoPartialResultTerm)
{
    ConvFpropErrorScale s;
    s.accumulationLength = 65536;
    s.inputMeanSquare = 1.0 / 3.0;
    s.weightMeanSquare = 1.0 / 3.0;
    const auto tol = calculateConvFpropOutputTolerance<float>(s);
    const double sigma = std::sqrt(65536.0) / 3.0;
    EXPECT_LT(tol.atol,
              1.001 * (4.0 * 256.0 * std::ldexp(1.0, -24) * sigma + std::ldexp(1.0, -126)));
}

// A tree-ordered sum and the exact sum stay within the tolerance of the reference's sequential
// float sum at every reduction length.
TEST(TestConvFpropOutputToleranceFloat, CorrectSumsInAnyOrderPass)
{
    for(const uint64_t k : {uint64_t{16}, uint64_t{1152}, uint64_t{65536}})
    {
        for(unsigned seed = 0; seed < 32; ++seed)
        {
            const auto p = makeDotProblem<float>(k, seed);
            const auto tol = calculateConvFpropOutputTolerance<float>(scaleOf(p));
            const auto ref = static_cast<double>(sequentialDot(p));
            const auto tree = static_cast<double>(pairwiseDot(p, 0, k, UINT64_MAX));
            EXPECT_TRUE(withinTolerance<float>(tree, ref, tol)) << "K=" << k << " seed=" << seed;
            EXPECT_TRUE(withinTolerance<float>(exactDot(p), ref, tol))
                << "K=" << k << " seed=" << seed;
        }
    }
}

// An output near zero has only the summation-order term to lean on: the sequential and the tree
// sum of products that cancel differ by a random walk of roundings.
TEST(TestConvFpropOutputToleranceFloat, NearZeroOutputsNeedTheSummationOrderTerm)
{
    for(const uint64_t k : {uint64_t{1152}, uint64_t{65536}})
    {
        double largestOverTerm = 0.0;
        for(unsigned seed = 0; seed < 32; ++seed)
        {
            const auto p = makeCancellingProblem<float>(k, seed);
            const auto tol = calculateConvFpropOutputTolerance<float>(scaleOf(p));
            const auto ref = static_cast<double>(sequentialDot(p));
            const auto tree = static_cast<double>(pairwiseDot(p, 0, k, UINT64_MAX));
            EXPECT_TRUE(withinTolerance<float>(tree, ref, tol)) << "K=" << k << " seed=" << seed;
            largestOverTerm = std::max(largestOverTerm, std::abs(tree - ref) / tol.atol);
        }
        // Check on the test itself: the sums really do differ, by a good part of atol, so that
        // removing the term fails the comparison above.
        EXPECT_GT(largestOverTerm, 0.02) << "K=" << k;
    }
}

// Leaving out the largest of 1152 products (an error of up to about 1) is far outside a float
// tolerance, at every draw.
TEST(TestConvFpropOutputToleranceFloat, MissingTermFails)
{
    const uint64_t k = 1152;
    for(unsigned seed = 0; seed < 32; ++seed)
    {
        const auto p = makeDotProblem<float>(k, seed);
        uint64_t largest = 0;
        for(uint64_t i = 0; i < k; ++i)
        {
            if(std::abs(p.x[i] * p.w[i]) > std::abs(p.x[largest] * p.w[largest]))
            {
                largest = i;
            }
        }
        const auto tol = calculateConvFpropOutputTolerance<float>(scaleOf(p));
        const auto withoutLargest = static_cast<double>(pairwiseDot(p, 0, k, largest));
        const auto ref = static_cast<double>(sequentialDot(p));
        EXPECT_FALSE(withinTolerance<float>(withoutLargest, ref, tol)) << "seed=" << seed;
    }
}

// C/groups times the filter volume, for 1D, 2D and 3D weights.
TEST(TestConvFpropAccumulationLength, MultipliesEveryWeightDimensionButTheFirst)
{
    EXPECT_EQ(convFpropAccumulationLength({32, 4}), 4u);
    EXPECT_EQ(convFpropAccumulationLength({32, 4, 3}), 12u);
    EXPECT_EQ(convFpropAccumulationLength({32, 4, 3, 3}), 36u);
    EXPECT_EQ(convFpropAccumulationLength({32, 4, 3, 3, 5}), 180u);
}

TEST(TestConvFpropAccumulationLength, RejectsWeightsWithoutAChannelDimension)
{
    EXPECT_THROW(convFpropAccumulationLength({}), std::invalid_argument);
    EXPECT_THROW(convFpropAccumulationLength({32}), std::invalid_argument);
}

TEST(TestConvFpropAccumulationLength, ReturnsZeroForAZeroExtent)
{
    EXPECT_EQ(convFpropAccumulationLength({32, 0, 3, 3}), 0u);
    EXPECT_EQ(convFpropAccumulationLength({32, 4, 0}), 0u);
}

TEST(TestConvFpropAccumulationLength, RejectsANegativeExtent)
{
    EXPECT_THROW(convFpropAccumulationLength({32, -1}), std::invalid_argument);
    EXPECT_THROW(convFpropAccumulationLength({32, 4, -3}), std::invalid_argument);
}
