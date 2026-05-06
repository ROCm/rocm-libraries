// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
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
