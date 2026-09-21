// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesLayerNorm.hpp>
#include <vector>

using namespace hipdnn_test_sdk::utilities::layernorm;
using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using hipdnn_test_sdk::utilities::computeGamma;

template <typename Out, typename In, typename Comp>
struct TypeTriple
{
    using OutputType = Out;
    using InputType = In;
    using ComputeType = Comp;
};

// =================================================================================================
// TestCalculateLayernormFpropTolerance
// =================================================================================================

struct LayernormFpropToleranceTestCase
{
    double xMin;
    double xMax;
    double scaleMin;
    double scaleMax;
    int64_t normalizedElementCount;
    double biasMin;
    double biasMax;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const LayernormFpropToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", scaleMin: " << tc.scaleMin
           << ", scaleMax: " << tc.scaleMax
           << ", normalizedElementCount: " << tc.normalizedElementCount
           << ", biasMin: " << tc.biasMin << ", biasMax: " << tc.biasMax
           << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<LayernormFpropToleranceTestCase> getLayernormFpropToleranceTestCases();

// LayerNorm tolerance has two error paths:
//   Path A (variance): gammaVar(2M)/2 * maxAbsScale  (sumAbsProductBound cancels)
//   Path B (mean):     gammaMean(M) * maxAbsScale
//   + NONLINEAR_OPS * u * maxAbsScale + maxAbsBias * u
// Combined: (gammaVar(2M)/2 + gammaMean(M) + 5u) * maxAbsScale + maxAbsBias * u

// Float / Float / Float
// For FP32 at small M, computeGamma uses linear bound: nU/(1-nU)
template <>
std::vector<LayernormFpropToleranceTestCase>
    getLayernormFpropToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gammaVar = [&](int64_t m) {
        return computeGamma(static_cast<uint64_t>(2) * static_cast<uint64_t>(m), u);
    };
    auto gammaMean = [&](int64_t m) { return computeGamma(static_cast<uint64_t>(m), u); };
    return {// normalizedElementCount = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // M=1, x=[-1,1], scale=[-1,1], no bias
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gammaVar(1) / 2.0) + gammaMean(1) + 5.0 * u},
            // M=10, x=[-1,1], scale=[-1,1], no bias
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gammaVar(10) / 2.0) + gammaMean(10) + 5.0 * u},
            // M=10, x=[-1000,1000], scale=[-1000,1000], no bias
            {-1000.0,
             1000.0,
             -1000.0,
             1000.0,
             10,
             0.0,
             0.0,
             ((gammaVar(10) / 2.0) + gammaMean(10) + 5.0 * u) * 1000.0},
            // M=3, x=[-1,1], scale=[-1,1], bias=[-0.5,0.5]
            {-1.0,
             1.0,
             -1.0,
             1.0,
             3,
             -0.5,
             0.5,
             (gammaVar(3) / 2.0) + gammaMean(3) + 5.0 * u + 0.5 * u}};
}

// Float / Double / Float (Input casting error: inputEpsilon(double) < epsilon(float))
// Variance path accTol = gammaVar * S + S * epsilon → (gammaVar + u)/2 after propagation
// Mean path unaffected by input casting (absorbed by NONLINEAR_OPS)
template <>
std::vector<LayernormFpropToleranceTestCase>
    getLayernormFpropToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gammaVar = [&](int64_t m) {
        return computeGamma(static_cast<uint64_t>(2) * static_cast<uint64_t>(m), u);
    };
    auto gammaMean = [&](int64_t m) { return computeGamma(static_cast<uint64_t>(m), u); };
    return {
        // normalizedElementCount = 0 => throws
        {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
        // M=1
        {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, ((gammaVar(1) + u) / 2.0) + gammaMean(1) + 5.0 * u},
        // M=10
        {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, ((gammaVar(10) + u) / 2.0) + gammaMean(10) + 5.0 * u},
        // Zero input: x=0, range=0, all terms vanish
        {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Half / Float / Float (Output casting error: outputEpsilon(half=2^-10) > epsilon(float=2^-23))
template <>
std::vector<LayernormFpropToleranceTestCase>
    getLayernormFpropToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gammaVar = [&](int64_t m) {
        return computeGamma(static_cast<uint64_t>(2) * static_cast<uint64_t>(m), u);
    };
    auto gammaMean = [&](int64_t m) { return computeGamma(static_cast<uint64_t>(m), u); };
    return {// normalizedElementCount = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // M=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             1,
             0.0,
             0.0,
             (gammaVar(1) / 2.0) + gammaMean(1) + 5.0 * u + 1.0 * uHalf},
            // M=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             10,
             0.0,
             0.0,
             (gammaVar(10) / 2.0) + gammaMean(10) + 5.0 * u + 1.0 * uHalf},
            // Zero input: range=0, output cast term from maxOutputMagnitude=0 vanishes
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Half / Half / Half
// computeGamma selects linear vs statistical based on nU threshold (0.01).
// half epsilon = 2^-10.  Variance path: 2*M=2 -> nU=0.0039<0.01 (linear).
//                        Mean path:     M=1   -> nU=0.002<0.01 (linear).
// At M=10: variance 2*M=20 -> nU=0.039>=0.01 (statistical).
//          mean     M=10   -> nU=0.020>=0.01 (statistical).
template <>
std::vector<LayernormFpropToleranceTestCase>
    getLayernormFpropToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gammaVar = [&](int64_t m) {
        return computeGamma(static_cast<uint64_t>(2) * static_cast<uint64_t>(m), u);
    };
    auto gammaMean = [&](int64_t m) { return computeGamma(static_cast<uint64_t>(m), u); };
    return {// normalizedElementCount = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // M=1
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gammaVar(1) / 2.0) + gammaMean(1) + 5.0 * u},
            // M=10
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gammaVar(10) / 2.0) + gammaMean(10) + 5.0 * u},
            // Zero input: all terms vanish
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Bfloat16 / Float / Float (Output casting error: outputEpsilon(bf16=2^-7) > epsilon(float=2^-23))
template <>
std::vector<LayernormFpropToleranceTestCase>
    getLayernormFpropToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gammaVar = [&](int64_t m) {
        return computeGamma(static_cast<uint64_t>(2) * static_cast<uint64_t>(m), u);
    };
    auto gammaMean = [&](int64_t m) { return computeGamma(static_cast<uint64_t>(m), u); };
    return {// normalizedElementCount = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // M=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             1,
             0.0,
             0.0,
             (gammaVar(1) / 2.0) + gammaMean(1) + 5.0 * u + 1.0 * uBf16},
            // M=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             10,
             0.0,
             0.0,
             (gammaVar(10) / 2.0) + gammaMean(10) + 5.0 * u + 1.0 * uBf16},
            // Zero input: range=0, output cast from maxOutputMagnitude=0 vanishes
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
// bf16 epsilon = 2^-7.  Variance: 2*M=2 -> nU=2*2*2^-7=0.03125>=0.01 -> statistical.
//                       Mean:     M=1   -> nU=2*1*2^-7=0.01563>=0.01 -> statistical.
template <>
std::vector<LayernormFpropToleranceTestCase>
    getLayernormFpropToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gammaVar = [&](int64_t m) {
        return computeGamma(static_cast<uint64_t>(2) * static_cast<uint64_t>(m), u);
    };
    auto gammaMean = [&](int64_t m) { return computeGamma(static_cast<uint64_t>(m), u); };
    return {// normalizedElementCount = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // M=1
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gammaVar(1) / 2.0) + gammaMean(1) + 5.0 * u},
            // M=10
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gammaVar(10) / 2.0) + gammaMean(10) + 5.0 * u},
            // Zero input: all terms vanish
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateLayernormFpropTolerance
    : public ::testing::TestWithParam<LayernormFpropToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateLayernormFpropTolerance<Out, In, Comp>(params.xMin,
                                                                 params.xMax,
                                                                 params.scaleMin,
                                                                 params.scaleMax,
                                                                 params.normalizedElementCount,
                                                                 params.biasMin,
                                                                 params.biasMax)),
                std::invalid_argument);
        }
        else
        {
            auto tol
                = calculateLayernormFpropTolerance<Out, In, Comp>(params.xMin,
                                                                  params.xMax,
                                                                  params.scaleMin,
                                                                  params.scaleMax,
                                                                  params.normalizedElementCount,
                                                                  params.biasMin,
                                                                  params.biasMax);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalculateLayernormFpropToleranceFp32
    = TestCalculateLayernormFpropTolerance<float, float, float>;
TEST_P(TestCalculateLayernormFpropToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateLayernormFpropToleranceFp32,
    ::testing::ValuesIn(getLayernormFpropToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateLayernormFpropToleranceInputDouble
    = TestCalculateLayernormFpropTolerance<float, double, float>;
TEST_P(TestCalculateLayernormFpropToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateLayernormFpropToleranceInputDouble,
    ::testing::ValuesIn(getLayernormFpropToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateLayernormFpropToleranceComputeFloatFp16
    = TestCalculateLayernormFpropTolerance<half, float, float>;
TEST_P(TestCalculateLayernormFpropToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateLayernormFpropToleranceComputeFloatFp16,
    ::testing::ValuesIn(getLayernormFpropToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateLayernormFpropToleranceFp16
    = TestCalculateLayernormFpropTolerance<half, half, half>;
TEST_P(TestCalculateLayernormFpropToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateLayernormFpropToleranceFp16,
    ::testing::ValuesIn(getLayernormFpropToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalculateLayernormFpropToleranceComputeFloatBfp16
    = TestCalculateLayernormFpropTolerance<bfloat16, float, float>;
TEST_P(TestCalculateLayernormFpropToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateLayernormFpropToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getLayernormFpropToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateLayernormFpropToleranceBfp16
    = TestCalculateLayernormFpropTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateLayernormFpropToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateLayernormFpropToleranceBfp16,
    ::testing::ValuesIn(
        getLayernormFpropToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// Test that calculateLayernormFpropTolerance catches simulated wrong outputs
TEST(TestCalculateLayernormFpropTolerance, DetectsFailure)
{
    // M=100, x=[-1,1], scale=[-1,1]
    const std::vector<int64_t> dims = {1, 1, 10, 10};
    const std::vector<int64_t> strides = {100, 100, 10, 1};

    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.000001f); // Small error
    actualFailing->fillTensorWithValue(1.2f); // Large error

    auto tol = calculateLayernormFpropTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 100);

    // For FP32, M=100 (2M=200 accum): tolerance should be small
    EXPECT_LT(tol, 0.01f);
    EXPECT_GT(tol, 1e-6f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateLayernormFpropTolerance throws when gamma >= 0.5
TEST(TestCalculateLayernormFpropTolerance, ThrowsOnSingularity)
{
    // For float, epsilon = 2^-23 ~ 1.19e-7.
    // computeGamma uses 2M accumulations. Statistical bound: gamma = 6*sqrt(2*2M)*u
    // gamma >= 0.5 -> 2M >= (0.5/(6*u))^2 / 2 -> M >= (0.5/(6*u))^2 / 4
    // M >= (0.5/(6*1.19e-7))^2/4 ~ 1.47e11. Use 1.5e11.
    EXPECT_THROW((calculateLayernormFpropTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, 150000000000LL)),
                 std::overflow_error);

    // For bfloat16, epsilon = 2^-7. gamma >= 0.5 at much smaller M.
    // gamma = 6*sqrt(2*2M)*u >= 0.5 -> M >= (0.5/(6*2^-7))^2/4 ~ 17.1. Use M=50.
    EXPECT_THROW(
        (calculateLayernormFpropTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 50)),
        std::overflow_error);

    // For half, epsilon = 2^-10. gamma >= 0.5 -> M >= (0.5/(6*2^-10))^2/4 ~ 1093. Use M=2500.
    EXPECT_THROW((calculateLayernormFpropTolerance<half, half, half>(-1.0, 1.0, -1.0, 1.0, 2500)),
                 std::overflow_error);

    // float/double/float uses float epsilon -> same threshold as float/float/float
    EXPECT_THROW((calculateLayernormFpropTolerance<float, double, float>(
                     -1.0, 1.0, -1.0, 1.0, 150000000000LL)),
                 std::overflow_error);
}

// Test that calculateLayernormFpropTolerance throws when tolerance exceeds OutputType max
TEST(TestCalculateLayernormFpropTolerance, ThrowsOnOutputOverflow)
{
    // OutputType = half (max approx 65504)
    // Use very large scale to push tolerance above half max.
    // M=10, x=[-1,1], scale=[-1e8,1e8]
    // output cast term: maxAbsScale * uHalf = 1e8 * 2^-10 ~ 97656 >> 65504
    EXPECT_THROW((calculateLayernormFpropTolerance<half, float, float>(-1.0, 1.0, -1e8, 1e8, 10)),
                 std::overflow_error);

    // half/half/half with very large scale
    EXPECT_THROW((calculateLayernormFpropTolerance<half, half, half>(-1000, 1000, -1e7, 1e7, 10)),
                 std::overflow_error);
}

// Test zero input (x=0) produces a valid small tolerance
TEST(TestCalculateLayernormFpropTolerance, ZeroInput)
{
    // When x=0, range=0, all accumulation and mean path terms vanish.
    // propagatedTolerance = maxAbsBias * epsilon
    auto tol = calculateLayernormFpropTolerance<float, float, float>(0.0, 0.0, -1.0, 1.0, 10);
    EXPECT_EQ(tol, 0.0f);

    // With bias (float)
    auto tolBias
        = calculateLayernormFpropTolerance<float, float, float>(0.0, 0.0, -1.0, 1.0, 10, -0.5, 0.5);
    EXPECT_GT(tolBias, 0.0f);

    // Zero input with bfloat16 output casting: outputEpsilon(bf16) > epsilon(float), but
    // maxOutputMagnitude=0, so output cast term also vanishes
    auto tolBf16
        = calculateLayernormFpropTolerance<bfloat16, float, float>(0.0, 0.0, -1.0, 1.0, 10);
    EXPECT_EQ(tolBf16, 0.0f);

    // Zero input with bias and half output: bias term + output cast of bias
    auto uFloat = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto tolHalfBias
        = calculateLayernormFpropTolerance<half, float, float>(0.0, 0.0, -1.0, 1.0, 10, -0.5, 0.5);
    // Expected: maxAbsBias * epsilon + maxOutputMagnitude * outputEpsilon
    //         = 0.5 * uFloat + 0.5 * uHalf
    auto expectedHalfBias = static_cast<float>(0.5 * uFloat + 0.5 * uHalf);
    EXPECT_NEAR(tolHalfBias, expectedHalfBias, 1e-10f);
}

// Sanity check: tolerance scaling behavior with increasing M.
// For small M with FP32, computeGamma uses linear bound -> dominant term ~ 2M * u
// For BF16, computeGamma uses statistical bound -> dominant term ~ sqrt(2M) * u
TEST(TestCalculateLayernormFpropTolerance, ToleranceScalesWithM)
{
    // FP32: tolerance should grow roughly linearly with M (gamma/2 ~ M*u)
    auto tolFp32M1 = calculateLayernormFpropTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 1);
    auto tolFp32M10
        = calculateLayernormFpropTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 10);
    auto tolFp32M100
        = calculateLayernormFpropTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 100);

    EXPECT_LT(tolFp32M1, tolFp32M10) << "FP32: M=10 should have higher tolerance than M=1";
    EXPECT_LT(tolFp32M10, tolFp32M100) << "FP32: M=100 should have higher tolerance than M=10";

    // Ratio check: M=100 vs M=1 (use wide spread so gamma term dominates the constant 5u)
    auto ratioFp32 = static_cast<double>(tolFp32M100) / static_cast<double>(tolFp32M1);
    EXPECT_GT(ratioFp32, 10.0) << "FP32: ratio M100/M1 should reflect ~M growth";

    // BF16 (statistical): tolerance should grow ~ sqrt(M)
    // M=50 with 2*50=100 accum: gamma ~ 6*sqrt(200)*2^-7 ~ 0.663 >= 0.5, so use M=25 as upper
    auto tolBf16M1
        = calculateLayernormFpropTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 1);
    auto tolBf16M5
        = calculateLayernormFpropTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 5);
    auto tolBf16M25
        = calculateLayernormFpropTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 25);

    EXPECT_LT(tolBf16M1, tolBf16M5) << "BF16: M=5 should have higher tolerance than M=1";
    EXPECT_LT(tolBf16M5, tolBf16M25) << "BF16: M=25 should have higher tolerance than M=5";

    auto ratioBf16 = static_cast<double>(tolBf16M5) / static_cast<double>(tolBf16M1);
    EXPECT_GT(ratioBf16, 1.5) << "BF16: ratio M5/M1 should reflect sqrt(M) growth";
}
