// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesRMSNorm.hpp>
#include <vector>

using namespace hipdnn_test_sdk::utilities::rmsnorm;
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
// TestCalculateRMSNormFwdTolerance
// =================================================================================================

struct RMSNormFwdToleranceTestCase
{
    double xMin;
    double xMax;
    double scaleMin;
    double scaleMax;
    int64_t nChannels;
    double biasMin;
    double biasMax;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const RMSNormFwdToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", scaleMin: " << tc.scaleMin
           << ", scaleMax: " << tc.scaleMax << ", nChannels: " << tc.nChannels
           << ", biasMin: " << tc.biasMin << ", biasMax: " << tc.biasMax
           << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<RMSNormFwdToleranceTestCase> getRMSNormFwdToleranceTestCases();

// RMSNorm tests use computeGamma() from the shared utilities namespace.
// propTol = (computeGamma(C, u) * C * maxAbsX^2 / (2 * C * maxAbsX^2)) * maxAbsScale
//         + NONLINEAR_OPS * u * maxAbsScale
//         + maxAbsBias * u
//         = (gamma / 2 + 5u) * maxAbsScale + maxAbsBias * u

// Float / Float / Float
// For FP32 at small C, computeGamma uses linear bound: nU/(1-nU)
template <>
std::vector<RMSNormFwdToleranceTestCase>
    getRMSNormFwdToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// nChannels = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // C=1, x=[-1,1], scale=[-1,1], no bias
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gamma(1) / 2.0) + 5.0 * u},
            // C=10, x=[-1,1], scale=[-1,1], no bias
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gamma(10) / 2.0) + 5.0 * u},
            // C=10, x=[-1000,1000], scale=[-1000,1000], no bias
            {-1000.0,
             1000.0,
             -1000.0,
             1000.0,
             10,
             0.0,
             0.0,
             (gamma(10) / 2.0) * 1000.0 + 5.0 * u * 1000.0},
            // C=3, x=[-1,1], scale=[-1,1], bias=[-0.5,0.5]
            {-1.0, 1.0, -1.0, 1.0, 3, -0.5, 0.5, (gamma(3) / 2.0) + 5.0 * u + 0.5 * u}};
}

// Float / Double / Float (Input casting error: inputEpsilon(double) < epsilon(float))
// accTol = gamma * sumAbsProductBound + sumAbsProductBound * epsilon
template <>
std::vector<RMSNormFwdToleranceTestCase>
    getRMSNormFwdToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// nChannels = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // C=1
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, ((gamma(1) + u) / 2.0) + 5.0 * u},
            // C=10
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, ((gamma(10) + u) / 2.0) + 5.0 * u},
            // Zero input: x=0, all terms vanish
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Half / Float / Float (Output casting error: outputEpsilon(half=2^-10) > epsilon(float=2^-23))
template <>
std::vector<RMSNormFwdToleranceTestCase>
    getRMSNormFwdToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// nChannels = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // C=1
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gamma(1) / 2.0) + 5.0 * u + 1.0 * uHalf},
            // C=10
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gamma(10) / 2.0) + 5.0 * u + 1.0 * uHalf},
            // Zero input: output cast term from maxOutputMagnitude=0 vanishes
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Half / Half / Half
// computeGamma selects linear vs statistical based on nU threshold (0.01).
// half epsilon = 2^-10.  C=1: nU=0.00195<0.01 -> linear.  C=10: nU=0.0195>=0.01 -> statistical.
template <>
std::vector<RMSNormFwdToleranceTestCase>
    getRMSNormFwdToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// nChannels = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // C=1: gamma via computeGamma (linear at this nU)
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gamma(1) / 2.0) + 5.0 * u},
            // C=10: gamma via computeGamma (statistical at this nU)
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gamma(10) / 2.0) + 5.0 * u},
            // Zero input: all terms vanish
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Bfloat16 / Float / Float (Output casting error: outputEpsilon(bf16=2^-7) > epsilon(float=2^-23))
template <>
std::vector<RMSNormFwdToleranceTestCase>
    getRMSNormFwdToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// nChannels = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // C=1
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gamma(1) / 2.0) + 5.0 * u + 1.0 * uBf16},
            // C=10
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gamma(10) / 2.0) + 5.0 * u + 1.0 * uBf16},
            // Zero input: output cast from maxOutputMagnitude=0 vanishes
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
// bf16 epsilon = 2^-7.  C=1: nU=0.0156>=0.01 -> statistical.
template <>
std::vector<RMSNormFwdToleranceTestCase>
    getRMSNormFwdToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// nChannels = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 0.0, 0.0, 0.0, true},
            // C=1
            {-1.0, 1.0, -1.0, 1.0, 1, 0.0, 0.0, (gamma(1) / 2.0) + 5.0 * u},
            // C=10
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, (gamma(10) / 2.0) + 5.0 * u},
            // Zero input: all terms vanish
            {0.0, 0.0, -1.0, 1.0, 10, 0.0, 0.0, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateRMSNormFwdTolerance
    : public ::testing::TestWithParam<RMSNormFwdToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW((calculateRMSNormFwdTolerance<Out, In, Comp>(params.xMin,
                                                                      params.xMax,
                                                                      params.scaleMin,
                                                                      params.scaleMax,
                                                                      params.nChannels,
                                                                      params.biasMin,
                                                                      params.biasMax)),
                         std::invalid_argument);
        }
        else
        {
            auto tol = calculateRMSNormFwdTolerance<Out, In, Comp>(params.xMin,
                                                                   params.xMax,
                                                                   params.scaleMin,
                                                                   params.scaleMax,
                                                                   params.nChannels,
                                                                   params.biasMin,
                                                                   params.biasMax);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalculateRMSNormFwdToleranceFp32 = TestCalculateRMSNormFwdTolerance<float, float, float>;
TEST_P(TestCalculateRMSNormFwdToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateRMSNormFwdToleranceFp32,
    ::testing::ValuesIn(getRMSNormFwdToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateRMSNormFwdToleranceInputDouble
    = TestCalculateRMSNormFwdTolerance<float, double, float>;
TEST_P(TestCalculateRMSNormFwdToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateRMSNormFwdToleranceInputDouble,
    ::testing::ValuesIn(getRMSNormFwdToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateRMSNormFwdToleranceComputeFloatFp16
    = TestCalculateRMSNormFwdTolerance<half, float, float>;
TEST_P(TestCalculateRMSNormFwdToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateRMSNormFwdToleranceComputeFloatFp16,
    ::testing::ValuesIn(getRMSNormFwdToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateRMSNormFwdToleranceFp16 = TestCalculateRMSNormFwdTolerance<half, half, half>;
TEST_P(TestCalculateRMSNormFwdToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateRMSNormFwdToleranceFp16,
    ::testing::ValuesIn(getRMSNormFwdToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalculateRMSNormFwdToleranceComputeFloatBfp16
    = TestCalculateRMSNormFwdTolerance<bfloat16, float, float>;
TEST_P(TestCalculateRMSNormFwdToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateRMSNormFwdToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getRMSNormFwdToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateRMSNormFwdToleranceBfp16
    = TestCalculateRMSNormFwdTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateRMSNormFwdToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateRMSNormFwdToleranceBfp16,
    ::testing::ValuesIn(
        getRMSNormFwdToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// Test that calculateRMSNormFwdTolerance catches simulated wrong outputs
TEST(TestCalculateRMSNormFwdTolerance, DetectsFailure)
{
    // C=100, x=[-1,1], scale=[-1,1]
    const std::vector<int64_t> dims = {1, 1, 10, 10};
    const std::vector<int64_t> strides = {100, 100, 10, 1};

    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.000001f); // Small error (1e-6 < tol ~1.25e-5)
    actualFailing->fillTensorWithValue(1.2f); // Large error

    auto tol = calculateRMSNormFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 100);

    // For FP32, C=100: tolerance should be small (dominated by nonlinear ops term)
    EXPECT_LT(tol, 0.01f);
    EXPECT_GT(tol, 1e-6f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateRMSNormFwdTolerance throws when gamma >= 0.5
TEST(TestCalculateRMSNormFwdTolerance, ThrowsOnSingularity)
{
    // For float, epsilon = 2^-23 ~ 1.19e-7.
    // computeGamma switches to statistical at nU >= 0.01.
    // C=5,000,000: nU = 2*5e6*2^-23 ~ 1.19 >= 0.01 -> statistical.
    // gamma = 6 * sqrt(2*5e6) * 2^-23 ~ 6 * 3162 * 1.19e-7 ~ 0.00226 < 0.5 (won't throw).
    // Need much larger C for gamma >= 0.5 with float statistical bound.
    // gamma = 6 * sqrt(2C) * u >= 0.5 -> C >= (0.5 / (6*u))^2 / 2
    // C >= (0.5 / (6 * 1.19e-7))^2 / 2 ~ 2.93e11. Use 3e11.
    EXPECT_THROW(
        (calculateRMSNormFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 300000000000LL)),
        std::overflow_error);

    // For bfloat16, epsilon = 2^-7.  gamma >= 0.5 at much smaller C.
    // gamma = 6*sqrt(2C)*u >= 0.5 -> C >= (0.5/(6*2^-7))^2/2 ~ 68.3. Use C=100.
    EXPECT_THROW(
        (calculateRMSNormFwdTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 100)),
        std::overflow_error);

    // For half, epsilon = 2^-10.  gamma >= 0.5 -> C >= (0.5/(6*2^-10))^2/2 ~ 4370. Use C=5000.
    EXPECT_THROW((calculateRMSNormFwdTolerance<half, half, half>(-1.0, 1.0, -1.0, 1.0, 5000)),
                 std::overflow_error);

    // float/double/float uses float epsilon -> same threshold as float/float/float
    EXPECT_THROW(
        (calculateRMSNormFwdTolerance<float, double, float>(-1.0, 1.0, -1.0, 1.0, 300000000000LL)),
        std::overflow_error);
}

// Test that calculateRMSNormFwdTolerance throws when tolerance exceeds OutputType max
TEST(TestCalculateRMSNormFwdTolerance, ThrowsOnOutputOverflow)
{
    // OutputType = half (max approx 65504)
    // Use very large scale to push tolerance above half max.
    // C=10, x=[-1,1], scale=[-1e8,1e8]
    // output cast term: maxAbsScale * uHalf = 1e8 * 2^-10 ~ 97656 >> 65504
    EXPECT_THROW((calculateRMSNormFwdTolerance<half, float, float>(-1.0, 1.0, -1e8, 1e8, 10)),
                 std::overflow_error);

    // half/half/half with very large scale: propagated tolerance exceeds half max (65504).
    // C=10, x=[-1000,1000], scale=[-1e7,1e7]:
    // gamma(10) ~ 0.0262, tol ~ gamma/2 * 1e7 + 5 * 2^-10 * 1e7 ~ 1.8e5 > 65504
    EXPECT_THROW((calculateRMSNormFwdTolerance<half, half, half>(-1000, 1000, -1e7, 1e7, 10)),
                 std::overflow_error);
}

// Test zero input (x=0) produces a valid small tolerance
TEST(TestCalculateRMSNormFwdTolerance, ZeroInput)
{
    // When x=0, maxAbsX=0, sumAbsProductBound=0, accTol=0
    // propagatedTolerance = 0 + 0 + maxAbsBias * epsilon
    auto tol = calculateRMSNormFwdTolerance<float, float, float>(0.0, 0.0, -1.0, 1.0, 10);
    EXPECT_EQ(tol, 0.0f);

    // With bias (float)
    auto tolBias
        = calculateRMSNormFwdTolerance<float, float, float>(0.0, 0.0, -1.0, 1.0, 10, -0.5, 0.5);
    EXPECT_GT(tolBias, 0.0f);

    // Zero input with bfloat16 output casting: outputEpsilon(bf16) > epsilon(float), but
    // maxOutputMagnitude=0, so output cast term also vanishes
    auto tolBf16 = calculateRMSNormFwdTolerance<bfloat16, float, float>(0.0, 0.0, -1.0, 1.0, 10);
    EXPECT_EQ(tolBf16, 0.0f);

    // Zero input with bias and half output: bias term + output cast of bias
    auto uFloat = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto tolHalfBias
        = calculateRMSNormFwdTolerance<half, float, float>(0.0, 0.0, -1.0, 1.0, 10, -0.5, 0.5);
    // Expected: maxAbsBias * epsilon + maxOutputMagnitude * outputEpsilon
    //         = 0.5 * uFloat + 0.5 * uHalf
    auto expectedHalfBias = static_cast<float>(0.5 * uFloat + 0.5 * uHalf);
    EXPECT_NEAR(tolHalfBias, expectedHalfBias, 1e-10f);
}

// Sanity check: tolerance scaling behavior with increasing C.
// For small C with FP32, computeGamma uses linear bound -> dominant term ~ C * u
// For BF16, computeGamma uses statistical bound -> dominant term ~ sqrt(C) * u
TEST(TestCalculateRMSNormFwdTolerance, ToleranceScalesWithChannels)
{
    // FP32: tolerance should grow roughly linearly with C (gamma/2 ~ C*u)
    auto tolFp32C1 = calculateRMSNormFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 1);
    auto tolFp32C10 = calculateRMSNormFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 10);
    auto tolFp32C100 = calculateRMSNormFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 100);

    EXPECT_LT(tolFp32C1, tolFp32C10) << "FP32: C=10 should have higher tolerance than C=1";
    EXPECT_LT(tolFp32C10, tolFp32C100) << "FP32: C=100 should have higher tolerance than C=10";

    // Ratio check: C=100 vs C=1 (use wider spread so gamma term dominates the constant 5u)
    auto ratioFp32 = static_cast<double>(tolFp32C100) / static_cast<double>(tolFp32C1);
    EXPECT_GT(ratioFp32, 10.0) << "FP32: ratio C100/C1 should reflect ~C growth";

    // BF16 (statistical): tolerance should grow ~ sqrt(C)
    // C=100 exceeds gamma >= 0.5 for bf16 (gamma ~ 0.663), so use C=50 as upper bound
    auto tolBf16C1
        = calculateRMSNormFwdTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 1);
    auto tolBf16C10
        = calculateRMSNormFwdTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 10);
    auto tolBf16C50
        = calculateRMSNormFwdTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, -1.0, 1.0, 50);

    EXPECT_LT(tolBf16C1, tolBf16C10) << "BF16: C=10 should have higher tolerance than C=1";
    EXPECT_LT(tolBf16C10, tolBf16C50) << "BF16: C=50 should have higher tolerance than C=10";

    auto ratioBf16 = static_cast<double>(tolBf16C10) / static_cast<double>(tolBf16C1);
    EXPECT_GT(ratioBf16, 1.5) << "BF16: ratio C10/C1 should reflect sqrt(C) growth";
}
