// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesSdpa.hpp>
#include <optional>
#include <vector>

using namespace hipdnn_test_sdk::utilities::sdpa;
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
// TestCalculateSdpaFwdTolerance
// =================================================================================================

struct SdpaFwdToleranceTestCase
{
    double qMin;
    double qMax;
    double kMin;
    double kMax;
    double vMin;
    double vMax;
    int64_t headDim;
    int64_t seqKv;
    std::optional<double> scale;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const SdpaFwdToleranceTestCase& tc)
    {
        os << "qMin: " << tc.qMin << ", qMax: " << tc.qMax << ", kMin: " << tc.kMin
           << ", kMax: " << tc.kMax << ", vMin: " << tc.vMin << ", vMax: " << tc.vMax
           << ", headDim: " << tc.headDim << ", seqKv: " << tc.seqKv << ", scale: ";
        if(tc.scale.has_value())
        {
            os << tc.scale.value();
        }
        else
        {
            os << "nullopt";
        }
        os << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<SdpaFwdToleranceTestCase> getSdpaFwdToleranceTestCases();

// Helper: compute the expected SDPA tolerance given parameters and epsilon.
// totalTolerance = scoreError * maxAbsV + softmaxError * maxAbsV + pvAccumError
//
// scoreError = gammaD * D * maxAbsQ * maxAbsK * absScale
//            + D * maxAbsQ * maxAbsK * absScale * epsilon
// softmaxError = gammaSkv + (Skv + 1) * epsilon
// pvAccumError = gammaSkv * maxAbsV
static double expectedSdpaTolerance(double maxAbsQ,
                                    double maxAbsK,
                                    double maxAbsV,
                                    int64_t headDim,
                                    int64_t seqKv,
                                    double absScale,
                                    double epsilon)
{
    const auto d = static_cast<uint64_t>(headDim);
    const auto skv = static_cast<uint64_t>(seqKv);

    const auto gammaD = computeGamma(d, epsilon);
    const auto gammaSkv = computeGamma(skv, epsilon);

    const double sumAbsProductBoundQk = static_cast<double>(d) * maxAbsQ * maxAbsK;

    const double scoreError
        = gammaD * sumAbsProductBoundQk * absScale + sumAbsProductBoundQk * absScale * epsilon;

    const double softmaxError = gammaSkv + (static_cast<double>(skv) + 1.0) * epsilon;

    const double pvAccumError = gammaSkv * maxAbsV;

    return scoreError * maxAbsV + softmaxError * maxAbsV + pvAccumError;
}

// Float / Float / Float
// For FP32 at small D and Skv, computeGamma uses linear bound: nU/(1-nU)
template <>
std::vector<SdpaFwdToleranceTestCase>
    getSdpaFwdToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());

    // headDim=0 => throws
    // seqKv=0 => throws
    // D=8, Skv=4, x/k/v in [-1,1], default scale = 1/sqrt(8)
    const double absScaleDefault8 = 1.0 / std::sqrt(8.0);

    return {// headDim = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 4, std::nullopt, 0.0, true},
            // seqKv = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 0, std::nullopt, 0.0, true},
            // D=8, Skv=4, all in [-1,1], default scale
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, absScaleDefault8, u)},
            // D=64, Skv=128, all in [-1,1], default scale
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             64,
             128,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 64, 128, 1.0 / std::sqrt(64.0), u)},
            // D=8, Skv=4, all in [0,1], default scale (matches test tensor init range)
            {0.0,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, absScaleDefault8, u)},
            // Custom scale = 1.0
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             1.0,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, 1.0, u)}};
}

// Float / Double / Float (Input casting: inputEpsilon(double) < epsilon(float))
// Adds per-stage casting errors:
//   castErrorQkt = 2 * sumAbsProductBoundQk * absScale * epsilon -> * maxAbsV
//   castErrorPv  = 1 * maxAbsV * epsilon
template <>
std::vector<SdpaFwdToleranceTestCase>
    getSdpaFwdToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    const double absScaleDefault8 = 1.0 / std::sqrt(8.0);

    auto expectedWithCast = [&](int64_t headDimVal, int64_t seqKvVal, double absScale) {
        const double base = expectedSdpaTolerance(1.0, 1.0, 1.0, headDimVal, seqKvVal, absScale, u);
        const double sumAbsProductBoundQk = static_cast<double>(headDimVal) * 1.0 * 1.0;
        const double castQkt = 2.0 * sumAbsProductBoundQk * absScale * u; // * maxAbsV=1
        const double castPv = 1.0 * 1.0 * u; // maxAbsV * epsilon
        return base + castQkt * 1.0 + castPv;
    };

    return {// headDim = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 4, std::nullopt, 0.0, true},
            // D=8, Skv=4, all in [-1,1], default scale
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedWithCast(8, 4, absScaleDefault8)},
            // D=64, Skv=128, all in [-1,1]
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             64,
             128,
             std::nullopt,
             expectedWithCast(64, 128, 1.0 / std::sqrt(64.0))},
            // Zero V: all error terms involving maxAbsV vanish
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4, std::nullopt, 0.0}};
}

// Half / Float / Float (Output casting: outputEpsilon(half=2^-10) > epsilon(float=2^-23))
// Adds: maxAbsV * outputEpsilon
template <>
std::vector<SdpaFwdToleranceTestCase> getSdpaFwdToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double absScaleDefault8 = 1.0 / std::sqrt(8.0);

    return {// headDim = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 4, std::nullopt, 0.0, true},
            // D=8, Skv=4, all in [-1,1]
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, absScaleDefault8, u) + 1.0 * uHalf},
            // Zero V: base tolerance vanishes, output cast also vanishes (maxOutputMagnitude=0)
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4, std::nullopt, 0.0}};
}

// Half / Half / Half
// half epsilon = 2^-10. D=8: nU = 2*8*2^-10 = 0.015625 >= 0.01 -> statistical.
template <>
std::vector<SdpaFwdToleranceTestCase> getSdpaFwdToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    const double absScaleDefault8 = 1.0 / std::sqrt(8.0);

    return {// headDim = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 4, std::nullopt, 0.0, true},
            // D=8, Skv=4, all in [-1,1]
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, absScaleDefault8, u)},
            // Zero V: all terms vanish
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4, std::nullopt, 0.0}};
}

// Bfloat16 / Float / Float (Output casting: outputEpsilon(bf16=2^-7) > epsilon(float=2^-23))
template <>
std::vector<SdpaFwdToleranceTestCase>
    getSdpaFwdToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    const double absScaleDefault8 = 1.0 / std::sqrt(8.0);

    return {// headDim = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 4, std::nullopt, 0.0, true},
            // D=8, Skv=4, all in [-1,1]
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, absScaleDefault8, u) + 1.0 * uBf16},
            // Zero V: output cast vanishes too
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4, std::nullopt, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
// bf16 epsilon = 2^-7. D=1: nU = 2*1*2^-7 = 0.015625 >= 0.01 -> statistical.
template <>
std::vector<SdpaFwdToleranceTestCase>
    getSdpaFwdToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    const double absScaleDefault8 = 1.0 / std::sqrt(8.0);

    return {// headDim = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 4, std::nullopt, 0.0, true},
            // D=8, Skv=4, all in [-1,1]
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             8,
             4,
             std::nullopt,
             expectedSdpaTolerance(1.0, 1.0, 1.0, 8, 4, absScaleDefault8, u)},
            // Zero V: all terms vanish
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4, std::nullopt, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateSdpaFwdTolerance : public ::testing::TestWithParam<SdpaFwdToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW((calculateSdpaFwdTolerance<Out, In, Comp>(params.qMin,
                                                                   params.qMax,
                                                                   params.kMin,
                                                                   params.kMax,
                                                                   params.vMin,
                                                                   params.vMax,
                                                                   params.headDim,
                                                                   params.seqKv,
                                                                   params.scale)),
                         std::invalid_argument);
        }
        else
        {
            auto tol = calculateSdpaFwdTolerance<Out, In, Comp>(params.qMin,
                                                                params.qMax,
                                                                params.kMin,
                                                                params.kMax,
                                                                params.vMin,
                                                                params.vMax,
                                                                params.headDim,
                                                                params.seqKv,
                                                                params.scale);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalculateSdpaFwdToleranceFp32 = TestCalculateSdpaFwdTolerance<float, float, float>;
TEST_P(TestCalculateSdpaFwdToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateSdpaFwdToleranceFp32,
    ::testing::ValuesIn(getSdpaFwdToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalculateSdpaFwdToleranceInputDouble
    = TestCalculateSdpaFwdTolerance<float, double, float>;
TEST_P(TestCalculateSdpaFwdToleranceInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateSdpaFwdToleranceInputDouble,
    ::testing::ValuesIn(getSdpaFwdToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalculateSdpaFwdToleranceComputeFloatFp16
    = TestCalculateSdpaFwdTolerance<half, float, float>;
TEST_P(TestCalculateSdpaFwdToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateSdpaFwdToleranceComputeFloatFp16,
    ::testing::ValuesIn(getSdpaFwdToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalculateSdpaFwdToleranceFp16 = TestCalculateSdpaFwdTolerance<half, half, half>;
TEST_P(TestCalculateSdpaFwdToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateSdpaFwdToleranceFp16,
    ::testing::ValuesIn(getSdpaFwdToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalculateSdpaFwdToleranceComputeFloatBfp16
    = TestCalculateSdpaFwdTolerance<bfloat16, float, float>;
TEST_P(TestCalculateSdpaFwdToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateSdpaFwdToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getSdpaFwdToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalculateSdpaFwdToleranceBfp16
    = TestCalculateSdpaFwdTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalculateSdpaFwdToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateSdpaFwdToleranceBfp16,
    ::testing::ValuesIn(getSdpaFwdToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// Test that calculateSdpaFwdTolerance catches simulated wrong outputs
TEST(TestCalculateSdpaFwdTolerance, DetectsFailure)
{
    // D=8, Skv=4, all in [-1,1]
    const std::vector<int64_t> dims = {1, 1, 4, 8}; // [B, H, Sq, D]
    const std::vector<int64_t> strides = {32, 32, 8, 1};

    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

    baseline->fillTensorWithValue(0.5f);
    actualPassing->fillTensorWithValue(0.500001f); // Small error
    actualFailing->fillTensorWithValue(1.5f); // Large error

    auto tol
        = calculateSdpaFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 4);

    // Tolerance should be small but nonzero for FP32
    EXPECT_LT(tol, 0.01f);
    EXPECT_GT(tol, 1e-7f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that calculateSdpaFwdTolerance throws when gamma >= 0.5
TEST(TestCalculateSdpaFwdTolerance, ThrowsOnSingularity)
{
    // For float, epsilon = 2^-23. gamma >= 0.5 with statistical bound at C ~ 2.93e11.
    // headDim triggers gamma_D validation first.
    EXPECT_THROW((calculateSdpaFwdTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 300000000000LL, 4)),
                 std::overflow_error);

    // seqKv triggers gamma_Skv validation.
    EXPECT_THROW((calculateSdpaFwdTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 300000000000LL)),
                 std::overflow_error);

    // For bfloat16, epsilon = 2^-7. gamma >= 0.5 at C ~ 68.
    EXPECT_THROW((calculateSdpaFwdTolerance<bfloat16, bfloat16, bfloat16>(
                     -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 100, 4)),
                 std::overflow_error);

    // For half, epsilon = 2^-10. gamma >= 0.5 at C ~ 4370.
    EXPECT_THROW(
        (calculateSdpaFwdTolerance<half, half, half>(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 5000, 4)),
        std::overflow_error);
}

// Test that calculateSdpaFwdTolerance throws when tolerance exceeds OutputType max
TEST(TestCalculateSdpaFwdTolerance, ThrowsOnOutputOverflow)
{
    // OutputType = half (max approx 65504)
    // Use very large V to push output cast term above half max.
    // maxAbsV = 1e8, output cast = 1e8 * 2^-10 ~ 97656 > 65504
    EXPECT_THROW(
        (calculateSdpaFwdTolerance<half, float, float>(-1.0, 1.0, -1.0, 1.0, -1e8, 1e8, 8, 4)),
        std::overflow_error);
}

// Test zero V input produces zero tolerance
TEST(TestCalculateSdpaFwdTolerance, ZeroVInput)
{
    // When V=0, maxAbsV=0. All output terms are proportional to maxAbsV => tolerance = 0.
    auto tol = calculateSdpaFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4);
    EXPECT_EQ(tol, 0.0f);

    // With bfloat16 output: output cast of maxAbsV=0 also vanishes
    auto tolBf16
        = calculateSdpaFwdTolerance<bfloat16, float, float>(-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4);
    EXPECT_EQ(tolBf16, 0.0f);

    // With input casting (double -> float): cast errors also scale by maxAbsV => 0
    auto tolCast
        = calculateSdpaFwdTolerance<float, double, float>(-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 8, 4);
    EXPECT_EQ(tolCast, 0.0f);
}

// Sanity check: tolerance scaling behavior with increasing dimensions.
// Stage 1 (Q@K^T) scales with D. Stages 2+3 (softmax + P@V) scale with Skv.
TEST(TestCalculateSdpaFwdTolerance, ToleranceScalesWithDimensions)
{
    // FP32: varying headDim (D) with fixed Skv=4, scale=1.0 to isolate Stage 1
    auto tolD1 = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1, 4, 1.0);
    auto tolD10 = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 4, 1.0);
    auto tolD100 = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 100, 4, 1.0);

    EXPECT_LT(tolD1, tolD10) << "FP32: D=10 should have higher tolerance than D=1";
    EXPECT_LT(tolD10, tolD100) << "FP32: D=100 should have higher tolerance than D=10";

    // Verify D dominates Stage 1: ratio D100/D1 should reflect D growth
    auto ratioD = static_cast<double>(tolD100) / static_cast<double>(tolD1);
    EXPECT_GT(ratioD, 5.0) << "FP32: ratio D100/D1 should reflect D growth in Stage 1";

    // FP32: varying seqKv (Skv) with fixed D=8, scale=1.0 to isolate Stages 2+3
    auto tolSkv1 = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 1, 1.0);
    auto tolSkv10 = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 10, 1.0);
    auto tolSkv100 = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 8, 100, 1.0);

    EXPECT_LT(tolSkv1, tolSkv10) << "FP32: Skv=10 should have higher tolerance than Skv=1";
    EXPECT_LT(tolSkv10, tolSkv100) << "FP32: Skv=100 should have higher tolerance than Skv=10";

    // BF16: verify tolerance grows with D (statistical bound -> sqrt(D) growth)
    auto tolBf16D1 = calculateSdpaFwdTolerance<bfloat16, bfloat16, bfloat16>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1, 4, 1.0);
    auto tolBf16D10 = calculateSdpaFwdTolerance<bfloat16, bfloat16, bfloat16>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 4, 1.0);
    // D=50 stays under gamma_D < 0.5 for bf16 (gamma = 6*sqrt(100)*2^-7 ~ 0.469)
    auto tolBf16D50 = calculateSdpaFwdTolerance<bfloat16, bfloat16, bfloat16>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 50, 4, 1.0);

    EXPECT_LT(tolBf16D1, tolBf16D10) << "BF16: D=10 should have higher tolerance than D=1";
    EXPECT_LT(tolBf16D10, tolBf16D50) << "BF16: D=50 should have higher tolerance than D=10";
}

// Test custom scale vs default scale
TEST(TestCalculateSdpaFwdTolerance, CustomScaleVsDefault)
{
    // Default scale = 1/sqrt(64) = 0.125
    auto tolDefault
        = calculateSdpaFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 64, 128);

    // Custom scale = 1.0 (8x larger than default for D=64)
    auto tolLargeScale = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 64, 128, 1.0);

    // Larger scale amplifies Stage 1 score error, so tolerance should be larger
    EXPECT_GT(tolLargeScale, tolDefault)
        << "Larger scale should produce larger tolerance (Stage 1 amplified)";

    // Custom scale = 0.0 should produce a valid (small) tolerance, not trigger default
    auto tolZeroScale = calculateSdpaFwdTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 64, 128, 0.0);

    EXPECT_LT(tolZeroScale, tolDefault)
        << "scale=0.0 zeroes Stage 1, should be smaller than default";
}

// D=1 and Skv=1: minimal accumulation, tolerance should be very small for FP32
TEST(TestCalculateSdpaFwdTolerance, MinimalDimensions)
{
    auto tol
        = calculateSdpaFwdTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1, 1);

    // Should be nonzero but small
    EXPECT_GT(tol, 0.0f);
    EXPECT_LT(tol, 1e-5f) << "D=1, Skv=1 with FP32 should produce very small tolerance";
}
