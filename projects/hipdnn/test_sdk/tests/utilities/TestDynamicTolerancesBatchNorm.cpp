// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesBatchNorm.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerancesCommon.hpp>
#include <vector>

using namespace hipdnn_test_sdk::utilities::batchnorm;
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
// TestCalculateBatchnormTrainingTolerance
// =================================================================================================

struct BnTrainToleranceTestCase
{
    double xMin;
    double xMax;
    double scaleMin;
    double scaleMax;
    double biasMin;
    double biasMax;
    int64_t nElementsPerChannel;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnTrainToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", scaleMin: " << tc.scaleMin
           << ", scaleMax: " << tc.scaleMax << ", biasMin: " << tc.biasMin
           << ", biasMax: " << tc.biasMax << ", NHW: " << tc.nElementsPerChannel
           << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnTrainToleranceTestCase> getBnTrainToleranceTestCases();

// BN Training tolerance formula (no casting, maxAbsX>0):
//   accTol = (gamma + 2u) * S     (Higham + 2u for div/sub variance ops)
//   propTol = accTol/(2*S) * scale + gamma * scale + 6u * scale + u * bias
//           = ((gamma + 2u)/2 + gamma + 6u) * maxAbsScale + u * maxAbsBias
//           = (3/2*gamma + u + 6u) * maxAbsScale + u * maxAbsBias
//           = (3/2*gamma + 7u) * maxAbsScale + u * maxAbsBias
// Output casting: maxOutputMagnitude = maxAbsScale * maxAbsX * invVarEst + maxAbsBias
//   where invVarEst = 1/sqrt(max(maxAbsX^2, 1e-5) + 1e-5)
// With input casting (InputType=double, ComputeType=float):
//   accTol = (gamma + 2u + u) * S -> variance path = (gamma + 3u)/2
//   total = ((gamma + 3u)/2 + gamma + 6u) * maxAbsScale + u * maxAbsBias
//         = (3/2*gamma + 3u/2 + 6u) * maxAbsScale + u * maxAbsBias

// Helper to compute output cast contribution for y tolerance
static double yOutputCast(double maxAbsX,
                          double maxAbsScale,
                          double maxAbsBias,
                          double outputEpsilon,
                          double computeEpsilon,
                          double epsBn = 1e-5)
{
    if(outputEpsilon <= computeEpsilon)
    {
        return 0.0;
    }
    const double varEst = std::max(maxAbsX * maxAbsX, epsBn);
    const double invVarEst = 1.0 / std::sqrt(varEst + epsBn);
    const double maxOut = (maxAbsX > 0.0 ? maxAbsScale * maxAbsX * invVarEst : 0.0) + maxAbsBias;
    return maxOut * outputEpsilon;
}

// Float / Float / Float
template <>
std::vector<BnTrainToleranceTestCase>
    getBnTrainToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0, 0.0, true},
            // NHW=1, x=[-1,1], scale=[-1,1], no bias
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1, (1.5 * gamma(1) + 7.0 * u) * 1.0},
            // NHW=10, x=[-1,1], scale=[-1,1], no bias
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, (1.5 * gamma(10) + 7.0 * u) * 1.0},
            // NHW=10, x=[-1000,1000], scale=[-1000,1000], no bias
            // Tolerance scales with |scale|, not |x| (normalization cancels |x|)
            {-1000.0, 1000.0, -1000.0, 1000.0, 0.0, 0.0, 10, (1.5 * gamma(10) + 7.0 * u) * 1000.0},
            // NHW=3, x=[-1,1], scale=[-1,1], bias=[-0.5,0.5]
            {-1.0, 1.0, -1.0, 1.0, -0.5, 0.5, 3, (1.5 * gamma(3) + 7.0 * u) * 1.0 + 0.5 * u}};
}

// Float / Double / Float (Input casting: inputEpsilon(double) < epsilon(float))
// accTol = (gamma + 2u) * S + u * S (input cast) = (gamma + 3u) * S
// variance path = (gamma + 3u)/2, then + gamma (mean) + 6u (ops)
// total = (3/2*gamma + 3u/2 + 6u) * scale
template <>
std::vector<BnTrainToleranceTestCase>
    getBnTrainToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0, 0.0, true},
            // NHW=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1,
             ((gamma(1) + 3.0 * u) / 2.0 + gamma(1) + 6.0 * u) * 1.0},
            // NHW=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             10,
             ((gamma(10) + 3.0 * u) / 2.0 + gamma(10) + 6.0 * u) * 1.0},
            // Zero input: all terms vanish
            {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10, 0.0}};
}

// Half / Float / Float (Output casting: outputEpsilon(half=2^-10) > epsilon(float=2^-23))
// maxOutputMagnitude = maxAbsScale * maxAbsX * invVarEst (conservative, not assuming xHat~O(1))
template <>
std::vector<BnTrainToleranceTestCase> getBnTrainToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    auto yCast = [&](double maxAbsX, double maxAbsScale, double maxAbsBias) {
        return yOutputCast(maxAbsX, maxAbsScale, maxAbsBias, uHalf, u);
    };
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0, 0.0, true},
            // NHW=1: base + output cast
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1,
             (1.5 * gamma(1) + 7.0 * u) * 1.0 + yCast(1.0, 1.0, 0.0)},
            // NHW=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             10,
             (1.5 * gamma(10) + 7.0 * u) * 1.0 + yCast(1.0, 1.0, 0.0)},
            // Zero input: output cast from maxOutputMagnitude=0 vanishes
            {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10, 0.0}};
}

// Half / Half / Half
template <>
std::vector<BnTrainToleranceTestCase> getBnTrainToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1, (1.5 * gamma(1) + 7.0 * u) * 1.0},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, (1.5 * gamma(10) + 7.0 * u) * 1.0},
            // Zero input
            {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10, 0.0}};
}

// Bfloat16 / Float / Float (Output casting: outputEpsilon(bf16=2^-7) > epsilon(float=2^-23))
template <>
std::vector<BnTrainToleranceTestCase>
    getBnTrainToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    auto yCast = [&](double maxAbsX, double maxAbsScale, double maxAbsBias) {
        return yOutputCast(maxAbsX, maxAbsScale, maxAbsBias, uBf16, u);
    };
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0, 0.0, true},
            // NHW=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1,
             (1.5 * gamma(1) + 7.0 * u) * 1.0 + yCast(1.0, 1.0, 0.0)},
            // NHW=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             10,
             (1.5 * gamma(10) + 7.0 * u) * 1.0 + yCast(1.0, 1.0, 0.0)},
            // Zero input
            {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnTrainToleranceTestCase>
    getBnTrainToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1, (1.5 * gamma(1) + 7.0 * u) * 1.0},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, (1.5 * gamma(10) + 7.0 * u) * 1.0},
            // Zero input
            {0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnTrainTolerance : public ::testing::TestWithParam<BnTrainToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateBatchnormTrainingTolerance<Out, In, Comp>(params.xMin,
                                                                    params.xMax,
                                                                    params.scaleMin,
                                                                    params.scaleMax,
                                                                    params.biasMin,
                                                                    params.biasMax,
                                                                    params.nElementsPerChannel)),
                std::invalid_argument);
        }
        else
        {
            auto tol
                = calculateBatchnormTrainingTolerance<Out, In, Comp>(params.xMin,
                                                                     params.xMax,
                                                                     params.scaleMin,
                                                                     params.scaleMax,
                                                                     params.biasMin,
                                                                     params.biasMax,
                                                                     params.nElementsPerChannel);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnTrainTolFp32 = TestCalculateBnTrainTolerance<float, float, float>;
TEST_P(TestCalcBnTrainTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnTrainTolFp32,
    ::testing::ValuesIn(getBnTrainToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnTrainTolInputDouble = TestCalculateBnTrainTolerance<float, double, float>;
TEST_P(TestCalcBnTrainTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnTrainTolInputDouble,
    ::testing::ValuesIn(getBnTrainToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnTrainTolComputeFloatFp16 = TestCalculateBnTrainTolerance<half, float, float>;
TEST_P(TestCalcBnTrainTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnTrainTolComputeFloatFp16,
    ::testing::ValuesIn(getBnTrainToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnTrainTolFp16 = TestCalculateBnTrainTolerance<half, half, half>;
TEST_P(TestCalcBnTrainTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnTrainTolFp16,
    ::testing::ValuesIn(getBnTrainToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnTrainTolComputeFloatBfp16 = TestCalculateBnTrainTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnTrainTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnTrainTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnTrainToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnTrainTolBfp16 = TestCalculateBnTrainTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnTrainTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnTrainTolBfp16,
    ::testing::ValuesIn(getBnTrainToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// TestCalculateBatchnormMeanTolerance
// =================================================================================================

struct BnMeanToleranceTestCase
{
    double xMin;
    double xMax;
    int64_t nElementsPerChannel;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnMeanToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", NHW: " << tc.nElementsPerChannel
           << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnMeanToleranceTestCase> getBnMeanToleranceTestCases();

// Mean tolerance formula (no casting):
//   tolerance = (gamma + u) * maxAbsX
// With input casting (double->float):
//   tolerance += computeEpsilon * maxAbsX (from inputCastError / NHW)
//   total = (gamma + 2u) * maxAbsX

// Float / Float / Float
template <>
std::vector<BnMeanToleranceTestCase> getBnMeanToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, (gamma(1) + u) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, (gamma(10) + u) * 1.0},
            // Large x range
            {-1000.0, 1000.0, 10, (gamma(10) + u) * 1000.0},
            // Zero input
            {0.0, 0.0, 10, 0.0}};
}

// Float / Double / Float
template <>
std::vector<BnMeanToleranceTestCase> getBnMeanToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1: input casting adds u * maxAbsX
            {-1.0, 1.0, 1, (gamma(1) + 2.0 * u) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, (gamma(10) + 2.0 * u) * 1.0},
            // Zero input
            {0.0, 0.0, 10, 0.0}};
}

// Half / Float / Float (Output casting)
template <>
std::vector<BnMeanToleranceTestCase> getBnMeanToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1: base + output casting (maxAbsX * uHalf)
            {-1.0, 1.0, 1, (gamma(1) + u) * 1.0 + 1.0 * uHalf},
            // NHW=10
            {-1.0, 1.0, 10, (gamma(10) + u) * 1.0 + 1.0 * uHalf},
            // Zero input
            {0.0, 0.0, 10, 0.0}};
}

// Half / Half / Half
template <>
std::vector<BnMeanToleranceTestCase> getBnMeanToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, (gamma(1) + u) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, (gamma(10) + u) * 1.0},
            // Zero input
            {0.0, 0.0, 10, 0.0}};
}

// Bfloat16 / Float / Float (Output casting)
template <>
std::vector<BnMeanToleranceTestCase>
    getBnMeanToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, (gamma(1) + u) * 1.0 + 1.0 * uBf16},
            // NHW=10
            {-1.0, 1.0, 10, (gamma(10) + u) * 1.0 + 1.0 * uBf16},
            // Zero input
            {0.0, 0.0, 10, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnMeanToleranceTestCase>
    getBnMeanToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, (gamma(1) + u) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, (gamma(10) + u) * 1.0},
            // Zero input
            {0.0, 0.0, 10, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnMeanTolerance : public ::testing::TestWithParam<BnMeanToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW((calculateBatchnormMeanTolerance<Out, In, Comp>(
                             params.xMin, params.xMax, params.nElementsPerChannel)),
                         std::invalid_argument);
        }
        else
        {
            auto tol = calculateBatchnormMeanTolerance<Out, In, Comp>(
                params.xMin, params.xMax, params.nElementsPerChannel);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnMeanTolFp32 = TestCalculateBnMeanTolerance<float, float, float>;
TEST_P(TestCalcBnMeanTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnMeanTolFp32,
    ::testing::ValuesIn(getBnMeanToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnMeanTolInputDouble = TestCalculateBnMeanTolerance<float, double, float>;
TEST_P(TestCalcBnMeanTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnMeanTolInputDouble,
    ::testing::ValuesIn(getBnMeanToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnMeanTolComputeFloatFp16 = TestCalculateBnMeanTolerance<half, float, float>;
TEST_P(TestCalcBnMeanTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnMeanTolComputeFloatFp16,
    ::testing::ValuesIn(getBnMeanToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnMeanTolFp16 = TestCalculateBnMeanTolerance<half, half, half>;
TEST_P(TestCalcBnMeanTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnMeanTolFp16,
    ::testing::ValuesIn(getBnMeanToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnMeanTolComputeFloatBfp16 = TestCalculateBnMeanTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnMeanTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnMeanTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnMeanToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnMeanTolBfp16 = TestCalculateBnMeanTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnMeanTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnMeanTolBfp16,
    ::testing::ValuesIn(getBnMeanToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// TestCalculateBatchnormInvVarianceTolerance
// =================================================================================================

struct BnInvVarToleranceTestCase
{
    double xMin;
    double xMax;
    int64_t nElementsPerChannel;
    double epsilonBn;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnInvVarToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", NHW: " << tc.nElementsPerChannel
           << ", epsilonBn: " << tc.epsilonBn << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnInvVarToleranceTestCase> getBnInvVarToleranceTestCases();

// InvVar tolerance formula (corrected, see plan Section 15):
//   deltaVariance = (3*gamma + 2u) * maxAbsX^2
//   varEstimate = max(maxAbsX^2, epsilonBn)
//   varPlusEps = varEstimate + epsilonBn
//   tolerance = deltaVariance / (2 * varPlusEps^{3/2}) + 3u / sqrt(varPlusEps)
// With output casting: + invVarEstimate * outputEpsilon
//   where invVarEstimate = 1/sqrt(varPlusEps)
// For x=0: varEstimate = epsilonBn, varPlusEps = 2*epsilonBn

// Helper to compute expected invVar tolerance
static double expectedInvVarTol(double maxAbsX,
                                double epsBn,
                                double gamma,
                                double u,
                                double outputEpsilon = 0.0,
                                double computeEpsilon = 0.0)
{
    const double varEst = std::max(maxAbsX * maxAbsX, epsBn);
    const double varPlusEps = varEst + epsBn;
    const double deltaVar = (3.0 * gamma + 2.0 * u) * maxAbsX * maxAbsX;
    const double invVarEst = 1.0 / std::sqrt(varPlusEps);
    double tol = deltaVar / (2.0 * varPlusEps * std::sqrt(varPlusEps)) + 3.0 * u * invVarEst;
    if(outputEpsilon > computeEpsilon)
    {
        tol += invVarEst * outputEpsilon;
    }
    return tol;
}

// Float / Float / Float
template <>
std::vector<BnInvVarToleranceTestCase>
    getBnInvVarToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1, x=[-1,1]
            {-1.0, 1.0, 1, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(1), u)},
            // NHW=10
            {-1.0, 1.0, 10, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(10), u)},
            // Large x range: tolerance now properly scales down with maxAbsX
            {-1000.0, 1000.0, 10, 1e-5, expectedInvVarTol(1000.0, 1e-5, gamma(10), u)},
            // Zero input: varEstimate = epsBn, varPlusEps = 2*epsBn
            {0.0, 0.0, 10, 1e-5, expectedInvVarTol(0.0, 1e-5, gamma(10), u)}};
}

// Float / Double / Float (no output casting, no input casting for invVar)
template <>
std::vector<BnInvVarToleranceTestCase>
    getBnInvVarToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(1), u)},
            // NHW=10
            {-1.0, 1.0, 10, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(10), u)},
            // Zero input
            {0.0, 0.0, 10, 1e-5, expectedInvVarTol(0.0, 1e-5, gamma(10), u)}};
}

// Half / Float / Float (Output casting)
template <>
std::vector<BnInvVarToleranceTestCase>
    getBnInvVarToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(1), u, uHalf, u)},
            // NHW=10
            {-1.0, 1.0, 10, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(10), u, uHalf, u)},
            // Zero input
            {0.0, 0.0, 10, 1e-5, expectedInvVarTol(0.0, 1e-5, gamma(10), u, uHalf, u)}};
}

// Half / Half / Half
template <>
std::vector<BnInvVarToleranceTestCase> getBnInvVarToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(1), u)},
            // NHW=10
            {-1.0, 1.0, 10, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(10), u)},
            // Zero input
            {0.0, 0.0, 10, 1e-5, expectedInvVarTol(0.0, 1e-5, gamma(10), u)}};
}

// Bfloat16 / Float / Float
template <>
std::vector<BnInvVarToleranceTestCase>
    getBnInvVarToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(1), u, uBf16, u)},
            // NHW=10
            {-1.0, 1.0, 10, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(10), u, uBf16, u)},
            // Zero input
            {0.0, 0.0, 10, 1e-5, expectedInvVarTol(0.0, 1e-5, gamma(10), u, uBf16, u)}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnInvVarToleranceTestCase>
    getBnInvVarToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(1), u)},
            // NHW=10
            {-1.0, 1.0, 10, 1e-5, expectedInvVarTol(1.0, 1e-5, gamma(10), u)},
            // Zero input
            {0.0, 0.0, 10, 1e-5, expectedInvVarTol(0.0, 1e-5, gamma(10), u)}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnInvVarTolerance : public ::testing::TestWithParam<BnInvVarToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateBatchnormInvVarianceTolerance<Out, In, Comp>(
                    params.xMin, params.xMax, params.nElementsPerChannel, params.epsilonBn)),
                std::invalid_argument);
        }
        else
        {
            auto tol = calculateBatchnormInvVarianceTolerance<Out, In, Comp>(
                params.xMin, params.xMax, params.nElementsPerChannel, params.epsilonBn);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnInvVarTolFp32 = TestCalculateBnInvVarTolerance<float, float, float>;
TEST_P(TestCalcBnInvVarTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInvVarTolFp32,
    ::testing::ValuesIn(getBnInvVarToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnInvVarTolInputDouble = TestCalculateBnInvVarTolerance<float, double, float>;
TEST_P(TestCalcBnInvVarTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInvVarTolInputDouble,
    ::testing::ValuesIn(getBnInvVarToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnInvVarTolComputeFloatFp16 = TestCalculateBnInvVarTolerance<half, float, float>;
TEST_P(TestCalcBnInvVarTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInvVarTolComputeFloatFp16,
    ::testing::ValuesIn(getBnInvVarToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnInvVarTolFp16 = TestCalculateBnInvVarTolerance<half, half, half>;
TEST_P(TestCalcBnInvVarTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInvVarTolFp16,
    ::testing::ValuesIn(getBnInvVarToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnInvVarTolComputeFloatBfp16 = TestCalculateBnInvVarTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnInvVarTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInvVarTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnInvVarToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnInvVarTolBfp16 = TestCalculateBnInvVarTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnInvVarTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInvVarTolBfp16,
    ::testing::ValuesIn(getBnInvVarToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// Standalone Tests
// =================================================================================================

// Test that calculateBatchnormTrainingTolerance catches simulated wrong outputs
TEST(TestCalculateBnTrainTolerance, DetectsFailure)
{
    // NHW=100 (shape [1,1,10,10]), x=[-1,1], scale=[-1,1]
    const std::vector<int64_t> dims = {1, 1, 10, 10};
    const std::vector<int64_t> strides = {100, 100, 10, 1};

    auto baseline = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualPassing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);
    auto actualFailing = hipdnn_data_sdk::utilities::createTensor<float>(dims, strides);

    baseline->fillTensorWithValue(1.0f);
    actualPassing->fillTensorWithValue(1.000001f); // Small error
    actualFailing->fillTensorWithValue(1.2f); // Large error

    auto tol = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 100);

    EXPECT_LT(tol, 0.01f);
    EXPECT_GT(tol, 1e-6f);

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tol, 0);

    bool valid = validator->allClose(*baseline, *actualPassing);
    EXPECT_TRUE(valid);

    valid = validator->allClose(*baseline, *actualFailing);
    EXPECT_FALSE(valid);
}

// Test that tolerance throws when gamma >= 0.5
TEST(TestCalculateBnTrainTolerance, ThrowsOnSingularity)
{
    // float: gamma >= 0.5 at C ~ 3e11 (same as RMSNorm)
    EXPECT_THROW((calculateBatchnormTrainingTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 300000000000LL)),
                 std::overflow_error);

    // bfloat16: gamma >= 0.5 at C ~ 69. Use C=100.
    EXPECT_THROW((calculateBatchnormTrainingTolerance<bfloat16, bfloat16, bfloat16>(
                     -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 100)),
                 std::overflow_error);

    // half: gamma >= 0.5 at C ~ 4370. Use C=5000.
    EXPECT_THROW((calculateBatchnormTrainingTolerance<half, half, half>(
                     -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 5000)),
                 std::overflow_error);
}

// Test that tolerance throws when it exceeds OutputType max
TEST(TestCalculateBnTrainTolerance, ThrowsOnOutputOverflow)
{
    // half output with very large scale: output cast term >> 65504
    EXPECT_THROW((calculateBatchnormTrainingTolerance<half, float, float>(
                     -1.0, 1.0, -1e8, 1e8, 0.0, 0.0, 10)),
                 std::overflow_error);
}

// Test zero input produces correct minimal tolerance
TEST(TestCalculateBnTrainTolerance, ZeroInput)
{
    // No bias: all terms vanish
    auto tol = calculateBatchnormTrainingTolerance<float, float, float>(
        0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10);
    EXPECT_EQ(tol, 0.0f);

    // With bias: only bias term survives
    auto tolBias = calculateBatchnormTrainingTolerance<float, float, float>(
        0.0, 0.0, -1.0, 1.0, -0.5, 0.5, 10);
    EXPECT_GT(tolBias, 0.0f);

    // Zero input with bf16 output casting: maxOutputMagnitude=0 -> no output cast
    auto tolBf16 = calculateBatchnormTrainingTolerance<bfloat16, float, float>(
        0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 10);
    EXPECT_EQ(tolBf16, 0.0f);

    // Zero input with bias and half output: bias term + output cast of bias
    auto uFloat = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto tolHalfBias = calculateBatchnormTrainingTolerance<half, float, float>(
        0.0, 0.0, -1.0, 1.0, -0.5, 0.5, 10);
    // Expected: maxAbsBias * epsilon + maxOutputMagnitude * outputEpsilon
    //         = 0.5 * uFloat + 0.5 * uHalf  (maxOutputMagnitude = 0 + maxAbsBias = 0.5)
    auto expectedHalfBias
        = static_cast<float>(0.5 * uFloat + yOutputCast(0.0, 1.0, 0.5, uHalf, uFloat));
    EXPECT_NEAR(tolHalfBias, expectedHalfBias, 1e-10f);
}

// Sanity check: tolerance scaling behavior with increasing NHW
TEST(TestCalculateBnTrainTolerance, ToleranceScalesWithNHW)
{
    // FP32: tolerance should grow roughly linearly with NHW (gamma ~ NHW*u for small NHW)
    auto tolC1 = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1);
    auto tolC10 = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10);
    auto tolC100 = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 100);

    EXPECT_LT(tolC1, tolC10) << "FP32: NHW=10 should have higher tolerance than NHW=1";
    EXPECT_LT(tolC10, tolC100) << "FP32: NHW=100 should have higher tolerance than NHW=10";

    auto ratioFp32 = static_cast<double>(tolC100) / static_cast<double>(tolC1);
    EXPECT_GT(ratioFp32, 10.0) << "FP32: ratio NHW100/NHW1 should reflect ~NHW growth";

    // BF16 (statistical): tolerance should grow ~ sqrt(NHW)
    auto tolBf16C1 = calculateBatchnormTrainingTolerance<bfloat16, bfloat16, bfloat16>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1);
    auto tolBf16C10 = calculateBatchnormTrainingTolerance<bfloat16, bfloat16, bfloat16>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10);
    auto tolBf16C50 = calculateBatchnormTrainingTolerance<bfloat16, bfloat16, bfloat16>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 50);

    EXPECT_LT(tolBf16C1, tolBf16C10) << "BF16: NHW=10 should have higher tolerance than NHW=1";
    EXPECT_LT(tolBf16C10, tolBf16C50) << "BF16: NHW=50 should have higher tolerance than NHW=10";

    auto ratioBf16 = static_cast<double>(tolBf16C10) / static_cast<double>(tolBf16C1);
    EXPECT_GT(ratioBf16, 1.5) << "BF16: ratio NHW10/NHW1 should reflect sqrt(NHW) growth";
}

// Sanity: mean tolerance scales with NHW
TEST(TestCalculateBnMeanTolerance, ToleranceScalesWithNHW)
{
    auto tol1 = calculateBatchnormMeanTolerance<float, float, float>(-1.0, 1.0, 1);
    auto tol10 = calculateBatchnormMeanTolerance<float, float, float>(-1.0, 1.0, 10);
    auto tol100 = calculateBatchnormMeanTolerance<float, float, float>(-1.0, 1.0, 100);

    EXPECT_LT(tol1, tol10);
    EXPECT_LT(tol10, tol100);
}

// Sanity: mean zero input
TEST(TestCalculateBnMeanTolerance, ZeroInput)
{
    auto tol = calculateBatchnormMeanTolerance<float, float, float>(0.0, 0.0, 10);
    EXPECT_EQ(tol, 0.0f);
}

// Sanity: invVar tolerance scales with NHW
TEST(TestCalculateBnInvVarTolerance, ToleranceScalesWithNHW)
{
    auto tol1 = calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 1);
    auto tol10 = calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 10);
    auto tol100 = calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 100);

    EXPECT_LT(tol1, tol10);
    EXPECT_LT(tol10, tol100);
}

// Sanity: invVar zero input gives finite positive tolerance
TEST(TestCalculateBnInvVarTolerance, ZeroInput)
{
    auto tol = calculateBatchnormInvVarianceTolerance<float, float, float>(0.0, 0.0, 10);
    EXPECT_GT(tol, 0.0f);
    EXPECT_LT(tol, 1.0f); // Should be a reasonable small value
}

// Asymmetric x range: maxAbsX = max(|xMin|, |xMax|)
TEST(TestCalculateBnTrainTolerance, AsymmetricXRange)
{
    // xMin=-0.5, xMax=2.0 => maxAbsX=2.0 (same as symmetric [-2,2])
    auto tolAsym = calculateBatchnormTrainingTolerance<float, float, float>(
        -0.5, 2.0, -1.0, 1.0, 0.0, 0.0, 10);
    auto tolSym = calculateBatchnormTrainingTolerance<float, float, float>(
        -2.0, 2.0, -1.0, 1.0, 0.0, 0.0, 10);

    // Both have same maxAbsX=2.0, so tolerance should be identical
    EXPECT_EQ(tolAsym, tolSym);

    // One-sided: xMin=0, xMax=5.0 => maxAbsX=5.0
    auto tolOneSided = calculateBatchnormTrainingTolerance<float, float, float>(
        0.0, 5.0, -1.0, 1.0, 0.0, 0.0, 10);
    auto tolSymFive = calculateBatchnormTrainingTolerance<float, float, float>(
        -5.0, 5.0, -1.0, 1.0, 0.0, 0.0, 10);

    EXPECT_EQ(tolOneSided, tolSymFive);

    // Mean tolerance: same behavior
    auto meanAsym = calculateBatchnormMeanTolerance<float, float, float>(-0.5, 2.0, 10);
    auto meanSym = calculateBatchnormMeanTolerance<float, float, float>(-2.0, 2.0, 10);
    EXPECT_EQ(meanAsym, meanSym);

    // InvVar tolerance: same behavior
    auto invVarAsym = calculateBatchnormInvVarianceTolerance<float, float, float>(-0.5, 2.0, 10);
    auto invVarSym = calculateBatchnormInvVarianceTolerance<float, float, float>(-2.0, 2.0, 10);
    EXPECT_EQ(invVarAsym, invVarSym);
}

// Scale and bias both zero: y tolerance should be zero
TEST(TestCalculateBnTrainTolerance, ZeroScaleAndBias)
{
    auto tol = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10);
    EXPECT_EQ(tol, 0.0f);

    // With half output casting: maxOutputMagnitude = 0 + 0 = 0, so cast term also 0
    auto tolHalf = calculateBatchnormTrainingTolerance<half, float, float>(
        -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 10);
    EXPECT_EQ(tolHalf, 0.0f);
}

// Negative NHW should throw (validates the < 1 check catches negatives)
TEST(TestCalculateBnTrainTolerance, ThrowsOnNegativeNHW)
{
    EXPECT_THROW((calculateBatchnormTrainingTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, -1)),
                 std::invalid_argument);
    EXPECT_THROW((calculateBatchnormMeanTolerance<float, float, float>(-1.0, 1.0, -1)),
                 std::invalid_argument);
    EXPECT_THROW((calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, -1)),
                 std::invalid_argument);
}

// InvVar: small x where eps_bn dominates variance estimate
TEST(TestCalculateBnInvVarTolerance, SmallXEpsilonDominates)
{
    // maxAbsX = 0.001, eps_bn = 1e-5
    // maxAbsX^2 = 1e-6 < eps_bn = 1e-5, so varEstimate = eps_bn
    // varPlusEps = 2 * eps_bn = 2e-5
    // This tests that the variance floor prevents underestimation
    auto tol = calculateBatchnormInvVarianceTolerance<float, float, float>(-0.001, 0.001, 10, 1e-5);
    EXPECT_GT(tol, 0.0f);

    // Compare with zero-input case: should be similar magnitude since eps_bn dominates both
    auto tolZero = calculateBatchnormInvVarianceTolerance<float, float, float>(0.0, 0.0, 10, 1e-5);
    EXPECT_GT(tolZero, 0.0f);

    // Small-x tolerance should be >= zero-input tolerance (has additional deltaVar contribution)
    EXPECT_GE(tol, tolZero);

    // InvVar tolerance should decrease as maxAbsX increases (larger var -> less amplification)
    auto tolLargeX
        = calculateBatchnormInvVarianceTolerance<float, float, float>(-10.0, 10.0, 10, 1e-5);
    EXPECT_LT(tolLargeX, tol) << "Larger x should give smaller invVar tolerance";
}

// InvVar: tolerance properly scales down for large maxAbsX
TEST(TestCalculateBnInvVarTolerance, LargeXScalesDown)
{
    auto tol1 = calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 10);
    auto tol100 = calculateBatchnormInvVarianceTolerance<float, float, float>(-100.0, 100.0, 10);
    auto tol1000 = calculateBatchnormInvVarianceTolerance<float, float, float>(-1000.0, 1000.0, 10);

    // Tolerance should decrease as maxAbsX increases
    EXPECT_LT(tol100, tol1) << "x=100 should give smaller invVar tolerance than x=1";
    EXPECT_LT(tol1000, tol100) << "x=1000 should give smaller invVar tolerance than x=100";

    // The ratio should reflect the 1/maxAbsX scaling in the gamma term
    auto ratio = static_cast<double>(tol1) / static_cast<double>(tol100);
    EXPECT_GT(ratio, 10.0) << "Tolerance ratio should reflect ~1/maxAbsX scaling";
}

// Mean and invVar also throw on singularity
TEST(TestCalculateBnMeanTolerance, ThrowsOnSingularity)
{
    EXPECT_THROW((calculateBatchnormMeanTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, 100)),
                 std::overflow_error);
}

TEST(TestCalculateBnInvVarTolerance, ThrowsOnSingularity)
{
    EXPECT_THROW(
        (calculateBatchnormInvVarianceTolerance<bfloat16, bfloat16, bfloat16>(-1.0, 1.0, 100)),
        std::overflow_error);
}

// epsilonBn <= 0 should throw for both y and invVar
TEST(TestCalculateBnTrainTolerance, ThrowsOnInvalidEpsilon)
{
    EXPECT_THROW((calculateBatchnormTrainingTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, 0.0)),
                 std::invalid_argument);
    EXPECT_THROW((calculateBatchnormTrainingTolerance<float, float, float>(
                     -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, -1e-5)),
                 std::invalid_argument);
}

TEST(TestCalculateBnInvVarTolerance, ThrowsOnInvalidEpsilon)
{
    EXPECT_THROW((calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 10, 0.0)),
                 std::invalid_argument);
    EXPECT_THROW(
        (calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 10, -1e-5)),
        std::invalid_argument);
}

// InvVar: tolerance changes with different epsilon values
TEST(TestCalculateBnInvVarTolerance, NonDefaultEpsilon)
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma10 = computeGamma(10, u);

    // Large epsilon (1e-3): larger variance floor -> smaller invVar -> smaller tolerance
    auto tolLargeEps
        = calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 10, 1e-3);
    auto expectedLargeEps = expectedInvVarTol(1.0, 1e-3, gamma10, u);
    EXPECT_NEAR(
        tolLargeEps,
        static_cast<float>(expectedLargeEps),
        std::max(static_cast<float>(expectedLargeEps) * 0.01f, std::numeric_limits<float>::min()));

    // Small epsilon (1e-8): varEstimate = maxAbsX^2 dominates, epsilon barely matters
    auto tolSmallEps
        = calculateBatchnormInvVarianceTolerance<float, float, float>(-1.0, 1.0, 10, 1e-8);
    auto expectedSmallEps = expectedInvVarTol(1.0, 1e-8, gamma10, u);
    EXPECT_NEAR(
        tolSmallEps,
        static_cast<float>(expectedSmallEps),
        std::max(static_cast<float>(expectedSmallEps) * 0.01f, std::numeric_limits<float>::min()));

    // With maxAbsX=0: epsilon dominates entirely, larger eps -> smaller tolerance
    auto tolZeroLargeEps
        = calculateBatchnormInvVarianceTolerance<float, float, float>(0.0, 0.0, 10, 1e-3);
    auto tolZeroSmallEps
        = calculateBatchnormInvVarianceTolerance<float, float, float>(0.0, 0.0, 10, 1e-8);
    EXPECT_LT(tolZeroLargeEps, tolZeroSmallEps)
        << "Larger eps_bn -> larger variance floor -> smaller invVar tolerance for zero input";
}

// =================================================================================================
// TestCalculateBatchnormInferenceTolerance
// =================================================================================================

struct BnInfToleranceTestCase
{
    double xMin;
    double xMax;
    double meanMin;
    double meanMax;
    double invVarMin;
    double invVarMax;
    double scaleMin;
    double scaleMax;
    double biasMin;
    double biasMax;
    double expectedTolerance;

    friend std::ostream& operator<<(std::ostream& os, const BnInfToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", meanMin: " << tc.meanMin
           << ", meanMax: " << tc.meanMax << ", invVarMin: " << tc.invVarMin
           << ", invVarMax: " << tc.invVarMax << ", scaleMin: " << tc.scaleMin
           << ", scaleMax: " << tc.scaleMax << ", biasMin: " << tc.biasMin
           << ", biasMax: " << tc.biasMax << ", expectedTolerance: " << tc.expectedTolerance;
        return os;
    }
};

template <typename T>
std::vector<BnInfToleranceTestCase> getBnInfToleranceTestCases();

// BN Inference tolerance formula (no casting, invVar variant):
//   maxAbsDiff = maxAbsX + maxAbsMean (triangle inequality)
//   P = maxAbsDiff * maxAbsInvVar * maxAbsScale
//   chainTol = 3u * P
//   biasTol = u * (P + maxAbsBias)
//   total = 4u * P + u * maxAbsBias
// With input casting (InputType=double, ComputeType=float):
//   total += computeEpsilon * P (input cast for 1 tensor)
//   = (4u + u) * P + u * maxAbsBias = 5u * P + u * B
// With output casting (OutputType=half/bf16):
//   maxOutMag = P + maxAbsBias
//   total += maxOutMag * outputEpsilon

// Helper to compute expected inference tolerance.
// inputTypeEpsilon/outputTypeEpsilon are the epsilon of the InputType/OutputType respectively.
// They are used only to detect whether casting occurs (by comparing against computeEpsilon).
// The actual casting error magnitude is always bounded by the destination type's epsilon.
static double expectedBnInfTol(double maxAbsX,
                               double maxAbsMean,
                               double maxAbsInvVar,
                               double maxAbsScale,
                               double maxAbsBias,
                               double computeEpsilon,
                               double inputTypeEpsilon = 0.0,
                               double outputTypeEpsilon = 0.0)
{
    const double maxAbsDiff = maxAbsX + maxAbsMean;
    const double mainProduct = maxAbsDiff * maxAbsInvVar * maxAbsScale;
    double tol = 3.0 * computeEpsilon * mainProduct; // chain ops
    tol += computeEpsilon * (mainProduct + maxAbsBias); // bias addition
    // Input casting: higher-precision input downcast to compute type
    if(inputTypeEpsilon > 0.0 && inputTypeEpsilon < computeEpsilon)
    {
        tol += computeEpsilon * mainProduct;
    }
    // Output casting: compute type downcast to lower-precision output
    if(outputTypeEpsilon > computeEpsilon)
    {
        tol += (mainProduct + maxAbsBias) * outputTypeEpsilon;
    }
    return tol;
}

// Float / Float / Float
template <>
std::vector<BnInfToleranceTestCase> getBnInfToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    // Test ranges matching BatchnormFwdTensorBundle: x=[0,1], mean=[0,1], invVar=[1,3], scale=[0,1], bias=[0,1]
    return {// Standard test: x=[0,1], mean=[0,1], invVar=[1,3], scale=[0,1], bias=[0,1]
            // maxAbsDiff = 1+1=2, P = 2*3*1=6
            {0.0,
             1.0,
             0.0,
             1.0,
             1.0,
             3.0,
             0.0,
             1.0,
             0.0,
             1.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 1.0, u)},
            // Symmetric x, no bias
            {-1.0,
             1.0,
             -1.0,
             1.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 0.0, u)},
            // Zero input: P=0, only bias tolerance survives
            {0.0,
             0.0,
             0.0,
             0.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             -0.5,
             0.5,
             expectedBnInfTol(0.0, 0.0, 3.0, 1.0, 0.5, u)},
            // All zero: tolerance = 0
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            // Large invVar: P scales with invVar
            {-1.0,
             1.0,
             -1.0,
             1.0,
             1.0,
             10.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             expectedBnInfTol(1.0, 1.0, 10.0, 1.0, 0.0, u)}};
}

// Float / Double / Float (Input casting: inputEpsilon(double) < epsilon(float), adds u*P)
template <>
std::vector<BnInfToleranceTestCase> getBnInfToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uDouble = std::numeric_limits<double>::epsilon();
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             1.0,
             3.0,
             0.0,
             1.0,
             0.0,
             1.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 1.0, u, uDouble)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             expectedBnInfTol(0.0, 0.0, 3.0, 1.0, 0.0, u, uDouble)}};
}

// Half / Float / Float (Output casting: uHalf > uFloat)
template <>
std::vector<BnInfToleranceTestCase> getBnInfToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             1.0,
             3.0,
             0.0,
             1.0,
             0.0,
             1.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 1.0, u, 0.0, uHalf)},
            // Zero input with bias: output cast from bias magnitude
            {0.0,
             0.0,
             0.0,
             0.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             -0.5,
             0.5,
             expectedBnInfTol(0.0, 0.0, 3.0, 1.0, 0.5, u, 0.0, uHalf)}};
}

// Half / Half / Half
template <>
std::vector<BnInfToleranceTestCase> getBnInfToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             1.0,
             3.0,
             0.0,
             1.0,
             0.0,
             1.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 1.0, u)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             expectedBnInfTol(0.0, 0.0, 3.0, 1.0, 0.0, u)}};
}

// Bfloat16 / Float / Float (Output casting: uBf16 > uFloat)
template <>
std::vector<BnInfToleranceTestCase> getBnInfToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             1.0,
             3.0,
             0.0,
             1.0,
             0.0,
             1.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 1.0, u, 0.0, uBf16)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             expectedBnInfTol(0.0, 0.0, 3.0, 1.0, 0.0, u, 0.0, uBf16)}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnInfToleranceTestCase>
    getBnInfToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             1.0,
             3.0,
             0.0,
             1.0,
             0.0,
             1.0,
             expectedBnInfTol(1.0, 1.0, 3.0, 1.0, 1.0, u)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             1.0,
             3.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             expectedBnInfTol(0.0, 0.0, 3.0, 1.0, 0.0, u)}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnInfTolerance : public ::testing::TestWithParam<BnInfToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        auto tol = calculateBatchnormInferenceTolerance<Out, In, Comp>(params.xMin,
                                                                       params.xMax,
                                                                       params.meanMin,
                                                                       params.meanMax,
                                                                       params.invVarMin,
                                                                       params.invVarMax,
                                                                       params.scaleMin,
                                                                       params.scaleMax,
                                                                       params.biasMin,
                                                                       params.biasMax);

        auto expected = static_cast<float>(params.expectedTolerance);

        EXPECT_NEAR(tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
    }
};

using TestCalcBnInfTolFp32 = TestCalculateBnInfTolerance<float, float, float>;
TEST_P(TestCalcBnInfTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfTolFp32,
    ::testing::ValuesIn(getBnInfToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnInfTolInputDouble = TestCalculateBnInfTolerance<float, double, float>;
TEST_P(TestCalcBnInfTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfTolInputDouble,
    ::testing::ValuesIn(getBnInfToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnInfTolComputeFloatFp16 = TestCalculateBnInfTolerance<half, float, float>;
TEST_P(TestCalcBnInfTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfTolComputeFloatFp16,
    ::testing::ValuesIn(getBnInfToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnInfTolFp16 = TestCalculateBnInfTolerance<half, half, half>;
TEST_P(TestCalcBnInfTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfTolFp16,
    ::testing::ValuesIn(getBnInfToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnInfTolComputeFloatBfp16 = TestCalculateBnInfTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnInfTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnInfToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnInfTolBfp16 = TestCalculateBnInfTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnInfTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfTolBfp16,
    ::testing::ValuesIn(getBnInfToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// TestCalculateBatchnormInferenceWithVarianceTolerance
// =================================================================================================

struct BnInfVarToleranceTestCase
{
    double xMin;
    double xMax;
    double meanMin;
    double meanMax;
    double varMin;
    double varMax;
    double scaleMin;
    double scaleMax;
    double biasMin;
    double biasMax;
    double epsilonBn;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnInfVarToleranceTestCase& tc)
    {
        os << "xMin: " << tc.xMin << ", xMax: " << tc.xMax << ", meanMin: " << tc.meanMin
           << ", meanMax: " << tc.meanMax << ", varMin: " << tc.varMin << ", varMax: " << tc.varMax
           << ", scaleMin: " << tc.scaleMin << ", scaleMax: " << tc.scaleMax
           << ", biasMin: " << tc.biasMin << ", biasMax: " << tc.biasMax
           << ", epsBn: " << tc.epsilonBn << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnInfVarToleranceTestCase> getBnInfVarToleranceTestCases();

// BN Inference with variance tolerance formula (no casting):
//   maxAbsDiff = maxAbsX + maxAbsMean
//   smallestVar = varMin (must be >= 0, throws otherwise)
//   maxAbsInvVar = 1/sqrt(smallestVar + epsBn)
//   P = maxAbsDiff * maxAbsInvVar * maxAbsScale
//   chainTol = 6u * P (3 for invVar computation + 3 for normalize)
//   biasTol = u * (P + maxAbsBias)
//   total = 7u * P + u * maxAbsBias

// Helper to compute expected inference with variance tolerance.
// See expectedBnInfTol for inputTypeEpsilon/outputTypeEpsilon semantics.
static double expectedBnInfVarTol(double maxAbsX,
                                  double maxAbsMean,
                                  double varMin,
                                  double maxAbsScale,
                                  double maxAbsBias,
                                  double epsBn,
                                  double computeEpsilon,
                                  double inputTypeEpsilon = 0.0,
                                  double outputTypeEpsilon = 0.0)
{
    const double maxAbsDiff = maxAbsX + maxAbsMean;
    const double maxAbsInvVar = 1.0 / std::sqrt(varMin + epsBn);
    const double mainProduct = maxAbsDiff * maxAbsInvVar * maxAbsScale;
    double tol = 6.0 * computeEpsilon * mainProduct; // 6 chain ops
    tol += computeEpsilon * (mainProduct + maxAbsBias); // bias addition
    // Input casting: higher-precision input downcast to compute type
    if(inputTypeEpsilon > 0.0 && inputTypeEpsilon < computeEpsilon)
    {
        tol += computeEpsilon * mainProduct;
    }
    // Output casting: compute type downcast to lower-precision output
    if(outputTypeEpsilon > computeEpsilon)
    {
        tol += (mainProduct + maxAbsBias) * outputTypeEpsilon;
    }
    return tol;
}

// Float / Float / Float
template <>
std::vector<BnInfVarToleranceTestCase>
    getBnInfVarToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    // Test ranges matching BatchnormFwdWithVarianceTensorBundle: x=[0,1], mean=[0,1], var=[0.1,1], scale=[0,1], bias=[0,1]
    return {// Standard: var=[0.1,1.0], invVar = 1/sqrt(0.1+1e-5), P = 2*invVar*1
            {0.0,
             1.0,
             0.0,
             1.0,
             0.1,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 1.0, 1e-5, u)},
            // Symmetric ranges
            {-1.0,
             1.0,
             -1.0,
             1.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 0.0, 1e-5, u)},
            // Zero input: P=0, only bias tolerance survives
            {0.0,
             0.0,
             0.0,
             0.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             -0.5,
             0.5,
             1e-5,
             expectedBnInfVarTol(0.0, 0.0, 0.1, 1.0, 0.5, 1e-5, u)},
            // All zero (except epsBn): invVar = 1/sqrt(epsBn), but P=0
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e-5, 0.0},
            // eps = 0 => throws
            {0.0, 1.0, 0.0, 1.0, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, true},
            // eps < 0 => throws
            {0.0, 1.0, 0.0, 1.0, 0.1, 1.0, 0.0, 1.0, 0.0, 1.0, -1e-5, 0.0, true}};
}

// Float / Double / Float
template <>
std::vector<BnInfVarToleranceTestCase>
    getBnInfVarToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uDouble = std::numeric_limits<double>::epsilon();
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             0.1,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 1.0, 1e-5, u, uDouble)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1e-5,
             expectedBnInfVarTol(0.0, 0.0, 0.1, 1.0, 0.0, 1e-5, u, uDouble)}};
}

// Half / Float / Float
template <>
std::vector<BnInfVarToleranceTestCase>
    getBnInfVarToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             0.1,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 1.0, 1e-5, u, 0.0, uHalf)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1e-5,
             expectedBnInfVarTol(0.0, 0.0, 0.1, 1.0, 0.0, 1e-5, u, 0.0, uHalf)}};
}

// Half / Half / Half
template <>
std::vector<BnInfVarToleranceTestCase> getBnInfVarToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             0.1,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 1.0, 1e-5, u)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1e-5,
             expectedBnInfVarTol(0.0, 0.0, 0.1, 1.0, 0.0, 1e-5, u)}};
}

// Bfloat16 / Float / Float
template <>
std::vector<BnInfVarToleranceTestCase>
    getBnInfVarToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             0.1,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 1.0, 1e-5, u, 0.0, uBf16)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1e-5,
             expectedBnInfVarTol(0.0, 0.0, 0.1, 1.0, 0.0, 1e-5, u, 0.0, uBf16)}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnInfVarToleranceTestCase>
    getBnInfVarToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// Standard
            {0.0,
             1.0,
             0.0,
             1.0,
             0.1,
             1.0,
             0.0,
             1.0,
             0.0,
             1.0,
             1e-5,
             expectedBnInfVarTol(1.0, 1.0, 0.1, 1.0, 1.0, 1e-5, u)},
            // Zero input
            {0.0,
             0.0,
             0.0,
             0.0,
             0.1,
             1.0,
             -1.0,
             1.0,
             0.0,
             0.0,
             1e-5,
             expectedBnInfVarTol(0.0, 0.0, 0.1, 1.0, 0.0, 1e-5, u)}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnInfVarTolerance : public ::testing::TestWithParam<BnInfVarToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateBatchnormInferenceWithVarianceTolerance<Out, In, Comp>(params.xMin,
                                                                                 params.xMax,
                                                                                 params.meanMin,
                                                                                 params.meanMax,
                                                                                 params.varMin,
                                                                                 params.varMax,
                                                                                 params.scaleMin,
                                                                                 params.scaleMax,
                                                                                 params.biasMin,
                                                                                 params.biasMax,
                                                                                 params.epsilonBn)),
                std::invalid_argument);
        }
        else
        {
            auto tol
                = calculateBatchnormInferenceWithVarianceTolerance<Out, In, Comp>(params.xMin,
                                                                                  params.xMax,
                                                                                  params.meanMin,
                                                                                  params.meanMax,
                                                                                  params.varMin,
                                                                                  params.varMax,
                                                                                  params.scaleMin,
                                                                                  params.scaleMax,
                                                                                  params.biasMin,
                                                                                  params.biasMax,
                                                                                  params.epsilonBn);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnInfVarTolFp32 = TestCalculateBnInfVarTolerance<float, float, float>;
TEST_P(TestCalcBnInfVarTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfVarTolFp32,
    ::testing::ValuesIn(getBnInfVarToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnInfVarTolInputDouble = TestCalculateBnInfVarTolerance<float, double, float>;
TEST_P(TestCalcBnInfVarTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfVarTolInputDouble,
    ::testing::ValuesIn(getBnInfVarToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnInfVarTolComputeFloatFp16 = TestCalculateBnInfVarTolerance<half, float, float>;
TEST_P(TestCalcBnInfVarTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfVarTolComputeFloatFp16,
    ::testing::ValuesIn(getBnInfVarToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnInfVarTolFp16 = TestCalculateBnInfVarTolerance<half, half, half>;
TEST_P(TestCalcBnInfVarTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfVarTolFp16,
    ::testing::ValuesIn(getBnInfVarToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnInfVarTolComputeFloatBfp16 = TestCalculateBnInfVarTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnInfVarTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfVarTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnInfVarToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnInfVarTolBfp16 = TestCalculateBnInfVarTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnInfVarTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnInfVarTolBfp16,
    ::testing::ValuesIn(getBnInfVarToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// Standalone Tests - BN Inference
// =================================================================================================

// BN Inference: zero scale and bias gives zero tolerance
TEST(TestCalculateBnInfTolerance, ZeroScaleAndBias)
{
    auto tol = calculateBatchnormInferenceTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_EQ(tol, 0.0f);

    // With half output: maxOutputMagnitude = 0, output cast = 0
    auto tolHalf = calculateBatchnormInferenceTolerance<half, float, float>(
        -1.0, 1.0, -1.0, 1.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0);
    EXPECT_EQ(tolHalf, 0.0f);
}

// BN Inference: tolerance scales with invVar magnitude
TEST(TestCalculateBnInfTolerance, ScalesWithInvVar)
{
    auto tolSmall = calculateBatchnormInferenceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0);
    auto tolLarge = calculateBatchnormInferenceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 10.0, -1.0, 1.0, 0.0, 0.0);

    EXPECT_LT(tolSmall, tolLarge)
        << "Larger invVar should produce larger tolerance (P scales with invVar)";
}

// BN Inference: tolerance scales with scale magnitude
TEST(TestCalculateBnInfTolerance, ScalesWithScale)
{
    auto tolSmall = calculateBatchnormInferenceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.1, 0.0, 0.0);
    auto tolLarge = calculateBatchnormInferenceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 3.0, -1.0, 1.0, 0.0, 0.0);

    EXPECT_LT(tolSmall, tolLarge) << "Larger scale should produce larger tolerance";
}

// BN Inference: asymmetric x range
TEST(TestCalculateBnInfTolerance, AsymmetricXRange)
{
    auto tolAsym = calculateBatchnormInferenceTolerance<float, float, float>(
        -0.5, 2.0, 0.0, 0.0, 1.0, 3.0, -1.0, 1.0, 0.0, 0.0);
    auto tolSym = calculateBatchnormInferenceTolerance<float, float, float>(
        -2.0, 2.0, 0.0, 0.0, 1.0, 3.0, -1.0, 1.0, 0.0, 0.0);

    // Both have maxAbsX=2.0, so tolerance should be identical
    EXPECT_EQ(tolAsym, tolSym);
}

// BN Inference: variance variant throws on invalid epsilon
TEST(TestCalculateBnInfVarTolerance, ThrowsOnInvalidEpsilon)
{
    EXPECT_THROW((calculateBatchnormInferenceWithVarianceTolerance<float, float, float>(
                     -1.0, 1.0, 0.0, 0.0, 0.1, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0)),
                 std::invalid_argument);
    EXPECT_THROW((calculateBatchnormInferenceWithVarianceTolerance<float, float, float>(
                     -1.0, 1.0, 0.0, 0.0, 0.1, 1.0, -1.0, 1.0, 0.0, 0.0, -1e-5)),
                 std::invalid_argument);
}

// BN Inference: variance variant throws on negative varianceMin
TEST(TestCalculateBnInfVarTolerance, ThrowsOnNegativeVariance)
{
    EXPECT_THROW((calculateBatchnormInferenceWithVarianceTolerance<float, float, float>(
                     -1.0, 1.0, 0.0, 0.0, -0.1, 1.0, -1.0, 1.0, 0.0, 0.0, 1e-5)),
                 std::invalid_argument);
}

// BN Inference with variance: tolerance larger than invVar variant (more ops)
TEST(TestCalculateBnInfVarTolerance, LargerThanInvVarVariant)
{
    // For comparable inputs, the variance variant should produce a larger tolerance
    // because it has 6 chain ops vs 3 for the invVar variant.
    // Use var=1.0 so invVar = 1/sqrt(1+1e-5) ~ 1.0 (comparable to invVar range [1,1])
    auto tolInvVar = calculateBatchnormInferenceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0);
    auto tolVar = calculateBatchnormInferenceWithVarianceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 1e-5);

    EXPECT_LT(tolInvVar, tolVar)
        << "Variance variant should have larger tolerance due to more chain ops";
}

// BN Inference with variance: smaller variance -> larger tolerance (larger invVar)
TEST(TestCalculateBnInfVarTolerance, SmallVarianceLargerTolerance)
{
    auto tolLargeVar = calculateBatchnormInferenceWithVarianceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 1.0, 10.0, -1.0, 1.0, 0.0, 0.0, 1e-5);
    auto tolSmallVar = calculateBatchnormInferenceWithVarianceTolerance<float, float, float>(
        -1.0, 1.0, 0.0, 0.0, 0.01, 0.1, -1.0, 1.0, 0.0, 0.0, 1e-5);

    EXPECT_LT(tolLargeVar, tolSmallVar) << "Smaller variance -> larger invVar -> larger tolerance";
}

// BN Inference: output overflow detection
TEST(TestCalculateBnInfTolerance, ThrowsOnOutputOverflow)
{
    // half output with very large invVar and scale: output magnitude >> 65504
    EXPECT_THROW((calculateBatchnormInferenceTolerance<half, float, float>(
                     -1.0, 1.0, 0.0, 0.0, 1e4, 1e4, 1e4, 1e4, 0.0, 0.0)),
                 std::overflow_error);
}

// Y tolerance: non-default epsilon affects output casting magnitude
TEST(TestCalculateBnTrainTolerance, NonDefaultEpsilon)
{
    // For maxAbsX=1, scale=1, the main tolerance (3/2*gamma + 7u) is independent
    // of eps_bn. Only the output casting magnitude changes via invVarEst.
    auto tolDefault = calculateBatchnormTrainingTolerance<half, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, 1e-5);
    auto tolLargeEps = calculateBatchnormTrainingTolerance<half, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, 1e-3);

    // Larger eps_bn -> smaller invVarEst -> smaller maxOutputMagnitude -> smaller output cast
    EXPECT_LT(tolLargeEps, tolDefault) << "Larger eps_bn should reduce output casting contribution";

    // For float/float/float (no output casting), eps_bn has no effect
    auto tolFp32Default = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, 1e-5);
    auto tolFp32LargeEps = calculateBatchnormTrainingTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, 1e-3);
    EXPECT_EQ(tolFp32Default, tolFp32LargeEps)
        << "Without output casting, eps_bn should not affect y tolerance";
}

// =================================================================================================
// TestCalculateBatchnormBackwardDbiasTolerance
// =================================================================================================

struct BnBwdDbiasToleranceTestCase
{
    double dyMin;
    double dyMax;
    int64_t nElementsPerChannel;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnBwdDbiasToleranceTestCase& tc)
    {
        os << "dyMin: " << tc.dyMin << ", dyMax: " << tc.dyMax
           << ", NHW: " << tc.nElementsPerChannel << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnBwdDbiasToleranceTestCase> getBnBwdDbiasToleranceTestCases();

// dbias tolerance formula (no casting):
//   tolerance = gamma_NHW * max|dy|
// With input casting (InputType=double, ComputeType=float):
//   tolerance += computeEpsilon * NHW * max|dy| (input cast for 1 tensor)
// With output casting (OutputType=half/bf16):
//   tolerance += NHW * max|dy| * outputEpsilon

// Float / Float / Float
template <>
std::vector<BnBwdDbiasToleranceTestCase>
    getBnBwdDbiasToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, gamma(1) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, gamma(10) * 1.0},
            // Large dy range (scales linearly with max|dy|)
            {-1000.0, 1000.0, 10, gamma(10) * 1000.0},
            // Zero dy (tolerance=0)
            {0.0, 0.0, 10, 0.0}};
}

// Float / Double / Float (Input casting: inputEpsilon(double) < epsilon(float))
// tolerance = gamma * max|dy| + u * NHW * max|dy|
template <>
std::vector<BnBwdDbiasToleranceTestCase>
    getBnBwdDbiasToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1: gamma * 1 + u * 1 * 1
            {-1.0, 1.0, 1, gamma(1) * 1.0 + u * 1.0 * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, gamma(10) * 1.0 + u * 10.0 * 1.0},
            // Zero dy
            {0.0, 0.0, 10, 0.0}};
}

// Half / Float / Float (Output casting: outputEpsilon(half) > epsilon(float))
// tolerance = gamma * max|dy| + NHW * max|dy| * uHalf
template <>
std::vector<BnBwdDbiasToleranceTestCase>
    getBnBwdDbiasToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, gamma(1) * 1.0 + 1.0 * 1.0 * uHalf},
            // NHW=10
            {-1.0, 1.0, 10, gamma(10) * 1.0 + 10.0 * 1.0 * uHalf},
            // Zero dy
            {0.0, 0.0, 10, 0.0}};
}

// Half / Half / Half
template <>
std::vector<BnBwdDbiasToleranceTestCase>
    getBnBwdDbiasToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, gamma(1) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, gamma(10) * 1.0},
            // Zero dy
            {0.0, 0.0, 10, 0.0}};
}

// Bfloat16 / Float / Float (Output casting: outputEpsilon(bf16) > epsilon(float))
template <>
std::vector<BnBwdDbiasToleranceTestCase>
    getBnBwdDbiasToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, gamma(1) * 1.0 + 1.0 * 1.0 * uBf16},
            // NHW=10
            {-1.0, 1.0, 10, gamma(10) * 1.0 + 10.0 * 1.0 * uBf16},
            // Zero dy
            {0.0, 0.0, 10, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnBwdDbiasToleranceTestCase>
    getBnBwdDbiasToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    auto gamma = [&](int64_t c) { return computeGamma(static_cast<uint64_t>(c), u); };
    return {// NHW = 0 => throws
            {-1.0, 1.0, 0, 0.0, true},
            // NHW=1
            {-1.0, 1.0, 1, gamma(1) * 1.0},
            // NHW=10
            {-1.0, 1.0, 10, gamma(10) * 1.0},
            // Zero dy
            {0.0, 0.0, 10, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnBwdDbiasTolerance
    : public ::testing::TestWithParam<BnBwdDbiasToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW((calculateBatchnormBackwardDbiasTolerance<Out, In, Comp>(
                             params.dyMin, params.dyMax, params.nElementsPerChannel)),
                         std::invalid_argument);
        }
        else
        {
            auto tol = calculateBatchnormBackwardDbiasTolerance<Out, In, Comp>(
                params.dyMin, params.dyMax, params.nElementsPerChannel);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnBwdDbiasTolFp32 = TestCalculateBnBwdDbiasTolerance<float, float, float>;
TEST_P(TestCalcBnBwdDbiasTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDbiasTolFp32,
    ::testing::ValuesIn(getBnBwdDbiasToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnBwdDbiasTolInputDouble = TestCalculateBnBwdDbiasTolerance<float, double, float>;
TEST_P(TestCalcBnBwdDbiasTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDbiasTolInputDouble,
    ::testing::ValuesIn(getBnBwdDbiasToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnBwdDbiasTolComputeFloatFp16 = TestCalculateBnBwdDbiasTolerance<half, float, float>;
TEST_P(TestCalcBnBwdDbiasTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDbiasTolComputeFloatFp16,
    ::testing::ValuesIn(getBnBwdDbiasToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnBwdDbiasTolFp16 = TestCalculateBnBwdDbiasTolerance<half, half, half>;
TEST_P(TestCalcBnBwdDbiasTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDbiasTolFp16,
    ::testing::ValuesIn(getBnBwdDbiasToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnBwdDbiasTolComputeFloatBfp16
    = TestCalculateBnBwdDbiasTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnBwdDbiasTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDbiasTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnBwdDbiasToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnBwdDbiasTolBfp16 = TestCalculateBnBwdDbiasTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnBwdDbiasTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDbiasTolBfp16,
    ::testing::ValuesIn(
        getBnBwdDbiasToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// TestCalculateBatchnormBackwardDscaleTolerance
// =================================================================================================

struct BnBwdDscaleToleranceTestCase
{
    double dyMin;
    double dyMax;
    double xMin;
    double xMax;
    int64_t nElementsPerChannel;
    double epsilonBn;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnBwdDscaleToleranceTestCase& tc)
    {
        os << "dyMin: " << tc.dyMin << ", dyMax: " << tc.dyMax << ", xMin: " << tc.xMin
           << ", xMax: " << tc.xMax << ", NHW: " << tc.nElementsPerChannel
           << ", epsBn: " << tc.epsilonBn << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnBwdDscaleToleranceTestCase> getBnBwdDscaleToleranceTestCases();

// dscale tolerance formula (no casting, maxAbsDy > 0):
//   varEst = max(maxAbsX^2, epsBn)
//   invVarEst = 1/sqrt(varEst + epsBn)
//   maxAbsXHat = maxAbsX * invVarEst
//   tolerance = (gamma + u) * maxAbsXHat * max|dy|
// With input casting (double->float): + 2 * u * NHW * maxAbsXHat * max|dy|
// With output casting: + NHW * maxAbsXHat * max|dy| * outputEpsilon

// Helper to compute expected dscale tolerance
static double expectedDscaleTol(double maxAbsDy,
                                double maxAbsX,
                                int64_t nhw,
                                double epsBn,
                                double u,
                                double outputEpsilon = 0.0,
                                double computeEpsilon = 0.0,
                                double inputEpsilon = 0.0)
{
    if(maxAbsDy == 0.0)
    {
        return 0.0;
    }
    auto gamma = computeGamma(static_cast<uint64_t>(nhw), u);
    const double varEst = std::max(maxAbsX * maxAbsX, epsBn);
    const double invVarEst = 1.0 / std::sqrt(varEst + epsBn);
    const double maxAbsXHat = maxAbsX * invVarEst;
    const double signalBound = static_cast<double>(nhw) * maxAbsXHat * maxAbsDy;
    double tol = (gamma + u) * maxAbsXHat * maxAbsDy;
    // Input casting (factor=2 for two-tensor product)
    if(inputEpsilon > 0.0 && inputEpsilon < computeEpsilon)
    {
        tol += 2.0 * computeEpsilon * signalBound;
    }
    // Output casting
    if(outputEpsilon > computeEpsilon && computeEpsilon > 0.0)
    {
        tol += signalBound * outputEpsilon;
    }
    return tol;
}

// Float / Float / Float
template <>
std::vector<BnBwdDscaleToleranceTestCase>
    getBnBwdDscaleToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDscaleTol(1.0, 1.0, 1, 1e-5, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDscaleTol(1.0, 1.0, 10, 1e-5, u)},
            // Large x range
            {-1.0, 1.0, -1000.0, 1000.0, 10, 1e-5, expectedDscaleTol(1.0, 1000.0, 10, 1e-5, u)},
            // Zero dy (tolerance=0)
            {0.0, 0.0, -1.0, 1.0, 10, 1e-5, 0.0},
            // Zero x (xHat=0 -> tol=0 from function since dy>0 but xHat=0 means all products are 0)
            {-1.0, 1.0, 0.0, 0.0, 10, 1e-5, expectedDscaleTol(1.0, 0.0, 10, 1e-5, u)},
            // Invalid epsilon => throws
            {-1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, true}};
}

// Float / Double / Float (Input casting: inputEpsilon(double) < epsilon(float))
template <>
std::vector<BnBwdDscaleToleranceTestCase>
    getBnBwdDscaleToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uDouble = std::numeric_limits<double>::epsilon();
    return {
        // NHW = 0 => throws
        {-1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
        // NHW=1
        {-1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDscaleTol(1.0, 1.0, 1, 1e-5, u, 0.0, u, uDouble)},
        // NHW=10
        {-1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDscaleTol(1.0, 1.0, 10, 1e-5, u, 0.0, u, uDouble)},
        // Zero dy
        {0.0, 0.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Half / Float / Float (Output casting: outputEpsilon(half) > epsilon(float))
template <>
std::vector<BnBwdDscaleToleranceTestCase>
    getBnBwdDscaleToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDscaleTol(1.0, 1.0, 1, 1e-5, u, uHalf, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDscaleTol(1.0, 1.0, 10, 1e-5, u, uHalf, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Half / Half / Half
template <>
std::vector<BnBwdDscaleToleranceTestCase>
    getBnBwdDscaleToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDscaleTol(1.0, 1.0, 1, 1e-5, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDscaleTol(1.0, 1.0, 10, 1e-5, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Bfloat16 / Float / Float (Output casting)
template <>
std::vector<BnBwdDscaleToleranceTestCase>
    getBnBwdDscaleToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDscaleTol(1.0, 1.0, 1, 1e-5, u, uBf16, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDscaleTol(1.0, 1.0, 10, 1e-5, u, uBf16, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnBwdDscaleToleranceTestCase>
    getBnBwdDscaleToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDscaleTol(1.0, 1.0, 1, 1e-5, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDscaleTol(1.0, 1.0, 10, 1e-5, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnBwdDscaleTolerance
    : public ::testing::TestWithParam<BnBwdDscaleToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW((calculateBatchnormBackwardDscaleTolerance<Out, In, Comp>(
                             params.dyMin,
                             params.dyMax,
                             params.xMin,
                             params.xMax,
                             params.nElementsPerChannel,
                             params.epsilonBn)),
                         std::invalid_argument);
        }
        else
        {
            auto tol = calculateBatchnormBackwardDscaleTolerance<Out, In, Comp>(
                params.dyMin,
                params.dyMax,
                params.xMin,
                params.xMax,
                params.nElementsPerChannel,
                params.epsilonBn);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnBwdDscaleTolFp32 = TestCalculateBnBwdDscaleTolerance<float, float, float>;
TEST_P(TestCalcBnBwdDscaleTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDscaleTolFp32,
    ::testing::ValuesIn(getBnBwdDscaleToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnBwdDscaleTolInputDouble = TestCalculateBnBwdDscaleTolerance<float, double, float>;
TEST_P(TestCalcBnBwdDscaleTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDscaleTolInputDouble,
    ::testing::ValuesIn(getBnBwdDscaleToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnBwdDscaleTolComputeFloatFp16
    = TestCalculateBnBwdDscaleTolerance<half, float, float>;
TEST_P(TestCalcBnBwdDscaleTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDscaleTolComputeFloatFp16,
    ::testing::ValuesIn(getBnBwdDscaleToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnBwdDscaleTolFp16 = TestCalculateBnBwdDscaleTolerance<half, half, half>;
TEST_P(TestCalcBnBwdDscaleTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDscaleTolFp16,
    ::testing::ValuesIn(getBnBwdDscaleToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnBwdDscaleTolComputeFloatBfp16
    = TestCalculateBnBwdDscaleTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnBwdDscaleTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDscaleTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnBwdDscaleToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnBwdDscaleTolBfp16 = TestCalculateBnBwdDscaleTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnBwdDscaleTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDscaleTolBfp16,
    ::testing::ValuesIn(
        getBnBwdDscaleToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// TestCalculateBatchnormBackwardDxTolerance
// =================================================================================================

struct BnBwdDxToleranceTestCase
{
    double dyMin;
    double dyMax;
    double xMin;
    double xMax;
    double scaleMin;
    double scaleMax;
    int64_t nElementsPerChannel;
    double epsilonBn;
    double expectedTolerance;
    bool expectThrow = false;

    friend std::ostream& operator<<(std::ostream& os, const BnBwdDxToleranceTestCase& tc)
    {
        os << "dyMin: " << tc.dyMin << ", dyMax: " << tc.dyMax << ", xMin: " << tc.xMin
           << ", xMax: " << tc.xMax << ", scaleMin: " << tc.scaleMin
           << ", scaleMax: " << tc.scaleMax << ", NHW: " << tc.nElementsPerChannel
           << ", epsBn: " << tc.epsilonBn << ", expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

template <typename T>
std::vector<BnBwdDxToleranceTestCase> getBnBwdDxToleranceTestCases();

// dx tolerance formula (no casting):
//   varEst = max(maxAbsX^2, epsBn), invVarEst = 1/sqrt(varEst + epsBn)
//   maxAbsXHat = maxAbsX * invVarEst
//   scalarCoef = maxAbsScale * invVarEst
//   deltaMeanDy = (gamma + u) * max|dy|
//   deltaMeanDyXhat = (gamma + 2u) * maxAbsXHat * max|dy|
//   maxAbsE = (2 + maxAbsXHat^2) * max|dy|
//   tolerance = scalarCoef * (deltaMeanDy + maxAbsXHat * deltaMeanDyXhat + 3u * maxAbsE)
//             + 2u * scalarCoef * maxAbsE

// Helper to compute expected dx tolerance
static double expectedDxTol(double maxAbsDy,
                            double maxAbsX,
                            double maxAbsScale,
                            int64_t nhw,
                            double epsBn,
                            double u,
                            double outputEpsilon = 0.0,
                            double computeEpsilon = 0.0,
                            double inputEpsilon = 0.0)
{
    if(maxAbsDy == 0.0)
    {
        return 0.0;
    }
    auto gamma = computeGamma(static_cast<uint64_t>(nhw), u);
    const double varEst = std::max(maxAbsX * maxAbsX, epsBn);
    const double invVarEst = 1.0 / std::sqrt(varEst + epsBn);
    const double maxAbsXHat = maxAbsX * invVarEst;
    const double scalarCoef = maxAbsScale * invVarEst;
    const double deltaMeanDy = (gamma + u) * maxAbsDy;
    const double deltaMeanDyXhat = (gamma + 2.0 * u) * maxAbsXHat * maxAbsDy;
    const double maxAbsE = (2.0 + maxAbsXHat * maxAbsXHat) * maxAbsDy;
    double tol = scalarCoef * (deltaMeanDy + maxAbsXHat * deltaMeanDyXhat + 3.0 * u * maxAbsE)
                 + 2.0 * u * scalarCoef * maxAbsE;
    const double maxAbsDx = scalarCoef * maxAbsE;
    // Input casting (factor=2)
    if(inputEpsilon > 0.0 && inputEpsilon < computeEpsilon)
    {
        tol += 2.0 * computeEpsilon * maxAbsDx;
    }
    // Output casting
    if(outputEpsilon > computeEpsilon && computeEpsilon > 0.0)
    {
        tol += maxAbsDx * outputEpsilon;
    }
    return tol;
}

// Float / Float / Float
template <>
std::vector<BnBwdDxToleranceTestCase>
    getBnBwdDxToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDxTol(1.0, 1.0, 1.0, 1, 1e-5, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDxTol(1.0, 1.0, 1.0, 10, 1e-5, u)},
            // Zero dy (tol=0)
            {0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, 0.0},
            // Zero scale (scalarCoef=0 -> tol=0)
            {-1.0, 1.0, -1.0, 1.0, 0.0, 0.0, 10, 1e-5, expectedDxTol(1.0, 1.0, 0.0, 10, 1e-5, u)},
            // Invalid epsilon => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 0.0, 0.0, true}};
}

// Float / Double / Float (Input casting)
template <>
std::vector<BnBwdDxToleranceTestCase>
    getBnBwdDxToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uDouble = std::numeric_limits<double>::epsilon();
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             1,
             1e-5,
             expectedDxTol(1.0, 1.0, 1.0, 1, 1e-5, u, 0.0, u, uDouble)},
            // NHW=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             10,
             1e-5,
             expectedDxTol(1.0, 1.0, 1.0, 10, 1e-5, u, 0.0, u, uDouble)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Half / Float / Float (Output casting)
template <>
std::vector<BnBwdDxToleranceTestCase> getBnBwdDxToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             1,
             1e-5,
             expectedDxTol(1.0, 1.0, 1.0, 1, 1e-5, u, uHalf, u)},
            // NHW=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             10,
             1e-5,
             expectedDxTol(1.0, 1.0, 1.0, 10, 1e-5, u, uHalf, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Half / Half / Half
template <>
std::vector<BnBwdDxToleranceTestCase> getBnBwdDxToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDxTol(1.0, 1.0, 1.0, 1, 1e-5, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDxTol(1.0, 1.0, 1.0, 10, 1e-5, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Bfloat16 / Float / Float (Output casting)
template <>
std::vector<BnBwdDxToleranceTestCase>
    getBnBwdDxToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             1,
             1e-5,
             expectedDxTol(1.0, 1.0, 1.0, 1, 1e-5, u, uBf16, u)},
            // NHW=10
            {-1.0,
             1.0,
             -1.0,
             1.0,
             -1.0,
             1.0,
             10,
             1e-5,
             expectedDxTol(1.0, 1.0, 1.0, 10, 1e-5, u, uBf16, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

// Bfloat16 / Bfloat16 / Bfloat16
template <>
std::vector<BnBwdDxToleranceTestCase>
    getBnBwdDxToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());
    return {// NHW = 0 => throws
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0, 1e-5, 0.0, true},
            // NHW=1
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1, 1e-5, expectedDxTol(1.0, 1.0, 1.0, 1, 1e-5, u)},
            // NHW=10
            {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, expectedDxTol(1.0, 1.0, 1.0, 10, 1e-5, u)},
            // Zero dy
            {0.0, 0.0, -1.0, 1.0, -1.0, 1.0, 10, 1e-5, 0.0}};
}

template <typename Out, typename In, typename Comp>
class TestCalculateBnBwdDxTolerance : public ::testing::TestWithParam<BnBwdDxToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();

        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateBatchnormBackwardDxTolerance<Out, In, Comp>(params.dyMin,
                                                                      params.dyMax,
                                                                      params.xMin,
                                                                      params.xMax,
                                                                      params.scaleMin,
                                                                      params.scaleMax,
                                                                      params.nElementsPerChannel,
                                                                      params.epsilonBn)),
                std::invalid_argument);
        }
        else
        {
            auto tol
                = calculateBatchnormBackwardDxTolerance<Out, In, Comp>(params.dyMin,
                                                                       params.dyMax,
                                                                       params.xMin,
                                                                       params.xMax,
                                                                       params.scaleMin,
                                                                       params.scaleMax,
                                                                       params.nElementsPerChannel,
                                                                       params.epsilonBn);

            auto expected = static_cast<float>(params.expectedTolerance);

            EXPECT_NEAR(
                tol, expected, std::max(expected * 0.01f, std::numeric_limits<float>::min()));
        }
    }
};

using TestCalcBnBwdDxTolFp32 = TestCalculateBnBwdDxTolerance<float, float, float>;
TEST_P(TestCalcBnBwdDxTolFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDxTolFp32,
    ::testing::ValuesIn(getBnBwdDxToleranceTestCases<TypeTriple<float, float, float>>()));

using TestCalcBnBwdDxTolInputDouble = TestCalculateBnBwdDxTolerance<float, double, float>;
TEST_P(TestCalcBnBwdDxTolInputDouble, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDxTolInputDouble,
    ::testing::ValuesIn(getBnBwdDxToleranceTestCases<TypeTriple<float, double, float>>()));

using TestCalcBnBwdDxTolComputeFloatFp16 = TestCalculateBnBwdDxTolerance<half, float, float>;
TEST_P(TestCalcBnBwdDxTolComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDxTolComputeFloatFp16,
    ::testing::ValuesIn(getBnBwdDxToleranceTestCases<TypeTriple<half, float, float>>()));

using TestCalcBnBwdDxTolFp16 = TestCalculateBnBwdDxTolerance<half, half, half>;
TEST_P(TestCalcBnBwdDxTolFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDxTolFp16,
    ::testing::ValuesIn(getBnBwdDxToleranceTestCases<TypeTriple<half, half, half>>()));

using TestCalcBnBwdDxTolComputeFloatBfp16 = TestCalculateBnBwdDxTolerance<bfloat16, float, float>;
TEST_P(TestCalcBnBwdDxTolComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDxTolComputeFloatBfp16,
    ::testing::ValuesIn(getBnBwdDxToleranceTestCases<TypeTriple<bfloat16, float, float>>()));

using TestCalcBnBwdDxTolBfp16 = TestCalculateBnBwdDxTolerance<bfloat16, bfloat16, bfloat16>;
TEST_P(TestCalcBnBwdDxTolBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalcBnBwdDxTolBfp16,
    ::testing::ValuesIn(getBnBwdDxToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()));

// =================================================================================================
// Standalone Tests - BN Backward
// =================================================================================================

// dbias tolerance is simpler than dscale (pure sum vs dot-product)
TEST(TestCalculateBnBwdDbiasTolerance, DbiasToleranceSmallerThanDscale)
{
    auto dbiasTol = calculateBatchnormBackwardDbiasTolerance<float, float, float>(-1.0, 1.0, 100);
    auto dscaleTol
        = calculateBatchnormBackwardDscaleTolerance<float, float, float>(-1.0, 1.0, -1.0, 1.0, 100);

    EXPECT_LT(dbiasTol, dscaleTol)
        << "dbias (pure sum) should have smaller tolerance than dscale (dot-product)";
}

// Doubling scale approximately doubles dx tolerance
TEST(TestCalculateBnBwdDxTolerance, DxToleranceScalesWithScale)
{
    auto tolScale1 = calculateBatchnormBackwardDxTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 100);
    auto tolScale2 = calculateBatchnormBackwardDxTolerance<float, float, float>(
        -1.0, 1.0, -1.0, 1.0, -2.0, 2.0, 100);

    // scalarCoef = scale * invVar, so doubling scale should approximately double tolerance
    auto ratio = static_cast<double>(tolScale2) / static_cast<double>(tolScale1);
    EXPECT_GT(ratio, 1.8) << "Doubling scale should approximately double dx tolerance";
    EXPECT_LT(ratio, 2.2) << "Ratio should be close to 2.0";
}
