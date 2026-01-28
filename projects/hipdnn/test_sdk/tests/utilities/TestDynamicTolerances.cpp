// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
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

template <typename Out, typename Comp>
struct TypePair
{
    using OutputType = Out;
    using ComputeType = Comp;
};

template <typename T>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases();

// Float / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<float, float>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -23)},
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 65.0 * std::pow(2.0, -23)}};
}

// HipBfloat16 / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<hip_bfloat16, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -7)},
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 3.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -7)},
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         55.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -7)}};
}

// HipBfloat16 / HipBfloat16
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypePair<hip_bfloat16, hip_bfloat16>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -7)},
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -7)},
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 65.0 * std::pow(2.0, -7)}};
}

// Half / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<half, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
        {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -10)},
        {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 3.0 * std::pow(2.0, -23) + 2.0 * std::pow(2.0, -10)},
        {-1.0,
         1.0,
         -1.0,
         1.0,
         {10, 1, 1, 1},
         55.0 * std::pow(2.0, -23) + 10.0 * std::pow(2.0, -10)}};
}

// Half / Half
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<half, half>>()
{
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1}, 0.0, true},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -10)},
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -10)},
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 65.0 * std::pow(2.0, -10)}};
}

template <typename Out, typename Comp>
class TestCalculateConvWrwTolerance : public ::testing::TestWithParam<ConvWrwToleranceTestCase>
{
protected:
    void verifyTolerance()
    {
        const auto& params = GetParam();
        if(params.expectThrow)
        {
            EXPECT_THROW(
                (calculateConvWrwTolerance<Out, Comp>(
                    params.inputMin, params.inputMax, params.dyMin, params.dyMax, params.dyDims)),
                std::invalid_argument)
                << "Failed to throw for dims size: " << params.dyDims.size();
        }
        else
        {
            auto tol = calculateConvWrwTolerance<Out, Comp>(
                params.inputMin, params.inputMax, params.dyMin, params.dyMax, params.dyDims);
            EXPECT_NEAR(static_cast<float>(tol), params.expectedTolerance, 1e-5)
                << "Failed for dims size: " << params.dyDims.size();
        }
    }
};

using TestCalculateConvWrwToleranceFp32 = TestCalculateConvWrwTolerance<float, float>;
TEST_P(TestCalculateConvWrwToleranceFp32, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceFp32,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<float, float>>()));

using TestCalculateConvWrwToleranceComputeFloatBfp16
    = TestCalculateConvWrwTolerance<hip_bfloat16, float>;
TEST_P(TestCalculateConvWrwToleranceComputeFloatBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceComputeFloatBfp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<hip_bfloat16, float>>()));

using TestCalculateConvWrwToleranceBfp16
    = TestCalculateConvWrwTolerance<hip_bfloat16, hip_bfloat16>;
TEST_P(TestCalculateConvWrwToleranceBfp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceBfp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<hip_bfloat16, hip_bfloat16>>()));

using TestCalculateConvWrwToleranceComputeFloatFp16 = TestCalculateConvWrwTolerance<half, float>;
TEST_P(TestCalculateConvWrwToleranceComputeFloatFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(
    Smoke,
    TestCalculateConvWrwToleranceComputeFloatFp16,
    ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<half, float>>()));

using TestCalculateConvWrwToleranceFp16 = TestCalculateConvWrwTolerance<half, half>;
TEST_P(TestCalculateConvWrwToleranceFp16, VerifyTolerance)
{
    this->verifyTolerance();
}
INSTANTIATE_TEST_SUITE_P(Smoke,
                         TestCalculateConvWrwToleranceFp16,
                         ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<half, half>>()));
