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

    friend std::ostream& operator<<(std::ostream& os, const ConvWrwToleranceTestCase& tc)
    {
        os << "inputMin: " << tc.inputMin << ", inputMax: " << tc.inputMax
           << ", dyMin: " << tc.dyMin << ", dyMax: " << tc.dyMax << ", dyDims: [";
        for(size_t i = 0; i < tc.dyDims.size(); ++i)
        {
            os << tc.dyDims[i] << (i < tc.dyDims.size() - 1 ? ", " : "");
        }
        os << "], expectedTolerance: " << tc.expectedTolerance;
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
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -23)},
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 65.0 * std::pow(2.0, -23)}};
}

// HipBfloat16 / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<hip_bfloat16, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0},
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
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -7)},
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -7)},
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 65.0 * std::pow(2.0, -7)}};
}

// Half / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<half, float>>()
{
    return {
        {-1.0, 1.0, -1.0, 1.0, {}, 0.0},
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
    return {{-1.0, 1.0, -1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, -1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -10)},
            {-1.0, 1.0, -1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -10)},
            {-1.0, 1.0, -1.0, 1.0, {10, 1, 1, 1}, 65.0 * std::pow(2.0, -10)}};
}

template <typename Out, typename Comp>
class TestCalculateConvWrwTolerance : public ::testing::TestWithParam<ConvWrwToleranceTestCase>
{
};

#define REGISTER_CONV_TEST(OutType, CompType, Name)                                  \
    using TestConv##Name = TestCalculateConvWrwTolerance<OutType, CompType>;         \
    TEST_P(TestConv##Name, VerifyTolerance)                                          \
    {                                                                                \
        auto tol = calculateConvWrwTolerance<OutType, CompType>(GetParam().inputMin, \
                                                                GetParam().inputMax, \
                                                                GetParam().dyMin,    \
                                                                GetParam().dyMax,    \
                                                                GetParam().dyDims);  \
        EXPECT_NEAR(static_cast<float>(tol), GetParam().expectedTolerance, 1e-5)     \
            << "Failed for dims size: " << GetParam().dyDims.size();                 \
    }                                                                                \
    INSTANTIATE_TEST_SUITE_P(                                                        \
        Name,                                                                        \
        TestConv##Name,                                                              \
        ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<OutType, CompType>>()));

REGISTER_CONV_TEST(float, float, FloatFloat)
REGISTER_CONV_TEST(hip_bfloat16, float, HipBfloat16Float)
REGISTER_CONV_TEST(hip_bfloat16, hip_bfloat16, HipBfloat16HipBfloat16)
REGISTER_CONV_TEST(half, float, HalfFloat)
REGISTER_CONV_TEST(half, half, HalfHalf)
