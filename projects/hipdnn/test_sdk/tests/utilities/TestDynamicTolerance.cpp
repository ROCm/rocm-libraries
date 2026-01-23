// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_test_sdk/utilities/DynamicTolerance.hpp>
#include <hipdnn_test_sdk/utilities/VectorLoggingUtils.hpp>
#include <vector>

using namespace hipdnn_test_sdk::utilities;

// =================================================================================================
// TestGetPrecision
// =================================================================================================

struct GetPrecisionTestCase
{
    double value;
    double expectedPrecision;
};

std::ostream& operator<<(std::ostream& os, const GetPrecisionTestCase& tc)
{
    os << "{ value: " << tc.value << ", expectedPrecision: " << tc.expectedPrecision << " }";
    return os;
}

template <typename T>
std::vector<GetPrecisionTestCase> getPrecisionTestCases();

template <>
std::vector<GetPrecisionTestCase> getPrecisionTestCases<float>()
{
    return {{1.0, std::pow(2.0, -23)},
            {1.5, std::pow(2.0, -23)},
            {1.99, std::pow(2.0, -23)},
            {2.0, std::pow(2.0, -22)},
            {3.0, std::pow(2.0, -22)},
            {4.0, std::pow(2.0, -21)},
            {0.5, std::pow(2.0, -24)},
            {0.25, std::pow(2.0, -25)},
            {0.0, 0.0}};
}

template <>
std::vector<GetPrecisionTestCase> getPrecisionTestCases<hip_bfloat16>()
{
    return {{1.0, std::pow(2.0, -7)},
            {1.5, std::pow(2.0, -7)},
            {1.99, std::pow(2.0, -7)},
            {2.0, std::pow(2.0, -6)},
            {3.0, std::pow(2.0, -6)},
            {4.0, std::pow(2.0, -5)},
            {0.5, std::pow(2.0, -8)},
            {0.25, std::pow(2.0, -9)},
            {0.0, 0.0}};
}

template <>
std::vector<GetPrecisionTestCase> getPrecisionTestCases<half>()
{
    return {{1.0, std::pow(2.0, -10)},
            {1.5, std::pow(2.0, -10)},
            {1.99, std::pow(2.0, -10)},
            {2.0, std::pow(2.0, -9)},
            {3.0, std::pow(2.0, -9)},
            {4.0, std::pow(2.0, -8)},
            {0.5, std::pow(2.0, -11)},
            {0.25, std::pow(2.0, -12)},
            {0.0, 0.0}};
}

template <typename T>
class TestGetPrecision : public ::testing::TestWithParam<GetPrecisionTestCase>
{
};

#define REGISTER_PRECISION_TEST(Type, Name)                         \
    using TestGetPrecision##Name = TestGetPrecision<Type>;          \
    TEST_P(TestGetPrecision##Name, VerifyPrecision)                 \
    {                                                               \
        double precision = getPrecision<Type>(GetParam().value);    \
        EXPECT_NEAR(precision, GetParam().expectedPrecision, 1e-10) \
            << "Failed for value: " << GetParam().value;            \
    }                                                               \
    INSTANTIATE_TEST_SUITE_P(                                       \
        Name, TestGetPrecision##Name, ::testing::ValuesIn(getPrecisionTestCases<Type>()));

REGISTER_PRECISION_TEST(float, Float)
REGISTER_PRECISION_TEST(hip_bfloat16, HipBfloat16)
REGISTER_PRECISION_TEST(half, Half)

// =================================================================================================
// TestCalculateConvWrwTolerance
// =================================================================================================

struct ConvWrwToleranceTestCase
{
    double inputMin;
    double inputMax;
    std::vector<int64_t> dyDims;
    double expectedTolerance;
};

std::ostream& operator<<(std::ostream& os, const ConvWrwToleranceTestCase& tc)
{
    os << "{ inputMin: " << tc.inputMin << ", inputMax: " << tc.inputMax
       << ", dyDims: " << fmt::format("{}", tc.dyDims)
       << ", expectedTolerance: " << tc.expectedTolerance << " }";
    return os;
}

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
    return {{-1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -23)},
            {-1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -23)},
            {-1.0, 1.0, {10, 1, 1, 1}, 53.0 * std::pow(2.0, -23)}};
}

// HipBfloat16 / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<hip_bfloat16, float>>()
{
    return {{-1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, {1, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -7)},
            {-1.0, 1.0, {2, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -22) + std::pow(2.0, -6)},
            {-1.0, 1.0, {10, 1, 1, 1}, 45.0 * std::pow(2.0, -23) + std::pow(2.0, -4)}};
}

// HipBfloat16 / HipBfloat16
template <>
std::vector<ConvWrwToleranceTestCase>
    getConvWrwToleranceTestCases<TypePair<hip_bfloat16, hip_bfloat16>>()
{
    return {{-1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -7)},
            {-1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -7)},
            {-1.0, 1.0, {10, 1, 1, 1}, 53.0 * std::pow(2.0, -7)}};
}

// Half / Float
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<half, float>>()
{
    return {{-1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, {1, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -10)},
            {-1.0, 1.0, {2, 1, 1, 1}, std::pow(2.0, -23) + std::pow(2.0, -22) + std::pow(2.0, -9)},
            {-1.0, 1.0, {10, 1, 1, 1}, 45.0 * std::pow(2.0, -23) + std::pow(2.0, -7)}};
}

// Half / Half
template <>
std::vector<ConvWrwToleranceTestCase> getConvWrwToleranceTestCases<TypePair<half, half>>()
{
    return {{-1.0, 1.0, {}, 0.0},
            {-1.0, 1.0, {1, 1, 1, 1}, 2.0 * std::pow(2.0, -10)},
            {-1.0, 1.0, {2, 1, 1, 1}, 5.0 * std::pow(2.0, -10)},
            {-1.0, 1.0, {10, 1, 1, 1}, 53.0 * std::pow(2.0, -10)}};
}

template <typename Out, typename Comp>
class TestCalculateConvWrwTolerance : public ::testing::TestWithParam<ConvWrwToleranceTestCase>
{
};

#define REGISTER_CONV_TEST(OutType, CompType, Name)                              \
    using TestConv##Name = TestCalculateConvWrwTolerance<OutType, CompType>;     \
    TEST_P(TestConv##Name, VerifyTolerance)                                      \
    {                                                                            \
        auto tol = calculateConvWrwTolerance<OutType, CompType>(                 \
            GetParam().inputMin, GetParam().inputMax, GetParam().dyDims);        \
        EXPECT_NEAR(static_cast<float>(tol), GetParam().expectedTolerance, 1e-5) \
            << "Failed for dims size: " << GetParam().dyDims.size();             \
    }                                                                            \
    INSTANTIATE_TEST_SUITE_P(                                                    \
        Name,                                                                    \
        TestConv##Name,                                                          \
        ::testing::ValuesIn(getConvWrwToleranceTestCases<TypePair<OutType, CompType>>()));

REGISTER_CONV_TEST(float, float, FloatFloat)
REGISTER_CONV_TEST(hip_bfloat16, float, HipBfloat16Float)
REGISTER_CONV_TEST(hip_bfloat16, hip_bfloat16, HipBfloat16HipBfloat16)
REGISTER_CONV_TEST(half, float, HalfFloat)
REGISTER_CONV_TEST(half, half, HalfHalf)
