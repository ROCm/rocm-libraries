// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <vector>

using namespace hipdnn_data_sdk::types;

template <typename Out, typename In, typename Comp>
struct TypeTriple
{
    using OutputType = Out;
    using InputType = In;
    using ComputeType = Comp;
};

// =================================================================================================
// TestCalculateMatmulTolerance
// =================================================================================================

using namespace hipdnn_test_sdk::utilities::matmul;

struct MatmulToleranceTestCase
{
    std::vector<int64_t> aDims;
    std::vector<int64_t> bDims;
    std::vector<double> aRowValues;
    std::vector<double> bRowValues;
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
        os << "], aRowValues: [";
        for(size_t i = 0; i < tc.aRowValues.size(); ++i)
        {
            os << tc.aRowValues[i] << (i < tc.aRowValues.size() - 1 ? ", " : "");
        }
        os << "], bRowValues: [";
        for(size_t i = 0; i < tc.bRowValues.size(); ++i)
        {
            os << tc.bRowValues[i] << (i < tc.bRowValues.size() - 1 ? ", " : "");
        }
        os << "], expectedTolerance: " << tc.expectedTolerance
           << ", expectThrow: " << (tc.expectThrow ? "true" : "false");
        return os;
    }
};

namespace
{
// Helper to create a tensor where each row is filled with a per-row constant value.
// rowValues[i] specifies the constant value for all elements in row i.
// This ensures different rows have different sums, exercising the max-row-sum logic
// in computeMatrixInfNorm.
// Supports batched (>2D) tensors via iterateAlongDimensions.
template <typename T>
hipdnn_data_sdk::utilities::Tensor<T>
    createTensorFromRowValues(const std::vector<int64_t>& dims,
                              const std::vector<double>& rowValues)
{
    using hipdnn_data_sdk::utilities::iterateAlongDimensions;

    hipdnn_data_sdk::utilities::Tensor<T> tensor(dims);

    auto cols = dims.back();

    // outerDims = [batch..., rows] — everything except the last dim (cols)
    auto outerDims = std::vector<int64_t>(dims.begin(), dims.end() - 1);

    iterateAlongDimensions(outerDims, [&](const std::vector<int64_t>& outerIndices) {
        auto row = outerIndices.back();
        auto fullIndices = outerIndices;
        fullIndices.push_back(0);

        for(int64_t j = 0; j < cols; ++j)
        {
            fullIndices.back() = j;
            tensor.setHostValue(static_cast<T>(rowValues[static_cast<size_t>(row)]), fullIndices);
        }
    });

    return tensor;
}
} // namespace

using hipdnn_test_sdk::utilities::computeGamma;

template <typename T>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases();

// Float / Float / Float (High Precision: Linear)
// Tolerance = gamma(K, u_float) * ||A||_inf * ||B||_inf
// Row-varying values ensure ||A||_inf != entry_sum / rows
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<float, float, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());

    auto bRowValues100 = std::vector<double>(100, 1.0);

    return {// K=1: A=2x1, rows={1,2}. ||A||_inf=2, B=1x2, rows={1}. ||B||_inf=2
            {{2, 1}, {1, 2}, {1.0, 2.0}, {1.0}, computeGamma(1, u) * 2.0 * 2.0},
            // K=3: A=2x3, rows={1,2}. ||A||_inf=6, B=3x4, rows={1,3,0.5}. ||B||_inf=12
            {{2, 3}, {3, 4}, {1.0, 2.0}, {1.0, 3.0, 0.5}, computeGamma(3, u) * 6.0 * 12.0},
            // K=10: A=2x10, rows={1,2}. ||A||_inf=20, B=10x2, rows={1..5,1..5}. ||B||_inf=10
            {{2, 10},
             {10, 2},
             {1.0, 2.0},
             {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
             computeGamma(10, u) * 20.0 * 10.0},
            // K=100: A=2x100, rows={1,2}. ||A||_inf=200, B=100x2, all 1.0. ||B||_inf=2
            {{2, 100}, {100, 2}, {1.0, 2.0}, bRowValues100, computeGamma(100, u) * 200.0 * 2.0},
            // Batched K=3: A={2,2,3}, B={2,3,4}. Same row values as 2D K=3 case.
            // Batch dim doesn't change max row sum: ||A||_inf=6, ||B||_inf=12
            {{2, 2, 3}, {2, 3, 4}, {1.0, 2.0}, {1.0, 3.0, 0.5}, computeGamma(3, u) * 6.0 * 12.0},
            // Batched K=10: A={3,2,10}, B={3,10,2}. 3 batches.
            // ||A||_inf=20, ||B||_inf=10 (same as 2D K=10)
            {{3, 2, 10},
             {3, 10, 2},
             {1.0, 2.0},
             {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
             computeGamma(10, u) * 20.0 * 10.0}};
}

// Float / Double / Float (Input casting error)
// Tolerance = gamma(K, u_float) * ||A||_inf * ||B||_inf + 2 * ||A||_inf * ||B||_inf * u_float
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<float, double, float>>()
{
    auto u = static_cast<double>(std::numeric_limits<float>::epsilon());

    return {
        // K=1: ||A||_inf=2, ||B||_inf=2. Tol = gamma(1,u)*4 + 2*4*u
        {{2, 1}, {1, 2}, {1.0, 2.0}, {1.0}, computeGamma(1, u) * 2.0 * 2.0 + 2.0 * 2.0 * 2.0 * u},
        // K=10: ||A||_inf=20, ||B||_inf=10. Tol = gamma(10,u)*200 + 2*200*u
        {{2, 10},
         {10, 2},
         {1.0, 2.0},
         {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
         computeGamma(10, u) * 20.0 * 10.0 + 2.0 * 20.0 * 10.0 * u}};
}

// Half / Float / Float (Output casting error)
// Tolerance = gamma(K, u_float) * ||A||_inf * ||B||_inf + ||A||_inf * ||B||_inf * u_half
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<half, float, float>>()
{
    auto uFloat = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uHalf = static_cast<double>(std::numeric_limits<half>::epsilon());

    return {// K=1: ||A||_inf=2, ||B||_inf=2. Tol = gamma(1,uFloat)*4 + 4*uHalf
            {{2, 1},
             {1, 2},
             {1.0, 2.0},
             {1.0},
             computeGamma(1, uFloat) * 2.0 * 2.0 + 2.0 * 2.0 * uHalf},
            // K=10: ||A||_inf=20, ||B||_inf=10. Tol = gamma(10,uFloat)*200 + 200*uHalf
            {{2, 10},
             {10, 2},
             {1.0, 2.0},
             {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
             computeGamma(10, uFloat) * 20.0 * 10.0 + 20.0 * 10.0 * uHalf}};
}

// Half / Half / Half (Low Precision: Statistical)
// Tolerance = gamma(K, u_half) * ||A||_inf * ||B||_inf
template <>
std::vector<MatmulToleranceTestCase> getMatmulToleranceTestCases<TypeTriple<half, half, half>>()
{
    auto u = static_cast<double>(std::numeric_limits<half>::epsilon());

    auto bRowValues100 = std::vector<double>(100, 1.0);

    return {// K=1: ||A||_inf=2, ||B||_inf=2
            {{2, 1}, {1, 2}, {1.0, 2.0}, {1.0}, computeGamma(1, u) * 2.0 * 2.0},
            // K=10: ||A||_inf=20, ||B||_inf=10
            {{2, 10},
             {10, 2},
             {1.0, 2.0},
             {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
             computeGamma(10, u) * 20.0 * 10.0},
            // K=100: ||A||_inf=200, ||B||_inf=2
            {{2, 100}, {100, 2}, {1.0, 2.0}, bRowValues100, computeGamma(100, u) * 200.0 * 2.0}};
}

// Bfloat16 / Float / Float (Output casting error)
// Tolerance = gamma(K, u_float) * ||A||_inf * ||B||_inf + ||A||_inf * ||B||_inf * u_bf16
template <>
std::vector<MatmulToleranceTestCase>
    getMatmulToleranceTestCases<TypeTriple<bfloat16, float, float>>()
{
    auto uFloat = static_cast<double>(std::numeric_limits<float>::epsilon());
    auto uBf16 = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());

    return {// K=1: ||A||_inf=2, ||B||_inf=2. Tol = gamma(1,uFloat)*4 + 4*uBf16
            {{2, 1},
             {1, 2},
             {1.0, 2.0},
             {1.0},
             computeGamma(1, uFloat) * 2.0 * 2.0 + 2.0 * 2.0 * uBf16},
            // K=10: ||A||_inf=20, ||B||_inf=10. Tol = gamma(10,uFloat)*200 + 200*uBf16
            {{2, 10},
             {10, 2},
             {1.0, 2.0},
             {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
             computeGamma(10, uFloat) * 20.0 * 10.0 + 20.0 * 10.0 * uBf16}};
}

// Bfloat16 / Bfloat16 / Bfloat16 (Low Precision: Statistical)
// Tolerance = gamma(K, u_bf16) * ||A||_inf * ||B||_inf
template <>
std::vector<MatmulToleranceTestCase>
    getMatmulToleranceTestCases<TypeTriple<bfloat16, bfloat16, bfloat16>>()
{
    auto u = static_cast<double>(std::numeric_limits<bfloat16>::epsilon());

    auto bRowValues50 = std::vector<double>(50, 1.0);

    return {// K=1: ||A||_inf=2, ||B||_inf=2
            {{2, 1}, {1, 2}, {1.0, 2.0}, {1.0}, computeGamma(1, u) * 2.0 * 2.0},
            // K=10: ||A||_inf=20, ||B||_inf=10
            {{2, 10},
             {10, 2},
             {1.0, 2.0},
             {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
             computeGamma(10, u) * 20.0 * 10.0},
            // K=50: ||A||_inf=100, ||B||_inf=2 (K=100 exceeds gamma>=0.5 for bf16)
            {{2, 50}, {50, 2}, {1.0, 2.0}, bRowValues50, computeGamma(50, u) * 100.0 * 2.0}};
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
            auto a = createTensorFromRowValues<In>(tc.aDims, tc.aRowValues);
            auto b = createTensorFromRowValues<In>(tc.bDims, tc.bRowValues);

            EXPECT_THROW((calculateMatmulTolerance<Out, In, Comp>(a, b)), std::exception) << tc;
        }
        else
        {
            auto a = createTensorFromRowValues<In>(tc.aDims, tc.aRowValues);
            auto b = createTensorFromRowValues<In>(tc.bDims, tc.bRowValues);

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
    auto a = createTensorFromRowValues<float>({2, 3}, {1.0, 2.0});
    auto b = createTensorFromRowValues<float>({4, 5}, {1.0, 2.0, 3.0, 4.0});

    EXPECT_THROW((calculateMatmulTolerance<float, float, float>(a, b)), std::invalid_argument);
}

TEST(TestCalculateMatmulTolerance, ThrowsOn1dTensorA)
{
    // A is 1D (vector), B is 2D — must reject A
    hipdnn_data_sdk::utilities::Tensor<float> a({4});
    auto b = createTensorFromRowValues<float>({4, 2}, {1.0, 1.0, 1.0, 1.0});

    EXPECT_THROW((calculateMatmulTolerance<float, float, float>(a, b)), std::invalid_argument);
}

TEST(TestCalculateMatmulTolerance, ThrowsOn1dTensorB)
{
    // A is 2D, B is 1D (vector) — must reject B
    auto a = createTensorFromRowValues<float>({2, 4}, {1.0, 2.0});
    hipdnn_data_sdk::utilities::Tensor<float> b({4});

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
    auto bRowValues100 = std::vector<double>(100, 1.0);
    auto a = createTensorFromRowValues<bfloat16>({2, 100}, {1.0, 2.0});
    auto b = createTensorFromRowValues<bfloat16>({100, 2}, bRowValues100);

    EXPECT_THROW((calculateMatmulTolerance<bfloat16, bfloat16, bfloat16>(a, b)),
                 std::overflow_error);
}

// Test that extreme values cause output overflow
TEST(TestCalculateMatmulTolerance, ThrowsOnOutputOverflow)
{
    // Create matrices with very large values that will cause tolerance > half::max
    // A = 2x10, rows={1e5, 1e5}. ||A||_inf = 10 * 1e5 = 1e6
    // B = 10x2, all rows 1e5. ||B||_inf = 2 * 1e5 = 2e5
    // Product will exceed half max (65504)
    auto bRowValues10 = std::vector<double>(10, 1.0e5);
    auto a = createTensorFromRowValues<float>({2, 10}, {1.0e5, 1.0e5});
    auto b = createTensorFromRowValues<float>({10, 2}, bRowValues10);

    // OutputType = half, so max ≈ 65504
    EXPECT_THROW((calculateMatmulTolerance<half, float, float>(a, b)), std::overflow_error);
}
