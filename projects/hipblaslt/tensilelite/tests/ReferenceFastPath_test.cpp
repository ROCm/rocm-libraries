// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    ContractionProblemGemm makePackedProblem(rocisa::DataType typeA,
                                             rocisa::DataType typeB,
                                             rocisa::DataType typeC,
                                             size_t           M,
                                             size_t           N,
                                             size_t           K)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            typeA,
                                                            typeB,
                                                            typeC,
                                                            typeC,
                                                            M,
                                                            N,
                                                            K,
                                                            1,
                                                            M,
                                                            M * K,
                                                            K,
                                                            K * N,
                                                            M,
                                                            M * N,
                                                            M,
                                                            M * N,
                                                            0.0);
        problem.setComputeInputTypeA(typeA);
        problem.setComputeInputTypeB(typeB);
        problem.setAlphaType(typeC);
        problem.setBetaType(typeC);
        return problem;
    }
} // namespace

TEST(ReferenceFastPath, PreservesDoublePrecisionForF64)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(rocisa::DataType::Double,
                                     rocisa::DataType::Double,
                                     rocisa::DataType::Double,
                                     M,
                                     N,
                                     K);
    ASSERT_TRUE(isFastPathEligible(problem));

    const double a0 = 1.0 + std::ldexp(1.0, -40);
    const double a1 = 1.0 + std::ldexp(1.0, -41);
    const double b0 = 1.0 + std::ldexp(1.0, -42);
    const double b1 = 1.0 + std::ldexp(1.0, -43);

    std::vector<double> a = {a0, a1};
    std::vector<double> b = {b0, b1};
    std::vector<double> c = {0.0};
    std::vector<double> d = {0.0};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0, 0.0);
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    const double expected = a0 * b0 + a1 * b1;
    ASSERT_NE(static_cast<double>(static_cast<float>(expected)), expected);
    EXPECT_EQ(d[0], expected);
}

TEST(ReferenceFastPath, AppliesXFloat32OperandMathOpToBothOperands)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(rocisa::DataType::Float,
                                     rocisa::DataType::Float,
                                     rocisa::DataType::Float,
                                     M,
                                     N,
                                     K);
    problem.setF32XdlMathOp(rocisa::DataType::XFloat32);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<float> a = {1.234567f, -2.345678f};
    std::vector<float> b = {3.456789f, 4.567891f};
    std::vector<float> c = {0.0f};
    std::vector<float> d = {0.0f};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    auto xf32 = [](float v) { return static_cast<float>(XFloat32(v)); };
    const float expected = xf32(a[0]) * xf32(b[0]) + xf32(a[1]) * xf32(b[1]);
    const float fullF32  = a[0] * b[0] + a[1] * b[1];

    ASSERT_NE(expected, fullF32);
    EXPECT_EQ(d[0], expected);
}

TEST(ReferenceFastPath, DeviceScalarAlphaOverridesInlineZeroInSlowPath)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseScaleAlphaVec(1);
    problem.setParams().setDeviceScalarAlpha(true);

    std::vector<float> a           = {1.0f, 2.0f};
    std::vector<float> b           = {3.0f, 4.0f};
    std::vector<float> c           = {0.0f};
    std::vector<float> d           = {0.0f};
    float              deviceAlpha = 2.0f;

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 0.0f, 0.0f);
    inputs.scaleAlphaVec = &deviceAlpha;

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);

    EXPECT_EQ(d[0], 22.0f);
}

TEST(ReferenceFastPath, DeviceScalarAlphaOverridesInlineZeroInFastPath)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseScaleAlphaVec(1);
    problem.setParams().setDeviceScalarAlpha(true);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<float> a           = {1.0f, 2.0f};
    std::vector<float> b           = {3.0f, 4.0f};
    std::vector<float> c           = {0.0f};
    std::vector<float> d           = {0.0f};
    float              deviceAlpha = 2.0f;

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 0.0f, 0.0f);
    inputs.scaleAlphaVec = &deviceAlpha;

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    EXPECT_EQ(d[0], 22.0f);
}

TEST(ReferenceFastPath, DeviceScalarAlphaZeroSkipsNaNReductionInFastPath)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseScaleAlphaVec(1);
    problem.setParams().setDeviceScalarAlpha(true);
    ASSERT_TRUE(isFastPathEligible(problem));

    const float        nan         = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> a           = {nan, nan};
    std::vector<float> b           = {1.0f, 1.0f};
    std::vector<float> c           = {7.0f};
    std::vector<float> d           = {nan};
    float              deviceAlpha = 0.0f;

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 2.0f, 3.0f);
    inputs.scaleAlphaVec = &deviceAlpha;

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    EXPECT_FALSE(std::isnan(d[0]));
    EXPECT_EQ(d[0], 21.0f);
}

TEST(ReferenceFastPath, DeviceScalarAlphaZeroAllowsNullInputsInFastPath)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseScaleAlphaVec(1);
    problem.setParams().setDeviceScalarAlpha(true);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<float> c           = {3.0f};
    std::vector<float> d           = {0.0f};
    float              deviceAlpha = 0.0f;

    ContractionInputs inputs(nullptr, nullptr, c.data(), d.data(), 2.0f, 4.0f);
    inputs.scaleAlphaVec = &deviceAlpha;

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    EXPECT_EQ(d[0], 12.0f);
}

TEST(ReferenceFastPath, DeviceScalarAlphaZeroAllowsNullInputsInSlowPath)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseScaleAlphaVec(1);
    problem.setParams().setDeviceScalarAlpha(true);

    std::vector<float> c           = {3.0f};
    std::vector<float> d           = {0.0f};
    float              deviceAlpha = 0.0f;

    ContractionInputs inputs(nullptr, nullptr, c.data(), d.data(), 2.0f, 4.0f);
    inputs.scaleAlphaVec = &deviceAlpha;

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);

    EXPECT_EQ(d[0], 12.0f);
}
