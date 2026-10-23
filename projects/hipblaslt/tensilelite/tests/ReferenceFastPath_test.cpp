// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <cmath>
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

TEST(ReferenceBiasReduction, BatchedInputsRespectSourceAndOutputStrides)
{
    using Tensor = ContractionProblemGemm::TENSOR;
    const size_t m = 3, n = 5, batches = 2;
    for(auto source : {Tensor::A, Tensor::B})
    for(bool transpose : {false, true})
    for(size_t k : {size_t{0}, size_t{7}})
    for(size_t padding : {size_t{0}, size_t{4}})
    {
        auto problem = ContractionProblemGemm::GEMM(
            transpose, transpose, m, n, k, transpose ? std::max(k, size_t{1}) : m,
            transpose ? n : std::max(k, size_t{1}), m, 1.0, false, batches);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setUseBias(1);
        problem.setUseGradient(true);
        size_t length = source == Tensor::A ? m : n;
        size_t stride = length + padding;
        problem.setBias(rocisa::DataType::Float, length, stride, true, source);
        std::vector<float> a(problem.a().totalAllocatedElements(), 0);
        std::vector<float> b(problem.b().totalAllocatedElements(), 0);
        auto const& tensor = source == Tensor::A ? problem.a() : problem.b();
        auto& input = source == Tensor::A ? a : b;
        const size_t reduce = source == Tensor::A ? (transpose ? 0 : 1)
                                                  : (transpose ? 1 : 0);
        for(size_t batch = 0; batch < batches; ++batch)
        for(size_t i = 0; i < length; ++i)
        for(size_t j = 0; j < k; ++j)
        {
            std::vector<size_t> coord(3);
            coord[reduce] = j;
            coord[1 - reduce] = i;
            coord[2] = batch;
            input[tensor.index(coord)] = float(100 * batch + 10 * i + j);
        }
        std::vector<float> c(m * n * batches, 1), d(c.size(), -123);
        std::vector<float> bias(stride * batches, -123);
        ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 0.0f, 1.0f);
        inputs.bias = bias.data();
        SolveGemmCPU(problem, inputs, size_t(-1), false);
        for(size_t batch = 0; batch < batches; ++batch)
        for(size_t i = 0; i < stride; ++i)
        {
            const float expected = i < length
                ? float(k * (100 * batch + 10 * i) + (k ? k * (k - 1) / 2 : 0)) : -123;
            EXPECT_EQ(bias[batch * stride + i], expected);
        }
        for(float value : d)
            EXPECT_EQ(value, 1);
        problem.setBias(rocisa::DataType::Float, length, length - 1, true, source);
        EXPECT_THROW(SolveGemmCPU(problem, inputs, size_t(-1), false), std::runtime_error);
    }
}
