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

namespace
{
    template <typename T>
    std::vector<T> f16AccumulationOperand(std::vector<float> const& values)
    {
        std::vector<T> out;
        if constexpr(std::is_same<T, Float4x2>::value)
            for(size_t i = 0; i < values.size(); i += 2)
                out.push_back(Float4x2(values[i], values[i + 1]));
        else
            for(float v : values)
                out.push_back(T(v));
        return out;
    }

    // One K step of 64. Each group of 16 products has one of +-96 (+-36 for fp4 x fp4) and fifteen of 2^-8 (0.25);
    // the 2^-8 products are below their group's truncation step, so f16 accumulation gives 0 where f32 accumulation
    // gives 60 * 2^-8. The fp4 x fp4 products are all kept: 15.
    template <typename TA, typename TB, typename TD>
    void expectF16AccumulationCell(rocisa::DataType typeA,
                                   rocisa::DataType typeB,
                                   rocisa::DataType computeA,
                                   rocisa::DataType computeB,
                                   rocisa::DataType typeD)
    {
        constexpr bool fp4A = std::is_same<TA, Float4x2>::value;
        constexpr bool fp4B = std::is_same<TB, Float4x2>::value;
        const size_t   K    = 64;

        auto problem = ContractionProblemGemm::GEMM_Strides(
            false, false, typeA, typeB, typeD, typeD, 1, 1, K, 1, 1, K, K, K, 1, 1, 1, 1, 0.0);
        problem.setComputeInputTypeA(computeA);
        problem.setComputeInputTypeB(computeB);
        problem.setAlphaType(rocisa::DataType::Half);
        problem.setBetaType(rocisa::DataType::Half);
        problem.setHighPrecisionAccumulate(false);
        ASSERT_TRUE(isF16AccumulationProblem(problem));
        ASSERT_FALSE(isFastPathEligible(problem));

        std::vector<float> a(K), b(K);
        for(size_t k = 0; k < K; k++)
        {
            bool  big  = k % 16 == 0;
            float sign = (k / 16) % 2 ? -1.0f : 1.0f;
            a[k]       = big ? sign * (fp4A ? 6.0f : 16.0f) : (fp4A ? 0.5f : 0.0078125f);
            b[k]       = big ? (fp4B ? 6.0f : 16.0f) : (fp4B ? 0.5f : 0.0078125f);
        }
        auto            aData = f16AccumulationOperand<TA>(a);
        auto            bData = f16AccumulationOperand<TB>(b);
        std::vector<TD> c(1, TD(0.0f));
        std::vector<TD> d(1, TD(0.0f));

        ContractionInputs inputs(aData.data(), bData.data(), c.data(), d.data(), Half(1.0f), Half(0.0f));
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

        EXPECT_EQ(static_cast<float>(d[0]), fp4A && fp4B ? 15.0f : 0.0f);
    }
} // namespace

// Every fp4 cell of f16 accumulation, with the compute input types the library's problem types carry: the reference
// must dispatch it to the f16-accumulate model.
TEST(ReferenceF16Accumulation, Fp4CellsUseTheF16AccumulateModel)
{
    using DT = rocisa::DataType;
    expectF16AccumulationCell<Float4x2, Float8, Half>(DT::Float4, DT::Float8, DT::Float4, DT::Float8, DT::Half);
    expectF16AccumulationCell<Float8, Float4x2, Half>(DT::Float8, DT::Float4, DT::Float8, DT::Float4, DT::Half);
    expectF16AccumulationCell<Float4x2, Float4x2, Half>(DT::Float4, DT::Float4, DT::Float8, DT::Float8, DT::Half);
    expectF16AccumulationCell<Float4x2, Float8, Float8>(DT::Float4, DT::Float8, DT::Float4, DT::Float8, DT::Float8);
    expectF16AccumulationCell<Float8, Float4x2, Float8>(DT::Float8, DT::Float4, DT::Float8, DT::Float4, DT::Float8);
    expectF16AccumulationCell<Float4x2, Float4x2, Float8>(DT::Float4, DT::Float4, DT::Float8, DT::Float8, DT::Float8);
    expectF16AccumulationCell<Float4x2, Float8, BFloat16>(
        DT::Float4, DT::Float8, DT::Float4, DT::Float8, DT::BFloat16);
    expectF16AccumulationCell<Float8, Float4x2, BFloat16>(
        DT::Float8, DT::Float4, DT::Float8, DT::Float4, DT::BFloat16);
    expectF16AccumulationCell<Float4x2, Float4x2, BFloat16>(
        DT::Float4, DT::Float4, DT::Float8, DT::Float8, DT::BFloat16);
}
