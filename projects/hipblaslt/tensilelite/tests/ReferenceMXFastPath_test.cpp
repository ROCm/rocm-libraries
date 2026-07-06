// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <cmath>
#include <random>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    ContractionProblemGemm makeMXFP8Problem(size_t M, size_t N, size_t K, int mxBlock)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            rocisa::DataType::Float8,
                                                            rocisa::DataType::Float8,
                                                            rocisa::DataType::Float,
                                                            rocisa::DataType::Float,
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

        problem.setMXScaleA(rocisa::DataType::E8, mxBlock, {}, /*padScaleTensor=*/false);
        problem.setMXScaleB(rocisa::DataType::E8, mxBlock, {}, /*padScaleTensor=*/false);
        problem.setComputeInputTypeA(rocisa::DataType::Float8);
        problem.setComputeInputTypeB(rocisa::DataType::Float8);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        return problem;
    }

    void fillBinary(std::vector<Float8>& buf, std::mt19937& gen)
    {
        std::uniform_int_distribution<> coin(0, 1);
        for(auto& v : buf)
            v = Float8(coin(gen) ? 1.0f : -1.0f);
    }

    void fillScales(std::vector<E8>& buf, std::mt19937& gen)
    {
        std::uniform_real_distribution<float> mag(1.0f, 4.0f);
        for(auto& v : buf)
            v = E8(mag(gen));
    }

    float maxAbsDiff(std::vector<float> const& a, std::vector<float> const& b)
    {
        EXPECT_EQ(a.size(), b.size());
        float maxDiff = 0.0f;
        for(size_t i = 0; i < a.size(); ++i)
            maxDiff = std::max(maxDiff, std::fabs(a[i] - b[i]));
        return maxDiff;
    }
} // namespace

#ifdef TENSILE_USE_FP8_BF8

TEST(ReferenceMXFastPath, MatchesSlowPathForScaledFP8Gemm)
{
    const size_t M       = 64;
    const size_t N       = 64;
    const size_t K       = 128;
    const int    mxBlock = 32;

    auto problem = makeMXFP8Problem(M, N, K, mxBlock);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  dSlow(M * N, 0.0f);
    std::vector<float>  dFast(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

    std::mt19937 gen(12345);
    fillBinary(a, gen);
    fillBinary(b, gen);
    fillScales(mxsa, gen);
    fillScales(mxsb, gen);

    ContractionInputs inputsSlow(a.data(), b.data(), c.data(), dSlow.data(), 1.0f, 0.0f);
    inputsSlow.mxsa = mxsa.data();
    inputsSlow.mxsb = mxsb.data();

    ContractionInputs inputsFast(a.data(), b.data(), c.data(), dFast.data(), 1.0f, 0.0f);
    inputsFast.mxsa = mxsa.data();
    inputsFast.mxsb = mxsb.data();

    SolveGemmCPU(problem, inputsSlow, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
    SolveGemmCPU(problem, inputsFast, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    EXPECT_LT(maxAbsDiff(dSlow, dFast), 1e-3f);
}

TEST(ReferenceMXFastPath, MatchesSlowPathWithBetaAndBias)
{
    const size_t M       = 48;
    const size_t N       = 32;
    const size_t K       = 96;
    const int    mxBlock = 32;

    auto problem = makeMXFP8Problem(M, N, K, mxBlock);
    problem.setUseBias(1);
    problem.setBias(rocisa::DataType::Float, M, M);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N);
    std::vector<float>  dSlow(M * N, 0.0f);
    std::vector<float>  dFast(M * N, 0.0f);
    std::vector<float>  bias(M);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

    std::mt19937 gen(54321);
    fillBinary(a, gen);
    fillBinary(b, gen);
    fillScales(mxsa, gen);
    fillScales(mxsb, gen);
    for(auto& v : c)
        v = 0.25f;
    for(auto& v : bias)
        v = 0.5f;

    ContractionInputs inputsSlow(a.data(), b.data(), c.data(), dSlow.data(), 1.0f, 0.5f);
    inputsSlow.mxsa  = mxsa.data();
    inputsSlow.mxsb  = mxsb.data();
    inputsSlow.bias  = bias.data();

    ContractionInputs inputsFast(a.data(), b.data(), c.data(), dFast.data(), 1.0f, 0.5f);
    inputsFast.mxsa  = mxsa.data();
    inputsFast.mxsb  = mxsb.data();
    inputsFast.bias  = bias.data();

    SolveGemmCPU(problem, inputsSlow, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
    SolveGemmCPU(problem, inputsFast, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    EXPECT_LT(maxAbsDiff(dSlow, dFast), 1e-3f);
}

#else

TEST(ReferenceMXFastPath, DisabledWithoutFP8Support)
{
    GTEST_SKIP() << "TENSILE_USE_FP8_BF8 not enabled";
}

#endif
