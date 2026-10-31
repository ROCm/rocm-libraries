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
    ContractionProblemGemm makeMXProblem(rocisa::DataType typeA,
                                         rocisa::DataType typeB,
                                         size_t           M,
                                         size_t           N,
                                         size_t           K,
                                         int              mxBlock)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            typeA,
                                                            typeB,
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
        problem.setComputeInputTypeA(typeA);
        problem.setComputeInputTypeB(typeB);
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

    // Float8 MX problem (float C/D) with K not necessarily a multiple of the MX
    // blocks. A K-contiguous operand (transA, or B not transposed) has leading
    // dimension K + kPad, so the elements just past K are padding.
    constexpr size_t kPad = 8;

    ContractionProblemGemm makeMXTailProblem(
        bool transA, bool transB, size_t M, size_t N, size_t K, int mxBlockA, int mxBlockB)
    {
        const size_t lda     = transA ? K + kPad : M;
        const size_t ldb     = transB ? N : K + kPad;
        auto         problem = ContractionProblemGemm::GEMM_Strides(transA,
                                                            transB,
                                                            rocisa::DataType::Float8,
                                                            rocisa::DataType::Float8,
                                                            rocisa::DataType::Float,
                                                            rocisa::DataType::Float,
                                                            M,
                                                            N,
                                                            K,
                                                            1,
                                                            lda,
                                                            lda * (transA ? M : K),
                                                            ldb,
                                                            ldb * (transB ? K : N),
                                                            M,
                                                            M * N,
                                                            M,
                                                            M * N,
                                                            0.0);

        problem.setMXScaleA(rocisa::DataType::E8, mxBlockA, {}, /*padScaleTensor=*/false);
        problem.setMXScaleB(rocisa::DataType::E8, mxBlockB, {}, /*padScaleTensor=*/false);
        problem.setComputeInputTypeA(rocisa::DataType::Float8);
        problem.setComputeInputTypeB(rocisa::DataType::Float8);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        return problem;
    }
} // namespace

#ifndef _WIN32

TEST(ReferenceMXFastPath, RejectsMixedInputTypesWithMXFP4)
{
    const size_t M       = 64;
    const size_t N       = 64;
    const size_t K       = 128;
    const int    mxBlock = 32;

    auto problemA = makeMXProblem(
        rocisa::DataType::Float4, rocisa::DataType::Float, M, N, K, mxBlock);
    EXPECT_FALSE(isFastPathEligible(problemA));

    auto problemB = makeMXProblem(
        rocisa::DataType::Float, rocisa::DataType::Float4, M, N, K, mxBlock);
    EXPECT_FALSE(isFastPathEligible(problemB));

    auto problemBoth = makeMXProblem(
        rocisa::DataType::Float4, rocisa::DataType::Float4, M, N, K, mxBlock);
    EXPECT_TRUE(isFastPathEligible(problemBoth));
}

#endif

TEST(ReferenceMXFastPath, PartialLastMXBlockIsEligible)
{
    // K = 333 / 255 / 63 with block 32 as in mxfp4_mxfp4_bf16_tn_act, K smaller
    // than the block, and K a multiple of only one side's block.
    for(size_t K : {333, 255, 63, 16, 5})
    {
        EXPECT_TRUE(isFastPathEligible(makeMXTailProblem(true, false, 64, 64, K, 32, 32)))
            << "K=" << K;
    }
    EXPECT_TRUE(isFastPathEligible(makeMXTailProblem(false, false, 64, 64, 96, 32, 64)));
    EXPECT_TRUE(isFastPathEligible(makeMXTailProblem(false, false, 64, 64, 96, 64, 32)));

    // Blocks that are not a multiple of the fast path's K tile stay rejected.
    EXPECT_FALSE(isFastPathEligible(makeMXTailProblem(true, false, 64, 64, 333, 4, 4)));
}

TEST(ReferenceMXFastPath, PartialLastMXBlockSumsOnlyElementsBelowK)
{
    // Every case has a partial last MX block on A, on B, or on both. A and B
    // are +-1 below K and 4 elsewhere (the K padding of a K-contiguous operand
    // and the elements past the K-slow operand's last K row), and the scales
    // are powers of two, so the expected D is exact and any product past K
    // changes it.
    struct Case
    {
        size_t K;
        int    mxBlockA;
        int    mxBlockB;
    };
    const Case cases[] = {
        {333, 32, 32},
        {255, 32, 32},
        {63, 32, 32},
        {16, 32, 32},
        {5, 32, 32},
        {40, 16, 32},
        {40, 32, 16},
        {96, 32, 64},
        {96, 64, 32},
    };
    const size_t M = 37;
    const size_t N = 35;

    std::mt19937                    gen(2024);
    std::uniform_int_distribution<> coin(0, 1);
    std::uniform_int_distribution<> expDist(0, 3);

    for(auto const& tc : cases)
    {
        for(bool transA : {false, true})
        {
            for(bool transB : {false, true})
            {
                auto problem
                    = makeMXTailProblem(transA, transB, M, N, tc.K, tc.mxBlockA, tc.mxBlockB);
                ASSERT_TRUE(isFastPathEligible(problem));

                const size_t lda = problem.a().strides()[1];
                const size_t ldb = problem.b().strides()[1];
                auto         aIndex
                    = [&](size_t m, size_t k) { return transA ? m * lda + k : k * lda + m; };
                auto bIndex
                    = [&](size_t k, size_t n) { return transB ? k * ldb + n : n * ldb + k; };

                // One K-block of padding after each operand's allocated elements.
                const size_t        tailA = std::max(tc.mxBlockA, tc.mxBlockB) * (transA ? 1 : lda);
                const size_t        tailB = std::max(tc.mxBlockA, tc.mxBlockB) * (transB ? ldb : 1);
                std::vector<Float8> a(problem.a().totalAllocatedElements() + tailA, Float8(4.0f));
                std::vector<Float8> b(problem.b().totalAllocatedElements() + tailB, Float8(4.0f));
                for(size_t m = 0; m < M; ++m)
                    for(size_t k = 0; k < tc.K; ++k)
                        a[aIndex(m, k)] = Float8(coin(gen) ? 1.0f : -1.0f);
                for(size_t n = 0; n < N; ++n)
                    for(size_t k = 0; k < tc.K; ++k)
                        b[bIndex(k, n)] = Float8(coin(gen) ? 1.0f : -1.0f);

                std::vector<E8> mxsa(problem.mxsa().totalAllocatedElements());
                std::vector<E8> mxsb(problem.mxsb().totalAllocatedElements());
                for(auto& v : mxsa)
                    v = E8(std::ldexp(1.0f, expDist(gen)));
                for(auto& v : mxsb)
                    v = E8(std::ldexp(1.0f, expDist(gen)));
                const size_t mxsaStrideM   = problem.mxsa().strides()[problem.freeIndicesA()[0].i];
                const size_t mxsaStrideBlk = problem.mxsa().strides()[problem.boundIndices()[0].a];
                const size_t mxsbStrideN   = problem.mxsb().strides()[problem.freeIndicesB()[0].i];
                const size_t mxsbStrideBlk = problem.mxsb().strides()[problem.boundIndices()[0].b];

                std::vector<float> expected(M * N, 0.0f);
                for(size_t n = 0; n < N; ++n)
                {
                    for(size_t m = 0; m < M; ++m)
                    {
                        float sum = 0.0f;
                        for(size_t k = 0; k < tc.K; ++k)
                        {
                            float sa = static_cast<float>(
                                mxsa[m * mxsaStrideM + (k / tc.mxBlockA) * mxsaStrideBlk]);
                            float sb = static_cast<float>(
                                mxsb[n * mxsbStrideN + (k / tc.mxBlockB) * mxsbStrideBlk]);
                            sum += static_cast<float>(a[aIndex(m, k)])
                                   * static_cast<float>(b[bIndex(k, n)]) * sa * sb;
                        }
                        expected[n * M + m] = sum;
                    }
                }

                std::vector<float> c(M * N, 0.0f);
                std::vector<float> d(M * N, 0.0f);
                ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
                inputs.mxsa = mxsa.data();
                inputs.mxsb = mxsb.data();
                SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

                EXPECT_EQ(maxAbsDiff(expected, d), 0.0f)
                    << "K=" << tc.K << " mxBlockA=" << tc.mxBlockA << " mxBlockB=" << tc.mxBlockB
                    << " transA=" << transA << " transB=" << transB;
            }
        }
    }
}

TEST(ReferenceMXFastPath, MatchesSlowPathForScaledFP8Gemm)
{
    const size_t M       = 64;
    const size_t N       = 64;
    const size_t K       = 128;
    const int    mxBlock = 32;

    auto problem = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);
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

    auto problem = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);
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
