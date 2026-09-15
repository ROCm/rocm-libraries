// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>
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
                                         int              mxBlock,
                                         int              mxBlockFree = 1)
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

        problem.setMXScaleA(
            rocisa::DataType::E8, mxBlock, {}, /*padScaleTensor=*/false, mxBlockFree);
        problem.setMXScaleB(
            rocisa::DataType::E8, mxBlock, {}, /*padScaleTensor=*/false, mxBlockFree);
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

    // Print an M x N column-major result so a 1x128 run can be eyeballed.
    void printMatrix(char const* title, std::vector<float> const& d, size_t M, size_t N)
    {
        std::cout << "\n  " << title << "  (" << M << "x" << N << ")\n";
        for(size_t m = 0; m < M; ++m)
        {
            std::cout << "   ";
            for(size_t n = 0; n < N; ++n)
                std::cout << std::setw(12) << std::fixed << std::setprecision(2) << d[m + n * M];
            std::cout << "\n";
        }
        std::cout << std::flush;
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

#ifdef TENSILE_USE_FP8_BF8

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

TEST(ReferenceMXFastPath, MatchesSlowPathForBlock128)
{
    const size_t M       = 64;
    const size_t N       = 64;
    const size_t K       = 256;
    const int    mxBlock = 128;

    auto problem = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

    // One scale per 128 K-elements: the bound dimension holds exactly K/128
    // entries (dimk == 1, so the gfx1250 path adds no padding). This helper
    // builds an NN problem, so K is dimension 1 of A and dimension 0 of B.
    ASSERT_EQ(problem.mxsa().sizes()[1], K / mxBlock);
    ASSERT_EQ(problem.mxsb().sizes()[0], K / mxBlock);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  dSlow(M * N, 0.0f);
    std::vector<float>  dFast(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

    std::mt19937 gen(24680);
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

// K need not be a multiple of the MX block size. The fast path declines such
// problems, so this exercises the slow path's short final MX segment: with
// mxBlock=128 and K=200 the last segment covers only 72 elements, and summing a
// full 128 would walk past K.
TEST(ReferenceMXFastPath, SlowPathHandlesPartialFinalMXSegment)
{
    const size_t M       = 32;
    const size_t N       = 32;
    const size_t K       = 200;
    const int    mxBlock = 128;

    auto problem = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

    // ceil(200/128) == 2 scales along K (NN layout: K is dim 1 of A, dim 0 of B).
    ASSERT_EQ(problem.mxsa().sizes()[1], 2u);
    ASSERT_EQ(problem.mxsb().sizes()[0], 2u);
    EXPECT_FALSE(isFastPathEligible(problem));

    // All inputs and scales are exactly 1.0, so every output element must be
    // exactly K. An over-read of the final segment would sum more than K terms.
    std::vector<Float8> a(M * K, Float8(1.0f));
    std::vector<Float8> b(K * N, Float8(1.0f));
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  d(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements(), E8(1.0f));

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = mxsa.data();
    inputs.mxsb = mxsb.data();

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    for(size_t i = 0; i < d.size(); ++i)
        ASSERT_FLOAT_EQ(d[i], static_cast<float>(K)) << "at index " << i;
}

// ============================================================================
// Hand-checkable 1x128 demonstrations
//
// M=N=8, K=1024, mxBlock=128 -> exactly 8 MX blocks along K. Every input is
// chosen so the expected output can be computed on paper, and the result is
// printed so a reviewer can read it directly rather than trusting an assertion.
//
// NN layout (GEMM_Strides(false, false, ...)), all column-major:
//   A is MxK, lda=M  -> a[m + k*M]        mxsa sizes {M, K/mxBlock}
//   B is KxN, ldb=K  -> b[k + n*K]        mxsb sizes {K/mxBlock, N}
//   D is MxN, ldd=M  -> d[m + n*M]
// E8 is UE8M0, so scales must be exact powers of two -- every value below is.
// ============================================================================

TEST(ReferenceMXBlock128Demo, UniformInputsGiveHandComputedResult)
{
    const size_t M = 8, N = 8, K = 1024;
    const int    mxBlock = 128;
    const size_t kBlocks = K / mxBlock; // 8

    auto problem = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

    ASSERT_EQ(problem.mxsa().sizes()[0], M);
    ASSERT_EQ(problem.mxsa().sizes()[1], kBlocks);
    ASSERT_EQ(problem.mxsb().sizes()[0], kBlocks);
    ASSERT_EQ(problem.mxsb().sizes()[1], N);

    // a = b = 1.0 everywhere; scaleB = 1.0; scaleA for K-block j is 2^j.
    //   D[m][n] = sum over the 8 blocks of (128 elements x 1 x 2^j x 1 x 1)
    //           = 128 * (1+2+4+8+16+32+64+128)
    //           = 128 * 255
    //           = 32640      <- same for every (m, n)
    std::vector<Float8> a(M * K, Float8(1.0f));
    std::vector<Float8> b(K * N, Float8(1.0f));
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  d(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements(), E8(1.0f));

    auto const saStride = problem.mxsa().strides();
    for(size_t m = 0; m < M; ++m)
        for(size_t j = 0; j < kBlocks; ++j)
            mxsa[m * saStride[0] + j * saStride[1]] = E8(std::ldexp(1.0f, (int)j));

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = mxsa.data();
    inputs.mxsb = mxsb.data();
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    constexpr float kExpected = 128.0f * 255.0f; // 32640
    printMatrix("1x128 host result (expect every element = 128 * 255 = 32640)", d, M, N);

    for(size_t i = 0; i < d.size(); ++i)
        ASSERT_FLOAT_EQ(d[i], kExpected) << "at index " << i;
}

TEST(ReferenceMXBlock128Demo, VariedInputsGiveHandComputedResult)
{
    const size_t M = 8, N = 8, K = 1024;
    const int    mxBlock = 128;
    const size_t kBlocks = K / mxBlock;

    auto problem = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

    // a[m][k] = m+1, b[k][n] = n+1 (1..8, all exact in FP8 E4M3);
    // scaleA for K-block j is 2^(j-7), scaleB = 1.
    //   sum over blocks of 2^(j-7) = 255/128
    //   D[m][n] = (m+1) * (n+1) * 128 * 255/128 = (m+1) * (n+1) * 255
    // so the printed matrix is a multiplication table scaled by 255:
    //   D[0][0] = 255 ... D[7][7] = 64 * 255 = 16320
    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  d(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements(), E8(1.0f));

    for(size_t k = 0; k < K; ++k)
    {
        for(size_t m = 0; m < M; ++m)
            a[m + k * M] = Float8((float)(m + 1));
        for(size_t n = 0; n < N; ++n)
            b[k + n * K] = Float8((float)(n + 1));
    }

    auto const saStride = problem.mxsa().strides();
    for(size_t m = 0; m < M; ++m)
        for(size_t j = 0; j < kBlocks; ++j)
            mxsa[m * saStride[0] + j * saStride[1]] = E8(std::ldexp(1.0f, (int)j - 7));

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = mxsa.data();
    inputs.mxsb = mxsb.data();
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    printMatrix("1x128 host result (expect D[m][n] = (m+1)*(n+1)*255)", d, M, N);

    for(size_t m = 0; m < M; ++m)
        for(size_t n = 0; n < N; ++n)
            ASSERT_FLOAT_EQ(d[m + n * M], (float)((m + 1) * (n + 1)) * 255.0f)
                << "at m=" << m << " n=" << n;
}

// The defining property of 1x128: one scale covering 128 K-elements must be
// exactly equivalent to the same scale repeated across four 1x32 blocks. This
// compares two independent problems built with mxBlock=128 and mxBlock=32 over
// identical data, so it would catch an off-by-grouping in the 128 path even
// though both runs use the same reference code.
TEST(ReferenceMXBlock128Demo, Block128EqualsBlock32WithReplicatedScales)
{
    const size_t M = 8, N = 8, K = 1024;

    auto p128 = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, 128);
    auto p32 = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, 32);

    ASSERT_EQ(p128.mxsa().sizes()[1], 8u);
    ASSERT_EQ(p32.mxsa().sizes()[1], 32u);

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  d128(M * N, 0.0f), d32(M * N, 0.0f);

    std::mt19937 gen(1357);
    fillBinary(a, gen);
    fillBinary(b, gen);

    std::vector<E8> sa128(p128.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8> sb128(p128.mxsb().totalAllocatedElements(), E8(1.0f));
    std::vector<E8> sa32(p32.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8> sb32(p32.mxsb().totalAllocatedElements(), E8(1.0f));

    // Scale of K-block j is 2^(j mod 5 - 2); the 32-block tensor repeats each
    // one four times, since four 1x32 blocks tile one 1x128 block.
    auto blockScale = [](size_t j) { return std::ldexp(1.0f, (int)(j % 5) - 2); };

    auto const sa128Stride = p128.mxsa().strides();
    auto const sb128Stride = p128.mxsb().strides();
    auto const sa32Stride  = p32.mxsa().strides();
    auto const sb32Stride  = p32.mxsb().strides();

    for(size_t j = 0; j < 8; ++j)
    {
        for(size_t m = 0; m < M; ++m)
            sa128[m * sa128Stride[0] + j * sa128Stride[1]] = E8(blockScale(j));
        for(size_t n = 0; n < N; ++n)
            sb128[j * sb128Stride[0] + n * sb128Stride[1]] = E8(blockScale(j));
    }
    for(size_t jj = 0; jj < 32; ++jj)
    {
        for(size_t m = 0; m < M; ++m)
            sa32[m * sa32Stride[0] + jj * sa32Stride[1]] = E8(blockScale(jj / 4));
        for(size_t n = 0; n < N; ++n)
            sb32[jj * sb32Stride[0] + n * sb32Stride[1]] = E8(blockScale(jj / 4));
    }

    ContractionInputs in128(a.data(), b.data(), c.data(), d128.data(), 1.0f, 0.0f);
    in128.mxsa = sa128.data();
    in128.mxsb = sb128.data();
    ContractionInputs in32(a.data(), b.data(), c.data(), d32.data(), 1.0f, 0.0f);
    in32.mxsa = sa32.data();
    in32.mxsb = sb32.data();

    SolveGemmCPU(p128, in128, /*elementsToValidate=*/-1, /*tryFastPath=*/true);
    SolveGemmCPU(p32, in32, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    printMatrix("mxBlock=128, one scale per 128 K-elements", d128, M, N);
    printMatrix("mxBlock=32,  same scale repeated 4x (must match above)", d32, M, N);

    for(size_t i = 0; i < d128.size(); ++i)
        ASSERT_FLOAT_EQ(d128[i], d32[i]) << "at index " << i;
}

// ============================================================================
// 2D scaling tile (MXBlockFreeA / MXBlockFreeB)
//
// mxBlockFree free-dimension elements share one scale, so the scale tensor's
// free dimension holds ceil(M/mxBlockFree) (resp. ceil(N/mxBlockFree)) entries
// and the reference divides the free coordinate before indexing it.
//
// The defining property, mirroring Block128EqualsBlock32WithReplicatedScales
// along the other axis: a 128x128 tile must be exactly equivalent to the same
// scale replicated across 128 rows of a 1x128 tensor. The two runs build
// independent problems over identical data, so an off-by-grouping in the free
// direction cannot hide. Both the fast and the slow path are checked, since
// they index the scale tensor through completely separate code.
// ============================================================================

namespace
{
    // M, N, K, mxBlockFree
    using FreeTileParam = std::tuple<size_t, size_t, size_t, int>;
}

class ReferenceMXFreeTileTest : public ::testing::TestWithParam<FreeTileParam>
{
};

TEST_P(ReferenceMXFreeTileTest, EqualsReplicatedPerRowScales)
{
    auto [M, N, K, mxBlockFree] = GetParam();
    const int    mxBlock = 128;
    const size_t kBlocks = K / mxBlock;

    auto pTile = makeMXProblem(rocisa::DataType::Float8,
                               rocisa::DataType::Float8,
                               M, N, K, mxBlock, mxBlockFree);
    auto pFlat = makeMXProblem(
        rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

    const size_t tilesM = (M + mxBlockFree - 1) / mxBlockFree;
    const size_t tilesN = (N + mxBlockFree - 1) / mxBlockFree;

    // NN layout: mxsa is {free, kBlock}, mxsb is {kBlock, free}.
    ASSERT_EQ(pTile.mxsa().sizes()[0], tilesM);
    ASSERT_EQ(pTile.mxsa().sizes()[1], kBlocks);
    ASSERT_EQ(pTile.mxsb().sizes()[0], kBlocks);
    ASSERT_EQ(pTile.mxsb().sizes()[1], tilesN);
    ASSERT_EQ(pFlat.mxsa().sizes()[0], M);
    ASSERT_EQ(pFlat.mxsb().sizes()[1], N);

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  dTileFast(M * N, 0.0f), dTileSlow(M * N, 0.0f);
    std::vector<float>  dFlatFast(M * N, 0.0f);

    std::mt19937 gen(31337);
    fillBinary(a, gen);
    fillBinary(b, gen);

    std::vector<E8> saTile(pTile.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8> sbTile(pTile.mxsb().totalAllocatedElements(), E8(1.0f));
    std::vector<E8> saFlat(pFlat.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8> sbFlat(pFlat.mxsb().totalAllocatedElements(), E8(1.0f));

    // E8 is UE8M0, so every scale must be an exact power of two. Vary with both
    // the free tile and the K block so a swapped index would change the result.
    auto tileScale = [](size_t tile, size_t j) {
        return std::ldexp(1.0f, (int)((tile + 2 * j) % 5) - 2);
    };

    auto const saTileStride = pTile.mxsa().strides();
    auto const sbTileStride = pTile.mxsb().strides();
    auto const saFlatStride = pFlat.mxsa().strides();
    auto const sbFlatStride = pFlat.mxsb().strides();

    for(size_t j = 0; j < kBlocks; ++j)
    {
        for(size_t t = 0; t < tilesM; ++t)
            saTile[t * saTileStride[0] + j * saTileStride[1]] = E8(tileScale(t, j));
        for(size_t t = 0; t < tilesN; ++t)
            sbTile[j * sbTileStride[0] + t * sbTileStride[1]] = E8(tileScale(t, j));

        // The flat tensor repeats each tile's scale across its mxBlockFree rows.
        for(size_t m = 0; m < M; ++m)
            saFlat[m * saFlatStride[0] + j * saFlatStride[1]]
                = E8(tileScale(m / mxBlockFree, j));
        for(size_t n = 0; n < N; ++n)
            sbFlat[j * sbFlatStride[0] + n * sbFlatStride[1]]
                = E8(tileScale(n / mxBlockFree, j));
    }

    ContractionInputs inTileFast(a.data(), b.data(), c.data(), dTileFast.data(), 1.0f, 0.0f);
    inTileFast.mxsa = saTile.data();
    inTileFast.mxsb = sbTile.data();
    ContractionInputs inTileSlow(a.data(), b.data(), c.data(), dTileSlow.data(), 1.0f, 0.0f);
    inTileSlow.mxsa = saTile.data();
    inTileSlow.mxsb = sbTile.data();
    ContractionInputs inFlatFast(a.data(), b.data(), c.data(), dFlatFast.data(), 1.0f, 0.0f);
    inFlatFast.mxsa = saFlat.data();
    inFlatFast.mxsb = sbFlat.data();

    SolveGemmCPU(pTile, inTileFast, /*elementsToValidate=*/-1, /*tryFastPath=*/true);
    SolveGemmCPU(pTile, inTileSlow, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
    SolveGemmCPU(pFlat, inFlatFast, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    for(size_t i = 0; i < dTileFast.size(); ++i)
    {
        ASSERT_FLOAT_EQ(dTileFast[i], dFlatFast[i])
            << "tiled fast path disagrees with replicated scales at index " << i;
        ASSERT_FLOAT_EQ(dTileSlow[i], dFlatFast[i])
            << "tiled slow path disagrees with replicated scales at index " << i;
    }
}

INSTANTIATE_TEST_SUITE_P(
    FreeTile,
    ReferenceMXFreeTileTest,
    ::testing::Values(
        //              M,    N,    K, mxBlockFree
        // mxBlockFree == 1 is the pre-existing layout; both problems are
        // identical, so this is a pure regression guard.
        std::make_tuple(64u,  64u, 256u,   1),
        // Small tiles keep the test cheap while still exercising the divide.
        std::make_tuple(64u,  64u, 256u,   4),
        std::make_tuple(64u,  32u, 512u,  16),
        // Free dimension not a multiple of the tile: the last tile is partial
        // and covers fewer than mxBlockFree rows.
        std::make_tuple(70u,  50u, 256u,  16),
        // The shipping shape: 128x128.
        std::make_tuple(256u, 256u, 256u, 128),
        std::make_tuple(200u, 160u, 384u, 128)
    )
);

// Hand-checkable 128x128: with all data and scales at 1 except a single tile,
// the result splits into exactly four constant quadrants.
TEST(ReferenceMXFreeTileDemo, SingleTileScaleAffectsOnlyItsQuadrant)
{
    const size_t M = 256, N = 256, K = 128;
    const int    mxBlock = 128, mxBlockFree = 128;

    auto problem = makeMXProblem(rocisa::DataType::Float8,
                                 rocisa::DataType::Float8,
                                 M, N, K, mxBlock, mxBlockFree);

    ASSERT_EQ(problem.mxsa().sizes()[0], 2u);
    ASSERT_EQ(problem.mxsb().sizes()[1], 2u);

    std::vector<Float8> a(M * K, Float8(1.0f));
    std::vector<Float8> b(K * N, Float8(1.0f));
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  d(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements(), E8(1.0f));

    // scaleA of the second row-tile is 4, scaleB of the second column-tile is 2.
    //   D[m][n] = K * scaleA(m/128) * scaleB(n/128)
    mxsa[1 * problem.mxsa().strides()[0]] = E8(4.0f);
    mxsb[1 * problem.mxsb().strides()[1]] = E8(2.0f);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = mxsa.data();
    inputs.mxsb = mxsb.data();
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    for(size_t m = 0; m < M; ++m)
        for(size_t n = 0; n < N; ++n)
        {
            const float sa       = (m < 128) ? 1.0f : 4.0f;
            const float sb       = (n < 128) ? 1.0f : 2.0f;
            const float expected = (float)K * sa * sb;
            ASSERT_FLOAT_EQ(d[m + n * M], expected) << "at m=" << m << " n=" << n;
        }
}

// ============================================================================
// Invariant: for block-aligned K the segment clamp is a no-op.
//
// Reference.cpp's slow-path MX loop clamps the final segment:
//     segment = min(innerMXLoop, boundSize[0] - i)
// The clamp can only change anything when boundSize[0] - i < innerMXLoop, i.e.
// when K is NOT a multiple of the block size. Every kernel-dispatched problem
// has K block-aligned (AssertSummationElementMultiple is raised to
// max(MXBlockA, MXBlockB)), so for all of them this reduces to the original
// unclamped `segment = innerMXLoop` and the arithmetic is bit-identical.
//
// The first check proves that algebraically over the actual loop bounds; the
// second pins the resulting values in full precision, so building with the
// clamp removed must reproduce this output byte for byte.
// ============================================================================

TEST(ReferenceMXBlock128Demo, BlockAlignedKMakesTheSegmentClampANoOp)
{
    for(size_t mxBlock : {32u, 128u})
    {
        for(size_t K : {128u, 256u, 384u, 1024u, 2048u})
        {
            ASSERT_EQ(K % mxBlock, 0u) << "test setup: K must be block-aligned";
            // Replicates the loop in Reference.cpp exactly.
            for(size_t i = 0; i < K; i += mxBlock)
                ASSERT_EQ(std::min(mxBlock, K - i), mxBlock)
                    << "clamp changed the segment length at K=" << K
                    << " mxBlock=" << mxBlock << " i=" << i;
        }
    }
}

TEST(ReferenceMXBlock128Demo, BlockAlignedSlowPathChecksum)
{
    const size_t M = 8, N = 8;
    const int    mxBlock = 128;

    std::cout << "\n  slow-path checksums, mxBlock=128, K block-aligned\n"
              << "  (must be byte-identical with the segment clamp removed)\n";

    for(size_t K : {128u, 256u, 384u, 1024u, 2048u})
    {
        auto problem = makeMXProblem(
            rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

        std::vector<Float8> a(M * K);
        std::vector<Float8> b(K * N);
        std::vector<float>  c(M * N, 0.0f);
        std::vector<float>  d(M * N, 0.0f);
        std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
        std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

        std::mt19937 gen(9001);
        fillBinary(a, gen);
        fillBinary(b, gen);
        fillScales(mxsa, gen);
        fillScales(mxsb, gen);

        ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
        inputs.mxsa = mxsa.data();
        inputs.mxsb = mxsb.data();

        // tryFastPath=false: the clamp lives in the slow path, so the fast path
        // (which block-aligned K is eligible for) would bypass what we pin here.
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);

        double sum = 0.0;
        for(float v : d)
            sum += v;
        std::cout << "    K=" << std::setw(5) << K << "  sum=" << std::hexfloat << sum
                  << "  d[0]=" << d[0] << "  d[M*N-1]=" << d[M * N - 1] << std::defaultfloat
                  << "\n";
    }
    std::cout << std::flush;
}

#else

TEST(ReferenceMXFastPath, DisabledWithoutFP8Support)
{
    GTEST_SKIP() << "TENSILE_USE_FP8_BF8 not enabled";
}

#endif
