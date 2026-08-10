// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <roc/host_validation/comparison.hpp>
#include <roc/host_validation/generation.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    ContractionProblemGemm makeMXProblem(
        rocisa::DataType typeA, rocisa::DataType typeB, size_t M, size_t N, size_t K, int mxBlock)
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

    template <typename T>
    void generateValues(std::vector<T>&                             values,
                        roc::host_validation::ScalarType            type,
                        roc::host_validation::GenerationPatternSpec pattern,
                        std::uint64_t                               seed,
                        std::uint64_t                               stream)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        using namespace roc::host_validation;

        pattern.stream = stream;
        GenerationOptions options;
        options.seed = seed;
        options.real = std::move(pattern);
        generate(MutableTensorView(type,
                                   Layout::contiguous(Shape{values.size()}),
                                   std::as_writable_bytes(std::span<T>(values))),
                 options);
    }
} // namespace

#ifndef _WIN32

TEST(ReferenceMXFastPath, RejectsMixedInputTypesWithMXFP4)
{
    const size_t M       = 64;
    const size_t N       = 64;
    const size_t K       = 128;
    const int    mxBlock = 32;

    auto problemA
        = makeMXProblem(rocisa::DataType::Float4, rocisa::DataType::Float, M, N, K, mxBlock);
    EXPECT_FALSE(isFastPathEligible(problemA));

    auto problemB
        = makeMXProblem(rocisa::DataType::Float, rocisa::DataType::Float4, M, N, K, mxBlock);
    EXPECT_FALSE(isFastPathEligible(problemB));

    auto problemBoth
        = makeMXProblem(rocisa::DataType::Float4, rocisa::DataType::Float4, M, N, K, mxBlock);
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

    auto problem
        = makeMXProblem(rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);
    ASSERT_TRUE(isFastPathEligible(problem));

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  dSlow(M * N, 0.0f);
    std::vector<float>  dFast(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

    roc::host_validation::GenerationPatternSpec binary;
    binary.pattern    = roc::host_validation::GenerationPattern::CandidateSet;
    binary.candidates = {-1.0, 1.0};
    roc::host_validation::GenerationPatternSpec scale;
    scale.pattern    = roc::host_validation::GenerationPattern::CandidateSet;
    scale.candidates = {1.0, 2.0, 4.0};
    generateValues(a, roc::host_validation::ScalarType::Float8E4M3, binary, 12345, 0);
    generateValues(b, roc::host_validation::ScalarType::Float8E4M3, binary, 12345, 1);
    generateValues(mxsa, roc::host_validation::ScalarType::E8M0, scale, 12345, 2);
    generateValues(mxsb, roc::host_validation::ScalarType::E8M0, scale, 12345, 3);

    ContractionInputs inputsSlow(a.data(), b.data(), c.data(), dSlow.data(), 1.0f, 0.0f);
    inputsSlow.mxsa = mxsa.data();
    inputsSlow.mxsb = mxsb.data();

    ContractionInputs inputsFast(a.data(), b.data(), c.data(), dFast.data(), 1.0f, 0.0f);
    inputsFast.mxsa = mxsa.data();
    inputsFast.mxsb = mxsb.data();

    SolveGemmCPU(problem, inputsSlow, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
    ASSERT_TRUE(tryRuntimeTiledGemm(problem, inputsFast, /*elementsToValidate=*/-1));

    const auto comparison
        = roc::host_validation::compare(std::span<const float>(dFast),
                                        std::span<const float>(dSlow),
                                        roc::host_validation::nearComparisonOptions(1e-3));
    EXPECT_TRUE(comparison.passed())
        << "mismatches=" << comparison.mismatches
        << " max_absolute_difference=" << comparison.maxAbsoluteDifference;
}

TEST(ReferenceMXFastPath, MatchesSlowPathWithBetaAndBias)
{
    const size_t M       = 48;
    const size_t N       = 32;
    const size_t K       = 96;
    const int    mxBlock = 32;

    auto problem
        = makeMXProblem(rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);
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

    roc::host_validation::GenerationPatternSpec binary;
    binary.pattern    = roc::host_validation::GenerationPattern::CandidateSet;
    binary.candidates = {-1.0, 1.0};
    roc::host_validation::GenerationPatternSpec scale;
    scale.pattern    = roc::host_validation::GenerationPattern::CandidateSet;
    scale.candidates = {1.0, 2.0, 4.0};
    roc::host_validation::GenerationPatternSpec cPattern;
    cPattern.pattern    = roc::host_validation::GenerationPattern::Constant;
    cPattern.parameter0 = 0.25;
    roc::host_validation::GenerationPatternSpec biasPattern;
    biasPattern.pattern    = roc::host_validation::GenerationPattern::Constant;
    biasPattern.parameter0 = 0.5;
    generateValues(a, roc::host_validation::ScalarType::Float8E4M3, binary, 54321, 0);
    generateValues(b, roc::host_validation::ScalarType::Float8E4M3, binary, 54321, 1);
    generateValues(mxsa, roc::host_validation::ScalarType::E8M0, scale, 54321, 2);
    generateValues(mxsb, roc::host_validation::ScalarType::E8M0, scale, 54321, 3);
    generateValues(c, roc::host_validation::ScalarType::Float32, cPattern, 54321, 4);
    generateValues(bias, roc::host_validation::ScalarType::Float32, biasPattern, 54321, 5);

    ContractionInputs inputsSlow(a.data(), b.data(), c.data(), dSlow.data(), 1.0f, 0.5f);
    inputsSlow.mxsa = mxsa.data();
    inputsSlow.mxsb = mxsb.data();
    inputsSlow.bias = bias.data();

    ContractionInputs inputsFast(a.data(), b.data(), c.data(), dFast.data(), 1.0f, 0.5f);
    inputsFast.mxsa = mxsa.data();
    inputsFast.mxsb = mxsb.data();
    inputsFast.bias = bias.data();

    SolveGemmCPU(problem, inputsSlow, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
    ASSERT_TRUE(tryRuntimeTiledGemm(problem, inputsFast, /*elementsToValidate=*/-1));

    const auto comparison
        = roc::host_validation::compare(std::span<const float>(dFast),
                                        std::span<const float>(dSlow),
                                        roc::host_validation::nearComparisonOptions(1e-3));
    EXPECT_TRUE(comparison.passed())
        << "mismatches=" << comparison.mismatches
        << " max_absolute_difference=" << comparison.maxAbsoluteDifference;
}

#else

TEST(ReferenceMXFastPath, DisabledWithoutFP8Support)
{
    GTEST_SKIP() << "TENSILE_USE_FP8_BF8 not enabled";
}

#endif
