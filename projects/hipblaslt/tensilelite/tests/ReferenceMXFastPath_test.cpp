// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <roc/host_numerics/adapters/tensilelite/Reference.hpp>
#include <roc/host_numerics/adapters/tensilelite/TensileDataGeneration.hpp>
#include <roc/host_numerics/comparison.hpp>
#include <roc/host_numerics/generation.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    constexpr roc::host_numerics::GemmBackend requireBlockedExecution
        = roc::host_numerics::GemmBackend::Blocked;

    ContractionProblemGemm makeMXProblem(rocisa::DataType typeA,
                                         rocisa::DataType typeB,
                                         size_t           M,
                                         size_t           N,
                                         size_t           K,
                                         int              mxBlock,
                                         rocisa::DataType scaleType = rocisa::DataType::E8)
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

        problem.setMXScaleA(scaleType, mxBlock, {}, /*padScaleTensor=*/false);
        problem.setMXScaleB(scaleType, mxBlock, {}, /*padScaleTensor=*/false);
        problem.setComputeInputTypeA(typeA);
        problem.setComputeInputTypeB(typeB);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        return problem;
    }

    template <typename T>
    void generateValues(std::vector<T>&                                   values,
                        roc::host_numerics::ScalarType                  type,
                        roc::host_numerics::GenerationRecipe::Component component,
                        std::uint64_t                                     seed,
                        std::uint64_t                                     stream)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        using namespace roc::host_numerics;

        const auto recipe = GenerationRecipe::realOnly(
            std::move(component),
            tensilelite_adapter::dataInitializationSettings(seed, stream));
        Tensor generated(type, Layout::contiguousLastDimensionFastest(Shape{values.size()}));
        generate(generated, recipe);

        const std::span<std::byte> destination = std::as_writable_bytes(std::span<T>(values));
        if(generated.rawEncodedBackingStorage().size() != destination.size())
            throw std::runtime_error("Generated Tensor storage does not match test value storage.");
        std::ranges::copy(generated.rawEncodedBackingStorage(), destination.begin());
    }

#ifdef TENSILE_USE_FP8_BF8
    template <typename Scale>
    void expectBlockedNonE8MXScale(rocisa::DataType scaleType)
    {
        const size_t M       = 2;
        const size_t N       = 2;
        const size_t K       = 16;
        const int    mxBlock = 8;
        auto         problem = makeMXProblem(
            rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock, scaleType);

        std::vector<Float8> a(M * K, Float8(1.0f));
        std::vector<Float8> b(K * N, Float8(1.0f));
        std::vector<float>  c(M * N, 0.0f);
        std::vector<float>  d(M * N, -99.0f);
        std::vector<Scale>  mxsa(problem.mxsa().totalAllocatedElements(), Scale(2.0f));
        std::vector<Scale>  mxsb(problem.mxsb().totalAllocatedElements(), Scale(4.0f));

        ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
        inputs.mxsa = mxsa.data();
        inputs.mxsb = mxsb.data();

        executeReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, requireBlockedExecution);
        EXPECT_EQ(d, (std::vector<float>{128, 128, 128, 128}));
    }
#endif
} // namespace

#ifndef _WIN32

TEST(ReferenceMXFastPath, SupportsMixedInputTypesWithMXFP4)
{
    const size_t M       = 1;
    const size_t N       = 1;
    const size_t K       = 8;
    const int    mxBlock = 8;

    auto problem
        = makeMXProblem(rocisa::DataType::Float4, rocisa::DataType::Float, M, N, K, mxBlock);
    std::vector<Float4x2> a(K / 2, Float4x2(1.0f, 1.0f));
    std::vector<float>    b(K, 1.0f);
    std::vector<float>    c(1, 0.0f);
    std::vector<float>    d(1, -99.0f);
    std::vector<E8>       mxsa(problem.mxsa().totalAllocatedElements(), E8(2.0f));
    std::vector<E8>       mxsb(problem.mxsb().totalAllocatedElements(), E8(4.0f));

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = mxsa.data();
    inputs.mxsb = mxsb.data();

    executeReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, requireBlockedExecution);
    EXPECT_EQ(d[0], 64);
}

#endif

#ifdef TENSILE_USE_FP8_BF8

TEST(ReferenceMXFastPath, SupportsNonE8ScaleStorage)
{
    expectBlockedNonE8MXScale<E5M3>(rocisa::DataType::E5M3);
    expectBlockedNonE8MXScale<Float8>(rocisa::DataType::Float8);
}

TEST(ReferenceMXFastPath, TreatsE8RawZeroAsZero)
{
    constexpr size_t K       = 32;
    constexpr int    mxBlock = 32;
    auto problem
        = makeMXProblem(rocisa::DataType::Float8, rocisa::DataType::Float8, 1, 1, K, mxBlock);

    std::vector<Float8> a(K, Float8(1.0f));
    std::vector<Float8> b(K, Float8(1.0f));
    std::vector<float>  c(1, 0.0f);
    std::vector<float>  d(1, -99.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements(), E8(uint8_t{0}));
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements(), E8(1.0f));
    ContractionInputs   inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = mxsa.data();
    inputs.mxsb = mxsb.data();

    executeReferenceGemm(problem, inputs, /*elementsToValidate=*/-1);
    EXPECT_EQ(d[0], 0.0f);
}

TEST(ReferenceMXFastPath, MatchesPointwiseForScaledFP8Gemm)
{
    const size_t M       = 64;
    const size_t N       = 64;
    const size_t K       = 128;
    const int    mxBlock = 32;

    auto problem
        = makeMXProblem(rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N, 0.0f);
    std::vector<float>  dPointwise(M * N, 0.0f);
    std::vector<float>  dBlocked(M * N, 0.0f);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

    const auto binaryValues
        = roc::host_numerics::GenerationRecipe::candidateSet({.values = {-1.0, 1.0}});
    const auto scaleValues
        = roc::host_numerics::GenerationRecipe::candidateSet({.values = {1.0, 2.0, 4.0}});
    generateValues(a, roc::host_numerics::ScalarType::Float8E4M3, binaryValues, 12345, 0);
    generateValues(b, roc::host_numerics::ScalarType::Float8E4M3, binaryValues, 12345, 1);
    generateValues(mxsa, roc::host_numerics::ScalarType::E8M0Zero, scaleValues, 12345, 2);
    generateValues(mxsb, roc::host_numerics::ScalarType::E8M0Zero, scaleValues, 12345, 3);

    EXPECT_TRUE(std::ranges::any_of(a, [](Float8 value) { return float(value) != 0.0f; }));
    EXPECT_TRUE(std::ranges::any_of(b, [](Float8 value) { return float(value) != 0.0f; }));
    EXPECT_TRUE(std::ranges::any_of(mxsa, [](E8 value) { return float(value) != 0.0f; }));
    EXPECT_TRUE(std::ranges::any_of(mxsb, [](E8 value) { return float(value) != 0.0f; }));

    ContractionInputs inputsPointwise(a.data(), b.data(), c.data(), dPointwise.data(), 1.0f, 0.0f);
    inputsPointwise.mxsa = mxsa.data();
    inputsPointwise.mxsb = mxsb.data();

    ContractionInputs inputsBlocked(a.data(), b.data(), c.data(), dBlocked.data(), 1.0f, 0.0f);
    inputsBlocked.mxsa = mxsa.data();
    inputsBlocked.mxsb = mxsb.data();

    executeReferenceGemm(problem, inputsPointwise, /*elementsToValidate=*/-1);
    executeReferenceGemm(problem, inputsBlocked, /*elementsToValidate=*/-1, requireBlockedExecution);

    const auto comparison = roc::host_numerics::compare(
        roc::host_numerics::Tensor::copyNativeStorage(std::span<const float>(dBlocked)),
        roc::host_numerics::Tensor::copyNativeStorage(std::span<const float>(dPointwise)),
        roc::host_numerics::nearComparisonOptions(1e-3));
    EXPECT_TRUE(comparison.passed())
        << "mismatches=" << comparison.mismatches
        << " max_absolute_difference=" << comparison.maxAbsoluteDifference;
}

TEST(ReferenceMXFastPath, MatchesPointwiseWithBetaAndBias)
{
    const size_t M       = 48;
    const size_t N       = 32;
    const size_t K       = 96;
    const int    mxBlock = 32;

    auto problem
        = makeMXProblem(rocisa::DataType::Float8, rocisa::DataType::Float8, M, N, K, mxBlock);
    problem.setUseBias(1);
    problem.setBias(rocisa::DataType::Float, M, M);

    std::vector<Float8> a(M * K);
    std::vector<Float8> b(K * N);
    std::vector<float>  c(M * N);
    std::vector<float>  dPointwise(M * N, 0.0f);
    std::vector<float>  dBlocked(M * N, 0.0f);
    std::vector<float>  bias(M);
    std::vector<E8>     mxsa(problem.mxsa().totalAllocatedElements());
    std::vector<E8>     mxsb(problem.mxsb().totalAllocatedElements());

    const auto binaryValues
        = roc::host_numerics::GenerationRecipe::candidateSet({.values = {-1.0, 1.0}});
    const auto scaleValues
        = roc::host_numerics::GenerationRecipe::candidateSet({.values = {1.0, 2.0, 4.0}});
    const auto cValues    = roc::host_numerics::GenerationRecipe::constant({.value = 0.25});
    const auto biasValues = roc::host_numerics::GenerationRecipe::constant({.value = 0.5});
    generateValues(a, roc::host_numerics::ScalarType::Float8E4M3, binaryValues, 54321, 0);
    generateValues(b, roc::host_numerics::ScalarType::Float8E4M3, binaryValues, 54321, 1);
    generateValues(mxsa, roc::host_numerics::ScalarType::E8M0Zero, scaleValues, 54321, 2);
    generateValues(mxsb, roc::host_numerics::ScalarType::E8M0Zero, scaleValues, 54321, 3);
    generateValues(c, roc::host_numerics::ScalarType::Float32, cValues, 54321, 4);
    generateValues(bias, roc::host_numerics::ScalarType::Float32, biasValues, 54321, 5);

    EXPECT_TRUE(std::ranges::any_of(a, [](Float8 value) { return float(value) != 0.0f; }));
    EXPECT_TRUE(std::ranges::any_of(b, [](Float8 value) { return float(value) != 0.0f; }));
    EXPECT_TRUE(std::ranges::any_of(mxsa, [](E8 value) { return float(value) != 0.0f; }));
    EXPECT_TRUE(std::ranges::any_of(mxsb, [](E8 value) { return float(value) != 0.0f; }));
    EXPECT_EQ(c.front(), 0.25f);
    EXPECT_EQ(bias.front(), 0.5f);

    ContractionInputs inputsPointwise(a.data(), b.data(), c.data(), dPointwise.data(), 1.0f, 0.5f);
    inputsPointwise.mxsa = mxsa.data();
    inputsPointwise.mxsb = mxsb.data();
    inputsPointwise.bias = bias.data();

    ContractionInputs inputsBlocked(a.data(), b.data(), c.data(), dBlocked.data(), 1.0f, 0.5f);
    inputsBlocked.mxsa = mxsa.data();
    inputsBlocked.mxsb = mxsb.data();
    inputsBlocked.bias = bias.data();

    executeReferenceGemm(problem, inputsPointwise, /*elementsToValidate=*/-1);
    executeReferenceGemm(problem, inputsBlocked, /*elementsToValidate=*/-1, requireBlockedExecution);

    const auto comparison = roc::host_numerics::compare(
        roc::host_numerics::Tensor::copyNativeStorage(std::span<const float>(dBlocked)),
        roc::host_numerics::Tensor::copyNativeStorage(std::span<const float>(dPointwise)),
        roc::host_numerics::nearComparisonOptions(1e-3));
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
