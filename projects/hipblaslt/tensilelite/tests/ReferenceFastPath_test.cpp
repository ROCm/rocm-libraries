// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <roc/host_validation/adapters/tensilelite/GemmInvocationAdapter.hpp>
#include <roc/host_validation/validation.hpp>

#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <variant>
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

TEST(ReferenceBlockedBackend, PreservesDoublePrecisionForF64)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Double, rocisa::DataType::Double, rocisa::DataType::Double, M, N, K);

    const double a0 = 1.0 + std::ldexp(1.0, -40);
    const double a1 = 1.0 + std::ldexp(1.0, -41);
    const double b0 = 1.0 + std::ldexp(1.0, -42);
    const double b1 = 1.0 + std::ldexp(1.0, -43);

    std::vector<double> a = {a0, a1};
    std::vector<double> b = {b0, b1};
    std::vector<double> c = {0.0};
    std::vector<double> d = {0.0};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0, 0.0);
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));

    const double expected = a0 * b0 + a1 * b1;
    ASSERT_NE(static_cast<double>(static_cast<float>(expected)), expected);
    EXPECT_EQ(d[0], expected);
}

TEST(ReferenceBlockedBackend, AppliesXFloat32OperandMathOpToBothOperands)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setF32XdlMathOp(rocisa::DataType::XFloat32);

    std::vector<float> a = {1.234567f, -2.345678f};
    std::vector<float> b = {3.456789f, 4.567891f};
    std::vector<float> c = {0.0f};
    std::vector<float> d = {0.0f};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));

    auto        xf32     = [](float v) { return static_cast<float>(XFloat32(v)); };
    const float expected = xf32(a[0]) * xf32(b[0]) + xf32(a[1]) * xf32(b[1]);
    const float fullF32  = a[0] * b[0] + a[1] * b[1];

    ASSERT_NE(expected, fullF32);
    EXPECT_EQ(d[0], expected);
}

TEST(ReferenceBlockedBackend, DelegatesDenseRuntimeGemm)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{5, 6, 7, 8};
    std::vector<float> c(M * N, 1);
    std::vector<float> d(M * N, -99);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 2.0f, 3.0f);

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(d, (std::vector<float>{49, 71, 65, 95}));
}

TEST(ReferenceBlockedBackend, SupportsAffineLayoutsAndMixedRealStorage)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 2;

    ContractionProblemGemm::FreeIndices freeIndices{
        {true, 0, 0, 0},
        {false, 1, 1, 1},
    };
    ContractionProblemGemm::BatchIndices batchIndices{{2, 2, 2, 2}};
    ContractionProblemGemm::BoundIndices boundIndices{{1, 0, false, false}};
    TensorOps                            noOperations;
    auto problem = ContractionProblemGemm::FromIndexSizes(freeIndices,
                                                          batchIndices,
                                                          boundIndices,
                                                          {M, N, 1, K},
                                                          rocisa::DataType::Half,
                                                          {2, 7, 20},
                                                          noOperations,
                                                          rocisa::DataType::BFloat16,
                                                          {3, 8, 20},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {4, 9, 20},
                                                          noOperations,
                                                          rocisa::DataType::Double,
                                                          {3, 10, 20},
                                                          noOperations,
                                                          1.0);
    problem.setComputeInputTypeA(rocisa::DataType::Half);
    problem.setComputeInputTypeB(rocisa::DataType::BFloat16);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);

    std::vector<Half> a(problem.a().totalAllocatedElements(), Half(-99));
    a[0] = Half(1);
    a[2] = Half(2);
    a[7] = Half(3);
    a[9] = Half(4);
    std::vector<BFloat16> b(problem.b().totalAllocatedElements(), BFloat16(-99));
    b[0]  = BFloat16(5);
    b[3]  = BFloat16(6);
    b[8]  = BFloat16(7);
    b[11] = BFloat16(8);
    std::vector<float> c(problem.c().totalAllocatedElements(), -99);
    c[0]  = 1;
    c[4]  = 1;
    c[9]  = 1;
    c[13] = 1;
    std::vector<double> d(problem.d().totalAllocatedElements(), -99);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 2.0f, 3.0f);
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(d[0], 49);
    EXPECT_EQ(d[3], 71);
    EXPECT_EQ(d[10], 65);
    EXPECT_EQ(d[13], 95);
}

TEST(ReferenceBlockedBackend, PromotesHalfDestinationAccumulationToFloat)
{
    const size_t K       = 64;
    auto         problem = makePackedProblem(
        rocisa::DataType::Half, rocisa::DataType::Half, rocisa::DataType::Half, 1, 1, K);
    std::vector<Half> a(K, Half(0.1f));
    std::vector<Half> b(K, Half(0.1f));
    std::vector<Half> c(1, Half(0));
    std::vector<Half> d(1, Half(-99));
    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), Half(1), Half(0));

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    float expected = 0;
    for(size_t reduction = 0; reduction < K; ++reduction)
        expected += static_cast<float>(a[reduction]) * static_cast<float>(b[reduction]);
    EXPECT_EQ(d[0], Half(expected));
}

TEST(ReferenceGemmSelection, KeepsSparseHalfRequestsPointwise)
{
    const size_t M       = 1;
    const size_t N       = 10;
    const size_t K       = 64;
    auto         problem = makePackedProblem(
        rocisa::DataType::Half, rocisa::DataType::Half, rocisa::DataType::Half, M, N, K);
    std::vector<Half> a(M * K, Half(0.1f));
    std::vector<Half> b(K * N, Half(0.1f));
    std::vector<Half> c(M * N, Half(0));
    std::vector<Half> d(M * N, Half(-99));
    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), Half(1), Half(0));

    Half  pointwiseExpected = Half(0);
    float blockedExpected   = 0;
    for(size_t reduction = 0; reduction < K; ++reduction)
    {
        pointwiseExpected = Half(pointwiseExpected + Half(a[reduction] * b[reduction]));
        blockedExpected += static_cast<float>(a[reduction]) * static_cast<float>(b[reduction]);
    }
    ASSERT_NE(pointwiseExpected, Half(blockedExpected));

    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/1);
    EXPECT_EQ(d[0], pointwiseExpected);
    for(size_t index = 1; index < d.size(); ++index)
        EXPECT_EQ(d[index], Half(-99)) << "index=" << index;
}

TEST(ReferenceGemmSelection, ZeroRequestedElementsUsesSelectAllPolicy)
{
    const size_t M       = 1;
    const size_t N       = 2;
    const size_t K       = 64;
    auto         problem = makePackedProblem(
        rocisa::DataType::Half, rocisa::DataType::Half, rocisa::DataType::Half, M, N, K);
    std::vector<Half> a(M * K, Half(0.1f));
    std::vector<Half> b(K * N, Half(0.1f));
    std::vector<Half> c(M * N, Half(0));
    std::vector<Half> d(M * N, Half(-99));
    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), Half(1), Half(0));

    Half  pointwiseExpected = Half(0);
    float blockedExpected   = 0;
    for(size_t reduction = 0; reduction < K; ++reduction)
    {
        pointwiseExpected = Half(pointwiseExpected + Half(a[reduction] * b[reduction]));
        blockedExpected += static_cast<float>(a[reduction]) * static_cast<float>(b[reduction]);
    }
    ASSERT_NE(pointwiseExpected, Half(blockedExpected));

    const auto runInfo = SolveGemmCPU(problem, inputs, /*elementsToValidate=*/0);
    EXPECT_EQ(runInfo.backendUsed, roc::host_validation::GemmBackend::Blocked);
    EXPECT_EQ(d, std::vector<Half>(M * N, Half(blockedExpected)));
}

TEST(ReferenceGemmSelection, UsesPointwiseForSparseFloatValidation)
{
    const size_t M                  = 384;
    const size_t N                  = 384;
    const size_t K                  = 32;
    const size_t elementsToValidate = 128;
    auto         problem            = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    std::vector<float> a(M * K, 1.0f);
    std::vector<float> b(K * N, 1.0f);
    std::vector<float> c(M * N, 0.0f);
    std::vector<float> d(M * N, -99.0f);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    const auto runInfo = SolveGemmCPU(problem, inputs, elementsToValidate);
    EXPECT_EQ(runInfo.backendUsed, roc::host_validation::GemmBackend::Pointwise);
    EXPECT_EQ(runInfo.outputElementsWritten, elementsToValidate);
    EXPECT_EQ(runInfo.outputElementsCovered, elementsToValidate);
    EXPECT_LT(runInfo.outputElementsCovered, M * N);

    const auto selection
        = roc::host_validation::OutputSelection::primeStride(problem.d().totalLogicalElements(),
                                                             problem.d().totalAllocatedElements(),
                                                             elementsToValidate);
    const std::vector<size_t> selected = selection.indices(problem.d().totalLogicalElements());
    ASSERT_EQ(selected.size(), elementsToValidate);

    std::vector<bool> expectedWritten(M * N, false);
    for(const size_t index : selected)
    {
        expectedWritten[index] = true;
        EXPECT_EQ(d[index], static_cast<float>(K)) << "index=" << index;
    }
    size_t written = 0;
    for(size_t index = 0; index < d.size(); ++index)
    {
        if(d[index] != -99.0f)
        {
            EXPECT_TRUE(expectedWritten[index]) << "index=" << index;
            ++written;
        }
    }
    EXPECT_EQ(written, selected.size());

    std::vector<float> observedSelected;
    observedSelected.reserve(selected.size());
    std::vector<float> expectedSelected(selected.size(), static_cast<float>(K));
    for(const size_t index : selected)
        observedSelected.push_back(d[index]);
    const auto selectedLayout
        = roc::host_validation::Layout::contiguous(roc::host_validation::Shape{selected.size()});
    const auto matching = roc::host_validation::compare(
        roc::host_validation::Tensor::fromNative<float>(selectedLayout,
                                                        std::span<const float>(observedSelected)),
        roc::host_validation::Tensor::fromNative<float>(selectedLayout,
                                                        std::span<const float>(expectedSelected)));
    ASSERT_TRUE(matching.passed());

    observedSelected[selected.size() / 2] += 1.0f;
    const auto injectedFailure = roc::host_validation::compare(
        roc::host_validation::Tensor::fromNative<float>(selectedLayout,
                                                        std::span<const float>(observedSelected)),
        roc::host_validation::Tensor::fromNative<float>(selectedLayout,
                                                        std::span<const float>(expectedSelected)));
    EXPECT_FALSE(injectedFailure.passed());
    EXPECT_EQ(injectedFailure.mismatches, 1);
}

TEST(ReferenceOutputSelection, ComputesPrimeStrideSubset)
{
    const size_t M = 4;
    const size_t N = 3;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);

    std::vector<float> a(M * K, 1.0f);
    std::vector<float> b(K * N, 1.0f);
    std::vector<float> c(M * N, 0.0f);
    std::vector<float> d(M * N, -99.0f);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/3));

    for(size_t index = 0; index < d.size(); ++index)
    {
        const bool selected = index == 0 || index == 5 || index == 10;
        EXPECT_EQ(d[index], selected ? 2.0f : -99.0f) << "index=" << index;
    }
}

TEST(ReferenceStandaloneEpilogue, HandlesEAndAmaxScaleAndGate)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 1;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseBias(1);
    problem.setBias(rocisa::DataType::Float, M, M);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides(), true);
    problem.setOutputAmaxD(true);
    problem.setAmaxD(rocisa::DataType::Float, true);
    problem.setUseScaleCD(true);
    problem.setScaleC(rocisa::DataType::Float);
    problem.setScaleD(rocisa::DataType::Float);
    problem.setUseGateResidual(true);
    problem.setGateResidual(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides());
    problem.setActivationType(ActivationType::Relu);

    std::vector<float> a{1, 2};
    std::vector<float> b{3, 4};
    std::vector<float> c(M * N, 1);
    std::vector<float> d(M * N, -99);
    std::vector<float> e(M * N, -99);
    std::vector<float> bias{1, -10};
    std::vector<float> gate{0.5f, 2.0f, -1.0f, 0.25f};
    float              scaleC = 2;
    float              scaleD = 3;
    float              amaxD  = 0;

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.5f);
    inputs.e            = e.data();
    inputs.bias         = bias.data();
    inputs.scaleC       = &scaleC;
    inputs.scaleD       = &scaleD;
    inputs.gateResidual = gate.data();
    inputs.amaxD        = &amaxD;

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(e, (std::vector<float>{5, -3, 6, -1}));
    EXPECT_EQ(d, (std::vector<float>{8, 2, -19, 0.25f}));
    EXPECT_EQ(amaxD, 6);
}

TEST(ReferenceStandaloneEpilogue, CompletesDForPartialAmax)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 1;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setOutputAmaxD(true);
    problem.setAmaxD(rocisa::DataType::Float, true);

    std::vector<float> a{1, 2};
    std::vector<float> b{1, 10};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    float              amaxD = 0;

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.amaxD = &amaxD;

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/2));
    EXPECT_EQ(d, (std::vector<float>{1, 2, 10, 20}));
    EXPECT_EQ(amaxD, 20);
}

TEST(ReferenceStandaloneEpilogue, AccumulatesAmaxAcrossStridedBatches)
{
    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                        false,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        1,
                                                        2,
                                                        1,
                                                        2,
                                                        1,
                                                        1,
                                                        1,
                                                        2,
                                                        1,
                                                        2,
                                                        1,
                                                        2,
                                                        0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setOutputAmaxD(true);
    problem.setAmaxD(rocisa::DataType::Float, true);

    std::vector<float> a{10, 1};
    std::vector<float> b{1, 2, 2, 3};
    std::vector<float> c(4, 0);
    std::vector<float> d(4, -99);
    float              amaxD = 999;

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.amaxD = &amaxD;

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/1));
    EXPECT_EQ(d, (std::vector<float>{10, 20, 2, 3}));
    EXPECT_EQ(amaxD, 20);
}

TEST(ReferenceStandaloneEpilogue, HandlesGradientAuxiliaryInput)
{
    const size_t M = 2;
    const size_t N = 1;
    const size_t K = 1;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides());
    problem.setUseGradient(true);
    problem.setActivationType(ActivationType::Relu);

    std::vector<float> a{10, 20};
    std::vector<float> b{1};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    std::vector<float> e{-1, 2};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.e = e.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d, (std::vector<float>{0, 20}));
    EXPECT_EQ(e, (std::vector<float>{-1, 2}));
}

TEST(ReferenceStandaloneEpilogue, UsesZeroAuxiliaryWhenGradientEIsDisabled)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, 2, 1, 1);
    problem.setUseGradient(true);
    problem.setActivationType(ActivationType::Relu);

    std::vector<float> a{10, 20};
    std::vector<float> b{1};
    std::vector<float> c(2, 0);
    std::vector<float> d(2, -99);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d, (std::vector<float>{0, 0}));
}

TEST(ReferenceStandaloneEpilogue, HandlesGradientBiasReduction)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 1;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides());
    problem.setUseGradient(true);
    problem.setActivationType(ActivationType::Relu);
    problem.setUseBias(1);
    problem.setBias(rocisa::DataType::Float, M, M, true, ContractionProblemGemm::D, 0);

    std::vector<float> a{1, 2};
    std::vector<float> b{3, 4};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    std::vector<float> e(M * N, 1);
    std::vector<float> bias(M, 0);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.e    = e.data();
    inputs.bias = bias.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/2));
    EXPECT_EQ(d, (std::vector<float>{3, 6, 4, 8}));
    EXPECT_EQ(bias, (std::vector<float>{7, 14}));
}

TEST(ReferenceStandaloneEpilogue, WritesPointerArrayGradientBiasOutputs)
{
    const size_t M       = 2;
    const size_t N       = 1;
    const size_t K       = 1;
    const size_t batches = 2;
    auto         problem = ContractionProblemGemm::GEMM_Strides(false,
                                                                false,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                M,
                                                                N,
                                                                K,
                                                                batches,
                                                                M,
                                                                M * K,
                                                                K,
                                                                K * N,
                                                                M,
                                                                M * N,
                                                                M,
                                                                M * N,
                                                                0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setUseGradient(true);
    problem.setUseBias(1);
    problem.setBias(
        rocisa::DataType::Float, M, M, true, ContractionProblemGemm::A, /*factorDim=*/0);

    std::vector<float> a0{1, 2};
    std::vector<float> a1{3, 4};
    std::vector<float> b0{1};
    std::vector<float> b1{1};
    std::vector<float> c0(M * N, 0);
    std::vector<float> c1(M * N, 0);
    std::vector<float> d0(M * N, -99);
    std::vector<float> d1(M * N, -99);
    std::vector<float> bias0(M, 0);
    std::vector<float> bias1(M, 0);
    const void*        batchA[]    = {a0.data(), a1.data()};
    const void*        batchB[]    = {b0.data(), b1.data()};
    const void*        batchC[]    = {c0.data(), c1.data()};
    void*              batchD[]    = {d0.data(), d1.data()};
    const void*        batchBias[] = {bias0.data(), bias1.data()};

    ContractionInputs inputs(nullptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    inputs.batchA    = batchA;
    inputs.batchB    = batchB;
    inputs.batchC    = batchC;
    inputs.batchD    = batchD;
    inputs.batchBias = batchBias;

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d0, a0);
    EXPECT_EQ(d1, a1);
    EXPECT_EQ(bias0, a0);
    EXPECT_EQ(bias1, a1);
}

TEST(ReferenceStandaloneEpilogue, SamplesGradientBiasSourceA)
{
    const size_t M = 2;
    const size_t N = 3;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseGradient(true);
    problem.setUseBias(1);
    problem.setBias(
        rocisa::DataType::Float, M, M, true, ContractionProblemGemm::A, /*factorDim=*/0);

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b(K * N, 1);
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    std::vector<float> bias(M, 0);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 0.0f, 0.0f);
    inputs.bias = bias.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/2));
    EXPECT_EQ(d, (std::vector<float>{0, -99, -99, 0, -99, -99}));
    EXPECT_EQ(bias, (std::vector<float>{4, 6}));
}

TEST(ReferenceStandaloneEpilogue, SamplesGradientBiasSourceB)
{
    const size_t M = 2;
    const size_t N = 3;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseGradient(true);
    problem.setUseBias(1);
    problem.setBias(
        rocisa::DataType::Float, N, N, true, ContractionProblemGemm::B, /*factorDim=*/1);

    std::vector<float> a(M * K, 1);
    std::vector<float> b{1, 2, 3, 4, 5, 6};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    std::vector<float> bias(N, 0);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 0.0f, 0.0f);
    inputs.bias = bias.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/2));
    EXPECT_EQ(d, (std::vector<float>{0, -99, -99, 0, -99, -99}));
    EXPECT_EQ(bias, (std::vector<float>{3, 7, 11}));
}

TEST(ReferenceStandaloneEpilogue, PreservesSharedFactorAxisWhenMEqualsN)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 1;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseBias(1);
    problem.setBias(rocisa::DataType::Float, N, N, false, ContractionProblemGemm::D, 1);
    problem.setUseScaleAlphaVec(1);
    problem.setScaleAlphaVec(rocisa::DataType::Float, N, 1);

    std::vector<float> a{1, 2};
    std::vector<float> b{3, 4};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    std::vector<float> bias{1, 2};
    std::vector<float> scaleAlpha{10, 100};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.bias          = bias.data();
    inputs.scaleAlphaVec = scaleAlpha.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d, (std::vector<float>{31, 61, 402, 802}));
}

TEST(ReferenceStandaloneEpilogue, PreservesPartialOutputSelection)
{
    const size_t M = 2;
    const size_t N = 2;
    const size_t K = 1;

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, M, N, K);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides(), true);

    std::vector<float> a{1, 2};
    std::vector<float> b{3, 4};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    std::vector<float> e(M * N, -99);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.e = e.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/2));
    EXPECT_EQ(d, (std::vector<float>{3, -99, 4, -99}));
    EXPECT_EQ(e, (std::vector<float>{3, -99, 4, -99}));
}

TEST(ReferenceGemmSelection, FallsBackToPointwiseForInt32Accumulation)
{
    const size_t M = 2;
    const size_t N = 1;
    const size_t K = 2;

    auto problem = makePackedProblem(
        rocisa::DataType::Int8, rocisa::DataType::Int8, rocisa::DataType::Int32, M, N, K);

    std::vector<int8_t>  a{1, 2, 3, 4};
    std::vector<int8_t>  b{5, 6};
    std::vector<int32_t> c(M * N, 0);
    std::vector<int32_t> d(M * N, -99);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(0));
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1);
    EXPECT_EQ(d, (std::vector<int32_t>{23, 34}));
}

TEST(ReferenceRuntimePointwise, SaturatesInt8Destination)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Int8, rocisa::DataType::Int8, rocisa::DataType::Int8, 1, 4, 1);
    problem.setAlphaType(rocisa::DataType::Int32);
    problem.setBetaType(rocisa::DataType::Int32);

    std::vector<int8_t> a{100};
    std::vector<int8_t> b{2, -2, 1, -1};
    std::vector<int8_t> c(4, 0);
    std::vector<int8_t> d(4, 0);

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(0));
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d, (std::vector<int8_t>{127, -128, 100, -100}));
}

TEST(ReferenceStandaloneEpilogue, SaturatesInt8Destination)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Int8, rocisa::DataType::Int8, rocisa::DataType::Int8, 1, 1, 1);
    problem.setAlphaType(rocisa::DataType::Int32);
    problem.setBetaType(rocisa::DataType::Int32);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Int8, problem.d().sizes(), problem.d().strides(), true);

    std::vector<int8_t> a{100};
    std::vector<int8_t> b{2};
    std::vector<int8_t> c{0};
    std::vector<int8_t> d{0};
    std::vector<int8_t> e{0};

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(0));
    inputs.e = e.data();
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d[0], 127);
}

TEST(ReferenceBlockedBackend, SupportsMirroredBoundIndex)
{
    const size_t M     = 2;
    const size_t N     = 1;
    const size_t K     = 2;
    const size_t batch = 1;

    ContractionProblemGemm::FreeIndices freeIndices{
        {true, 0, 0, 0},
        {false, 1, 1, 1},
    };
    ContractionProblemGemm::BatchIndices batchIndices{{2, 2, 2, 2}};
    ContractionProblemGemm::BoundIndices boundIndices{{1, 0, true, false}};
    TensorOps                            noOperations;
    auto problem = ContractionProblemGemm::FromIndexSizes(freeIndices,
                                                          batchIndices,
                                                          boundIndices,
                                                          {M, N, batch, K},
                                                          rocisa::DataType::Float,
                                                          {1, M, M * K},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, K, K * N},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, M, M * N},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, M, M * N},
                                                          noOperations,
                                                          0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{5, 6};
    std::vector<float> c(M * N, 0);
    std::vector<float> d(M * N, -99);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(d, (std::vector<float>{21, 32}));
}

#ifdef TENSILE_USE_FP8_BF8
TEST(ReferenceRuntimePointwise, MirrorsBlockScalesWithTheBoundIndex)
{
    const size_t                        K = 32;
    ContractionProblemGemm::FreeIndices freeIndices{
        {true, 0, 0, 0},
        {false, 1, 1, 1},
    };
    ContractionProblemGemm::BatchIndices batchIndices{{2, 2, 2, 2}};
    ContractionProblemGemm::BoundIndices boundIndices{{1, 0, true, false}};
    TensorOps                            noOperations;
    auto problem = ContractionProblemGemm::FromIndexSizes(freeIndices,
                                                          batchIndices,
                                                          boundIndices,
                                                          {1, 1, 1, K},
                                                          rocisa::DataType::Float8,
                                                          {1, 1, K},
                                                          noOperations,
                                                          rocisa::DataType::Float8,
                                                          {1, K, K},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, 1, 1},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, 1, 1},
                                                          noOperations,
                                                          0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float8);
    problem.setComputeInputTypeB(rocisa::DataType::Float8);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setMXScaleA(rocisa::DataType::E8, 8);
    problem.setMXScaleB(rocisa::DataType::E8, 8);

    std::vector<Float8> a(K);
    std::vector<Float8> b(K, Float8(1.0f));
    for(size_t reduction = 0; reduction < K; ++reduction)
        a[reduction] = Float8(static_cast<float>(reduction / 8 + 1));
    std::vector<float>         c{0};
    std::vector<float>         d{-99};
    std::vector<E8>            scaleA(problem.mxsa().totalAllocatedElements(), E8(1.0f));
    std::vector<E8>            scaleB(problem.mxsb().totalAllocatedElements(), E8(1.0f));
    std::vector<int64_t>       scaleCoordinate(problem.mxsa().dimensions(), 0);
    const std::array<float, 4> scales{1, 2, 4, 8};
    for(size_t block = 0; block < scales.size(); ++block)
    {
        scaleCoordinate[1]                            = static_cast<int64_t>(block);
        scaleA[problem.mxsa().index(scaleCoordinate)] = E8(scales[block]);
    }

    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.mxsa = scaleA.data();
    inputs.mxsb = scaleB.data();
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d[0], 392);
}
#endif

TEST(ReferenceRuntimePointwise, HandlesPointerArrayBatches)
{
    const size_t M       = 2;
    const size_t N       = 1;
    const size_t K       = 1;
    const size_t batches = 2;

    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                        false,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        M,
                                                        N,
                                                        K,
                                                        batches,
                                                        M,
                                                        M * K,
                                                        K,
                                                        K * N,
                                                        M,
                                                        M * N,
                                                        M,
                                                        M * N,
                                                        0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setUseBias(1);
    problem.setBias(rocisa::DataType::Float, M, M);
    problem.setUseGateResidual(true);
    problem.setGateResidual(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides());

    std::vector<float> a0{-99, 1, 2};
    std::vector<float> a1{-99, 4, 5};
    std::vector<float> b0{-99, 3};
    std::vector<float> b1{-99, 2};
    std::vector<float> c0{-99, 0, 0};
    std::vector<float> c1{-99, 0, 0};
    std::vector<float> d0{-99, -99, -99};
    std::vector<float> d1{-99, -99, -99};
    std::vector<float> bias0{1, 2};
    std::vector<float> bias1{3, 4};
    std::vector<float> gate0{1, 1};
    std::vector<float> gate1{2, 2};
    const void*        batchA[]    = {a0.data(), a1.data()};
    const void*        batchB[]    = {b0.data(), b1.data()};
    const void*        batchC[]    = {c0.data(), c1.data()};
    void*              batchD[]    = {d0.data(), d1.data()};
    const void*        batchBias[] = {bias0.data(), bias1.data()};
    const void*        batchGate[] = {gate0.data(), gate1.data()};

    ContractionInputs inputs(nullptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    inputs.batchA            = batchA;
    inputs.batchB            = batchB;
    inputs.batchC            = batchC;
    inputs.batchD            = batchD;
    inputs.batchBias         = batchBias;
    inputs.batchGateResidual = batchGate;
    inputs.batchOffsetA      = sizeof(float);
    inputs.batchOffsetB      = sizeof(float);
    inputs.batchOffsetC      = sizeof(float);
    inputs.batchOffsetD      = sizeof(float);

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d0, (std::vector<float>{-99, 5, 9}));
    EXPECT_EQ(d1, (std::vector<float>{-99, 24, 30}));
}

TEST(ReferenceRuntimePointwise, PreservesEarlierStridedBatchOutputs)
{
    const size_t batches = 2;
    auto         problem = ContractionProblemGemm::GEMM_Strides(false,
                                                                false,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                rocisa::DataType::Float,
                                                                1,
                                                                1,
                                                                1,
                                                                batches,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                1,
                                                                0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides(), true);

    std::vector<float> a{2, 3};
    std::vector<float> b{4, 5};
    std::vector<float> c{0, 0};
    std::vector<float> d{-99, -99};
    std::vector<float> e{-99, -99};
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.e = e.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d, (std::vector<float>{8, 15}));
    EXPECT_EQ(e, d);
}

TEST(ReferenceBlockedBackend, RejectsInvalidPointerBatchBeforeWriting)
{
    const size_t M       = 1;
    const size_t N       = 1;
    const size_t K       = 1;
    const size_t batches = 2;

    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                        false,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        M,
                                                        N,
                                                        K,
                                                        batches,
                                                        M,
                                                        M * K,
                                                        K,
                                                        K * N,
                                                        M,
                                                        M * N,
                                                        M,
                                                        M * N,
                                                        0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);

    std::vector<float> a0{2};
    std::vector<float> a1{3};
    std::vector<float> b0{4};
    std::vector<float> b1{5};
    std::vector<float> c0{0};
    std::vector<float> c1{0};
    std::vector<float> d0{-99};
    const void*        batchA[] = {a0.data(), a1.data()};
    const void*        batchB[] = {b0.data(), b1.data()};
    const void*        batchC[] = {c0.data(), c1.data()};
    void*              batchD[] = {d0.data(), nullptr};

    ContractionInputs inputs(nullptr, nullptr, nullptr, nullptr, 1.0f, 0.0f);
    inputs.batchA = batchA;
    inputs.batchB = batchB;
    inputs.batchC = batchC;
    inputs.batchD = batchD;

    const auto translation
        = reference_adapter::translateGemmInvocation(problem, inputs, /*elementsToValidate=*/-1);
    ASSERT_TRUE(std::holds_alternative<reference_adapter::TranslationFailure>(translation));
    EXPECT_EQ(std::get<reference_adapter::TranslationFailure>(translation).code,
              reference_adapter::TranslationFailureCode::InvalidBatchPointer);

    EXPECT_FALSE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(d0[0], -99);
}

TEST(ReferenceInvocationAdapter, OwnsStandaloneTemporariesAcrossAdapterLifetime)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, 2, 2, 1);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides(), true);

    std::vector<float> a{1, 2};
    std::vector<float> b{3, 4};
    std::vector<float> c(4, 0);
    std::vector<float> d(4, -99);
    std::vector<float> e(4, -99);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.e = e.data();

    std::optional<reference_adapter::TranslatedGemmBatch> translated;
    {
        auto invocationTranslation
            = reference_adapter::translateGemmInvocation(problem, inputs, /*elementsToValidate=*/-1);
        ASSERT_TRUE(
            std::holds_alternative<reference_adapter::GemmInvocationAdapter>(invocationTranslation));
        auto adapter
            = std::move(std::get<reference_adapter::GemmInvocationAdapter>(invocationTranslation));

        auto batchTranslation = adapter.translateBatch(0, adapter.operationAccumulatorType());
        ASSERT_TRUE(
            std::holds_alternative<reference_adapter::TranslatedGemmBatch>(batchTranslation));
        translated.emplace(
            std::move(std::get<reference_adapter::TranslatedGemmBatch>(batchTranslation)));
    }

    ASSERT_TRUE(translated->intermediate);
    ASSERT_TRUE(translated->epilogue);
    roc::host_validation::referenceGemm(translated->gemm(),
                                        {
                                            .backend = roc::host_validation::GemmBackend::Pointwise,
                                            .requireRequestedBackend = true,
                                        });
    roc::host_validation::referenceEpilogue(*translated->epilogue);
    translated->copyOutputs();

    EXPECT_EQ(d, (std::vector<float>{3, 6, 4, 8}));
    EXPECT_EQ(e, d);
}

TEST(ReferenceInvocationAdapter, CopyOutputsPreservesUnselectedValuesAndPadding)
{
    auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                        false,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        rocisa::DataType::Float,
                                                        1,
                                                        2,
                                                        1,
                                                        1,
                                                        1,
                                                        1,
                                                        1,
                                                        2,
                                                        1,
                                                        2,
                                                        3,
                                                        4,
                                                        0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);

    std::vector<float> a{2};
    std::vector<float> b{3, 4};
    std::vector<float> c{0, 0};
    std::vector<float> d(4, -99);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    auto invocationTranslation
        = reference_adapter::translateGemmInvocation(problem, inputs, /*elementsToValidate=*/1);
    ASSERT_TRUE(
        std::holds_alternative<reference_adapter::GemmInvocationAdapter>(invocationTranslation));
    auto adapter
        = std::move(std::get<reference_adapter::GemmInvocationAdapter>(invocationTranslation));

    auto batchTranslation = adapter.translateBatch(0, adapter.operationAccumulatorType());
    ASSERT_TRUE(std::holds_alternative<reference_adapter::TranslatedGemmBatch>(batchTranslation));
    auto translated = std::move(std::get<reference_adapter::TranslatedGemmBatch>(batchTranslation));

    roc::host_validation::referenceGemm(translated.gemm());
    d[1] = 111;
    d[2] = 222;
    d[3] = 333;
    translated.copyOutputs();

    EXPECT_EQ(d, (std::vector<float>{6, 111, 222, 333}));
}

TEST(ReferenceInvocationAdapter, RejectsDescriptorStrideThatCannotFitPtrdiff)
{
    if constexpr(std::numeric_limits<size_t>::digits <= std::numeric_limits<ptrdiff_t>::digits)
        GTEST_SKIP() << "size_t has no values outside ptrdiff_t range";

    auto problem = makePackedProblem(
        rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, 1, 1, 1);
    const size_t oversizedStride = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) + 1;
    problem.resetTensor(
        ContractionProblemGemm::A, rocisa::DataType::Float, {1, 1, 1}, {1, oversizedStride, 1});

    std::vector<float> a{1};
    std::vector<float> b{1};
    std::vector<float> c{0};
    std::vector<float> d{-99};
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    const auto translation
        = reference_adapter::translateGemmInvocation(problem, inputs, /*elementsToValidate=*/0);
    ASSERT_TRUE(std::holds_alternative<reference_adapter::TranslationFailure>(translation));
    EXPECT_EQ(std::get<reference_adapter::TranslationFailure>(translation).code,
              reference_adapter::TranslationFailureCode::InvalidDescriptor);
}

TEST(ReferenceInvocationAdapter, RejectsMirroredOffsetMultiplicationOverflow)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 3;

    ContractionProblemGemm::FreeIndices freeIndices{
        {true, 0, 0, 0},
        {false, 1, 1, 1},
    };
    ContractionProblemGemm::BatchIndices batchIndices{{2, 2, 2, 2}};
    ContractionProblemGemm::BoundIndices boundIndices{{1, 0, true, false}};
    TensorOps                            noOperations;
    const size_t                         oversizedMirroredStride
        = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / 2 + 1;
    auto problem = ContractionProblemGemm::FromIndexSizes(freeIndices,
                                                          batchIndices,
                                                          boundIndices,
                                                          {M, N, 1, K},
                                                          rocisa::DataType::Float,
                                                          {1, oversizedMirroredStride, 1},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, K, K * N},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, M, M * N},
                                                          noOperations,
                                                          rocisa::DataType::Float,
                                                          {1, M, M * N},
                                                          noOperations,
                                                          0.0);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);

    std::vector<float> a(K, 1);
    std::vector<float> b(K, 1);
    std::vector<float> c{0};
    std::vector<float> d{-99};
    const void*        batchA[] = {a.data()};
    ContractionInputs  inputs(nullptr, b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.batchA = batchA;

    const auto translation
        = reference_adapter::translateGemmInvocation(problem, inputs, /*elementsToValidate=*/0);
    ASSERT_TRUE(std::holds_alternative<reference_adapter::TranslationFailure>(translation));
    EXPECT_EQ(std::get<reference_adapter::TranslationFailure>(translation).code,
              reference_adapter::TranslationFailureCode::InvalidDescriptor);
}

TEST(ReferenceRuntimePointwise, HandlesFloat16Accumulation)
{
    const size_t M = 1;
    const size_t N = 1;
    const size_t K = 64;

    auto problem = makePackedProblem(
        rocisa::DataType::Half, rocisa::DataType::Half, rocisa::DataType::Half, M, N, K);

    std::vector<Half> a(K, Half(0.1f));
    std::vector<Half> b(K, Half(0.1f));
    std::vector<Half> c(1, Half(0));
    std::vector<Half> d(1, Half(-99));
    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), Half(1), Half(0));

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    Half expected = Half(0);
    for(size_t reduction = 0; reduction < K; ++reduction)
        expected = Half(expected + Half(a[reduction] * b[reduction]));
    EXPECT_EQ(d[0], expected);
}

TEST(ReferenceRuntimePointwise, AppliesScalarScaleBeforeComputeQuantization)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Half, rocisa::DataType::Float, rocisa::DataType::Float, 1, 1, 1);
    problem.setComputeInputTypeA(rocisa::DataType::Float8);
    problem.setUseScaleAB("Scalar");
    problem.setScaleA(rocisa::DataType::Float, 1);
    problem.setScaleB(rocisa::DataType::Float, 1);

    std::vector<Half>  a{Half(1.1f)};
    std::vector<float> b{1};
    std::vector<float> c{0};
    std::vector<float> d{-99};
    float              scaleA = 3;
    float              scaleB = 1;
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.scaleA = &scaleA;
    inputs.scaleB = &scaleB;

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d[0], 3.25f);
    d[0] = -99;
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(d[0], 3.25f);
}

TEST(ReferenceRuntimePointwise, AppliesVectorScaleBeforeComputeQuantization)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Half, rocisa::DataType::Float, rocisa::DataType::Float, 2, 1, 1);
    problem.setComputeInputTypeA(rocisa::DataType::Float8);
    problem.setUseScaleAB("Vector");
    problem.setScaleA(rocisa::DataType::Float, 2);
    problem.setScaleB(rocisa::DataType::Float, 1);

    std::vector<Half>  a{Half(1.1f), Half(1.1f)};
    std::vector<float> b{1};
    std::vector<float> c(2, 0);
    std::vector<float> d(2, -99);
    std::vector<float> scaleA{3, 4};
    std::vector<float> scaleB{1};
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.scaleA = scaleA.data();
    inputs.scaleB = scaleB.data();

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d, (std::vector<float>{3.25f, 4.5f}));
    d.assign(2, -99);
    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1, ReferenceGemmExecution::BlockedRequired));
    EXPECT_EQ(d, (std::vector<float>{3.25f, 4.5f}));
}

TEST(ReferenceRuntimePointwise, SupportsEveryConfiguredActivation)
{
    const std::array<ActivationType, 13> activations{
        ActivationType::Abs,
        ActivationType::Clippedrelu,
        ActivationType::Gelu,
        ActivationType::Geluscaling,
        ActivationType::Leakyrelu,
        ActivationType::Relu,
        ActivationType::Sigmoid,
        ActivationType::Tanh,
        ActivationType::DGelu,
        ActivationType::DRelu,
        ActivationType::Silu,
        ActivationType::Swish,
        ActivationType::Clamp,
    };

    for(const ActivationType activation : activations)
    {
        auto problem = makePackedProblem(
            rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, 1, 1, 1);
        problem.setActivationType(activation);

        std::vector<float> a{2};
        std::vector<float> b{1};
        std::vector<float> c{0};
        std::vector<float> d{-99};
        ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
        inputs.activationArgs = {0.5f, 1.5f};

        EXPECT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1))
            << "activation=" << ToString(activation);
        EXPECT_TRUE(std::isfinite(d[0])) << "activation=" << ToString(activation);
    }
}

TEST(ReferenceRuntimePointwise, NormalizesExplicitGradientActivations)
{
    for(const ActivationType activation : {ActivationType::DGelu, ActivationType::DRelu})
    {
        auto problem = makePackedProblem(
            rocisa::DataType::Float, rocisa::DataType::Float, rocisa::DataType::Float, 1, 1, 1);
        problem.setUseE(true);
        problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides());
        problem.setUseGradient(true);
        problem.setActivationType(activation);

        std::vector<float> a{2};
        std::vector<float> b{1};
        std::vector<float> c{0};
        std::vector<float> d{-99};
        std::vector<float> e{1};
        ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
        inputs.e = e.data();

        EXPECT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1))
            << "activation=" << ToString(activation);
        EXPECT_TRUE(std::isfinite(d[0])) << "activation=" << ToString(activation);
    }
}

#if !defined(_WIN32) && defined(TENSILE_USE_FP6)
TEST(ReferencePackedStorage, Float6MatchesComponentCodec)
{
    Float6x32 packed{};
    packed.data.v0 = 0x01;
    packed.data.v1 = 0x02;
    packed.data.v2 = 0x07;
    packed.data.v3 = 0x1f;
    packed.data.v4 = 0x3f;

    const roc::host_validation::Tensor component(
        roc::host_validation::ScalarType::Float6E2M3,
        roc::host_validation::Layout::contiguous(roc::host_validation::Shape{32}),
        std::as_bytes(std::span<const Float6x32>(&packed, 1)));
    for(size_t index = 0; index < 32; ++index)
        EXPECT_EQ(component.loadAs<float>({index}), packed.getElement(index)) << "index=" << index;
}

TEST(ReferenceRuntimePointwise, HandlesPackedFloat6Storage)
{
    auto problem = makePackedProblem(
        rocisa::DataType::Float6, rocisa::DataType::Float6, rocisa::DataType::Float, 1, 1, 32);
    std::vector<Float6x32> a{Float6x32(1.0f)};
    std::vector<Float6x32> b{Float6x32(2.0f)};
    std::vector<float>     c{0};
    std::vector<float>     d{-99};
    ContractionInputs      inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d[0], 64);
}
#endif

#if !defined(_WIN32) && defined(TENSILE_USE_BF6)
TEST(ReferencePackedStorage, BFloat6MatchesComponentCodec)
{
    BFloat6x32 packed{};
    packed.data.v0 = 0x01;
    packed.data.v1 = 0x02;
    packed.data.v2 = 0x07;
    packed.data.v3 = 0x1f;
    packed.data.v4 = 0x3f;

    const roc::host_validation::Tensor component(
        roc::host_validation::ScalarType::Float6E3M2,
        roc::host_validation::Layout::contiguous(roc::host_validation::Shape{32}),
        std::as_bytes(std::span<const BFloat6x32>(&packed, 1)));
    for(size_t index = 0; index < 32; ++index)
        EXPECT_EQ(component.loadAs<float>({index}), packed.getElement(index)) << "index=" << index;
}

TEST(ReferenceRuntimePointwise, HandlesPackedBFloat6Storage)
{
    auto problem = makePackedProblem(
        rocisa::DataType::BFloat6, rocisa::DataType::BFloat6, rocisa::DataType::Float, 1, 1, 32);
    std::vector<BFloat6x32> a{BFloat6x32(1.0f)};
    std::vector<BFloat6x32> b{BFloat6x32(2.0f)};
    std::vector<float>      c{0};
    std::vector<float>      d{-99};
    ContractionInputs       inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);

    ASSERT_TRUE(tryReferenceGemm(problem, inputs, /*elementsToValidate=*/-1));
    EXPECT_EQ(d[0], 64);
}
#endif
