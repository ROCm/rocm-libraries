// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
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

    // Int8 A/B with C/D type typeCD and alpha/beta type alphaType (one of the
    // reference's TypedGemm_I8_* ids when the pair is supported).
    ContractionProblemGemm makeInt8Problem(
        rocisa::DataType typeCD, rocisa::DataType alphaType, size_t M, size_t N, size_t K)
    {
        auto problem
            = makePackedProblem(rocisa::DataType::Int8, rocisa::DataType::Int8, typeCD, M, N, K);
        problem.setAlphaType(alphaType);
        problem.setBetaType(alphaType);
        return problem;
    }

    // A x B -> typeCD with compute-input types computeA/computeB and float
    // alpha/beta (one of the reference's Float8-output ids when typeCD is an
    // 8-bit float type and the combination is supported).
    ContractionProblemGemm makeFloat8OutputProblem(rocisa::DataType typeA,
                                                   rocisa::DataType typeB,
                                                   rocisa::DataType typeCD,
                                                   rocisa::DataType computeA,
                                                   rocisa::DataType computeB,
                                                   size_t           M,
                                                   size_t           N,
                                                   size_t           K)
    {
        auto problem = makePackedProblem(typeA, typeB, typeCD, M, N, K);
        problem.setComputeInputTypeA(computeA);
        problem.setComputeInputTypeB(computeB);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        return problem;
    }

    // M x 3 column-major Int8 A whose row m sums to rowSums[m] (each within
    // [-3 * 128, 3 * 127]); with B = {1, 1, 1} (K = 3, N = 1), D[m] = alpha * rowSums[m].
    std::vector<int8_t> int8RowsWithSums(std::vector<int> const& rowSums)
    {
        const size_t        M = rowSums.size();
        std::vector<int8_t> a(M * 3);
        for(size_t m = 0; m < M; ++m)
        {
            int rest = rowSums[m];
            for(size_t k = 0; k < 3; ++k)
            {
                const int v  = std::max(-128, std::min(127, rest));
                a[m + k * M] = static_cast<int8_t>(v);
                rest -= v;
            }
            EXPECT_EQ(rest, 0) << "row sum " << rowSums[m] << " is out of range";
        }
        return a;
    }

    // Runs the problem on the fast path (which must accept it) and the slow path,
    // and returns both D buffers (C is zero). setInputs sets further
    // ContractionInputs fields (scales) on both runs.
    template <typename DT, typename InT, typename SetInputs = void (*)(ContractionInputs&)>
    std::pair<std::vector<DT>, std::vector<DT>> solveFastAndSlow(
        ContractionProblemGemm const& problem,
        std::vector<InT> const&       a,
        std::vector<InT> const&       b,
        ConstantVariant               alpha,
        ConstantVariant               beta,
        SetInputs                     setInputs = [](ContractionInputs&) {})
    {
        const size_t    sizeD = problem.d().totalAllocatedElements();
        std::vector<DT> c(sizeD, DT(0));
        std::vector<DT> dFast(sizeD, DT(0));
        std::vector<DT> dSlow(sizeD, DT(0));

        EXPECT_TRUE(isFastPathEligible(problem));
        ContractionInputs fastInputs(a.data(), b.data(), c.data(), dFast.data(), alpha, beta);
        setInputs(fastInputs);
        SolveGemmCPU(problem, fastInputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);
        ContractionInputs slowInputs(a.data(), b.data(), c.data(), dSlow.data(), alpha, beta);
        setInputs(slowInputs);
        SolveGemmCPU(problem, slowInputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
        return {dFast, dSlow};
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

TEST(ReferenceFastPath, Int8EligibilityFollowsReferenceTypeIds)
{
    using DT = rocisa::DataType;

    // The six TypedGemm_I8_* ids: C/D type x alpha/beta type.
    for(auto [typeCD, alphaType] : {std::pair{DT::Int8, DT::Int32},
                                    std::pair{DT::Int32, DT::Int32},
                                    std::pair{DT::Int32, DT::Float},
                                    std::pair{DT::Int8, DT::Float},
                                    std::pair{DT::Half, DT::Float},
                                    std::pair{DT::BFloat16, DT::Float}})
    {
        EXPECT_TRUE(isFastPathEligible(makeInt8Problem(typeCD, alphaType, 4, 4, 4)))
            << "C/D " << TensileLite::ToString(typeCD) << ", alpha "
            << TensileLite::ToString(alphaType);
    }

    // Not a reference type id: Float output, Half output with int32 alpha.
    EXPECT_FALSE(isFastPathEligible(makeInt8Problem(DT::Float, DT::Float, 4, 4, 4)));
    EXPECT_FALSE(isFastPathEligible(makeInt8Problem(DT::Half, DT::Int32, 4, 4, 4)));

    // Mixed Int8 / Float inputs.
    auto mixed = makePackedProblem(DT::Int8, DT::Float, DT::Float, 4, 4, 4);
    EXPECT_FALSE(isFastPathEligible(mixed));
    auto mixedB = makePackedProblem(DT::Float, DT::Int8, DT::Int32, 4, 4, 4);
    EXPECT_FALSE(isFastPathEligible(mixedB));

    // Integer outputs remain unsupported for floating-point inputs.
    auto floatToInt = makePackedProblem(DT::Float, DT::Float, DT::Int32, 4, 4, 4);
    EXPECT_FALSE(isFastPathEligible(floatToInt));

    // useE is rejected before type-id classification can throw for mismatched scalars.
    auto useE = makeInt8Problem(DT::Int32, DT::Int32, 4, 4, 4);
    useE.setUseE(true);
    useE.setBetaType(DT::Float);
    EXPECT_NO_THROW(EXPECT_FALSE(isFastPathEligible(useE)));

    // Int8 storage with a non-Int8 compute-input type.
    auto computeF8 = makeInt8Problem(DT::Int32, DT::Int32, 4, 4, 4);
    computeF8.setComputeInputTypeA(DT::Float8);
    EXPECT_FALSE(isFastPathEligible(computeF8));

    // MX block scaling (floating-point MAC only).
    auto mx = makeInt8Problem(DT::Int32, DT::Int32, 4, 4, 32);
    ASSERT_TRUE(isFastPathEligible(mx));
    mx.setMXScaleA(DT::E8, 32, {}, /*padScaleTensor=*/false);
    mx.setMXScaleB(DT::E8, 32, {}, /*padScaleTensor=*/false);
    EXPECT_FALSE(isFastPathEligible(mx));
}

// int32_t epilogue (I8_I32_I32): the sum is exact above 2^24, where a float
// accumulator is not.
TEST(ReferenceFastPath, Int8Int32EpilogueIsExactAboveFloatRange)
{
    const size_t K = 1041;
    auto problem   = makeInt8Problem(rocisa::DataType::Int32, rocisa::DataType::Int32, 1, 1, K);

    std::vector<int8_t> a(K, 127);
    std::vector<int8_t> b(K, 127);
    const int32_t       exactSum = static_cast<int32_t>(K) * 127 * 127; // 16,790,289
    ASSERT_GT(exactSum, 1 << 24);
    ASSERT_NE(static_cast<int32_t>(static_cast<float>(exactSum)), exactSum);

    auto [dFast, dSlow] = solveFastAndSlow<int32_t>(problem, a, b, int32_t(1), int32_t(0));
    EXPECT_EQ(dFast[0], exactSum);
    EXPECT_EQ(dFast[0], dSlow[0]);
}

// float epilogue (I8_I32_S): D = int32_t(alpha * float(exact sum)). Here the exact
// sum is a float value. Both paths reduce in integers before the float epilogue.
// The former per-K float reduction gives 16,790,288 (4 short): the 1041st
// product rounds 16,790,289 down to
// even, and the last three products (+1 each) are lost above 2^24.
TEST(ReferenceFastPath, Int8FloatEpilogueStartsFromExactSum)
{
    const size_t K = 1044;
    auto problem   = makeInt8Problem(rocisa::DataType::Int32, rocisa::DataType::Float, 1, 1, K);

    std::vector<int8_t> a(K, 127);
    std::vector<int8_t> b(K, 127);
    std::fill(a.end() - 3, a.end(), int8_t(1));
    std::fill(b.end() - 3, b.end(), int8_t(1));
    const int32_t exactSum = 1041 * 127 * 127 + 3; // 16,790,292
    float         floatSum = 0.0f;
    for(size_t k = 0; k < K; ++k)
        floatSum += static_cast<float>(a[k]) * static_cast<float>(b[k]);
    ASSERT_EQ(static_cast<int32_t>(static_cast<float>(exactSum)), exactSum);
    ASSERT_NE(floatSum, static_cast<float>(exactSum));

    auto [dFast, dSlow] = solveFastAndSlow<int32_t>(problem, a, b, 1.0f, 0.0f);
    EXPECT_EQ(dFast[0], exactSum);
    EXPECT_EQ(dSlow[0], exactSum);
}

// int32_t epilogue (I8_I8_I32): the Int8 output is clamped to [-128, 127].
TEST(ReferenceFastPath, Int8OutputSaturatesFromInt32Epilogue)
{
    const std::vector<int>    sums     = {127, 128, -128, -129, 300, -300, 0, -5};
    const std::vector<int8_t> expected = {127, 127, -128, -128, 127, -128, 0, -5};

    auto problem
        = makeInt8Problem(rocisa::DataType::Int8, rocisa::DataType::Int32, sums.size(), 1, 3);
    std::vector<int8_t> a = int8RowsWithSums(sums);
    std::vector<int8_t> b(3, 1);

    auto [dFast, dSlow] = solveFastAndSlow<int8_t>(problem, a, b, int32_t(1), int32_t(0));
    for(size_t m = 0; m < sums.size(); ++m)
    {
        EXPECT_EQ(int(dFast[m]), int(expected[m])) << "sum " << sums[m];
        EXPECT_EQ(int(dFast[m]), int(dSlow[m])) << "sum " << sums[m];
    }
}

// float epilogue (I8_I8_S): the Int8 output is rounded half to even, then
// clamped to [-128, 127] (alpha 0.5 makes every odd sum a tie).
TEST(ReferenceFastPath, Int8OutputRoundsHalfToEvenThenSaturatesFromFloatEpilogue)
{
    const std::vector<int> sums = {255, 253, 251, -255, -257, -259, 5, 3, 1, -1, -3, 300, -300};
    // 127.5 -> 128 -> 127, 126.5 -> 126, 125.5 -> 126, -127.5 -> -128, -128.5 -> -128,
    // -129.5 -> -130 -> -128, 2.5 -> 2, 1.5 -> 2, 0.5 -> 0, -0.5 -> 0, -1.5 -> -2.
    const std::vector<int8_t> expected
        = {127, 126, 126, -128, -128, -128, 2, 2, 0, 0, -2, 127, -128};

    auto problem
        = makeInt8Problem(rocisa::DataType::Int8, rocisa::DataType::Float, sums.size(), 1, 3);
    std::vector<int8_t> a = int8RowsWithSums(sums);
    std::vector<int8_t> b(3, 1);

    auto [dFast, dSlow] = solveFastAndSlow<int8_t>(problem, a, b, 0.5f, 0.0f);
    for(size_t m = 0; m < sums.size(); ++m)
    {
        EXPECT_EQ(int(dFast[m]), int(expected[m])) << "sum " << sums[m];
        EXPECT_EQ(int(dFast[m]), int(dSlow[m])) << "sum " << sums[m];
    }
}

// float epilogue (I8_I32_S): alpha, the ScaleAB scales and scaleAlphaVec are applied
// in the slow path's order, (((alpha * scaleA) * scaleB) * sum) * scaleAlphaVec, each
// product rounded to float. The values are chosen so that the other fast-path types'
// folded order, (alpha * (scaleA * scaleB) * scaleAlphaVec) * sum, truncates to a
// different Int32 output.
TEST(ReferenceFastPath, Int8FloatEpilogueFollowsSlowPathOrder)
{
    using DT = rocisa::DataType;
    std::vector<int8_t> b(1, 50);

    {
        SCOPED_TRACE("scaleAlphaVec");
        auto problem = makeInt8Problem(DT::Int32, DT::Float, 1, 1, 1);
        problem.setUseScaleAlphaVec(1);
        problem.setScaleAlphaVec(DT::Float, 1, /*factorDim=*/0);
        std::vector<int8_t> a(1, 2); // sum 100
        std::vector<float>  alphaVec(1, 0.7f);
        const float         alpha = 0.7f;

        float slowOrder = alpha * 100.0f;
        slowOrder *= alphaVec[0];
        const float folded = (alpha * alphaVec[0]) * 100.0f;
        ASSERT_NE(static_cast<int32_t>(slowOrder), static_cast<int32_t>(folded));

        auto [dFast, dSlow]
            = solveFastAndSlow<int32_t>(problem, a, b, alpha, 0.0f, [&](ContractionInputs& in) {
                  in.scaleAlphaVec = alphaVec.data();
              });
        EXPECT_EQ(dFast[0], static_cast<int32_t>(slowOrder));
        EXPECT_EQ(dFast[0], dSlow[0]);
    }
    {
        SCOPED_TRACE("ScaleAB Scalar");
        auto problem = makeInt8Problem(DT::Int32, DT::Float, 1, 1, 1);
        problem.setUseScaleAB("Scalar");
        problem.setScaleA(DT::Float, 1);
        problem.setScaleB(DT::Float, 1);
        std::vector<int8_t> a(1, 60); // sum 3000
        std::vector<float>  scaleA(1, 0.1f);
        std::vector<float>  scaleB(1, 0.9f);
        const float         alpha = 0.1f;

        float alphaScaled = alpha;
        alphaScaled *= scaleA[0];
        alphaScaled *= scaleB[0];
        const float slowOrder = alphaScaled * 3000.0f;
        const float folded    = (alpha * (scaleA[0] * scaleB[0])) * 3000.0f;
        ASSERT_NE(static_cast<int32_t>(slowOrder), static_cast<int32_t>(folded));

        auto [dFast, dSlow]
            = solveFastAndSlow<int32_t>(problem, a, b, alpha, 0.0f, [&](ContractionInputs& in) {
                  in.scaleA = scaleA.data();
                  in.scaleB = scaleB.data();
              });
        EXPECT_EQ(dFast[0], static_cast<int32_t>(slowOrder));
        EXPECT_EQ(dFast[0], dSlow[0]);
    }
}

TEST(ReferenceFastPath, Int8InPlaceCDPreservesBetaInput)
{
    auto problem = makeInt8Problem(rocisa::DataType::Int32, rocisa::DataType::Int32, 1, 1, 1);
    int8_t a = 2, b = 3;
    int32_t cd = 5;
    ContractionInputs inputs(&a, &b, &cd, &cd, int32_t(1), int32_t(1));
    ASSERT_TRUE(isFastPathEligible(problem));
    SolveGemmCPU(problem, inputs, -1, true);
    EXPECT_EQ(cd, 11);
}

TEST(ReferenceFastPath, Int8DotProductWrapsAtInt32Boundary)
{
    constexpr size_t K = 140000;
    const std::vector<int8_t> a(K, -128);
    // The exact sums are 2,293,760,000 and -2,275,840,000. Subtracting or
    // adding 2^32 gives these signed results, both exactly representable in float.
    for(auto [bValue, expected] : {std::pair{int8_t(-128), int32_t(-2001207296)},
                                  std::pair{int8_t(127), int32_t(2019127296)}})
    {
        const std::vector<int8_t> b(K, bValue);
        auto check = [&](auto alpha) {
            using ScalarT = decltype(alpha);
            auto problem = makeInt8Problem(
                rocisa::DataType::Int32, TypeInfo<ScalarT>::Enum, 1, 1, K);
            int32_t c = 0, d = 0;
            ContractionInputs inputs(a.data(), b.data(), &c, &d, alpha, ScalarT(0));
            ASSERT_TRUE(isFastPathEligible(problem));
            SolveGemmCPU(problem, inputs, -1, true);
            EXPECT_EQ(d, expected) << "B=" << int(bValue)
                                   << ", alpha type=" << ToString(TypeInfo<ScalarT>::Enum);
        };
        check(int32_t(1));
        check(1.0f);
    }
}

TEST(ReferenceFastPath, Int8StridedCFollowsSlowReference)
{
    using DT     = rocisa::DataType;
    auto problem = ContractionProblemGemm::FromIndexSizes("Contraction_l_Ailk_Bljk_Cijk_Dijk",
                                                          {2, 1, 1, 1},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int32,
                                                          {2, 4, 4},
                                                          DT::Int32,
                                                          {},
                                                          1.0);
    problem.setComputeInputTypeA(DT::Int8);
    problem.setComputeInputTypeB(DT::Int8);
    problem.setAlphaType(DT::Int32);
    problem.setBetaType(DT::Int32);
    ASSERT_TRUE(isFastPathEligible(problem));
    const std::vector<int8_t>  a = {2, 3}, b = {5};
    const std::vector<int32_t> c        = {7, 99, 11};
    const std::vector<int32_t> expected = {17, 26};
    for(bool tryFast : {false, true})
    {
        SCOPED_TRACE(tryFast);
        std::vector<int32_t> d(2, -1);
        ContractionInputs    inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(1));
        SolveGemmCPU(problem, inputs, -1, tryFast);
        EXPECT_EQ(d, expected);
    }
}

TEST(ReferenceFastPath, Int8PermutedCIndicesFollowSlowReference)
{
    using DT     = rocisa::DataType;
    auto problem = ContractionProblemGemm::FromIndexSizes("Contraction_l_Ailk_Bljk_Cjik_Dijk",
                                                          {2, 2, 1, 1},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int32,
                                                          {},
                                                          DT::Int32,
                                                          {},
                                                          1.0);
    problem.setComputeInputTypeA(DT::Int8);
    problem.setComputeInputTypeB(DT::Int8);
    problem.setAlphaType(DT::Int32);
    problem.setBetaType(DT::Int32);
    ASSERT_TRUE(isFastPathEligible(problem));
    const std::vector<int8_t>  a = {2, 3}, b = {5, 7};
    const std::vector<int32_t> c        = {11, 13, 17, 19};
    const std::vector<int32_t> expected = {21, 32, 27, 40};
    for(bool tryFast : {false, true})
    {
        SCOPED_TRACE(tryFast);
        std::vector<int32_t> d(4, -1);
        ContractionInputs    inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(1));
        SolveGemmCPU(problem, inputs, -1, tryFast);
        EXPECT_EQ(d, expected);
    }
}

TEST(ReferenceFastPath, Int8PermutedCBatchIndexFollowsSlowReference)
{
    using DT     = rocisa::DataType;
    auto problem = ContractionProblemGemm::FromIndexSizes("Contraction_l_Ailk_Bljk_Ckij_Dijk",
                                                          {2, 1, 2, 1},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int32,
                                                          {},
                                                          DT::Int32,
                                                          {},
                                                          1.0);
    problem.setComputeInputTypeA(DT::Int8);
    problem.setComputeInputTypeB(DT::Int8);
    problem.setAlphaType(DT::Int32);
    problem.setBetaType(DT::Int32);
    ASSERT_TRUE(isFastPathEligible(problem));
    const std::vector<int8_t> a = {2, 3, 5, 7}, b = {11, 13};
    // Extra storage makes incorrect batch addressing read sentinels instead of out of bounds.
    const std::vector<int32_t> c        = {17, 19, 23, 29, 101, 103};
    const std::vector<int32_t> expected = {39, 56, 84, 120};
    for(bool tryFast : {false, true})
    {
        SCOPED_TRACE(tryFast);
        std::vector<int32_t> d(4, -1);
        ContractionInputs    inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(1));
        SolveGemmCPU(problem, inputs, -1, tryFast);
        EXPECT_EQ(d, expected);
    }
}

TEST(ReferenceFastPath, Int8MultipleBatchIndicesFollowSlowReference)
{
    using DT     = rocisa::DataType;
    auto problem = ContractionProblemGemm::FromIndexSizes("Contraction_m_Aimkl_Bmjkl_Cijkl_Dijkl",
                                                          {1, 1, 2, 2, 1},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int8,
                                                          {},
                                                          DT::Int32,
                                                          {},
                                                          DT::Int32,
                                                          {},
                                                          0.0);
    problem.setComputeInputTypeA(DT::Int8);
    problem.setComputeInputTypeB(DT::Int8);
    problem.setAlphaType(DT::Int32);
    problem.setBetaType(DT::Int32);
    EXPECT_FALSE(isFastPathEligible(problem));
    const std::vector<int8_t>  a = {2, 3, 5, 7}, b = {11, 13, 17, 19};
    const std::vector<int32_t> c(4, 0);
    const std::vector<int32_t> expected = {22, 39, 85, 133};
    for(bool tryFast : {false, true})
    {
        SCOPED_TRACE(tryFast);
        std::vector<int32_t> d(4, -1);
        ContractionInputs    inputs(a.data(), b.data(), c.data(), d.data(), int32_t(1), int32_t(0));
        SolveGemmCPU(problem, inputs, -1, tryFast);
        EXPECT_EQ(d, expected);
    }
}

TEST(ReferenceFastPath, ReversedDFreeIndicesUseSlowReference)
{
    auto check = [&](auto inputValue, auto outputValue) {
        using InputT  = decltype(inputValue);
        using OutputT = decltype(outputValue);
        auto problem  = ContractionProblemGemm::FromIndexSizes("Contraction_l_Ajlk_Blik_Cijk_Dijk",
                                                               {3, 2, 1, 1},
                                                              TypeInfo<InputT>::Enum,
                                                               {},
                                                              TypeInfo<InputT>::Enum,
                                                               {},
                                                              TypeInfo<OutputT>::Enum,
                                                               {},
                                                              TypeInfo<OutputT>::Enum,
                                                               {},
                                                              1.0);
        problem.setComputeInputTypeA(TypeInfo<InputT>::Enum);
        problem.setComputeInputTypeB(TypeInfo<InputT>::Enum);
        problem.setAlphaType(TypeInfo<OutputT>::Enum);
        problem.setBetaType(TypeInfo<OutputT>::Enum);
        EXPECT_FALSE(isFastPathEligible(problem));
        ASSERT_EQ(problem.d().totalAllocatedElements(), 6);
        const std::vector<InputT>  a = {2, 3}, b = {7, 11, 13};
        const std::vector<OutputT> c(6, 0);
        // The final four entries guard against writes beyond the D descriptor.
        const std::vector<OutputT> expected = {14, 22, 26, 21, 33, 39, -1, -1, -1, -1};
        for(bool tryFast : {false, true})
        {
            SCOPED_TRACE(tryFast);
            std::vector<OutputT> d(10, -1);
            ContractionInputs    inputs(
                a.data(), b.data(), c.data(), d.data(), OutputT(1), OutputT(0));
            SolveGemmCPU(problem, inputs, -1, tryFast);
            EXPECT_EQ(d, expected);
        }
    };
    check(int8_t(0), int32_t(0));
    check(0.0f, 0.0f);
}

TEST(ReferenceFastPathDeathTest, BetaZeroSkipsNullC)
{
    auto check = [&](auto outputValue, auto scalarValue) {
        using OutputT = decltype(outputValue);
        using ScalarT = decltype(scalarValue);
        auto problem  = makeInt8Problem(TypeInfo<OutputT>::Enum, TypeInfo<ScalarT>::Enum, 1, 1, 1);
        ASSERT_TRUE(isFastPathEligible(problem));
        int8_t a = 2, b = 3;
        for(bool tryFast : {false, true})
        {
            SCOPED_TRACE(tryFast);
            // Run in a child so an unused-C dereference reports a failed test.
            EXPECT_EXIT(
                {
                    OutputT           d(0);
                    ContractionInputs inputs(&a, &b, nullptr, &d, ScalarT(1), ScalarT(0));
                    SolveGemmCPU(problem, inputs, -1, tryFast);
                    std::exit(d == OutputT(6) ? 0 : 1);
                },
                ::testing::ExitedWithCode(0),
                "");
        }
    };
    check(int8_t(0), int32_t(0));
    check(int32_t(0), int32_t(0));
    check(int8_t(0), 0.0f);
    check(int32_t(0), 0.0f);
    check(TensileLite::Half(0.0f), 0.0f);
    check(TensileLite::BFloat16(0.0f), 0.0f);
}

TEST(ReferenceFastPath, Int8ConvertedOutputPreservesStridePadding)
{
    auto check = [&](auto outputValue, auto scalarValue) {
        using OutputT = decltype(outputValue);
        using ScalarT = decltype(scalarValue);
        using DT      = rocisa::DataType;
        auto problem  = ContractionProblemGemm::FromIndexSizes("Contraction_l_Ailk_Bljk_Cijk_Dijk",
                                                               {2, 2, 2, 1},
                                                              DT::Int8,
                                                               {},
                                                              DT::Int8,
                                                               {},
                                                              TypeInfo<OutputT>::Enum,
                                                               {},
                                                              TypeInfo<OutputT>::Enum,
                                                               {1, 3, 8},
                                                              1.0);
        problem.setComputeInputTypeA(DT::Int8);
        problem.setComputeInputTypeB(DT::Int8);
        problem.setAlphaType(TypeInfo<ScalarT>::Enum);
        problem.setBetaType(TypeInfo<ScalarT>::Enum);
        ASSERT_TRUE(isFastPathEligible(problem));
        ASSERT_EQ(problem.d().totalAllocatedElements(), 13);
        const std::vector<int8_t>  a = {2, 3, 2, 3}, b = {5, 7, 11, 13};
        const std::vector<OutputT> c(8, OutputT(0));
        // Preserve the column gap and the three elements between batches.
        std::vector<OutputT> expected;
        for(int value : {10, 15, 99, 14, 21, 99, 99, 99, 22, 33, 99, 26, 39})
            expected.emplace_back(value);
        for(bool tryFast : {false, true})
        {
            SCOPED_TRACE(tryFast);
            std::vector<OutputT> d(13, OutputT(99));
            ContractionInputs    inputs(
                a.data(), b.data(), c.data(), d.data(), ScalarT(1), ScalarT(0));
            SolveGemmCPU(problem, inputs, -1, tryFast);
            EXPECT_EQ(d, expected);
        }
    };
    check(int8_t(0), int32_t(0));
    check(int8_t(0), 0.0f);
    check(int32_t(0), 0.0f);
    check(TensileLite::Half(0.0f), 0.0f);
    check(TensileLite::BFloat16(0.0f), 0.0f);
}

TEST(ReferenceFastPath, Float8OutputEligibilityFollowsReferenceTypeIds)
{
    using DT = rocisa::DataType;
    struct Types
    {
        DT a, b, cd, computeA, computeB;
    };

    // The reference's Float8-output ids with float alpha/beta (TypedGemm_*_S).
    const DT F8 = DT::Float8, B8 = DT::BFloat8, F8N = DT::Float8_fnuz, B8N = DT::BFloat8_fnuz,
             H  = DT::Half;
    for(Types t : {Types{F8, F8, F8, F8, F8},          Types{F8, F8, B8, F8, F8},
                   Types{B8, B8, F8, B8, B8},          Types{B8, B8, B8, B8, B8},
                   Types{F8, B8, F8, F8, B8},          Types{F8, B8, B8, F8, B8},
                   Types{B8, F8, F8, B8, F8},          Types{B8, F8, B8, B8, F8},
                   Types{H, F8, F8, H, H},             Types{F8, H, F8, H, H},
                   Types{H, F8, F8, F8, F8},           Types{F8, H, F8, F8, F8},
                   Types{F8N, F8N, F8N, F8N, F8N},     Types{F8N, F8N, B8N, F8N, F8N},
                   Types{B8N, B8N, F8N, B8N, B8N},     Types{B8N, B8N, B8N, B8N, B8N},
                   Types{F8N, B8N, F8N, F8N, B8N},     Types{F8N, B8N, B8N, F8N, B8N},
                   Types{B8N, F8N, F8N, B8N, F8N},     Types{B8N, F8N, B8N, B8N, F8N},
                   Types{H, F8N, F8N, H, H},           Types{F8N, H, F8N, H, H},
                   Types{H, F8N, F8N, F8N, F8N},       Types{F8N, H, F8N, F8N, F8N}})
    {
        for(auto mode : {"", "Scalar", "Vector"})
        {
            auto problem = makeFloat8OutputProblem(t.a, t.b, t.cd, t.computeA, t.computeB, 4, 4, 4);
            problem.setUseScaleAB(mode);
            const bool narrowsHalf = (t.a == H && t.computeA != H) || (t.b == H && t.computeB != H);
            const bool expected    = std::string(mode) != "Scalar" || !narrowsHalf;
            EXPECT_EQ(isFastPathEligible(problem), expected)
                << TensileLite::ToString(t.a) << " x " << TensileLite::ToString(t.b) << " -> "
                << TensileLite::ToString(t.cd) << ", MAC " << TensileLite::ToString(t.computeA)
                << " x " << TensileLite::ToString(t.computeB) << ", ScaleAB=" << mode;
        }
    }

    // Not a reference Float8-output id: Float or Half inputs, an output of the other
    // F8 family, BFloat8 x Half, Float8 x Half with mixed MAC types, a Half
    // alpha/beta, or Float4 inputs (TypedGemm_F4_F8_S, not part of this fast path).
    for(Types t : {Types{DT::Float, DT::Float, F8, DT::Float, DT::Float},
                   Types{H, H, F8, H, H},
                   Types{F8, F8, F8N, F8, F8},
                   Types{F8N, F8N, B8, F8N, F8N},
                   Types{B8, H, B8, H, H},
                   Types{F8, H, F8, F8, H},
                   Types{DT::Float4, DT::Float4, F8, DT::Float4, DT::Float4}})
    {
        EXPECT_FALSE(isFastPathEligible(
            makeFloat8OutputProblem(t.a, t.b, t.cd, t.computeA, t.computeB, 4, 4, 4)))
            << TensileLite::ToString(t.a) << " x " << TensileLite::ToString(t.b) << " -> "
            << TensileLite::ToString(t.cd) << ", MAC " << TensileLite::ToString(t.computeA)
            << " x " << TensileLite::ToString(t.computeB);
    }
    auto halfAlpha = makeFloat8OutputProblem(F8, F8, F8, F8, F8, 4, 4, 4);
    halfAlpha.setAlphaType(DT::Half);
    halfAlpha.setBetaType(DT::Half);
    EXPECT_FALSE(isFastPathEligible(halfAlpha));
}

namespace
{
    template <typename F8>
    void checkScaledHalfToFloat8Output(bool halfA, bool scalar)
    {
        constexpr auto f8      = TypeInfo<F8>::Enum;
        auto           problem = makeFloat8OutputProblem(halfA ? rocisa::DataType::Half : f8,
                                               halfA ? f8 : rocisa::DataType::Half,
                                               f8,
                                               f8,
                                               f8,
                                               2,
                                               2,
                                               1);
        problem.setUseScaleAB(scalar ? "Scalar" : "Vector");
        problem.setScaleA(rocisa::DataType::Float, scalar ? 1 : 2);
        problem.setScaleB(rocisa::DataType::Float, scalar ? 1 : 2);
        EXPECT_EQ(isFastPathEligible(problem), !scalar);

        std::vector<Half>  h = {Half(0.9f), Half(-0.9f)};
        std::vector<F8>    f = {F8(1.0f), F8(-1.0f)};
        std::vector<F8>    c(4, F8(0.0f)), d(4, F8(0.0f));
        std::vector<float> halfScale  = {3.0f, -2.0f};
        std::vector<float> otherScale = {0.5f, 1.0f};
        ContractionInputs  inputs(halfA ? static_cast<void const*>(h.data()) : f.data(),
                                 halfA ? static_cast<void const*>(f.data()) : h.data(),
                                 c.data(),
                                 d.data(),
                                 1.0f,
                                 0.0f);
        inputs.scaleA = halfA ? halfScale.data() : otherScale.data();
        inputs.scaleB = halfA ? otherScale.data() : halfScale.data();
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

        // Scalar: Half(0.9) * 3 narrows to 2.75, then the other scale gives
        // 1.375. Narrowing first would instead give 1.25 after the final cast.
        // Vector: Half(0.9) narrows to 0.875, then BOTH vectors apply; ties
        // 1.3125 and -2.625 round to even F8 values 1.25 and -2.5.
        const std::vector<float> expected
            = scalar  ? std::vector<float>{1.375f, -1.375f, -1.375f, 1.375f}
              : halfA ? std::vector<float>{1.25f, 0.875f, -2.5f, -1.75f}
                      : std::vector<float>{1.25f, -2.5f, 0.875f, -1.75f};
        for(size_t i = 0; i < d.size(); ++i)
            EXPECT_EQ(float(d[i]), expected[i]) << "D[" << i << "]";
    }
}

TEST(ReferenceFastPath, Float8OutputScalarNarrowingFallsBack)
{
    for(bool halfA : {true, false})
    {
        SCOPED_TRACE(halfA ? "Half A" : "Half B");
        {
            SCOPED_TRACE("OCP");
            checkScaledHalfToFloat8Output<Float8>(halfA, true);
        }
        {
            SCOPED_TRACE("fnuz");
            checkScaledHalfToFloat8Output<Float8_fnuz>(halfA, true);
        }
    }
}

TEST(ReferenceFastPath, Float8OutputVectorNarrowingRemainsEligible)
{
    for(bool halfA : {true, false})
    {
        SCOPED_TRACE(halfA ? "Half A" : "Half B");
        {
            SCOPED_TRACE("OCP");
            checkScaledHalfToFloat8Output<Float8>(halfA, false);
        }
        {
            SCOPED_TRACE("fnuz");
            checkScaledHalfToFloat8Output<Float8_fnuz>(halfA, false);
        }
    }
}

namespace
{
    // A = B = 1 (K = 1, N = 1) with scaleAlphaVec[m] = cases[m].first, so the float
    // value before the output cast is exactly cases[m].first on both paths. Checks
    // that the Float8/BFloat8 output is cases[m].second on both paths, including
    // non-finite results from SaturateCast.
    template <typename OutT>
    void checkFloat8OutputCast(rocisa::DataType                           type,
                               std::vector<std::pair<float, float>> const& cases)
    {
        const size_t M       = cases.size();
        auto         problem = makeFloat8OutputProblem(type, type, type, type, type, M, 1, 1);
        problem.setUseScaleAlphaVec(1);
        problem.setScaleAlphaVec(rocisa::DataType::Float, M, /*factorDim=*/0);

        std::vector<float> alphaVec;
        for(auto const& c : cases)
            alphaVec.push_back(c.first);
        std::vector<OutT> a(M, OutT(1.0f));
        std::vector<OutT> b(1, OutT(1.0f));

        auto [dFast, dSlow]
            = solveFastAndSlow<OutT>(problem, a, b, 1.0f, 0.0f, [&](ContractionInputs& in) {
                  in.scaleAlphaVec = alphaVec.data();
              });
        for(size_t m = 0; m < M; ++m)
        {
            SCOPED_TRACE(::testing::Message() << "value " << cases[m].first);
            if(std::isnan(cases[m].second))
            {
                EXPECT_TRUE(std::isnan(static_cast<float>(dFast[m])));
                EXPECT_TRUE(std::isnan(static_cast<float>(dSlow[m])));
            }
            else
            {
                EXPECT_EQ(static_cast<float>(dFast[m]), cases[m].second);
                EXPECT_EQ(static_cast<float>(dSlow[m]), cases[m].second);
            }
        }
    }
} // namespace

// Finite Float8/BFloat8 outputs round to nearest even (ties between two F8 values,
// also subnormal ones) and saturate to the largest finite value (including values
// that round up past it). fnuz types have no negative zero: a tiny negative value
// is +0. OCP E5M2 preserves signed infinity; OCP E4M3 and both fnuz types map
// infinity to NaN. All four types preserve NaN.
TEST(ReferenceFastPath, Float8OutputRoundsToNearestEvenAndSaturates)
{
    const float tiny = std::ldexp(1.0f, -20);
    const float inf  = std::numeric_limits<float>::infinity();
    const float nan  = std::numeric_limits<float>::quiet_NaN();
    {
        SCOPED_TRACE("Float8 (E4M3, max 448, smallest subnormal 2^-9)");
        const float s = std::ldexp(1.0f, -9);
        checkFloat8OutputCast<Float8>(rocisa::DataType::Float8,
                                      {{1.0625f, 1.0f},
                                       {1.1875f, 1.25f},
                                       {-1.1875f, -1.25f},
                                       {1.0625f + tiny, 1.125f},
                                       {17.0f, 16.0f},
                                       {19.0f, 20.0f},
                                       {448.0f, 448.0f},
                                       {464.0f, 448.0f},
                                       {1000.0f, 448.0f},
                                       {-1.0e6f, -448.0f},
                                       {inf, nan},
                                       {-inf, nan},
                                       {nan, nan},
                                       {0.5f * s, 0.0f},
                                       {1.5f * s, 2.0f * s},
                                       {2.5f * s, 2.0f * s}});
    }
    {
        SCOPED_TRACE("BFloat8 (E5M2, max 57344, smallest subnormal 2^-16)");
        const float s = std::ldexp(1.0f, -16);
        checkFloat8OutputCast<BFloat8>(rocisa::DataType::BFloat8,
                                       {{1.125f, 1.0f},
                                        {1.375f, 1.5f},
                                        {-1.375f, -1.5f},
                                        {9.0f, 8.0f},
                                        {11.0f, 12.0f},
                                        {57344.0f, 57344.0f},
                                        {61440.0f, 57344.0f},
                                        {1.0e6f, 57344.0f},
                                        {-1.0e6f, -57344.0f},
                                        {inf, inf},
                                        {-inf, -inf},
                                        {nan, nan},
                                        {0.5f * s, 0.0f},
                                        {1.5f * s, 2.0f * s}});
    }
    {
        SCOPED_TRACE("Float8_fnuz (E4M3, max 240, smallest subnormal 2^-10)");
        const float s = std::ldexp(1.0f, -10);
        checkFloat8OutputCast<Float8_fnuz>(rocisa::DataType::Float8_fnuz,
                                           {{1.0625f, 1.0f},
                                            {1.1875f, 1.25f},
                                            {17.0f, 16.0f},
                                            {19.0f, 20.0f},
                                            {240.0f, 240.0f},
                                            {248.0f, 240.0f},
                                            {1000.0f, 240.0f},
                                            {-1.0e6f, -240.0f},
                                            {inf, nan},
                                            {-inf, nan},
                                            {nan, nan},
                                            {0.5f * s, 0.0f},
                                            {-0.5f * s, 0.0f},
                                            {1.5f * s, 2.0f * s}});
    }
    {
        SCOPED_TRACE("BFloat8_fnuz (E5M2, max 57344, smallest subnormal 2^-17)");
        const float s = std::ldexp(1.0f, -17);
        checkFloat8OutputCast<BFloat8_fnuz>(rocisa::DataType::BFloat8_fnuz,
                                            {{1.125f, 1.0f},
                                             {1.375f, 1.5f},
                                             {9.0f, 8.0f},
                                             {11.0f, 12.0f},
                                             {57344.0f, 57344.0f},
                                             {61440.0f, 57344.0f},
                                             {1.0e6f, 57344.0f},
                                             {-1.0e6f, -57344.0f},
                                             {inf, nan},
                                             {-inf, nan},
                                             {nan, nan},
                                             {0.5f * s, 0.0f},
                                             {-0.5f * s, 0.0f},
                                             {1.5f * s, 2.0f * s}});
    }
}

namespace
{
    // Complex (C_C_C or Z_Z_Z) M x N x K problem, A and B conjugated as requested.
    ContractionProblemGemm makeComplexProblem(
        rocisa::DataType type, size_t M, size_t N, size_t K, bool conjA, bool conjB)
    {
        auto problem = makePackedProblem(type, type, type, M, N, K);
        if(conjA)
            problem.setAOps({TensorOp::ComplexConjugate()});
        if(conjB)
            problem.setBOps({TensorOp::ComplexConjugate()});
        return problem;
    }

    // Runs a complex problem on the fast path (which must accept it) and the slow
    // path with the given C, and returns both D buffers.
    template <typename CT>
    std::pair<std::vector<CT>, std::vector<CT>>
        solveComplexFastAndSlow(ContractionProblemGemm const& problem,
                                std::vector<CT> const&        a,
                                std::vector<CT> const&        b,
                                std::vector<CT> const&        c,
                                CT                            alpha,
                                CT                            beta)
    {
        std::vector<CT> dFast(c.size());
        std::vector<CT> dSlow(c.size());

        EXPECT_TRUE(isFastPathEligible(problem));
        ContractionInputs fastInputs(a.data(), b.data(), c.data(), dFast.data(), alpha, beta);
        SolveGemmCPU(problem, fastInputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);
        ContractionInputs slowInputs(a.data(), b.data(), c.data(), dSlow.data(), alpha, beta);
        SolveGemmCPU(problem, slowInputs, /*elementsToValidate=*/-1, /*tryFastPath=*/false);
        return {dFast, dSlow};
    }

    // D = alpha * sum_k op(A[k]) * op(B[k]) + beta * C for M = N = 1, K = 2, with
    // op = conj where requested; small integer parts, so every value is exact.
    template <typename T>
    void checkComplexConjugation(rocisa::DataType type)
    {
        using CT                    = std::complex<T>;
        const std::vector<CT> a     = {CT(1, 2), CT(3, -1)};
        const std::vector<CT> b     = {CT(2, 1), CT(-1, 4)};
        const std::vector<CT> c     = {CT(1, -1)};
        const CT              alpha = CT(0, 1);
        const CT              beta  = CT(2, -3);

        std::vector<CT> results;
        for(bool conjA : {false, true})
        {
            for(bool conjB : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "conjA " << conjA << ", conjB " << conjB);
                auto     opA = [&](CT v) { return conjA ? std::conj(v) : v; };
                auto     opB = [&](CT v) { return conjB ? std::conj(v) : v; };
                const CT expected
                    = alpha * (opA(a[0]) * opB(b[0]) + opA(a[1]) * opB(b[1])) + beta * c[0];

                auto [dFast, dSlow] = solveComplexFastAndSlow(
                    makeComplexProblem(type, 1, 1, 2, conjA, conjB), a, b, c, alpha, beta);
                EXPECT_EQ(dFast[0], expected);
                EXPECT_EQ(dSlow[0], expected);
                results.push_back(expected);
            }
        }
        // The four conjugation combinations give four different results.
        for(size_t i = 0; i < results.size(); ++i)
            for(size_t j = i + 1; j < results.size(); ++j)
                EXPECT_NE(results[i], results[j]);
    }

    template <typename T>
    void checkComplexInPlaceCD(rocisa::DataType type)
    {
        using CT = std::complex<T>;
        struct Case
        {
            CT a, b, c, alpha, beta, expected;
        };
        const std::vector<Case> cases = {
            // Newling's counterexample: the original C contributes (5, 1).
            {CT(2, 0), CT(3, 0), CT(5, 1), CT(1, 0), CT(1, 0), CT(11, 1)},
            // Nonzero imaginary inputs and scalars: alpha * AB is (7, -9),
            // beta * the original C is (9, 7), and the result is (16, -2).
            {CT(2, 1), CT(3, -2), CT(5, 1), CT(1, -1), CT(2, 1), CT(16, -2)},
        };
        auto problem = makeComplexProblem(type, 1, 1, 1, false, false);
        ASSERT_TRUE(isFastPathEligible(problem));

        for(auto const& test : cases)
        {
            for(bool tryFastPath : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "A " << test.a << ", B " << test.b << ", C "
                                                << test.c << ", fast " << tryFastPath);
                std::vector<CT>   cd = {test.c};
                ContractionInputs inputs(
                    &test.a, &test.b, cd.data(), cd.data(), test.alpha, test.beta);
                SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, tryFastPath);
                EXPECT_EQ(cd[0], test.expected);
            }
        }
    }

    // Random (-1, 1) parts in A, B and C, complex alpha and beta, sizes that are not
    // multiples of the fast path's tiles: fast and slow D must be identical, for
    // each conjugation combination (both sum op(A) * op(B) in K order from zero and
    // apply alpha and beta the same way).
    template <typename T>
    void checkComplexFastMatchesSlow(rocisa::DataType type)
    {
        using CT       = std::complex<T>;
        const size_t M = 37, N = 45, K = 53;

        std::mt19937                      gen(7);
        std::uniform_real_distribution<T> dist(-1, 1);
        auto                              fill = [&](size_t n) {
            std::vector<CT> v(n);
            for(auto& x : v)
            {
                const T re = dist(gen);
                const T im = dist(gen);
                x          = CT(re, im);
            }
            return v;
        };
        const auto a = fill(M * K);
        const auto b = fill(K * N);
        const auto c = fill(M * N);

        for(bool conjA : {false, true})
        {
            for(bool conjB : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "conjA " << conjA << ", conjB " << conjB);
                auto [dFast, dSlow]
                    = solveComplexFastAndSlow(makeComplexProblem(type, M, N, K, conjA, conjB),
                                              a,
                                              b,
                                              c,
                                              CT(0.7, -1.3),
                                              CT(0.25, 0.5));
                size_t mismatches = 0;
                for(size_t i = 0; i < dFast.size(); ++i)
                    mismatches += (dFast[i] != dSlow[i]);
                EXPECT_EQ(mismatches, 0u);
            }
        }
    }
} // namespace

TEST(ReferenceFastPath, ComplexEligibilityFollowsReferenceTypeIds)
{
    using DT          = rocisa::DataType;
    const DT complexF = DT::ComplexFloat;
    const DT complexD = DT::ComplexDouble;

    for(DT t : {complexF, complexD})
    {
        SCOPED_TRACE(TensileLite::ToString(t));
        EXPECT_TRUE(isFastPathEligible(makeComplexProblem(t, 4, 4, 4, false, false)));
        EXPECT_TRUE(isFastPathEligible(makeComplexProblem(t, 4, 4, 4, true, true)));

        // Not C_C_C / Z_Z_Z.
        auto realAlpha = makeComplexProblem(t, 4, 4, 4, false, false);
        realAlpha.setAlphaType(DT::Float);
        realAlpha.setBetaType(DT::Float);
        EXPECT_FALSE(isFastPathEligible(realAlpha));
        EXPECT_FALSE(isFastPathEligible(makePackedProblem(t, t, DT::Float, 4, 4, 4)));
        EXPECT_FALSE(isFastPathEligible(makePackedProblem(DT::Float, DT::Float, t, 4, 4, 4)));

        // Epilogue inputs the slow path reads as zero (or, for activation, ignores)
        // for complex types stay on the slow path.
        auto bias = makeComplexProblem(t, 4, 4, 4, false, false);
        bias.setUseBias(1);
        bias.setBias(t, 4, 4);
        EXPECT_FALSE(isFastPathEligible(bias));
        for(const char* mode : {"Scalar", "Vector"})
        {
            auto scaleAB = makeComplexProblem(t, 4, 4, 4, false, false);
            scaleAB.setUseScaleAB(mode);
            EXPECT_FALSE(isFastPathEligible(scaleAB)) << "ScaleAB " << mode;
        }
        auto scaleAlphaVec = makeComplexProblem(t, 4, 4, 4, false, false);
        scaleAlphaVec.setUseScaleAlphaVec(1);
        scaleAlphaVec.setScaleAlphaVec(t, 4, /*factorDim=*/0);
        EXPECT_FALSE(isFastPathEligible(scaleAlphaVec));
        auto activation = makeComplexProblem(t, 4, 4, 4, false, false);
        activation.setActivationType(ActivationType::Relu);
        EXPECT_FALSE(isFastPathEligible(activation));
        auto gateResidual = makeComplexProblem(t, 4, 4, 4, false, false);
        gateResidual.setUseGateResidual(true);
        EXPECT_FALSE(isFastPathEligible(gateResidual));
    }
    EXPECT_FALSE(isFastPathEligible(makePackedProblem(complexF, complexD, complexD, 4, 4, 4)));
}

// TensorOp::ComplexConjugate on A and/or B conjugates that operand before the
// multiply (the slow path's Transform<T>::Input), on both paths.
TEST(ReferenceFastPath, ComplexConjugatesTheRequestedOperands)
{
    {
        SCOPED_TRACE("ComplexFloat");
        checkComplexConjugation<float>(rocisa::DataType::ComplexFloat);
    }
    {
        SCOPED_TRACE("ComplexDouble");
        checkComplexConjugation<double>(rocisa::DataType::ComplexDouble);
    }
}

TEST(ReferenceFastPath, ComplexInPlaceCDPreservesBetaInput)
{
    {
        SCOPED_TRACE("ComplexFloat");
        checkComplexInPlaceCD<float>(rocisa::DataType::ComplexFloat);
    }
    {
        SCOPED_TRACE("ComplexDouble");
        checkComplexInPlaceCD<double>(rocisa::DataType::ComplexDouble);
    }
}

TEST(ReferenceFastPath, ComplexMatchesSlowPathExactly)
{
    {
        SCOPED_TRACE("ComplexFloat");
        checkComplexFastMatchesSlow<float>(rocisa::DataType::ComplexFloat);
    }
    {
        SCOPED_TRACE("ComplexDouble");
        checkComplexFastMatchesSlow<double>(rocisa::DataType::ComplexDouble);
    }
}

namespace
{
    // Float M x N x K problem writing E (useE) of type typeE with strides strideE.
    ContractionProblemGemm makeUseEProblem(size_t                     M,
                                           size_t                     N,
                                           size_t                     K,
                                           rocisa::DataType           typeE,
                                           std::vector<size_t> const& stridesE)
    {
        using DT     = rocisa::DataType;
        auto problem = makePackedProblem(DT::Float, DT::Float, DT::Float, M, N, K);
        problem.setUseE(true);
        problem.setE(typeE, problem.d().sizes(), stridesE, /*isOutput=*/true);
        return problem;
    }
} // namespace

TEST(ReferenceFastPath, UseEEligibility)
{
    using DT = rocisa::DataType;
    EXPECT_TRUE(isFastPathEligible(makeUseEProblem(4, 4, 4, DT::Float, {})));
    EXPECT_TRUE(isFastPathEligible(makeUseEProblem(4, 4, 4, DT::Half, {})));

    // With useE the slow path requires equal alpha and beta types and throws
    // otherwise; the fast path leaves such problems to it. Without useE they are
    // eligible.
    auto mixedAlphaBeta = makeUseEProblem(4, 4, 4, DT::Float, {});
    mixedAlphaBeta.setBetaType(DT::Double);
    EXPECT_FALSE(isFastPathEligible(mixedAlphaBeta));
    std::vector<float> a(16, 1.0f), b(16, 1.0f), c(16, 0.0f), d(16, 0.0f), e(16, 0.0f);
    ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    inputs.e = e.data();
    EXPECT_THROW(SolveGemmCPU(mixedAlphaBeta, inputs, /*elementsToValidate=*/-1, true),
                 std::runtime_error);
    auto mixedAlphaBetaNoE = makePackedProblem(DT::Float, DT::Float, DT::Float, 4, 4, 4);
    mixedAlphaBetaNoE.setBetaType(DT::Double);
    EXPECT_TRUE(isFastPathEligible(mixedAlphaBetaNoE));
}

// E receives the value before activation (after bias), converted to E's own type
// (here Half, while D is Float) and stored at E's own strides (here a padded
// leading dimension), on both paths; D receives relu of that value.
TEST(ReferenceFastPath, UseEStoresPreActivationValue)
{
    using DT       = rocisa::DataType;
    const size_t M = 3, N = 2, K = 1, ldE = M + 2;
    auto problem   = makeUseEProblem(M, N, K, DT::Half, {1, ldE, ldE * N});
    problem.setUseBias(1);
    problem.setBias(DT::Float, M, M);
    problem.setActivationType(ActivationType::Relu);
    problem.setParams().setActivationEnum(ActivationType::Relu);
    ASSERT_EQ(problem.e().totalAllocatedElements(), ldE + M);

    // value(m, n) = a[m] * b[n] + bias[m]; 2049 is not a Half value (it rounds to
    // the even neighbour 2048).
    const std::vector<float> a    = {2049.0f, -3.0f, 0.5f};
    const std::vector<float> b    = {1.0f, -1.0f};
    const std::vector<float> bias = {0.0f, 1.0f, -2.0f};
    const std::vector<float> c(M * N, 0.0f);
    const std::vector<float> value = {2049.0f, -2.0f, -1.5f, -2049.0f, 4.0f, -2.5f};

    // E padding (rows M .. ldE - 1 of column 0) must keep its initial value.
    const Half sentinel(99.0f);

    const std::vector<float> expectedE
        = {2048.0f, -2.0f, -1.5f, 99.0f, 99.0f, -2048.0f, 4.0f, -2.5f};

    for(bool fast : {true, false})
    {
        SCOPED_TRACE(fast ? "fast path" : "slow path");
        if(fast)
            ASSERT_TRUE(isFastPathEligible(problem));
        std::vector<float> d(M * N, -1.0f);
        std::vector<Half>  e(problem.e().totalAllocatedElements(), sentinel);
        ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
        inputs.bias = bias.data();
        inputs.e    = e.data();
        SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/fast);

        for(size_t i = 0; i < M * N; ++i)
            EXPECT_EQ(d[i], std::max(value[i], 0.0f)) << "D element " << i;
        for(size_t i = 0; i < e.size(); ++i)
            EXPECT_EQ(static_cast<float>(e[i]), expectedE[i]) << "E element " << i;
    }
}

namespace
{
    // Float M x N x K gradient problem (useGradient) with batchCount batches.
    ContractionProblemGemm makeGradientProblem(
        size_t M, size_t N, size_t K, bool transA, bool transB, size_t batchCount = 1)
    {
        using DT     = rocisa::DataType;
        auto problem = ContractionProblemGemm::GEMM_Strides(transA,
                                                            transB,
                                                            DT::Float,
                                                            DT::Float,
                                                            DT::Float,
                                                            DT::Float,
                                                            M,
                                                            N,
                                                            K,
                                                            batchCount,
                                                            transA ? K : M,
                                                            -1,
                                                            transB ? N : K,
                                                            -1,
                                                            M,
                                                            -1,
                                                            M,
                                                            -1,
                                                            0.0);
        problem.setComputeInputTypeA(DT::Float);
        problem.setComputeInputTypeB(DT::Float);
        problem.setAlphaType(DT::Float);
        problem.setBetaType(DT::Float);
        problem.setUseGradient(true);
        return problem;
    }

    // Adds a Float bias gradient (an output) reduced from src: length N for B, else M.
    size_t setBiasGradient(ContractionProblemGemm& problem, ContractionProblemGemm::TENSOR src)
    {
        const size_t length
            = (src == ContractionProblemGemm::B) ? problem.freeSizeB(0) : problem.freeSizeA(0);
        problem.setUseBias(1);
        problem.setBias(rocisa::DataType::Float, length, length, /*isOutput=*/true, src);
        return length;
    }

    template <typename T>
    void checkBiasGradientAccumulation(rocisa::DataType type,
                                       rocisa::DataType scalarType,
                                       size_t           N,
                                       ConstantVariant  alpha,
                                       ConstantVariant  beta,
                                       bool             highPrecisionAccumulate,
                                       float            expectedBias)
    {
        auto problem = makePackedProblem(type, type, type, 1, N, 1);
        problem.setAlphaType(scalarType);
        problem.setBetaType(scalarType);
        problem.setHighPrecisionAccumulate(highPrecisionAccumulate);
        problem.setUseGradient(true);
        problem.setUseBias(1);
        problem.setBias(type, 1, 1, /*isOutput=*/true, ContractionProblemGemm::D);

        const std::vector<T> a(1, T(1.0f)), b(N, T(1.0f)), c(N, T(0.0f));
        for(bool fast : {true, false})
        {
            SCOPED_TRACE(fast ? "fast requested" : "slow requested");
            std::vector<T>    d(N, T(-1.0f)), bias(1, T(-1.0f));
            ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), alpha, beta);
            inputs.bias = bias.data();
            SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/fast);

            for(size_t i = 0; i < N; ++i)
                EXPECT_EQ(static_cast<float>(d[i]), 1.0f) << "D element " << i;
            EXPECT_EQ(static_cast<float>(bias[0]), expectedBias);
        }
        // Check eligibility after solving so the unfixed test exposes the numerical error.
        EXPECT_EQ(isFastPathEligible(problem), highPrecisionAccumulate);
    }
} // namespace

TEST(ReferenceFastPath, NonHpaBiasGradientKeepsNarrowAccumulation)
{
    using DT = rocisa::DataType;
    // Repeated additions of one stop changing at 2048 in Half and 256 in BFloat16.
    // The float reductions instead round once to 4096 and 512, respectively.
    {
        SCOPED_TRACE("Half");
        checkBiasGradientAccumulation<Half>(
            DT::Half, DT::Half, 4097, Half(1.0f), Half(0.0f), false, 2048.0f);
    }
    {
        SCOPED_TRACE("BFloat16");
        checkBiasGradientAccumulation<BFloat16>(
            DT::BFloat16, DT::Float, 513, 1.0f, 0.0f, false, 256.0f);
    }
}

TEST(ReferenceFastPath, HpaBiasGradientKeepsFloatAccumulation)
{
    using DT = rocisa::DataType;
    {
        SCOPED_TRACE("Half");
        checkBiasGradientAccumulation<Half>(
            DT::Half, DT::Half, 4097, Half(1.0f), Half(0.0f), true, 4096.0f);
    }
    {
        SCOPED_TRACE("BFloat16");
        checkBiasGradientAccumulation<BFloat16>(
            DT::BFloat16, DT::Float, 513, 1.0f, 0.0f, true, 512.0f);
    }
}

TEST(ReferenceFastPath, NarrowBiasGradientEligibilityRequiresHpa)
{
    using DT = rocisa::DataType;
    for(auto [type, scalarType] :
        {std::pair{DT::Half, DT::Half}, std::pair{DT::BFloat16, DT::Float}})
    {
        SCOPED_TRACE(TensileLite::ToString(type));
        for(auto src :
            {ContractionProblemGemm::A, ContractionProblemGemm::B, ContractionProblemGemm::D})
        {
            SCOPED_TRACE("bias source " + std::to_string(src));
            auto problem = makePackedProblem(type, type, type, 4, 4, 4);
            problem.setAlphaType(scalarType);
            problem.setBetaType(scalarType);
            problem.setUseGradient(true);
            problem.setUseBias(1);
            problem.setBias(type, 4, 4, /*isOutput=*/true, src);
            problem.setHighPrecisionAccumulate(true);
            EXPECT_TRUE(isFastPathEligible(problem));

            problem.setHighPrecisionAccumulate(false);
            EXPECT_FALSE(isFastPathEligible(problem));

            problem.setUseBias(0);
            EXPECT_TRUE(isFastPathEligible(problem));
            problem.setUseBias(1);
            problem.setUseGradient(false);
            EXPECT_TRUE(isFastPathEligible(problem));
        }
    }
}

TEST(ReferenceFastPath, GradientEligibility)
{
    using DT = rocisa::DataType;
    EXPECT_TRUE(isFastPathEligible(makeGradientProblem(4, 4, 4, false, false)));
    EXPECT_TRUE(isFastPathEligible(makeGradientProblem(4, 4, 4, false, false, 2)));

    auto withE = makeGradientProblem(4, 4, 4, false, false);
    withE.setUseE(true);
    withE.setE(DT::Half, withE.d().sizes(), {}, /*isOutput=*/false);
    EXPECT_TRUE(isFastPathEligible(withE));

    for(auto src :
        {ContractionProblemGemm::A, ContractionProblemGemm::B, ContractionProblemGemm::D})
    {
        SCOPED_TRACE("bias source " + std::to_string(src));
        auto single = makeGradientProblem(4, 4, 4, true, true);
        setBiasGradient(single, src);
        EXPECT_TRUE(isFastPathEligible(single));

        // ReductionCPU reduces one batch only; the slow path keeps the others.
        auto batched = makeGradientProblem(4, 4, 4, false, false, 2);
        setBiasGradient(batched, src);
        EXPECT_FALSE(isFastPathEligible(batched));
    }
}

// Gradient on both paths: D = (alpha * A * B + beta * C) * activation(E) with E an
// input (Half, at its own padded strides) that is left unchanged, and no bias added;
// then gate residual (D = gate * D + gate, Half gate at padded strides). The bias
// gradient is reduced from D (after the activation gradient, before gate residual), A
// or B. The data are small integers (E: halves; gate: -1, 0.5, 2), so the paths agree
// bit for bit.
TEST(ReferenceFastPath, GradientMatchesSlowPathExactly)
{
    using DT = rocisa::DataType;
    // Partial 32 x 32 tiles; K is not a multiple of the fast path's K block.
    const size_t                       M = 37, N = 35, K = 19, ldE = M + 3;
    std::mt19937                       gen(7);
    std::uniform_int_distribution<int> dist(-3, 3);
    auto                               fill = [&](auto& vec, float scale) {
        using T = typename std::decay_t<decltype(vec)>::value_type;
        for(auto& v : vec)
            v = static_cast<T>(static_cast<float>(dist(gen)) * scale);
    };

    for(auto act : {ActivationType::DGelu, ActivationType::Relu, ActivationType::DRelu})
    {
        for(auto src :
            {ContractionProblemGemm::A, ContractionProblemGemm::B, ContractionProblemGemm::D})
        {
            for(int trans = 0; trans < 4; ++trans)
            {
                const bool transA = trans & 1, transB = trans & 2;
                SCOPED_TRACE(TensileLite::ToString(act) + ", bias source " + std::to_string(src)
                             + ", transA " + std::to_string(transA) + ", transB "
                             + std::to_string(transB));
                auto problem = makeGradientProblem(M, N, K, transA, transB);
                problem.setActivationType(ActivationType::All);
                problem.setParams().setActivationEnum(act);
                problem.setUseE(true);
                problem.setE(DT::Half, problem.d().sizes(), {1, ldE, ldE * N}, false);
                problem.setUseGateResidual(true);
                problem.setGateResidual(DT::Half, problem.d().sizes(), {1, ldE, ldE * N});
                const size_t biasLength = setBiasGradient(problem, src);
                ASSERT_TRUE(isFastPathEligible(problem));

                std::vector<float> a(M * K), b(K * N), c(M * N);
                std::vector<Half>  e(problem.e().totalAllocatedElements());
                std::vector<Half>  gate(problem.gateResidual().totalAllocatedElements());
                fill(a, 1.0f);
                fill(b, 1.0f);
                fill(c, 1.0f);
                fill(e, 0.5f);
                const float gateValues[] = {-1.0f, 0.5f, 2.0f};
                for(size_t i = 0; i < gate.size(); ++i)
                    gate[i] = Half(gateValues[i % 3]);

                std::vector<float> d[2], bias[2];
                for(bool fast : {true, false})
                {
                    std::vector<float> dRun(M * N, -1.0f), biasRun(biasLength, -1.0f);
                    std::vector<Half>  eRun = e;
                    ContractionInputs  inputs(a.data(), b.data(), c.data(), dRun.data(), 2.0f, 0.5f);
                    inputs.bias         = biasRun.data();
                    inputs.e            = eRun.data();
                    inputs.gateResidual = gate.data();
                    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/fast);
                    EXPECT_EQ(0, std::memcmp(eRun.data(), e.data(), e.size() * sizeof(Half)))
                        << (fast ? "fast" : "slow") << " path changed E";
                    d[fast]    = dRun;
                    bias[fast] = biasRun;
                }
                EXPECT_EQ(d[true], d[false]);
                EXPECT_EQ(bias[true], bias[false]);
            }
        }
    }
}

// The bias gradient follows elementsToValidate as ReductionCPU does, on both paths:
// with fewer elements to validate than bias elements (but enough for the fast path),
// only every NextPrime(M / elementsToValidate)-th bias element is written.
TEST(ReferenceFastPath, GradientBiasFollowsElementsToValidate)
{
    // 5 of 20 elements: the fast path takes >= 20%; bias stride NextPrime(10 / 5) = 2.
    const size_t M = 10, N = 2, K = 3, elementsToValidate = 5;
    for(auto src : {ContractionProblemGemm::A, ContractionProblemGemm::D})
    {
        SCOPED_TRACE("bias source " + std::to_string(src));
        auto problem = makeGradientProblem(M, N, K, false, false);
        setBiasGradient(problem, src);
        ASSERT_TRUE(isFastPathEligible(problem));

        std::vector<float> a(M * K), b(K * N, 1.0f), c(M * N, 0.0f);
        for(size_t i = 0; i < a.size(); ++i)
            a[i] = static_cast<float>(i % 7) - 3.0f;

        std::vector<float> d[2], bias[2];
        for(bool fast : {true, false})
        {
            std::vector<float> dRun(M * N, -1.0f), biasRun(M, 99.0f);
            ContractionInputs  inputs(a.data(), b.data(), c.data(), dRun.data(), 1.0f, 0.0f);
            inputs.bias = biasRun.data();
            SolveGemmCPU(problem, inputs, elementsToValidate, /*tryFastPath=*/fast);
            for(size_t i = 0; i < M; ++i)
                EXPECT_EQ(biasRun[i] == 99.0f, i % 2 == 1)
                    << (fast ? "fast" : "slow") << " path, bias element " << i;
            d[fast]    = dRun;
            bias[fast] = biasRun;
        }
        EXPECT_EQ(bias[true], bias[false]);
        // From D the slow path computes every D element as well.
        if(src == ContractionProblemGemm::D)
            EXPECT_EQ(d[true], d[false]);
    }
}
