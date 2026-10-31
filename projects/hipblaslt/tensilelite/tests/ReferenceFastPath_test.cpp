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
#include <random>
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
// sum is a float value; accumulating the same products in float, as the slow path
// does, gives 16,790,288 (4 short): the 1041st product rounds 16,790,289 down to
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

    std::vector<int32_t> c(1, 0);
    std::vector<int32_t> d(1, 0);
    ASSERT_TRUE(isFastPathEligible(problem));
    ContractionInputs inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
    SolveGemmCPU(problem, inputs, /*elementsToValidate=*/-1, /*tryFastPath=*/true);

    EXPECT_EQ(d[0], exactSum);
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
        EXPECT_TRUE(isFastPathEligible(
            makeFloat8OutputProblem(t.a, t.b, t.cd, t.computeA, t.computeB, 4, 4, 4)))
            << TensileLite::ToString(t.a) << " x " << TensileLite::ToString(t.b) << " -> "
            << TensileLite::ToString(t.cd) << ", MAC " << TensileLite::ToString(t.computeA)
            << " x " << TensileLite::ToString(t.computeB);
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
    // A = B = 1 (K = 1, N = 1) with scaleAlphaVec[m] = cases[m].first, so the float
    // value before the output cast is exactly cases[m].first on both paths. Checks
    // that the Float8/BFloat8 output is cases[m].second on the fast path, and the
    // same as on the slow path (SaturateCast).
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
            EXPECT_EQ(static_cast<float>(dFast[m]), cases[m].second) << "value " << cases[m].first;
            EXPECT_EQ(static_cast<float>(dFast[m]), static_cast<float>(dSlow[m]))
                << "value " << cases[m].first;
        }
    }
} // namespace

// Float8/BFloat8 outputs round to nearest even (ties between two F8 values, also
// subnormal ones) and saturate to the largest finite value (including values that
// round up past it). fnuz types have no negative zero: a tiny negative value is +0.
TEST(ReferenceFastPath, Float8OutputRoundsToNearestEvenAndSaturates)
{
    const float tiny = std::ldexp(1.0f, -20);
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
