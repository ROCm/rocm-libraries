// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Reference.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <tuple>
#include <vector>

using namespace TensileLite;
using namespace TensileLite::Client;

namespace
{
    template <typename Input, typename Output, typename Accum>
    ContractionProblemGemm makeAmaxProblem(size_t M, size_t N, size_t batches)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            TypeInfo<Input>::Enum,
                                                            TypeInfo<Input>::Enum,
                                                            TypeInfo<Output>::Enum,
                                                            TypeInfo<Output>::Enum,
                                                            M,
                                                            N,
                                                            1,
                                                            batches,
                                                            M,
                                                            M,
                                                            1,
                                                            N,
                                                            M,
                                                            M * N,
                                                            M,
                                                            M * N,
                                                            0.0);
        problem.setComputeInputTypeA(TypeInfo<Input>::Enum);
        problem.setComputeInputTypeB(TypeInfo<Input>::Enum);
        problem.setAlphaType(TypeInfo<Accum>::Enum);
        problem.setBetaType(TypeInfo<Accum>::Enum);
        problem.setHighPrecisionAccumulate(true);
        problem.setOutputAmaxD(true);
        problem.setAmaxD(TypeInfo<Accum>::Enum, true);
        return problem;
    }

    // Independent integer-valued products avoid reduction-order rounding.
    // The final batch has the greatest magnitude, and 33x35 crosses both
    // fast tile edges. 1x1 also exercises a team with fewer tiles than workers.
    template <typename Input, typename Output, typename Accum>
    void checkAmaxProducts()
    {
        for(auto [M, N, batches] : {std::tuple{size_t(1), size_t(1), size_t(1)},
                                    std::tuple{size_t(33), size_t(35), size_t(3)}})
        {
            auto problem = makeAmaxProblem<Input, Output, Accum>(M, N, batches);
            ASSERT_TRUE(isFastPathEligible(problem));
            std::vector<Input>  a(M * batches), b(N * batches);
            std::vector<Output> c(M * N * batches, Output(0));
            for(size_t batch = 0; batch < batches; ++batch)
            {
                for(size_t m = 0; m < M; ++m)
                    a[batch * M + m] = Input((int(m % 5) - 2) * int(batch + 1));
                for(size_t n = 0; n < N; ++n)
                    b[batch * N + n] = Input(n % 2 ? -1 : 2);
            }
            for(bool fast : {false, true})
            {
                SCOPED_TRACE(testing::Message() << "fast " << fast << ", batches " << batches);
                std::vector<Output> d(c.size(), Output(-1));
                Accum               amax = Accum(-1), expectedAmax = Accum(0);
                ContractionInputs   inputs(
                    a.data(), b.data(), c.data(), d.data(), Accum(1), Accum(0));
                inputs.amaxD = &amax;
                SolveGemmCPU(problem, inputs, -1, fast);
                for(size_t batch = 0; batch < batches; ++batch)
                    for(size_t n = 0; n < N; ++n)
                        for(size_t m = 0; m < M; ++m)
                        {
                            Accum value
                                = Accum((int(m % 5) - 2) * int(batch + 1) * (n % 2 ? -1 : 2));
                            expectedAmax
                                = std::max(expectedAmax, value > Accum(0) ? value : -value);
                            EXPECT_EQ(double(d[batch * M * N + n * M + m]), double(Output(value)));
                        }
                EXPECT_EQ(amax, expectedAmax);
            }
        }
    }

    template <typename Input, typename Scalar>
    void checkNonHpaAmaxRounding()
    {
        auto problem = makeAmaxProblem<Input, Input, Scalar>(1, 1, 1);
        problem.setAmaxD(rocisa::DataType::Float, true);
        EXPECT_TRUE(isFastPathEligible(problem));
        problem.setHighPrecisionAccumulate(false);
        EXPECT_FALSE(isFastPathEligible(problem));

        // (1 + 2^-7)^2 = 1 + 2^-6 + 2^-14. The final term survives
        // in float but is lost in both the Half and BFloat16 epilogues.
        Input a(1.0078125f), b(1.0078125f), c(0.0f);
        for(bool fast : {false, true})
        {
            SCOPED_TRACE(testing::Message() << "tryFastPath " << fast);
            Input             d(-1.0f);
            float             amax = -1.0f;
            ContractionInputs inputs(&a, &b, &c, &d, Scalar(1), Scalar(0));
            inputs.amaxD = &amax;
            SolveGemmCPU(problem, inputs, -1, fast);
            EXPECT_EQ(float(d), 1.015625f);
            EXPECT_EQ(amax, 1.015625f);
        }
    }
}

TEST(ReferenceAmaxD, FloatProductsAndBatches)
{
    checkAmaxProducts<float, float, float>();
}

TEST(ReferenceAmaxD, DoubleProductsAndBatches)
{
    checkAmaxProducts<double, double, double>();
}

TEST(ReferenceAmaxD, HalfAndBFloat16ProductsAndBatches)
{
    checkAmaxProducts<Half, Half, float>();
    checkAmaxProducts<BFloat16, BFloat16, float>();
}

TEST(ReferenceAmaxD, Int8ProductsAndBatches)
{
    checkAmaxProducts<int8_t, int32_t, int32_t>();
    checkAmaxProducts<int8_t, int32_t, float>();
    checkAmaxProducts<int8_t, int8_t, float>();
}

TEST(ReferenceAmaxD, Float8ProductsAndBatches)
{
    checkAmaxProducts<Float8_fnuz, Float8_fnuz, float>();
    checkAmaxProducts<BFloat8_fnuz, BFloat8_fnuz, float>();
    checkAmaxProducts<Float8, Float8, float>();
    checkAmaxProducts<BFloat8, BFloat8, float>();
}

TEST(ReferenceAmaxD, ActivationPrecedesAmaxAndGateFollowsIt)
{
    auto problem = makeAmaxProblem<float, float, float>(3, 1, 1);
    problem.setActivationType(ActivationType::Relu);
    problem.setParams().setActivationEnum(ActivationType::Relu);
    problem.setUseGateResidual(true);
    problem.setGateResidual(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides());
    const std::vector<float> a = {-4, 2, 3}, b = {1}, c = {0, 2, -2}, gate = {10, 10, 10};
    // alpha=2, beta=1: pre-activation {-8,6,4}, post-activation {0,6,4}.
    // Gate then maps these to {10,70,50}; AMAX must be 6, not 8 or 70.
    for(bool fast : {false, true})
    {
        ASSERT_TRUE(isFastPathEligible(problem));
        std::vector<float> d(3, -1);
        float              amax = -1;
        ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 2.0f, 1.0f);
        inputs.gateResidual = gate.data();
        inputs.amaxD        = &amax;
        SolveGemmCPU(problem, inputs, -1, fast);
        EXPECT_EQ(d, (std::vector<float>{10, 70, 50}));
        EXPECT_EQ(amax, 6.0f);
    }
}

TEST(ReferenceAmaxD, GradientPrecedesAmax)
{
    auto problem = makeAmaxProblem<float, float, float>(3, 1, 1);
    problem.setUseGradient(true);
    problem.setUseE(true);
    problem.setE(rocisa::DataType::Float, problem.d().sizes(), problem.d().strides(), false);
    problem.setActivationType(ActivationType::Relu);
    problem.setParams().setActivationEnum(ActivationType::Relu);
    const std::vector<float> a = {-4, 2, 3}, b = {1}, c(3, 0), e = {-1, 3, 2};
    for(bool fast : {false, true})
    {
        ASSERT_TRUE(isFastPathEligible(problem));
        std::vector<float> d(3, -1);
        float              amax = -1;
        ContractionInputs  inputs(a.data(), b.data(), c.data(), d.data(), 1.0f, 0.0f);
        inputs.e     = const_cast<float*>(e.data());
        inputs.amaxD = &amax;
        SolveGemmCPU(problem, inputs, -1, fast);
        EXPECT_EQ(d, (std::vector<float>{0, 6, 6}));
        EXPECT_EQ(amax, 6.0f);
    }
}

TEST(ReferenceAmaxD, MaximumPrecedesOutputNarrowing)
{
    auto        problem = makeAmaxProblem<Float8_fnuz, Float8_fnuz, float>(1, 1, 1);
    Float8_fnuz a(1.25f), b(1.375f), c(0.0f);
    for(bool fast : {false, true})
    {
        ASSERT_TRUE(isFastPathEligible(problem));
        Float8_fnuz       d(-1.0f);
        float             amax = -1;
        ContractionInputs inputs(&a, &b, &c, &d, 1.0f, 0.0f);
        inputs.amaxD = &amax;
        SolveGemmCPU(problem, inputs, -1, fast);
        EXPECT_EQ(float(d), 1.75f);
        EXPECT_EQ(amax, 1.71875f);
    }
}

TEST(ReferenceAmaxD, NonHpaHalfAndBFloat16PreserveEpilogueRounding)
{
    {
        SCOPED_TRACE("Half");
        checkNonHpaAmaxRounding<Half, Half>();
    }
    {
        SCOPED_TRACE("BFloat16");
        checkNonHpaAmaxRounding<BFloat16, float>();
    }
}

TEST(ReferenceAmaxD, ZeroAndNaNComparisonSemantics)
{
    auto  problem = makeAmaxProblem<float, float, float>(1, 1, 1);
    float a = 1, b = 1, c = 0;
    for(float alpha : {0.0f, std::numeric_limits<float>::quiet_NaN()})
        for(bool fast : {false, true})
        {
            ASSERT_TRUE(isFastPathEligible(problem));
            float             d = -1, amax = -1;
            ContractionInputs inputs(&a, &b, &c, &d, alpha, 0.0f);
            inputs.amaxD = &amax;
            SolveGemmCPU(problem, inputs, -1, fast);
            if(std::isnan(alpha))
                EXPECT_TRUE(std::isnan(d));
            else
                EXPECT_EQ(d, 0.0f);
            EXPECT_EQ(amax, 0.0f);
        }
}

TEST(ReferenceAmaxD, ComplexAndScaleCDRemainIneligible)
{
    auto complex
        = makeAmaxProblem<std::complex<float>, std::complex<float>, std::complex<float>>(1, 1, 1);
    EXPECT_FALSE(isFastPathEligible(complex));
    auto scaled = makeAmaxProblem<float, float, float>(1, 1, 1);
    scaled.setUseScaleCD(true);
    EXPECT_FALSE(isFastPathEligible(scaled));
}
