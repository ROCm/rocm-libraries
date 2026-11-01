/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Regression test for the f64 unary operators of the elementwise operations and reductions.
// Earlier releases evaluated them with Composable Kernel's ck::math functions, whose double
// overloads of exp and log call themselves and whose double sqrt is the hardware approximation:
// EXP, LOG and SIGMOID returned Inf or NaN and SQRT was only accurate to about 1e-8. Every unary
// operator is checked here, in f64, against the host's long double math library, with a bound of
// 64 units in the last place of the terms (f32-level errors are about a million times larger).

#include "api_test_helpers.hpp"

#include <cfloat>
#include <cmath>
#include <string>

using hiptensor_test::ApiObjects;
using hiptensor_test::DeviceBuffer;

namespace
{
    constexpr int64_t M = 32, N = 24, count = M * N;

    struct UnaryCase
    {
        hiptensorOperator_t op;
        char const*         name;
        double              lo, hi; // input range where the operator is finite
    };

    UnaryCase const unaryCases[] = {
        {HIPTENSOR_OP_IDENTITY, "IDENTITY", -2, 2}, {HIPTENSOR_OP_SQRT, "SQRT", 0.25, 2},
        {HIPTENSOR_OP_RELU, "RELU", -2, 2},         {HIPTENSOR_OP_RCP, "RCP", 0.25, 2},
        {HIPTENSOR_OP_SIGMOID, "SIGMOID", -2, 2},   {HIPTENSOR_OP_TANH, "TANH", -2, 2},
        {HIPTENSOR_OP_EXP, "EXP", -2, 2},           {HIPTENSOR_OP_LOG, "LOG", 0.25, 2},
        {HIPTENSOR_OP_ABS, "ABS", -2, 2},           {HIPTENSOR_OP_NEG, "NEG", -2, 2},
        {HIPTENSOR_OP_SIN, "SIN", -2, 2},           {HIPTENSOR_OP_COS, "COS", -2, 2},
        {HIPTENSOR_OP_TAN, "TAN", -1, 1},           {HIPTENSOR_OP_SINH, "SINH", -2, 2},
        {HIPTENSOR_OP_COSH, "COSH", -2, 2},         {HIPTENSOR_OP_ASIN, "ASIN", -0.9, 0.9},
        {HIPTENSOR_OP_ACOS, "ACOS", -0.9, 0.9},     {HIPTENSOR_OP_ATAN, "ATAN", -2, 2},
        {HIPTENSOR_OP_ASINH, "ASINH", -2, 2},       {HIPTENSOR_OP_ACOSH, "ACOSH", 1.1, 3},
        {HIPTENSOR_OP_ATANH, "ATANH", -0.9, 0.9},   {HIPTENSOR_OP_CEIL, "CEIL", -2.5, 2.5},
        {HIPTENSOR_OP_FLOOR, "FLOOR", -2.5, 2.5},
    };

    long double reference(hiptensorOperator_t op, long double x)
    {
        switch(op)
        {
        case HIPTENSOR_OP_IDENTITY:
            return x;
        case HIPTENSOR_OP_SQRT:
            return sqrtl(x);
        case HIPTENSOR_OP_RELU:
            return x > 0 ? x : 0;
        case HIPTENSOR_OP_RCP:
            return 1.0L / x;
        case HIPTENSOR_OP_SIGMOID:
            return 1.0L / (1.0L + expl(-x));
        case HIPTENSOR_OP_TANH:
            return tanhl(x);
        case HIPTENSOR_OP_EXP:
            return expl(x);
        case HIPTENSOR_OP_LOG:
            return logl(x);
        case HIPTENSOR_OP_ABS:
            return fabsl(x);
        case HIPTENSOR_OP_NEG:
            return -x;
        case HIPTENSOR_OP_SIN:
            return sinl(x);
        case HIPTENSOR_OP_COS:
            return cosl(x);
        case HIPTENSOR_OP_TAN:
            return tanl(x);
        case HIPTENSOR_OP_SINH:
            return sinhl(x);
        case HIPTENSOR_OP_COSH:
            return coshl(x);
        case HIPTENSOR_OP_ASIN:
            return asinl(x);
        case HIPTENSOR_OP_ACOS:
            return acosl(x);
        case HIPTENSOR_OP_ATAN:
            return atanl(x);
        case HIPTENSOR_OP_ASINH:
            return asinhl(x);
        case HIPTENSOR_OP_ACOSH:
            return acoshl(x);
        case HIPTENSOR_OP_ATANH:
            return atanhl(x);
        case HIPTENSOR_OP_CEIL:
            return ceill(x);
        case HIPTENSOR_OP_FLOOR:
            return floorl(x);
        default:
            ADD_FAILURE() << "unexpected operator " << op;
            return 0;
        }
    }

    // Deterministic values spread over [lo, hi].
    std::vector<double> inputs(double lo, double hi, int salt)
    {
        std::vector<double> v(count);
        for(int64_t i = 0; i < count; ++i)
        {
            v[i] = lo + (hi - lo) * double((i * 7919 + salt * 104729) % 1000 + 0.5) / 1000.0;
        }
        return v;
    }

    // |actual - expected| <= 64 ulp of `scale`, the sum of the magnitudes of the terms.
    void expectClose(std::vector<double> const&      actual,
                     std::vector<long double> const& expected,
                     std::vector<long double> const& scale,
                     std::string const&              what)
    {
        int64_t bad = 0;
        for(int64_t i = 0; i < count; ++i)
        {
            long double tolerance = 64.0L * DBL_EPSILON * scale[i];
            if(!std::isfinite(actual[i]) || fabsl(actual[i] - expected[i]) > tolerance)
            {
                if(bad++ < 3)
                {
                    ADD_FAILURE() << what << ": element " << i << " is " << actual[i]
                                  << ", expected " << double(expected[i]);
                }
            }
        }
        EXPECT_EQ(bad, 0) << what;
    }
} // namespace

TEST(ElementwiseF64UnaryOpTest, BinaryTrinaryAndPermutation)
{
    ApiObjects api;
    if(!api.supports(HIPTENSOR_R_64F))
    {
        GTEST_SKIP() << "No f64 support on this device";
    }
    auto          desc    = api.tensor({M, N}, {1, M}, HIPTENSOR_R_64F);
    int32_t const modes[] = {'m', 'n'};

    auto const           b = inputs(-2, 2, 1), c = inputs(-2, 2, 2);
    DeviceBuffer<double> dB(b), dC(c), dD(std::vector<double>(count, NAN));
    double const         alpha = 1.25, beta = 0.5, gamma = 0.75;

    for(auto const& u : unaryCases)
    {
        auto const           a = inputs(u.lo, u.hi, 0);
        DeviceBuffer<double> dA(a);

        std::vector<long double> opA(count);
        for(int64_t i = 0; i < count; ++i)
        {
            opA[i] = reference(u.op, a[i]);
        }

        // Binary: D = alpha * op(A) + gamma * C.
        {
            hiptensorOperationDescriptor_t op = nullptr;
            ASSERT_EQ(hiptensorCreateElementwiseBinary(api.handle,
                                                       &op,
                                                       desc,
                                                       modes,
                                                       u.op,
                                                       desc,
                                                       modes,
                                                       HIPTENSOR_OP_IDENTITY,
                                                       desc,
                                                       modes,
                                                       HIPTENSOR_OP_ADD,
                                                       HIPTENSOR_COMPUTE_DESC_64F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(op);
            hiptensorPlan_t plan = nullptr;
            ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);
            dD.write(std::vector<double>(count, NAN));
            ASSERT_EQ(hiptensorElementwiseBinaryExecute(
                          api.handle, plan, &alpha, dA.ptr, &gamma, dC.ptr, dD.ptr, nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            std::vector<long double> expected(count), scale(count);
            for(int64_t i = 0; i < count; ++i)
            {
                expected[i] = 1.25L * opA[i] + 0.75L * c[i];
                scale[i]    = fabsl(1.25L * opA[i]) + fabsl(0.75L * c[i]);
            }
            expectClose(dD.read(), expected, scale, std::string("binary ") + u.name);
        }

        // Trinary: D = (alpha * op(A) + beta * B) + gamma * C.
        {
            hiptensorOperationDescriptor_t op = nullptr;
            ASSERT_EQ(hiptensorCreateElementwiseTrinary(api.handle,
                                                        &op,
                                                        desc,
                                                        modes,
                                                        u.op,
                                                        desc,
                                                        modes,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        desc,
                                                        modes,
                                                        HIPTENSOR_OP_IDENTITY,
                                                        desc,
                                                        modes,
                                                        HIPTENSOR_OP_ADD,
                                                        HIPTENSOR_OP_ADD,
                                                        HIPTENSOR_COMPUTE_DESC_64F),
                      HIPTENSOR_STATUS_SUCCESS);
            api.keep(op);
            hiptensorPlan_t plan = nullptr;
            ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);
            dD.write(std::vector<double>(count, NAN));
            ASSERT_EQ(hiptensorElementwiseTrinaryExecute(api.handle,
                                                         plan,
                                                         &alpha,
                                                         dA.ptr,
                                                         &beta,
                                                         dB.ptr,
                                                         &gamma,
                                                         dC.ptr,
                                                         dD.ptr,
                                                         nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            std::vector<long double> expected(count), scale(count);
            for(int64_t i = 0; i < count; ++i)
            {
                expected[i] = 1.25L * opA[i] + 0.5L * b[i] + 0.75L * c[i];
                scale[i]    = fabsl(1.25L * opA[i]) + fabsl(0.5L * b[i]) + fabsl(0.75L * c[i]);
            }
            expectClose(dD.read(), expected, scale, std::string("trinary ") + u.name);
        }

        // Permutation: D = alpha * op(A), with the modes swapped.
        {
            auto                           descT    = api.tensor({N, M}, {1, N}, HIPTENSOR_R_64F);
            int32_t const                  modesT[] = {'n', 'm'};
            hiptensorOperationDescriptor_t op       = nullptr;
            ASSERT_EQ(
                hiptensorCreatePermutation(
                    api.handle, &op, desc, modes, u.op, descT, modesT, HIPTENSOR_COMPUTE_DESC_64F),
                HIPTENSOR_STATUS_SUCCESS);
            api.keep(op);
            hiptensorPlan_t plan = nullptr;
            ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);
            dD.write(std::vector<double>(count, NAN));
            ASSERT_EQ(hiptensorPermute(api.handle, plan, &alpha, dA.ptr, dD.ptr, nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            std::vector<long double> expected(count), scale(count);
            for(int64_t m = 0; m < M; ++m)
            {
                for(int64_t n = 0; n < N; ++n)
                {
                    expected[n + N * m] = 1.25L * opA[m + M * n];
                    scale[n + N * m]    = fabsl(expected[n + N * m]);
                }
            }
            expectClose(dD.read(), expected, scale, std::string("permutation ") + u.name);
        }
    }
}

// D = alpha * sum_n op(A[m, n]) + beta * C, in f64.
TEST(ElementwiseF64UnaryOpTest, Reduction)
{
    ApiObjects api;
    if(!api.supports(HIPTENSOR_R_64F))
    {
        GTEST_SKIP() << "No f64 support on this device";
    }
    auto          descA    = api.tensor({M, N}, {1, M}, HIPTENSOR_R_64F);
    auto          descCD   = api.tensor({M}, {1}, HIPTENSOR_R_64F);
    int32_t const modeA[]  = {'m', 'n'};
    int32_t const modeCD[] = {'m'};

    std::vector<double> c(M);
    for(int64_t m = 0; m < M; ++m)
    {
        c[m] = double(m % 9) - 4.0;
    }
    DeviceBuffer<double> dC(c), dD(std::vector<double>(M, NAN));
    double const         alpha = 1.25, beta = 0.5;

    for(auto const& u : unaryCases)
    {
        auto const           a = inputs(u.lo, u.hi, 3);
        DeviceBuffer<double> dA(a);

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &op,
                                           descA,
                                           modeA,
                                           u.op,
                                           descCD,
                                           modeCD,
                                           HIPTENSOR_OP_IDENTITY,
                                           descCD,
                                           modeCD,
                                           HIPTENSOR_OP_ADD,
                                           HIPTENSOR_COMPUTE_DESC_64F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        hiptensorPlan_t plan = nullptr;
        ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);
        dD.write(std::vector<double>(M, NAN));
        ASSERT_EQ(hiptensorReduce(
                      api.handle, plan, &alpha, dA.ptr, &beta, dC.ptr, dD.ptr, nullptr, 0, nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        CHECK_HIP_ERROR(hipDeviceSynchronize());

        auto const actual = dD.read();
        int64_t    bad    = 0;
        for(int64_t m = 0; m < M; ++m)
        {
            long double sum = 0, magnitude = 0;
            for(int64_t n = 0; n < N; ++n)
            {
                auto v = reference(u.op, a[m + M * n]);
                sum += v;
                magnitude += fabsl(v);
            }
            long double expected  = 1.25L * sum + 0.5L * c[m];
            long double tolerance = 64.0L * DBL_EPSILON * (1.25L * magnitude + fabsl(0.5L * c[m]));
            if(!std::isfinite(actual[m]) || fabsl(actual[m] - expected) > tolerance)
            {
                if(bad++ < 3)
                {
                    ADD_FAILURE() << "reduction " << u.name << ": m = " << m << " is " << actual[m]
                                  << ", expected " << double(expected);
                }
            }
        }
        EXPECT_EQ(bad, 0) << "reduction " << u.name;
    }
}
