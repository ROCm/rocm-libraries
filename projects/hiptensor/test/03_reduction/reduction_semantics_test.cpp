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

// Regression tests for hiptensorReduce, D = alpha * reduce(opA(A)) + beta * opC(C):
//  - C was copied to D with a blocking hipMemcpy on the null stream, so it did not wait for work
//    on the caller's stream (a non-blocking stream that is still writing C), and it copied the
//    packed byte count, which for a padded or strided D copies the wrong elements and writes
//    between them;
//  - a reduction that keeps every mode of A changed the plan's operator and ran as an elementwise
//    operation, which has no rank-1 or BF16 kernels;
//  - with alpha == 0 or beta == 0, A or C were still read and transformed, so NaN or Inf in them
//    reached D.
// The expected results come from plain loops here. Inputs are small integers and the scalars
// dyadic, so the results are exact.

#include "api_test_helpers.hpp"

#include <hip/hip_bfloat16.h>

#include <algorithm>
#include <cmath>
#include <limits>

using hiptensor_test::ApiObjects;
using hiptensor_test::DeviceBuffer;

namespace
{
    // Spins for about `milliseconds` on the device clock, then writes `values` to `out`.
    __global__ void delayedWrite(float* out, float const* values, int count, uint64_t ticks)
    {
        if(blockIdx.x == 0 && threadIdx.x == 0)
        {
            uint64_t start = wall_clock64();
            while(wall_clock64() - start < ticks)
            {
                __builtin_amdgcn_s_sleep(100);
            }
            for(int i = 0; i < count; ++i)
            {
                out[i] = values[i];
            }
        }
    }

    double valueA(int64_t i)
    {
        return static_cast<double>(i % 7 - 3);
    }
    double valueC(int64_t i)
    {
        return static_cast<double>((5 * i) % 9 - 4);
    }
} // namespace

// D (rank 1, strided with gaps) = 1.5 * sum_n A[m, n] + 0.5 * C, where C is written on a
// non-blocking stream by a kernel that runs for a while, and the reduction is enqueued on the same
// stream right after it. D's gaps must keep their contents.
TEST(ReductionSemanticsTest, CopiesCOnTheCallersStreamAndKeepsDGaps)
{
    constexpr int64_t M = 6, N = 5, gap = 3;

    ApiObjects api;
    auto       descA  = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    auto       descCD = api.tensor({M}, {gap}, HIPTENSOR_R_32F);
    int32_t const modeA[] = {'m', 'n'}, modeCD[] = {'m'};

    hiptensorOperationDescriptor_t op = nullptr;
    ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                       &op,
                                       descA,
                                       modeA,
                                       HIPTENSOR_OP_IDENTITY,
                                       descCD,
                                       modeCD,
                                       HIPTENSOR_OP_IDENTITY,
                                       descCD,
                                       modeCD,
                                       HIPTENSOR_OP_ADD,
                                       HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    api.keep(op);
    hiptensorPlan_t plan = nullptr;
    ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

    std::vector<float> a(M * N);
    for(int64_t i = 0; i < M * N; ++i)
    {
        a[i] = static_cast<float>(valueA(i));
    }
    constexpr float    sentinel = -1000.0f;
    std::vector<float> staleC(M * gap, 77.0f), newC(M * gap, sentinel);
    for(int64_t m = 0; m < M; ++m)
    {
        newC[m * gap] = static_cast<float>(valueC(m));
    }
    DeviceBuffer<float> dA(a), dC(staleC), dNewC(newC);
    DeviceBuffer<float> dD(std::vector<float>(M * gap, sentinel));

    hipStream_t stream;
    CHECK_HIP_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    int rateKHz = 0;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(&rateKHz, hipDeviceAttributeWallClockRate, 0));
    uint64_t const ticks = uint64_t(rateKHz) * 200; // about 200 ms

    hipLaunchKernelGGL(delayedWrite,
                       dim3(1),
                       dim3(64),
                       0,
                       stream,
                       dC.ptr,
                       dNewC.ptr,
                       static_cast<int>(M * gap),
                       ticks);
    CHECK_HIP_ERROR(hipGetLastError());

    float const alpha = 1.5f, beta = 0.5f;
    ASSERT_EQ(
        hiptensorReduce(
            api.handle, plan, &alpha, dA.ptr, &beta, dC.ptr, dD.ptr, nullptr, 0, stream),
        HIPTENSOR_STATUS_SUCCESS);
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    CHECK_HIP_ERROR(hipStreamDestroy(stream));

    auto d = dD.read();
    for(int64_t i = 0; i < M * gap; ++i)
    {
        if(i % gap != 0)
        {
            EXPECT_EQ(d[i], sentinel) << "gap element " << i << " was written";
            continue;
        }
        int64_t m   = i / gap;
        double  sum = 0;
        for(int64_t n = 0; n < N; ++n)
        {
            sum += valueA(m + M * n);
        }
        EXPECT_EQ(d[i], 1.5 * sum + 0.5 * valueC(m)) << "m = " << m;
    }
}

// alpha == 0: D = beta * opC(C) without reading A (poisoned, then nullptr); beta == 0:
// D = alpha * reduce(A) without reading C or D's previous contents; both zero: D = 0.
TEST(ReductionSemanticsTest, ZeroScalesDoNotReadTheirTensors)
{
    constexpr int64_t M = 8, N = 6;

    ApiObjects    api;
    auto          descA  = api.tensor({M, N}, {1, M}, HIPTENSOR_R_32F);
    auto          descCD = api.tensor({M}, {1}, HIPTENSOR_R_32F);
    int32_t const modeA[] = {'m', 'n'}, modeCD[] = {'m'};

    for(auto opReduce : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX})
    {
        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &op,
                                           descA,
                                           modeA,
                                           HIPTENSOR_OP_LOG,
                                           descCD,
                                           modeCD,
                                           HIPTENSOR_OP_IDENTITY,
                                           descCD,
                                           modeCD,
                                           opReduce,
                                           HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        hiptensorPlan_t plan = nullptr;
        ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

        // LOG of 0 and -1 is -Inf and NaN.
        std::vector<float> poisonedA(M * N), powers(M * N), c(M);
        for(int64_t i = 0; i < M * N; ++i)
        {
            poisonedA[i] = (i % 3 == 0) ? 0.0f : ((i % 3 == 1) ? -1.0f : NAN);
            powers[i]    = static_cast<float>(1 << (i % 4));
        }
        for(int64_t m = 0; m < M; ++m)
        {
            c[m] = static_cast<float>(valueC(m));
        }
        DeviceBuffer<float> dBadA(poisonedA), dPowers(powers), dC(c),
            dBadC(std::vector<float>(M, NAN)), dD(std::vector<float>(M, NAN));

        float const zero = 0.0f, alpha = 2.0f, beta = 0.25f;

        for(float const* aPtr : {static_cast<float const*>(dBadA.ptr), static_cast<float const*>(nullptr)})
        {
            dD.write(std::vector<float>(M, NAN));
            ASSERT_EQ(hiptensorReduce(
                          api.handle, plan, &zero, aPtr, &beta, dC.ptr, dD.ptr, nullptr, 0, nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            auto d = dD.read();
            for(int64_t m = 0; m < M; ++m)
            {
                EXPECT_EQ(d[m], 0.25 * valueC(m)) << "alpha = 0, m = " << m;
            }
        }

        for(float const* cPtr : {static_cast<float const*>(dBadC.ptr), static_cast<float const*>(nullptr)})
        {
            dD.write(std::vector<float>(M, NAN));
            ASSERT_EQ(hiptensorReduce(api.handle,
                                      plan,
                                      &alpha,
                                      dPowers.ptr,
                                      &zero,
                                      cPtr,
                                      dD.ptr,
                                      nullptr,
                                      0,
                                      nullptr),
                      HIPTENSOR_STATUS_SUCCESS);
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            auto d = dD.read();
            for(int64_t m = 0; m < M; ++m)
            {
                double r = std::log(double(powers[m]));
                for(int64_t n = 1; n < N; ++n)
                {
                    double v = std::log(double(powers[m + M * n]));
                    r        = opReduce == HIPTENSOR_OP_ADD ? r + v : std::max(r, v);
                }
                EXPECT_NEAR(d[m], 2.0 * r, 1e-5) << "beta = 0, m = " << m;
            }
        }

        dD.write(std::vector<float>(M, NAN));
        ASSERT_EQ(hiptensorReduce(
                      api.handle, plan, &zero, nullptr, &zero, nullptr, dD.ptr, nullptr, 0, nullptr),
                  HIPTENSOR_STATUS_SUCCESS);
        CHECK_HIP_ERROR(hipDeviceSynchronize());
        EXPECT_EQ(dD.read(), std::vector<float>(M, 0.0f));

        EXPECT_EQ(hiptensorReduce(
                      api.handle, plan, &alpha, nullptr, &beta, dC.ptr, dD.ptr, nullptr, 0, nullptr),
                  HIPTENSOR_STATUS_NOT_INITIALIZED);
    }
}

// A reduction whose output keeps every mode of A: D = alpha * opA(A) + beta * opC(C) whatever the
// reduction operator, for rank 1, for an output in another mode order, and for BF16; each plan
// runs twice with the same result.
TEST(ReductionSemanticsTest, KeepingEveryModeIsElementwise)
{
    ApiObjects api;

    // Rank 1, f32, MAX.
    {
        constexpr int64_t M = 37;
        auto              desc    = api.tensor({M}, {1}, HIPTENSOR_R_32F);
        int32_t const     modes[] = {'m'};

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &op,
                                           desc,
                                           modes,
                                           HIPTENSOR_OP_NEG,
                                           desc,
                                           modes,
                                           HIPTENSOR_OP_IDENTITY,
                                           desc,
                                           modes,
                                           HIPTENSOR_OP_MAX,
                                           HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        hiptensorPlan_t plan = nullptr;
        ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

        std::vector<float> a(M), c(M);
        for(int64_t i = 0; i < M; ++i)
        {
            a[i] = static_cast<float>(valueA(i));
            c[i] = static_cast<float>(valueC(i));
        }
        DeviceBuffer<float> dA(a), dC(c), dD(std::vector<float>(M, NAN));
        float const         alpha = 1.5f, beta = -0.5f;
        for(int run = 0; run < 2; ++run)
        {
            ASSERT_EQ(hiptensorReduce(
                          api.handle, plan, &alpha, dA.ptr, &beta, dC.ptr, dD.ptr, nullptr, 0, nullptr),
                      HIPTENSOR_STATUS_SUCCESS)
                << "run " << run;
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            auto d = dD.read();
            for(int64_t i = 0; i < M; ++i)
            {
                EXPECT_EQ(d[i], 1.5 * -valueA(i) - 0.5 * valueC(i)) << "run " << run << ", " << i;
            }
        }
    }

    // Rank 2, BF16 with f32 compute, D in the other mode order: D[n, m] = 2 * A[m, n] + C[n, m].
    {
        constexpr int64_t M = 9, N = 7;
        auto              descA    = api.tensor({M, N}, {1, M}, HIPTENSOR_R_16BF);
        auto              descCD   = api.tensor({N, M}, {1, N}, HIPTENSOR_R_16BF);
        int32_t const     modeA[]  = {'m', 'n'};
        int32_t const     modeCD[] = {'n', 'm'};

        hiptensorOperationDescriptor_t op = nullptr;
        ASSERT_EQ(hiptensorCreateReduction(api.handle,
                                           &op,
                                           descA,
                                           modeA,
                                           HIPTENSOR_OP_IDENTITY,
                                           descCD,
                                           modeCD,
                                           HIPTENSOR_OP_IDENTITY,
                                           descCD,
                                           modeCD,
                                           HIPTENSOR_OP_ADD,
                                           HIPTENSOR_COMPUTE_DESC_32F),
                  HIPTENSOR_STATUS_SUCCESS);
        api.keep(op);
        hiptensorPlan_t plan = nullptr;
        ASSERT_EQ(api.plan(op, plan), HIPTENSOR_STATUS_SUCCESS);

        std::vector<hip_bfloat16> a(M * N), c(M * N);
        for(int64_t i = 0; i < M * N; ++i)
        {
            a[i] = static_cast<hip_bfloat16>(static_cast<float>(valueA(i)));
            c[i] = static_cast<hip_bfloat16>(static_cast<float>(valueC(i)));
        }
        DeviceBuffer<hip_bfloat16> dA(a), dC(c),
            dD(std::vector<hip_bfloat16>(M * N, static_cast<hip_bfloat16>(-100.0f)));
        float const alpha = 2.0f, beta = 1.0f;
        for(int run = 0; run < 2; ++run)
        {
            ASSERT_EQ(hiptensorReduce(
                          api.handle, plan, &alpha, dA.ptr, &beta, dC.ptr, dD.ptr, nullptr, 0, nullptr),
                      HIPTENSOR_STATUS_SUCCESS)
                << "run " << run;
            CHECK_HIP_ERROR(hipDeviceSynchronize());
            auto d = dD.read();
            for(int64_t m = 0; m < M; ++m)
            {
                for(int64_t n = 0; n < N; ++n)
                {
                    EXPECT_EQ(static_cast<float>(d[n + N * m]),
                              2.0 * valueA(m + M * n) + valueC(n + N * m))
                        << "run " << run << ", m = " << m << ", n = " << n;
                }
            }
        }
    }
}
