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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Regression test for scalars passed with a BF16 compute descriptor being read as zero, which made
// a BF16 reduction return D = 0 * reduce(A) + 0 * C. The expected result is computed here with a
// plain loop instead of the library's CPU reference, which decodes the scalars the same way as
// the device path and so returned zero as well.
//
// Every input is a small integer and both scalars are short dyadic fractions, so the reduced
// value and the scaled sum are exact in BF16 and the results are compared exactly.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace
{
    template <typename T>
    struct DeviceBuffer
    {
        explicit DeviceBuffer(std::vector<T> const& host)
            : count(host.size())
        {
            CHECK_HIP_ERROR(hipMalloc(&ptr, count * sizeof(T)));
            CHECK_HIP_ERROR(hipMemcpy(ptr, host.data(), count * sizeof(T), hipMemcpyHostToDevice));
        }
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;

        std::vector<T> read() const
        {
            std::vector<T> host(count);
            CHECK_HIP_ERROR(hipMemcpy(host.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
            return host;
        }

        T*     ptr = nullptr;
        size_t count;
    };

    // Owns the hipTensor objects and the workspace of one reduction and releases them on every
    // exit path.
    struct Reduction
    {
        hiptensorHandle_t              handle   = nullptr;
        hiptensorTensorDescriptor_t    descA    = nullptr;
        hiptensorTensorDescriptor_t    descCD   = nullptr;
        hiptensorOperationDescriptor_t desc     = nullptr;
        hiptensorPlanPreference_t      pref     = nullptr;
        hiptensorPlan_t                plan     = nullptr;
        void*                          work     = nullptr;
        uint64_t                       workSize = 0;

        Reduction()
        {
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
        }
        ~Reduction()
        {
            if(work)
            {
                static_cast<void>(hipFree(work));
            }
            if(plan)
            {
                static_cast<void>(hiptensorDestroyPlan(plan));
            }
            if(pref)
            {
                static_cast<void>(hiptensorDestroyPlanPreference(pref));
            }
            if(desc)
            {
                static_cast<void>(hiptensorDestroyOperationDescriptor(desc));
            }
            if(descCD)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(descCD));
            }
            if(descA)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(descA));
            }
            static_cast<void>(hiptensorDestroy(handle));
        }
        Reduction(Reduction const&)            = delete;
        Reduction& operator=(Reduction const&) = delete;
    };
} // namespace

// D = alpha * reduce_n(A) + beta * C for BF16 tensors with a BF16 compute descriptor, where A is
// {m, n} and C and D are {m}.
TEST(ReductionBf16ScalarTest, AppliesBf16ComputeScalars)
{
    constexpr int64_t M = 6;
    constexpr int64_t N = 5;

    std::vector<int64_t> const extentA{M, N};
    std::vector<int64_t> const stridesA{1, M};
    std::vector<int32_t> const modeA{'m', 'n'};
    std::vector<int64_t> const extentCD{M};
    std::vector<int64_t> const stridesCD{1};
    std::vector<int32_t> const modeCD{'m'};

    // Small integers in [-3, 3] and [-4, 4].
    auto valueA = [](int64_t i) { return static_cast<double>(i % 7 - 3); };
    auto valueC = [](int64_t i) { return static_cast<double>((5 * i) % 9 - 4); };

    std::vector<hip_bfloat16> hostA, hostC;
    for(int64_t i = 0; i < M * N; ++i)
    {
        hostA.push_back(static_cast<hip_bfloat16>(static_cast<float>(valueA(i))));
    }
    for(int64_t i = 0; i < M; ++i)
    {
        hostC.push_back(static_cast<hip_bfloat16>(static_cast<float>(valueC(i))));
    }

    for(auto opReduce : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX})
    {
        DeviceBuffer<hip_bfloat16> dA(hostA);
        DeviceBuffer<hip_bfloat16> dC(hostC);
        DeviceBuffer<hip_bfloat16> dD(
            std::vector<hip_bfloat16>(M, static_cast<hip_bfloat16>(-100.0f)));

        Reduction r;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(r.handle,
                                                              &r.descA,
                                                              extentA.size(),
                                                              extentA.data(),
                                                              stridesA.data(),
                                                              HIPTENSOR_R_16BF,
                                                              0));
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(r.handle,
                                                              &r.descCD,
                                                              extentCD.size(),
                                                              extentCD.data(),
                                                              stridesCD.data(),
                                                              HIPTENSOR_R_16BF,
                                                              0));
        CHECK_HIPTENSOR_ERROR(hiptensorCreateReduction(r.handle,
                                                       &r.desc,
                                                       r.descA,
                                                       modeA.data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       r.descCD,
                                                       modeCD.data(),
                                                       HIPTENSOR_OP_IDENTITY,
                                                       r.descCD,
                                                       modeCD.data(),
                                                       opReduce,
                                                       HIPTENSOR_COMPUTE_DESC_16BF));
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
            r.handle, &r.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
        CHECK_HIPTENSOR_ERROR(hiptensorEstimateWorkspaceSize(
            r.handle, r.desc, r.pref, HIPTENSOR_WORKSPACE_DEFAULT, &r.workSize));
        if(r.workSize > 0)
        {
            CHECK_HIP_ERROR(hipMalloc(&r.work, r.workSize));
        }
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(r.handle, &r.plan, r.desc, r.pref, r.workSize));

        // Scalars are passed in the compute type.
        auto alpha = static_cast<hip_bfloat16>(1.25f);
        auto beta  = static_cast<hip_bfloat16>(0.5f);
        CHECK_HIPTENSOR_ERROR(hiptensorReduce(r.handle,
                                              r.plan,
                                              &alpha,
                                              dA.ptr,
                                              &beta,
                                              dC.ptr,
                                              dD.ptr,
                                              r.work,
                                              r.workSize,
                                              hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        auto const actual = dD.read();
        for(int64_t m = 0; m < M; ++m)
        {
            double reduced = valueA(m);
            for(int64_t n = 1; n < N; ++n)
            {
                auto a  = valueA(m + M * n);
                reduced = opReduce == HIPTENSOR_OP_ADD ? reduced + a : std::max(reduced, a);
            }
            double expected = 1.25 * reduced + 0.5 * valueC(m);
            EXPECT_EQ(static_cast<double>(static_cast<float>(actual[m])), expected)
                << "opReduce = " << opReduce << ", m = " << m;
        }
    }
}
