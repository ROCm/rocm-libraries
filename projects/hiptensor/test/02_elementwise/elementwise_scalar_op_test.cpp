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

// Regression tests for how the element-wise operations apply their scalars and operators:
//  - scalars passed with a BF16 compute descriptor were read as zero;
//  - the trinary operation applied opAB and opABC in swapped positions;
//  - a permutation plan first executed with alpha == 1 kept its pass-through kernel when it was
//    executed again with another alpha under HIPTENSOR_CACHE_MODE_NONE.
// The expected results are computed here with a plain loop instead of the library's CPU
// reference, which decodes the scalars and orders the operators the same way as the device path
// and so cannot see these defects.
//
// Every input is a small integer and every scalar a short dyadic fraction, so each intermediate
// value is exact in the tensor's data type and the results are compared exactly.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr int64_t M            = 8;
    constexpr int64_t N            = 6;
    constexpr int64_t elementCount = M * N;

    // Every tensor is {m, n} with packed column-major strides, given explicitly so the layout
    // does not depend on the library default. Element i of every tensor is then the same (m, n).
    std::vector<int64_t> const extents{M, N};
    std::vector<int64_t> const strides{1, M};
    std::vector<int32_t> const modes{'m', 'n'};

    // Small integers with different periods, so A, B and C do not line up.
    double valueA(int64_t i)
    {
        return static_cast<double>(i % 7 - 3);
    }
    double valueB(int64_t i)
    {
        return static_cast<double>((3 * i) % 5 - 2);
    }
    double valueC(int64_t i)
    {
        return static_cast<double>((5 * i) % 9 - 4);
    }

    double applyBinary(hiptensorOperator_t op, double x, double y)
    {
        switch(op)
        {
        case HIPTENSOR_OP_ADD:
            return x + y;
        case HIPTENSOR_OP_MUL:
            return x * y;
        case HIPTENSOR_OP_MAX:
            return std::max(x, y);
        case HIPTENSOR_OP_MIN:
            return std::min(x, y);
        default:
            ADD_FAILURE() << "Unexpected binary operator " << op;
            return 0.0;
        }
    }

    template <typename T>
    std::vector<T> hostTensor(double (*value)(int64_t))
    {
        std::vector<T> host;
        host.reserve(elementCount);
        for(int64_t i = 0; i < elementCount; ++i)
        {
            host.push_back(static_cast<T>(static_cast<float>(value(i))));
        }
        return host;
    }

    template <typename T>
    struct DeviceBuffer
    {
        explicit DeviceBuffer(std::vector<T> const& host)
            : count(host.size())
        {
            CHECK_HIP_ERROR(hipMalloc(&ptr, count * sizeof(T)));
            write(host);
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

        void write(std::vector<T> const& host)
        {
            CHECK_HIP_ERROR(hipMemcpy(ptr, host.data(), count * sizeof(T), hipMemcpyHostToDevice));
        }
        std::vector<T> read() const
        {
            std::vector<T> host(count);
            CHECK_HIP_ERROR(hipMemcpy(host.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
            return host;
        }

        T*     ptr = nullptr;
        size_t count;
    };

    // Owns the hipTensor objects of one operation and destroys them on every exit path.
    struct Operation
    {
        hiptensorHandle_t                        handle = nullptr;
        std::vector<hiptensorTensorDescriptor_t> tensors;
        hiptensorOperationDescriptor_t           desc = nullptr;
        hiptensorPlanPreference_t                pref = nullptr;
        hiptensorPlan_t                          plan = nullptr;

        Operation()
        {
            CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));
        }
        ~Operation()
        {
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
            for(auto tensor : tensors)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(tensor));
            }
            static_cast<void>(hiptensorDestroy(handle));
        }
        Operation(Operation const&)            = delete;
        Operation& operator=(Operation const&) = delete;

        hiptensorTensorDescriptor_t tensor(std::vector<int64_t> const& tensorExtents,
                                           std::vector<int64_t> const& tensorStrides,
                                           hiptensorDataType_t         type)
        {
            hiptensorTensorDescriptor_t tensorDesc = nullptr;
            CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                                  &tensorDesc,
                                                                  tensorExtents.size(),
                                                                  tensorExtents.data(),
                                                                  tensorStrides.data(),
                                                                  type,
                                                                  0));
            tensors.push_back(tensorDesc);
            return tensorDesc;
        }

        void createPlan(hiptensorCacheMode_t cacheMode)
        {
            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
                handle, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
            CHECK_HIPTENSOR_ERROR(hiptensorPlanPreferenceSetAttribute(
                handle, pref, HIPTENSOR_PLAN_PREFERENCE_CACHE_MODE, &cacheMode, sizeof(cacheMode)));
            CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, desc, pref, 0));
        }
    };

    template <typename T>
    void expectExact(std::vector<T> const&      actual,
                     std::vector<double> const& expected,
                     std::string const&         what)
    {
        ASSERT_EQ(actual.size(), expected.size()) << what;
        size_t mismatches = 0;
        size_t first      = 0;
        for(size_t i = 0; i < actual.size(); ++i)
        {
            if(static_cast<double>(static_cast<float>(actual[i])) != expected[i])
            {
                if(mismatches++ == 0)
                {
                    first = i;
                }
            }
        }
        EXPECT_EQ(mismatches, 0u) << what << ": first mismatch at element " << first << ", got "
                                  << static_cast<float>(actual[first]) << ", expected "
                                  << expected[first];
    }
} // namespace

// D = opAC(alpha * A, gamma * C) on BF16 tensors with a BF16 compute descriptor. Before the fix
// alpha and gamma were read as zero, so every output was opAC(0, 0) = 0.
TEST(ElementwiseBf16ScalarTest, BinaryAppliesBf16ComputeScalars)
{
    for(auto opAC : {HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX})
    {
        DeviceBuffer<hip_bfloat16> dA(hostTensor<hip_bfloat16>(valueA));
        DeviceBuffer<hip_bfloat16> dC(hostTensor<hip_bfloat16>(valueC));
        DeviceBuffer<hip_bfloat16> dD(
            std::vector<hip_bfloat16>(elementCount, static_cast<hip_bfloat16>(-100.0f)));

        Operation op;
        auto      descA = op.tensor(extents, strides, HIPTENSOR_R_16BF);
        auto      descC = op.tensor(extents, strides, HIPTENSOR_R_16BF);
        auto      descD = op.tensor(extents, strides, HIPTENSOR_R_16BF);
        CHECK_HIPTENSOR_ERROR(hiptensorCreateElementwiseBinary(op.handle,
                                                               &op.desc,
                                                               descA,
                                                               modes.data(),
                                                               HIPTENSOR_OP_IDENTITY,
                                                               descC,
                                                               modes.data(),
                                                               HIPTENSOR_OP_IDENTITY,
                                                               descD,
                                                               modes.data(),
                                                               opAC,
                                                               HIPTENSOR_COMPUTE_DESC_16BF));
        op.createPlan(HIPTENSOR_CACHE_MODE_PEDANTIC);

        // Scalars are passed in the compute type.
        auto alpha = static_cast<hip_bfloat16>(1.25f);
        auto gamma = static_cast<hip_bfloat16>(0.5f);
        CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryExecute(
            op.handle, op.plan, &alpha, dA.ptr, &gamma, dC.ptr, dD.ptr, hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        std::vector<double> expected(elementCount);
        for(int64_t i = 0; i < elementCount; ++i)
        {
            expected[i] = applyBinary(opAC, 1.25 * valueA(i), 0.5 * valueC(i));
        }
        expectExact(dD.read(), expected, "opAC = " + std::to_string(opAC));
    }
}

// The documented trinary operation is D = opABC(opAB(alpha * A, beta * B), gamma * C). It used
// to compute opAB(opABC(alpha * A, beta * B), gamma * C), which only agrees when the two
// operators are the same, so every pair here has opAB != opABC.
class ElementwiseTrinaryOpOrderTest
    : public ::testing::TestWithParam<std::pair<hiptensorOperator_t, hiptensorOperator_t>>
{
};

TEST_P(ElementwiseTrinaryOpOrderTest, AppliesOpABBeforeOpABC)
{
    auto const [opAB, opABC] = GetParam();

    float const alpha = 2.0f;
    float const beta  = 1.0f;
    float const gamma = 0.5f;

    std::vector<double> expected(elementCount);
    std::vector<double> swapped(elementCount);
    for(int64_t i = 0; i < elementCount; ++i)
    {
        auto a      = alpha * valueA(i);
        auto b      = beta * valueB(i);
        auto c      = gamma * valueC(i);
        expected[i] = applyBinary(opABC, applyBinary(opAB, a, b), c);
        swapped[i]  = applyBinary(opAB, applyBinary(opABC, a, b), c);
    }
    // The data has to tell the two orders apart for the check to mean anything.
    ASSERT_NE(expected, swapped);

    DeviceBuffer<float> dA(hostTensor<float>(valueA));
    DeviceBuffer<float> dB(hostTensor<float>(valueB));
    DeviceBuffer<float> dC(hostTensor<float>(valueC));
    DeviceBuffer<float> dD(std::vector<float>(elementCount, -100.0f));

    Operation op;
    auto      descA = op.tensor(extents, strides, HIPTENSOR_R_32F);
    auto      descB = op.tensor(extents, strides, HIPTENSOR_R_32F);
    auto      descC = op.tensor(extents, strides, HIPTENSOR_R_32F);
    auto      descD = op.tensor(extents, strides, HIPTENSOR_R_32F);
    CHECK_HIPTENSOR_ERROR(hiptensorCreateElementwiseTrinary(op.handle,
                                                            &op.desc,
                                                            descA,
                                                            modes.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descB,
                                                            modes.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descC,
                                                            modes.data(),
                                                            HIPTENSOR_OP_IDENTITY,
                                                            descD,
                                                            modes.data(),
                                                            opAB,
                                                            opABC,
                                                            HIPTENSOR_COMPUTE_DESC_32F));
    op.createPlan(HIPTENSOR_CACHE_MODE_PEDANTIC);

    CHECK_HIPTENSOR_ERROR(hiptensorElementwiseTrinaryExecute(op.handle,
                                                             op.plan,
                                                             &alpha,
                                                             dA.ptr,
                                                             &beta,
                                                             dB.ptr,
                                                             &gamma,
                                                             dC.ptr,
                                                             dD.ptr,
                                                             hipStreamPerThread));
    CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

    expectExact(dD.read(),
                expected,
                "opAB = " + std::to_string(opAB) + ", opABC = " + std::to_string(opABC));
}

INSTANTIATE_TEST_SUITE_P(DistinctOperatorPairs,
                         ElementwiseTrinaryOpOrderTest,
                         ::testing::Values(std::pair{HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL},
                                           std::pair{HIPTENSOR_OP_MUL, HIPTENSOR_OP_ADD},
                                           std::pair{HIPTENSOR_OP_MAX, HIPTENSOR_OP_ADD},
                                           std::pair{HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX},
                                           std::pair{HIPTENSOR_OP_MIN, HIPTENSOR_OP_MUL}));

// B = alpha * A^T executed several times through one plan. The first execution has alpha == 1,
// for which the library picks a pass-through kernel that cannot scale. The plan keeps that kernel
// for later executions, so with HIPTENSOR_CACHE_MODE_NONE the later alphas used to be ignored.
TEST(PermutationPlanReuseTest, AlphaChangeAfterPassThroughUnderCacheModeNone)
{
    std::vector<int64_t> const extentsB{N, M};
    std::vector<int64_t> const stridesB{1, N};
    std::vector<int32_t> const modesB{'n', 'm'};
    std::vector<float> const   sentinel(elementCount, -100.0f);

    DeviceBuffer<float> dA(hostTensor<float>(valueA));
    DeviceBuffer<float> dB(sentinel);

    Operation op;
    auto      descA = op.tensor(extents, strides, HIPTENSOR_R_32F);
    auto      descB = op.tensor(extentsB, stridesB, HIPTENSOR_R_32F);
    CHECK_HIPTENSOR_ERROR(hiptensorCreatePermutation(op.handle,
                                                     &op.desc,
                                                     descA,
                                                     modes.data(),
                                                     HIPTENSOR_OP_IDENTITY,
                                                     descB,
                                                     modesB.data(),
                                                     HIPTENSOR_COMPUTE_DESC_32F));
    op.createPlan(HIPTENSOR_CACHE_MODE_NONE);

    for(float alpha : {1.0f, 2.5f, -0.5f, 1.0f})
    {
        dB.write(sentinel);
        CHECK_HIPTENSOR_ERROR(
            hiptensorPermute(op.handle, op.plan, &alpha, dA.ptr, dB.ptr, hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        std::vector<double> expected(elementCount);
        for(int64_t m = 0; m < M; ++m)
        {
            for(int64_t n = 0; n < N; ++n)
            {
                expected[n + N * m] = alpha * valueA(m + M * n);
            }
        }
        expectExact(dB.read(), expected, "alpha = " + std::to_string(alpha));
    }
}
