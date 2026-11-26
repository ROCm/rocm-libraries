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

// hiptensorElementwiseTrinaryExecute computes
//     D = opABC(opAB(alpha * opA(A), beta * opB(B)), gamma * opC(C)).
// The expected result is computed on the host with a plain loop rather than with the library's
// CPU reference, which builds its operator from the same arguments as the device path. Each case
// uses distinct opAB and opABC so that applying them in the wrong positions changes the result.
//
// Every input is a small integer and every scalar a power of two, so each intermediate value is
// exact in FP32 and the results are compared exactly.

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr int64_t  M            = 8;
    constexpr int64_t  N            = 6;
    constexpr int64_t  elementCount = M * N;
    constexpr uint32_t alignment    = 256;

    // Every tensor is {m, n} with packed column-major strides, given explicitly so the layout does
    // not depend on the library default. Element i of every tensor is then the same (m, n).
    int64_t const extents[2] = {M, N};
    int64_t const strides[2] = {1, M};
    int32_t const modes[2]   = {'m', 'n'};

    // Small integers with different periods, so A, B, and C do not line up.
    float valueA(int64_t i)
    {
        return static_cast<float>(i % 7 - 3);
    }
    float valueB(int64_t i)
    {
        return static_cast<float>((3 * i) % 5 - 2);
    }
    float valueC(int64_t i)
    {
        return static_cast<float>((5 * i) % 9 - 4);
    }

    float applyBinary(hiptensorOperator_t op, float x, float y)
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
            ADD_FAILURE() << "unexpected binary operator " << op;
            return 0.0f;
        }
    }
} // namespace

class ElementwiseTrinaryOpOrderTest
    : public ::testing::TestWithParam<std::pair<hiptensorOperator_t, hiptensorOperator_t>>
{
protected:
    void SetUp() override
    {
        ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hiptensorCreateTensorDescriptor(
                      mHandle, &mDesc, 2, extents, strides, HIPTENSOR_R_32F, alignment),
                  HIPTENSOR_STATUS_SUCCESS);
        for(auto* buffer : {&mA, &mB, &mC, &mD})
        {
            ASSERT_EQ(hipMalloc(buffer, elementCount * sizeof(float)), hipSuccess);
        }
    }

    void TearDown() override
    {
        for(auto* buffer : {mA, mB, mC, mD})
        {
            if(buffer)
            {
                EXPECT_EQ(hipFree(buffer), hipSuccess);
            }
        }
        if(mPlan)
        {
            hiptensorDestroyPlan(mPlan);
        }
        if(mPlanPref)
        {
            hiptensorDestroyPlanPreference(mPlanPref);
        }
        if(mOpDesc)
        {
            hiptensorDestroyOperationDescriptor(mOpDesc);
        }
        if(mDesc)
        {
            hiptensorDestroyTensorDescriptor(mDesc);
        }
        if(mHandle)
        {
            hiptensorDestroy(mHandle);
        }
    }

    hiptensorHandle_t              mHandle{};
    hiptensorTensorDescriptor_t    mDesc{};
    hiptensorOperationDescriptor_t mOpDesc{};
    hiptensorPlanPreference_t      mPlanPref{};
    hiptensorPlan_t                mPlan{};

    void* mA{};
    void* mB{};
    void* mC{};
    void* mD{};
};

TEST_P(ElementwiseTrinaryOpOrderTest, AppliesOpABBeforeOpABC)
{
    auto const [opAB, opABC] = GetParam();

    float const alpha = 2.0f;
    float const beta  = 1.0f;
    float const gamma = 0.5f;

    std::vector<float> hostA(elementCount);
    std::vector<float> hostB(elementCount);
    std::vector<float> hostC(elementCount);
    std::vector<float> expected(elementCount);
    std::vector<float> swapped(elementCount);
    for(int64_t i = 0; i < elementCount; ++i)
    {
        hostA[i]    = valueA(i);
        hostB[i]    = valueB(i);
        hostC[i]    = valueC(i);
        auto a      = alpha * hostA[i];
        auto b      = beta * hostB[i];
        auto c      = gamma * hostC[i];
        expected[i] = applyBinary(opABC, applyBinary(opAB, a, b), c);
        swapped[i]  = applyBinary(opAB, applyBinary(opABC, a, b), c);
    }
    // The data has to tell the two orders apart for the check to mean anything.
    ASSERT_NE(expected, swapped);

    size_t const bytes = elementCount * sizeof(float);
    ASSERT_EQ(hipMemcpy(mA, hostA.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(mB, hostB.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
    ASSERT_EQ(hipMemcpy(mC, hostC.data(), bytes, hipMemcpyHostToDevice), hipSuccess);
    std::vector<float> result(elementCount, -100.0f);
    ASSERT_EQ(hipMemcpy(mD, result.data(), bytes, hipMemcpyHostToDevice), hipSuccess);

    ASSERT_EQ(hiptensorCreateElementwiseTrinary(mHandle,
                                                &mOpDesc,
                                                mDesc,
                                                modes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mDesc,
                                                modes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mDesc,
                                                modes,
                                                HIPTENSOR_OP_IDENTITY,
                                                mDesc,
                                                modes,
                                                opAB,
                                                opABC,
                                                HIPTENSOR_COMPUTE_DESC_32F),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreatePlanPreference(
                  mHandle, &mPlanPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hiptensorCreatePlan(mHandle, &mPlan, mOpDesc, mPlanPref, 0),
              HIPTENSOR_STATUS_SUCCESS);

    ASSERT_EQ(hiptensorElementwiseTrinaryExecute(
                  mHandle, mPlan, &alpha, mA, &beta, mB, &gamma, mC, mD, nullptr),
              HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    ASSERT_EQ(hipMemcpy(result.data(), mD, bytes, hipMemcpyDeviceToHost), hipSuccess);

    for(int64_t i = 0; i < elementCount; ++i)
    {
        EXPECT_EQ(result[i], expected[i])
            << "element " << i << " (opAB = " << opAB << ", opABC = " << opABC << ")";
    }
}

INSTANTIATE_TEST_SUITE_P(DistinctOperatorPairs,
                         ElementwiseTrinaryOpOrderTest,
                         ::testing::Values(std::pair{HIPTENSOR_OP_ADD, HIPTENSOR_OP_MUL},
                                           std::pair{HIPTENSOR_OP_MUL, HIPTENSOR_OP_ADD},
                                           std::pair{HIPTENSOR_OP_MAX, HIPTENSOR_OP_ADD},
                                           std::pair{HIPTENSOR_OP_ADD, HIPTENSOR_OP_MAX},
                                           std::pair{HIPTENSOR_OP_MIN, HIPTENSOR_OP_MUL}));
